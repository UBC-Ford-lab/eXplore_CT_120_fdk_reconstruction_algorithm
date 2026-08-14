"""
TIGRE-based iterative cone-beam CT reconstruction.

Provides OS-SART, SART, and SIRT reconstruction using the TIGRE toolbox
as an alternative backend to ASTRA. TIGRE handles GPU memory splitting
internally, potentially enabling larger-than-VRAM reconstructions on
memory-constrained GPUs.

The geometry mapping translates the FDK backprojection coordinate convention
(fdk.py lines 598-612) into TIGRE's geometry object.

Requires: tigre (optional dependency, built from source)
"""

import copy
import json
import time
from pathlib import Path

import numpy as np

try:
    import tigre
    from tigre.utilities.geometry import Geometry as TigreGeometry
    import tigre.algorithms as tigre_algs
    from tigre.utilities.im_3d_denoise import im3ddenoise as _im3ddenoise
except ImportError:
    tigre = None

try:
    from skimage.metrics import structural_similarity as _ssim_fn
except ImportError:
    _ssim_fn = None

from ...ct_core.calibration import default_mu_water, mu_to_hu
from ...ct_core.preprocessing import preprocess_sinogram
from ...ct_core.utils import query_gpu_memory

SUPPORTED_TIGRE_ALGORITHMS = ('ossart', 'sart', 'sirt', 'mlem')

_TIGRE_ALG_FUNCS = {
    'ossart': lambda: tigre_algs.ossart,
    'sart': lambda: tigre_algs.sart,
    'sirt': lambda: tigre_algs.sirt,
    'mlem': lambda: tigre_algs.mlem,
}


def _check_tigre_available():
    """Raise ImportError if TIGRE is not installed."""
    if tigre is None:
        raise ImportError(
            "TIGRE is required for TIGRE-based iterative reconstruction. "
            "Build from source: cd TIGRE && pip install --no-build-isolation ."
        )


def fdk_angles_to_tigre(angles_fdk):
    """
    Convert FDK projection angles to TIGRE angle convention.

    FDK convention (fdk.py:598-612):
        At beta=0: source at (-R_s, 0, 0), detector at (+R_d, 0, 0).
        Source position: (-R_s*cos(beta), -R_s*sin(beta), 0)

    TIGRE convention (plot_geometry.py:102):
        At alpha=0: source at (+DSO, 0, 0).
        Source position: (DSO*cos(alpha), DSO*sin(alpha), 0)

    Matching source positions:
        -R_s*cos(beta) = DSO*cos(alpha)  =>  cos(alpha) = -cos(beta) = cos(beta + pi)
        -R_s*sin(beta) = DSO*sin(alpha)  =>  sin(alpha) = -sin(beta) = sin(beta + pi)

    This gives: alpha = beta + pi

    Args:
        angles_fdk: 1D array of FDK projection angles in radians

    Returns:
        np.ndarray of TIGRE angles in radians
    """
    return np.asarray(angles_fdk, dtype=np.float64) + np.pi


def build_tigre_geometry(geometry, N_b, N_a, detector_psi_deg=None, cpa_raw=None):
    """
    Build a TIGRE Geometry object from the FDK geometry dict.

    TIGRE uses [z, y, x] ordering for volumes and [rows, cols] for detectors.

    Args:
        geometry: dict with R_s, R_d, da, db, vol_shape, dx, dz
        N_b: Number of detector rows (vertical)
        N_a: Number of detector columns (horizontal)

    Returns:
        tigre.utilities.geometry.Geometry
    """
    _check_tigre_available()

    geo = TigreGeometry()

    # Source-detector distances (mm) — must be Python floats, not arrays
    R_s = float(geometry['R_s'])
    R_d = float(geometry['R_d'])
    geo.DSD = R_s + R_d     # Source-to-detector distance
    geo.DSO = R_s            # Source-to-origin (isocenter) distance

    # Detector grid: [rows, cols] = [N_b, N_a]
    geo.nDetector = np.array([N_b, N_a], dtype=np.int64)
    geo.dDetector = np.array([geometry['db'], geometry['da']], dtype=np.float64)
    geo.sDetector = geo.nDetector * geo.dDetector

    # Volume: TIGRE uses [Nz, Ny, Nx] ordering
    Nx, Ny, Nz = geometry['vol_shape']
    dx = geometry['dx']
    dz = geometry['dz']

    # TIGRE CUDA kernels hang with non-zero geo.offOrigin. Workaround:
    # always reconstruct a centered volume (offOrigin=[0,0,0]) large enough
    # to contain the ROI, then crop to the ROI region after reconstruction.
    vol_origin = geometry.get('vol_origin', (0, 0, 0))
    ox, oy, oz = vol_origin

    if ox != 0 or oy != 0 or oz != 0:
        Nx_roi, Ny_roi, Nz_roi = Nx, Ny, Nz
        Nx = Nx_roi + 2 * int(np.ceil(abs(ox) / dx))
        Ny = Ny_roi + 2 * int(np.ceil(abs(oy) / dx))
        Nz = Nz_roi + 2 * int(np.ceil(abs(oz) / dz))
        print(f"  TIGRE offOrigin bug workaround: expanding to centered volume")
        print(f"    ROI center: ({ox:.2f}, {oy:.2f}, {oz:.2f}) mm")
        print(f"    ROI dims:   ({Nx_roi}, {Ny_roi}, {Nz_roi})")
        print(f"    Expanded:   ({Nx}, {Ny}, {Nz}) (centered at isocenter)")

    # TIGRE CUDA kernels hang when Nxy is below an empirical threshold.
    # Tested on 3500x2296 detector, 0.075mm voxels:
    #   Nxy=879 (Nz=764) → HANG, Nxy=900 (Nz=933) → HANG,
    #   Nxy=933 (Nz=933) → HANG, Nxy=1000 (Nz=933) → PASS.
    Nxy_min = max(1000, Nz)
    if min(Nx, Ny) < Nxy_min:
        Nx_before_pad, Ny_before_pad = Nx, Ny
        Nx = max(Nx, Nxy_min)
        Ny = max(Ny, Nxy_min)
        print(f"  WARNING: TIGRE CUDA kernels hang for Nxy < ~1000.")
        print(f"  Auto-padding volume from ({Nx_before_pad}, {Ny_before_pad}, {Nz}) "
              f"to ({Nx}, {Ny}, {Nz}).")

    geo.nVoxel = np.array([Nz, Ny, Nx], dtype=np.int64)
    geo.dVoxel = np.array([dz, dx, dx], dtype=np.float64)
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    # Always zero offOrigin — non-zero values cause TIGRE CUDA hangs.
    # ROI offset is handled by expanding the centered volume and cropping after.
    geo.offOrigin = np.array([0.0, 0.0, 0.0])
    geo.offDetector = np.array([0.0, 0.0])

    # ---- detector in-plane rotation + column centre of rotation -------------
    # Before 2026-08-11 both were hard-zero here, i.e. TIGRE assumed a perfectly
    # square, perfectly centred detector. muNeRF measures both reference-free
    # from the projections (ct_core/detector_psi.py); passing them in makes the
    # iterative pipeline reconstruct on the SAME geometry muNeRF does instead of
    # a different assumed one.
    #
    # CONVENTIONS VERIFIED EMPIRICALLY against this TIGRE build (a 5 deg roll of
    # an off-centre marker moved it (+1.495, +0.070) px where an in-plane roll
    # predicts (+1.482, +0.065); components [1] and [2] barely moved it):
    #   rotDetector[0]  = IN-PLANE ROLL about the detector normal  == psi
    #   offDetector[1]  = detector-space COLUMN shift in mm; +D mm moves the
    #                     projected image by -D/dDetector columns, so the iso
    #                     ray lands at centre - D/d. To put it at cpa_raw:
    #                         offDetector[1] = (centre_raw - cpa_raw) * da_raw
    #   (COR is an OBJECT-space axis shift and comes out magnified by DSD/DSO —
    #    measured -5.61 px for +2 mm vs offDetector's -5.00 — so it is NOT the
    #    right knob for a detector-referenced column centre.)
    #
    # detector_psi_deg / cpa_raw of None reproduce the pre-2026-08-11 behaviour
    # EXACTLY (both terms zero) — that is the revert path.
    psi = float(detector_psi_deg or 0.0)
    geo.rotDetector = np.array([np.radians(psi), 0.0, 0.0])
    if psi:
        print(f"  Detector in-plane rotation: psi = {psi:+.4f} deg "
              f"(rotDetector[0])")
    if cpa_raw is not None:
        centre_raw = (N_a - 1) / 2.0
        off_col_mm = (centre_raw - float(cpa_raw)) * float(geometry['da'])
        geo.offDetector = np.array([0.0, off_col_mm])
        print(f"  Column CoR: cpa {float(cpa_raw):.2f} raw "
              f"({float(cpa_raw) - centre_raw:+.2f} from centre) -> "
              f"offDetector[1] = {off_col_mm:+.5f} mm")

    # Accuracy for ray-tracing interpolation
    geo.accuracy = 0.5

    # Center of rotation correction
    geo.COR = 0.0

    geo.mode = 'cone'

    return geo


def _pwls_weight(sinogram, geo, angles, gpuids=None):
    """
    Combined geometric x statistical per-ray weight array for PWLS-SIRT,
    same shape as `sinogram` (N_angles, N_b, N_a). Pass as the `W` kwarg to
    tigre_algs.sirt/sart/ossart, which otherwise computes a purely-geometric
    W internally (IterativeReconAlg.set_w()) — this augments that geometric
    conditioning with a data-driven confidence weight instead of replacing it.

    Statistical factor: T_i = exp(-p_i) recovers the (possibly soft-clipped)
    transmission that produced line integral p_i = -ln(T_i) in
    preprocess_sinogram, directly from the sinogram actually being
    reconstructed — no need to re-derive from bright/dark fields. Physically,
    expected photon count on ray i is N_i = I_0 * T_i, and quantum-noise
    variance on p_i is ~1/N_i, so the statistically-motivated weight is
    w_i ~ N_i = I_0 * T_i. I_0 is a single global scale constant (flat-field
    correction already removed per-pixel gain nonuniformity), and the PWLS
    objective (Ax-p)^T diag(w) (Ax-p) is invariant to a uniform positive
    rescaling of w, so I_0 cancels out under the mean-normalization below and
    is never needed explicitly.

    Geometric factor: replicated verbatim from TIGRE's
    IterativeReconAlg.set_w() (tigre/algorithms/iterative_recon_alg.py) so
    that PWLS mode uses the identical per-ray path-length normalization
    plain SIRT/SART would use by default. Dropping it in favor of the
    statistical factor alone would destabilize convergence, since raw
    photon-count-scale weights span many orders of magnitude across a real
    sinogram.
    """
    T = np.clip(np.exp(-sinogram.astype(np.float64)), 1e-6, 2.0)
    w_stat = (T / T.mean()).astype(np.float32)

    geox = copy.deepcopy(geo)
    geox.sVoxel[1:] = geox.sVoxel[1:] * 1.1
    geox.sVoxel[0] = max(geox.sDetector[0], geox.sVoxel[0])
    geox.nVoxel = np.array([2, 2, 2])
    geox.dVoxel = geox.sVoxel / geox.nVoxel
    W_geom = tigre.Ax(np.ones(geox.nVoxel, dtype=np.float32), geox, angles,
                       "interpolated", gpuids=gpuids)
    W_geom[W_geom <= min(geo.dVoxel / 2)] = np.inf
    W_geom = 1.0 / W_geom

    return (W_geom * w_stat).astype(np.float32)


class TIGREReconstructor:
    """
    Iterative cone-beam CT reconstruction using the TIGRE toolbox.

    Supports OS-SART, SART, SIRT, and MLEM algorithms. TIGRE handles GPU
    memory splitting internally, enabling reconstruction of volumes that
    exceed GPU VRAM.

    The output volume convention matches FDK: (Nx, Ny, Nz) with voxel coordinates.
    """

    def __init__(self, projections, angles, geometry,
                 algorithm='ossart', iterations=100,
                 blocksize=15, lmbda=0.5, lmbda_red=0.97,
                 nonneg=True, gpu_index=0,
                 bright_field=None, dark_field=None,
                 geometry_autocal=True, detector_psi_deg=None,
                 clamp_mode='none', soft_clip_transmission=True,
                 soft_clip_sharpness=50.0, upper_clamp=True,
                 upper_clamp_value=1.05,
                 mu_water=None, output_hu=True,
                 bhc_coeffs=None,
                 ring_correction=False, ring_median_width=51,
                 air_normalization=True,
                 crossval=True, holdout_index=None,
                 withhold_eval=False,
                 eval_every=10, patience=3,
                 tv_lambda=0.0, tv_iters=50,
                 pwls=False,
                 checkpoint_dir=None, checkpoint_z_range=None,
                 checkpoint_xy_range=None,
                 log_fn=None, diag_fn=None):
        """
        Args:
            projections: Raw projections, shape (N_angles, N_b, N_a)
            angles: Projection angles in radians (FDK convention), shape (N_angles,)
            geometry: dict with R_s, R_d, da, db, vol_shape, vol_origin, dx, dz
            algorithm: TIGRE algorithm name ('ossart', 'sart', 'sirt', 'mlem').
                mlem is Maximum-Likelihood Expectation-Maximization: solves for
                the ML image under a Poisson noise model via the multiplicative
                update x_{k+1} = x_k * Atb(p / Ax_k) / Atb(1), instead of the
                least-squares gradient step SIRT/SART/OSSART use. Always
                full-batch (no ordered subsets — TIGRE's MLEM forces
                blocksize=N_angles internally); ignores lmbda/lmbda_red
                entirely (no relaxation parameter in the EM update); enforces
                non-negativity implicitly through the multiplicative form
                rather than via the nonneg flag. Not compatible with pwls
                (see pwls below).
            iterations: Number of iterations (default 100)
            blocksize: Number of projections per OS-SART block (default 15).
                Smaller = more subsets = faster convergence but noisier per-update.
                Ignored for mlem (always full-batch).
            lmbda: Relaxation parameter (default 0.5). Controls update step size;
                lower values give smoother convergence, less streak artifacts.
            lmbda_red: Relaxation reduction factor per iteration (default 0.97).
                Lambda decays as lmbda * lmbda_red^iter, annealing toward zero.
            nonneg: Enforce non-negativity constraint (default True)
            gpu_index: CUDA device index (default 0)
            bright_field: Unattenuated beam reference for flat-field correction
            dark_field: Electronic noise reference for flat-field correction
            clamp_mode: Line integral clamping mode ('none', 'soft', 'hard')
            soft_clip_transmission: Use soft clipping for transmission floor
            soft_clip_sharpness: Sharpness of soft clip transition
            upper_clamp: Clamp transmission from above
            upper_clamp_value: Maximum allowed transmission value
            mu_water: Linear attenuation coefficient of water (mm^-1)
            output_hu: Convert output to Hounsfield Units
            bhc_coeffs: BHC polynomial coefficients [c1, c2, ...] or None
            ring_correction: Apply sinogram-space ring artifact correction
            ring_median_width: Median filter width for ring correction (odd int)
            crossval: bool. If True (default), evaluate PSNR/SSIM/MSE against
                the evaluation projection every eval_every iterations.
                Reconstruction stops early when SSIM stops improving
                (patience-based). The saved volume is the one at peak SSIM,
                not the final iteration.
            holdout_index: int or None. Index of the evaluation projection.
                None (default) auto-selects the middle projection (N_angles // 2).
                Ignored when crossval=False.
            withhold_eval: bool. If True, the evaluation projection is REMOVED
                from the reconstruction input (true held-out validation, the
                pre-2026-08-13 behaviour). Default False: the projection is
                reconstructed from and evaluated against (diagnostic).
            eval_every: int. Evaluate holdout metrics every this many iterations.
                (default: 10)
            patience: int. Stop early if SSIM fails to improve for this many
                consecutive eval checkpoints. (default: 3)
            tv_lambda: TV denoising parameter passed to im3ddenoise after each
                iteration chunk. 0.0 (default) disables TV entirely.
                im3ddenoise normalises the volume to [0,1] internally, so this
                value is scan-independent. TIGRE minimises (1/2)||u-f||² +
                TV(u)/λ, so HIGHER λ = LESS smoothing (weaker TV), LOWER λ =
                MORE smoothing (stronger TV). λ=0 disables TV; λ→∞ approaches
                plain SIRT. Useful range for micro-CT: 10–50. λ=10 gives
                visible noise suppression; λ<5 is destructively over-smooth.
            tv_iters: Chambolle-Pock TV denoising iterations applied at each
                TV step (default 50). Higher values converge the TV sub-problem
                more fully; 50 matches the TIGRE OSSART-TV default.
            pwls: bool. If True, replaces the algorithm's default purely-
                geometric per-ray weight with a PWLS (penalized weighted
                least squares) weight that additionally down-weights rays
                with low estimated transmission (noisier measurements),
                approximating a maximum-likelihood weighting under a
                quantum-noise model. See _pwls_weight() for the derivation.
                Composes with tv_lambda (TV step is unaffected — it acts in
                the image domain, PWLS only reweights the data-fidelity
                term) and with crossval/early stopping. Default False
                (plain geometric weighting, i.e. standard SIRT/SART).
                Incompatible with algorithm='mlem': MLEM.__init__ (TIGRE's
                tigre/algorithms/statistical_algorithms.py) unconditionally
                overrides any passed-in W with its own Atb(ones) sensitivity
                map, so a PWLS W array would be silently discarded rather
                than applied — raises ValueError instead of failing silently.
                MLEM's Poisson likelihood already models per-ray photon
                statistics natively, so PWLS reweighting is redundant for it
                anyway.
            checkpoint_dir: str or None. If set and crossval=True, save a
                cropped copy of the reconstructed volume at every crossval
                eval checkpoint (every eval_every iterations), in the same
                on-disk orientation/HU-calibration as the final saved
                volume (see _to_disk_volume). Written as
                '{checkpoint_dir}/iter{i:04d}.npy' (float32). A
                'crossval_metrics.json' (self.crossval_metrics) is also
                written to this directory once reconstruction finishes, so
                each checkpoint's SSIM/PSNR/MSE is available without
                re-parsing logs. Intended for tracking how image-quality
                metrics (MTF/NPS/d') evolve over iterations, independent of
                the crossval SSIM used for early stopping. Default None
                (disabled). Ignored when crossval=False (no per-checkpoint
                volume exists to save).
            checkpoint_z_range: (z0, z1) tuple or None. z-slice bounds (in
                on-disk VFF z-index convention) to slice out of each
                checkpoint volume before saving. Required if checkpoint_dir
                is set.
            checkpoint_xy_range: (y0, y1, x0, x1) tuple or None. In-plane
                crop bounds (on-disk VFF y/x convention) applied to each
                checkpoint volume before saving. None (default) keeps the
                full xy plane.
        """
        _check_tigre_available()

        if algorithm not in SUPPORTED_TIGRE_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {SUPPORTED_TIGRE_ALGORITHMS}"
            )

        if algorithm == 'mlem' and pwls:
            raise ValueError(
                "pwls=True is not compatible with algorithm='mlem': MLEM's "
                "constructor unconditionally overrides any custom W weight "
                "array with its own Atb(ones) sensitivity map, so the PWLS "
                "weight would be silently ignored rather than applied. "
                "MLEM already models per-ray photon statistics natively "
                "through its Poisson likelihood, so PWLS is redundant here."
            )

        self.projections = projections
        self.angles = np.asarray(angles, dtype=np.float64)
        self.geometry = geometry
        self.algorithm = algorithm
        self.iterations = iterations
        self.blocksize = blocksize
        self.lmbda = lmbda
        self.lmbda_red = lmbda_red
        self.nonneg = nonneg
        self.gpu_index = gpu_index

        # Preprocessing parameters
        self.bright_field = bright_field
        self.dark_field = dark_field
        # Reference-free psi + column-CoR calibration before reconstruction.
        # False == the pre-2026-08-11 behaviour (square, centred detector).
        self.geometry_autocal = bool(geometry_autocal)
        # Externally supplied psi (e.g. from the shared half-scan calibration
        # JSON that muNeRF writes — the better estimator). When set, the
        # inline conjugate fit is skipped entirely; the CoR stays at the
        # geometric centre either way (fitted intercepts are estimator bias,
        # see run zsu85kc6 / 2026-08-12).
        self.detector_psi_deg = (None if detector_psi_deg is None
                                 else float(detector_psi_deg))
        self.clamp_mode = clamp_mode
        self.soft_clip_transmission = soft_clip_transmission
        self.soft_clip_sharpness = soft_clip_sharpness
        self.upper_clamp = upper_clamp
        self.upper_clamp_value = upper_clamp_value

        # HU conversion parameters
        self.mu_water = default_mu_water(mu_water, bhc_coeffs)
        self.output_hu = output_hu

        # BHC and ring correction
        self.bhc_coeffs = bhc_coeffs
        self.ring_correction = ring_correction
        self.air_normalization = air_normalization
        self.ring_median_width = ring_median_width

        # Evaluation projection (crossval machinery)
        self.crossval = crossval
        self.withhold_eval = withhold_eval
        # Optional live-metric sink: Callable[[dict, int], None], called at
        # every eval checkpoint. Keeps the backend wandb-free; the driver
        # passes ReconLogger.log so W&B charts update DURING the run.
        self.log_fn = log_fn
        # Optional projection-diagnostics sink: Callable[(pred, target,
        # step)], called with the FDK-convention prediction/measurement at
        # every eval checkpoint. The driver passes a wrapper around
        # ReconLogger.log_projection_diag (diag/* scalars + SSIM-heatmap and
        # power-spectrum figures). Also wandb-free here.
        self.diag_fn = diag_fn
        self.holdout_index = holdout_index
        self.eval_every = eval_every
        self.patience = patience
        self.tv_lambda = tv_lambda
        self.tv_iters = tv_iters
        self.pwls = pwls
        self.best_iter = None       # set during reconstruct() when crossval is on
        self.crossval_metrics = None  # dict of lists; set after reconstruct()

        # Per-checkpoint volume export (see checkpoint_dir docstring above)
        if checkpoint_dir is not None and checkpoint_z_range is None:
            raise ValueError("checkpoint_z_range is required when checkpoint_dir is set.")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_z_range = checkpoint_z_range
        self.checkpoint_xy_range = checkpoint_xy_range
        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.reconstructed_volume = None

        # Detector dimensions
        self.N_angles, self.N_b, self.N_a = self.projections.shape

    def _estimate_gpu_memory(self, geo):
        """
        Print informational GPU memory estimate.

        Unlike ASTRA, TIGRE handles memory splitting internally, so this
        is informational only (no hard OOM check).
        """
        Nz, Ny, Nx = geo.nVoxel  # TIGRE convention: [z, y, x]
        f32 = 4  # bytes per float32

        vol_bytes = Nx * Ny * Nz * f32
        sino_bytes = self.N_angles * self.N_b * self.N_a * f32

        print(f"\nGPU memory estimate (informational — TIGRE splits internally):")
        print(f"  Volume:    {vol_bytes / 2**30:.2f} GiB ({Nx}x{Ny}x{Nz} float32)")
        print(f"  Sinogram:  {sino_bytes / 2**30:.2f} GiB "
              f"({self.N_angles}x{self.N_b}x{self.N_a} float32)")
        print(f"  Total:     {(vol_bytes + sino_bytes) / 2**30:.2f} GiB (min)")

        gpu = query_gpu_memory(self.gpu_index)
        if gpu is None:
            print("  (nvidia-smi not available — skipping GPU info)")
        else:
            print(f"  GPU {self.gpu_index} ({gpu['name']}): "
                  f"{gpu['total_bytes'] / 2**30:.2f} GiB total, "
                  f"{gpu['free_bytes'] / 2**30:.2f} GiB free")

    def _crop_to_roi(self, vol_xyz, Nx_orig, Ny_orig, Nz_orig, verbose=False):
        """Center-crop a padded (x, y, z) volume back to the requested ROI.

        The reconstructed volume is centered at isocenter (offOrigin=0, the
        CUDA-hang workaround), so an off-center ROI crops at
        idx = (N_pad - N_roi)/2 + offset/voxel_size. Degenerates to a plain
        center-crop for full-FOV (vol_origin=0). Single definition used by
        both the final-volume path and the per-checkpoint export path.
        """
        Nx_pad, Ny_pad, Nz_pad = vol_xyz.shape
        if (Nx_pad, Ny_pad, Nz_pad) == (Nx_orig, Ny_orig, Nz_orig):
            return vol_xyz
        ox, oy, oz = self.geometry.get('vol_origin', (0, 0, 0))
        dx = self.geometry['dx']
        dz = self.geometry['dz']
        x0 = round((Nx_pad - Nx_orig) / 2 + ox / dx)
        y0 = round((Ny_pad - Ny_orig) / 2 + oy / dx)
        z0 = round((Nz_pad - Nz_orig) / 2 + oz / dz)
        vol_xyz = vol_xyz[x0:x0 + Nx_orig, y0:y0 + Ny_orig, z0:z0 + Nz_orig]
        if verbose:
            print(f"  Cropped ({Nx_pad}, {Ny_pad}, {Nz_pad}) → "
                  f"({Nx_orig}, {Ny_orig}, {Nz_orig}) "
                  f"[start: ({x0}, {y0}, {z0})]")
        return vol_xyz

    def _to_disk_volume(self, vol_tigre, Nx_orig, Ny_orig, Nz_orig):
        """
        Convert a raw TIGRE-convention volume (z, y, x) into the same
        orientation, ROI crop, and HU calibration as the final saved .vff:
          1. transpose to FDK convention (x, y, z) -- reconstruct() step 6
          2. center-crop the (possibly CUDA-hang-padded) volume down to the
             original ROI dimensions -- reconstruct() step 6b
          3. optional HU conversion (physics-based, skip_calibration
             convention: no two-point recalibration) -- reconstruct() step 7
          4. transpose + y-flip to on-disk VFF convention (z, y, x), matching
             ct_core/scan_setup.py's write step (vol.transpose(2,1,0)[:, ::-1, :])

        Shared by the end-of-reconstruct() final-volume path and the
        per-checkpoint export path so both are calibrated/oriented
        identically -- see checkpoint_dir docstring in __init__.

        Returns a float32 array in (z, y, x) on-disk convention, HU-clipped
        to [-1024, 4095] if output_hu, uncropped in xy/z (caller slices).
        """
        vol = vol_tigre.transpose(2, 1, 0).astype(np.float32)  # (x, y, z)
        vol = self._crop_to_roi(vol, Nx_orig, Ny_orig, Nz_orig)

        if self.output_hu:
            vol = mu_to_hu(vol, self.mu_water, verbose=False)

        return vol.transpose(2, 1, 0)[:, ::-1, :]  # (z, y, x), on-disk convention

    def _save_checkpoint(self, vol_tigre, i_done, Nx_orig, Ny_orig, Nz_orig):
        """Save a cropped on-disk-convention slab for one crossval checkpoint."""
        vol_disk = self._to_disk_volume(vol_tigre, Nx_orig, Ny_orig, Nz_orig)
        z0, z1 = self.checkpoint_z_range
        if self.checkpoint_xy_range is not None:
            y0, y1, x0, x1 = self.checkpoint_xy_range
            slab = vol_disk[z0:z1, y0:y1, x0:x1]
        else:
            slab = vol_disk[z0:z1]
        out_path = Path(self.checkpoint_dir) / f"iter{i_done:04d}.npy"
        np.save(out_path, slab)
        unit_str = f"[{slab.min():.0f}, {slab.max():.0f}] HU" if self.output_hu else \
            f"[{slab.min():.4f}, {slab.max():.4f}] (raw)"
        print(f"  [checkpoint] saved {out_path.name} {slab.shape} {unit_str}")

    def reconstruct(self):
        """
        Full reconstruction pipeline.

        Steps:
        1. Preprocess (flat-field + log transform)
        2. Build TIGRE geometry
        3. Convert angles from FDK to TIGRE convention
        4. GPU memory estimate (informational)
        5. Run TIGRE algorithm
        6. Transpose output from TIGRE (z,y,x) to FDK (x,y,z) convention
        7. Optional HU conversion

        Returns:
            self.reconstructed_volume — np.ndarray of shape (Nx, Ny, Nz)
        """
        _check_tigre_available()

        print("=" * 60)
        print(f"TIGRE Iterative Reconstruction: {self.algorithm.upper()}")
        print("=" * 60)

        # Step 1: Preprocess
        sinogram = preprocess_sinogram(
            projections=self.projections,
            bright_field=self.bright_field,
            dark_field=self.dark_field,
            clamp_mode=self.clamp_mode,
            soft_clip_transmission=self.soft_clip_transmission,
            soft_clip_sharpness=self.soft_clip_sharpness,
            upper_clamp=self.upper_clamp,
            upper_clamp_value=self.upper_clamp_value,
            bhc_coeffs=self.bhc_coeffs,
            ring_correction=self.ring_correction,
            air_normalization=self.air_normalization,
            ring_median_width=self.ring_median_width,
        )

        # Step 2: Build TIGRE geometry (may pad Nxy to avoid CUDA hang)
        Nx_orig, Ny_orig, Nz_orig = self.geometry['vol_shape']
        print("\nBuilding TIGRE geometry...")
        # ---- geometry auto-calibration (reference-free, ~1 s) ---------------
        # Measures the detector in-plane rotation psi and the column centre of
        # rotation from THIS scan's conjugate rays, before reconstructing. Both
        # were hard-zero here before 2026-08-11. Set geometry_autocal=False to
        # reproduce that exactly.
        _psi_deg, _cpa_raw = None, None
        if self.detector_psi_deg is not None:
            # External calibration (half-scan consistency JSON) wins over the
            # inline conjugate fit — it is the validated, unbiased estimator.
            _psi_deg = self.detector_psi_deg
            _cpa_raw = (self.N_a - 1) / 2.0
            print(f"\nGeometry: psi = {_psi_deg:+.4f} deg from external "
                  f"calibration (CoR at the geometric centre)")
        elif self.geometry_autocal:
            try:
                import torch as _torch
                from ...ct_core.detector_psi import estimate_psi_joint
                # self.angles (FDK convention) matches the sinogram AS IT IS
                # HERE — the TIGRE angle conversion and the column flip both
                # happen later, at Step 5.
                _ang = _torch.as_tensor(np.asarray(self.angles, dtype=np.float64))
                _sin = _torch.as_tensor(np.ascontiguousarray(sinogram))
                _g = dict(self.geometry)
                _g["central_pixel_a"] = (self.N_a - 1) / 2.0
                _g["central_pixel_b"] = (self.N_b - 1) / 2.0
                print("\nGeometry auto-calibration (conjugate rays, no reference "
                      "volume)...")
                _r = estimate_psi_joint(_sin, _ang, _g, downsample=1, verbose=False)
                from ...ct_core.detector_psi import MAX_PSI_DEG, MIN_JOINT_DEPTH
                if (abs(_r["psi_deg"]) <= MAX_PSI_DEG
                        and _r.get("joint_depth", 0.0) >= MIN_JOINT_DEPTH
                        and abs(_r["cpa0"] - _g["central_pixel_a"])
                            * float(self.geometry["da"]) <= 0.70):
                    # psi only. The fitted cpa0 intercept is estimator BIAS,
                    # not geometry: applying it (+2.12 raw cols on Scan_1510)
                    # split muNeRF run zsu85kc6's z=+23 mm tube into two
                    # overlapped half-discs, and the FBP tube test
                    # (2026-08-12) shows the geometric centre round at the
                    # midplane and both z extremes. The intercept is still
                    # fitted — the joint fit needs it as a nuisance
                    # parameter — but it is not APPLIED.
                    _psi_deg, _cpa_raw = _r["psi_deg"], _g["central_pixel_a"]
                    print(f"  psi = {_psi_deg:+.4f} deg  (fitted cpa0 "
                          f"{_r['cpa0']:.3f} NOT applied — known estimator "
                          f"bias; CoR stays at the geometric centre "
                          f"{_g['central_pixel_a']:.1f}), joint depth "
                          f"{_r.get('joint_depth', float('nan')):.2f}")
                else:
                    print(f"  REJECTED (psi {_r['psi_deg']:+.3f}, depth "
                          f"{_r.get('joint_depth', 0.0):.2f}) — using psi=0, "
                          f"centred CoR")
            except Exception as _e:
                print(f"  geometry auto-calibration FAILED ({type(_e).__name__}: "
                      f"{_e}) — using psi=0, centred CoR")

        # MIRROR BOTH TERMS. The sinogram is column-flipped for TIGRE at Step 5
        # (`sinogram[:, :, ::-1]`), AFTER this geometry is built, but psi and
        # cpa0 were measured on the UNFLIPPED array. Under a column mirror
        # c -> (N_a-1)-c:
        #     psi_tigre = -psi           (a mirror reverses an in-plane rotation)
        #     cpa_tigre = (N_a-1) - cpa  (the centre reflects about itself)
        # Skipping this would apply both corrections BACKWARDS — i.e. double the
        # error rather than remove it.
        if _psi_deg is not None:
            _psi_deg = -_psi_deg
            _cpa_raw = (self.N_a - 1) - _cpa_raw
            print(f"  mirrored for TIGRE's flipped detector: psi "
                  f"{_psi_deg:+.4f} deg, cpa {_cpa_raw:.3f}")
        geo = build_tigre_geometry(self.geometry, self.N_b, self.N_a,
                                   detector_psi_deg=_psi_deg, cpa_raw=_cpa_raw)
        print(f"  DSD={float(geo.DSD):.2f} mm, DSO={float(geo.DSO):.2f} mm")
        print(f"  Detector: {geo.nDetector} px, {geo.dDetector} mm/px")
        print(f"  Volume: {geo.nVoxel} voxels (z,y,x), {geo.dVoxel} mm/voxel")

        # Step 3: Convert angles
        tigre_angles = fdk_angles_to_tigre(self.angles)
        print(f"\nAngles: {len(tigre_angles)} projections, "
              f"range [{np.rad2deg(tigre_angles.min()):.1f}, "
              f"{np.rad2deg(tigre_angles.max()):.1f}] deg (TIGRE convention)")

        # Step 4: GPU memory estimate (uses actual TIGRE geo, including padding)
        self._estimate_gpu_memory(geo)

        # Step 5: Run TIGRE algorithm
        # TIGRE expects (N_angles, N_b, N_a) — same shape as FDK, but
        # TIGRE's detector U-axis (horizontal) is flipped relative to FDK/ASTRA.
        # Flip columns to match TIGRE's detector convention.
        sinogram = np.ascontiguousarray(sinogram[:, :, ::-1], dtype=np.float32)

        if self.checkpoint_dir is not None and not self.crossval:
            print("\nWARNING: checkpoint_dir is set but crossval=False — there is "
                  "no per-checkpoint loop to hook into, so no checkpoints will be "
                  "saved.")

        # Evaluation projection: extracted before reconstruction; withheld
        # from the input only when withhold_eval (true held-out validation).
        holdout_proj = None
        if self.crossval:
            if _ssim_fn is None:
                raise ImportError(
                    "scikit-image is required for cross-validation. "
                    "Install with: pip install scikit-image"
                )
            N_train = sinogram.shape[0]
            idx = self.holdout_index if self.holdout_index is not None else N_train // 2
            holdout_proj = sinogram[idx].copy()       # (N_b, N_a) TIGRE convention
            holdout_angle = tigre_angles[idx:idx + 1]
            holdout_deg = float(np.rad2deg(self.angles[idx]))
            if self.withhold_eval:
                sinogram = np.delete(sinogram, idx, axis=0)
                tigre_angles = np.delete(tigre_angles, idx, axis=0)
                print(f"\nCross-validation: WITHHOLDING projection {idx} "
                      f"({holdout_deg:.1f}° FDK) from the reconstruction; "
                      f"eval every {self.eval_every} iters, "
                      f"patience={self.patience} checkpoints.")
            else:
                print(f"\nDiagnostics: evaluating against projection {idx} "
                      f"({holdout_deg:.1f}° FDK), which STAYS in the "
                      f"reconstruction (pass withhold_eval for true "
                      f"validation); eval every {self.eval_every} iters, "
                      f"patience={self.patience} checkpoints.")

        if self.algorithm == 'mlem':
            # MLEM's ratio update x_{k+1} = x_k * Atb(p/Ax_k) / Atb(1) implicitly
            # assumes p >= 0 (it's the Poisson-emission-count EM update; TIGRE
            # only guards exact-zero denominators, not negative numerators).
            # preprocess_sinogram legitimately produces small negative line
            # integrals near air/background (soft-clip allows transmission up
            # to upper_clamp_value=1.05, i.e. p = -ln(T) slightly < 0) — harmless
            # for the least-squares algorithms but fatal here: wherever a
            # negative p lines up with a small Ax_k, the ratio blows up and,
            # being multiplicative, compounds geometrically every iteration
            # (observed: SSIM=0.69 at iter 10 -> float overflow by iter 20).
            # Floor at 0 before MLEM ever sees the data; not applied to
            # sirt/sart/ossart, which tolerate negative residuals fine.
            n_neg = int(np.sum(sinogram < 0.0))
            if n_neg > 0:
                neg_min = float(sinogram.min())
                print(f"\nMLEM: clipping {n_neg} negative sinogram values "
                      f"(min={neg_min:.4f}) to 0.0 — MLEM's ratio update is "
                      f"undefined for negative line integrals.")
            np.clip(sinogram, 0.0, None, out=sinogram)

        alg_func = _TIGRE_ALG_FUNCS[self.algorithm]()

        kwargs = {
            'lmbda': self.lmbda,
            'lmbda_red': self.lmbda_red,
            'verbose': True,
            'noneg': self.nonneg,
        }
        if self.algorithm in ('ossart', 'sart'):
            kwargs['blocksize'] = self.blocksize

        if self.gpu_index != 0:
            kwargs['gpuids'] = tigre.utilities.gpu.GpuIds(self.gpu_index)

        if self.pwls:
            print("\nComputing PWLS weights (geometric x statistical)...")
            W_pwls = _pwls_weight(sinogram, geo, tigre_angles,
                                   gpuids=kwargs.get('gpuids'))
            kwargs['W'] = W_pwls
            print(f"  W range: [{W_pwls.min():.4e}, {W_pwls.max():.4e}], "
                  f"mean={W_pwls.mean():.4e} (statistical factor alone "
                  f"has mean 1.0 by construction)")

        # Run 1 calibration iteration for time estimate.
        # NOTE: Python threading timeouts don't work for TIGRE hang detection
        # because TIGRE's C extension holds the GIL. Use bash-level 'timeout'
        # command to detect hangs (e.g., 'timeout 3600 python -m ...').
        print(f"\nCalibrating iteration speed (1 iteration)...")
        t0 = time.time()
        alg_func(sinogram, geo, tigre_angles, 1, **{**kwargs, 'verbose': False})
        dt_one = time.time() - t0
        est_total = dt_one * self.iterations
        print(f"  1 iteration: {dt_one:.1f}s → "
              f"{self.iterations} iterations estimated: "
              f"{est_total / 60:.1f} min")

        tv_str = (f", TV(λ={self.tv_lambda}, iters={self.tv_iters})"
                  if self.tv_lambda > 0 else "")
        pwls_str = ", PWLS" if self.pwls else ""
        if self.algorithm == 'mlem':
            # blocksize/lmbda/lmbda_red are accepted by TIGRE's MLEM kwargs
            # but never referenced in its update rule (full-batch, no
            # relaxation parameter) — omit them here to avoid implying they
            # have any effect.
            print(f"\nRunning MLEM "
                  f"({self.iterations} iterations, full-batch{tv_str})...")
        else:
            print(f"\nRunning {self.algorithm.upper()} "
                  f"({self.iterations} iterations, blocksize={self.blocksize}, "
                  f"lambda={self.lmbda}, lambda_red={self.lmbda_red}{tv_str}{pwls_str})...")

        t_start = time.time()

        if holdout_proj is None:
            if self.tv_lambda <= 0:
                # No TV, no crossval: run all iterations at once.
                vol_tigre = alg_func(sinogram, geo, tigre_angles, self.iterations, **kwargs)
            else:
                # TV + no crossval: run in chunks of eval_every, apply TV after each.
                vol_tigre = None
                chunk_kwargs = {**kwargs, 'verbose': False}
                i_done = 0
                while i_done < self.iterations:
                    chunk = min(self.eval_every, self.iterations - i_done)
                    chunk_kwargs['lmbda'] = self.lmbda * (self.lmbda_red ** i_done)
                    if vol_tigre is not None:
                        chunk_kwargs['init'] = vol_tigre
                    vol_tigre = alg_func(sinogram, geo, tigre_angles, chunk, **chunk_kwargs)
                    i_done += chunk
                    vol_tigre = _im3ddenoise(vol_tigre, self.tv_iters, self.tv_lambda)
                    print(f"  TV @ iter {i_done}: "
                          f"[{vol_tigre.min():.6f}, {vol_tigre.max():.6f}]")
        else:
            # Cross-validation: chunked loop with early stopping on SSIM.
            print(f"\n  {'Iter':>6}   {'PSNR (dB)':>10}   {'SSIM':>10}   "
                  f"{'Holdout MSE':>14}   {'':>4}")
            data_range = float(holdout_proj.max() - holdout_proj.min())
            if data_range < 1e-9:
                data_range = 1.0

            vol_tigre = None
            best_vol = None
            best_ssim = -1.0
            best_psnr = 0.0
            best_iter = 0
            patience_count = 0
            chunk_kwargs = {**kwargs, 'verbose': False}
            i_done = 0
            cv_iters, cv_ssim, cv_psnr, cv_mse = [], [], [], []

            while i_done < self.iterations:
                chunk = min(self.eval_every, self.iterations - i_done)

                # Correct starting lambda so decay is continuous across chunks.
                chunk_kwargs['lmbda'] = self.lmbda * (self.lmbda_red ** i_done)
                if vol_tigre is not None:
                    chunk_kwargs['init'] = vol_tigre

                vol_tigre = alg_func(sinogram, geo, tigre_angles, chunk, **chunk_kwargs)
                i_done += chunk

                # Apply TV regularization if enabled (before metrics, so SSIM
                # evaluates the regularized volume and best_vol is TV-denoised).
                if self.tv_lambda > 0:
                    vol_tigre = _im3ddenoise(vol_tigre, self.tv_iters, self.tv_lambda)

                # TIGRE's check_geo repmat-s offOrigin→(N,3), offDetector→(N,2),
                # rotDetector→(N,3), and COR→(N,) during reconstruction.
                # Reset all to scalar form so Ax passes check_geo for 1 angle.
                geo.offOrigin = np.array([0.0, 0.0, 0.0])
                geo.offDetector = np.array([0.0, 0.0])
                geo.rotDetector = np.array([0.0, 0.0, 0.0])
                geo.COR = 0.0

                # Forward-project at holdout angle and compute metrics.
                pred = tigre.Ax(vol_tigre, geo, holdout_angle, 'interpolated')[0]

                # Diagnostic: print ranges on first checkpoint to verify alignment.
                if i_done == self.eval_every:
                    print(f"  [diag] holdout: min={holdout_proj.min():.4f} "
                          f"max={holdout_proj.max():.4f} "
                          f"mean={holdout_proj.mean():.4f}")
                    print(f"  [diag] pred:    min={pred.min():.4f} "
                          f"max={pred.max():.4f} "
                          f"mean={pred.mean():.4f}")

                mse = float(np.mean((pred - holdout_proj) ** 2))
                psnr = (10.0 * np.log10(data_range ** 2 / mse)
                        if mse > 0 else float('inf'))
                ssim = float(_ssim_fn(holdout_proj, pred, data_range=data_range))

                cv_iters.append(i_done)
                cv_ssim.append(ssim)
                cv_psnr.append(psnr)
                cv_mse.append(mse)
                if self.diag_fn is not None:
                    # Columns back to the FDK detector convention, then cut to
                    # the detector window covered by the reconstruction domain
                    # (the same window the noise ceiling and the other
                    # backends' diagnostics use — outer rows leave the z-slab
                    # and outer columns miss the FOV cylinder, so both score
                    # FOV truncation, not the reconstruction).
                    #
                    # The [::-1] MUST come first: a0/a1 are ct_core column
                    # indices (they are derived from geometry's
                    # central_pixel_a) and this sinogram is still in TIGRE's
                    # reversed column order until the flip undoes it.
                    from ...ct_core.projection_diag import covered_detector_window
                    b0, b1, a0, a1 = covered_detector_window(
                        self.geometry, pred.shape[0], pred.shape[1])
                    self.diag_fn(pred[b0:b1, ::-1][:, a0:a1],
                                 holdout_proj[b0:b1, ::-1][:, a0:a1], i_done)
                elif self.log_fn is not None:
                    self.log_fn({"diag/ssim": ssim, "diag/psnr": psnr,
                                 "diag/mse": mse}, i_done)

                if ssim > best_ssim:
                    best_ssim = ssim
                    best_psnr = psnr
                    best_iter = i_done
                    best_vol = vol_tigre.copy()
                    patience_count = 0
                    marker = '★'
                else:
                    patience_count += 1
                    marker = f'-{patience_count}'

                print(f"  {i_done:>6}   {psnr:>10.4f}   {ssim:>10.6f}   "
                      f"{mse:>14.8e}   {marker}")

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(vol_tigre, i_done, Nx_orig, Ny_orig, Nz_orig)

                if patience_count >= self.patience:
                    print(f"\n  Early stopping: SSIM peaked at iter {best_iter} "
                          f"(SSIM={best_ssim:.6f}). "
                          f"No improvement for {self.patience} checkpoints.")
                    break

            else:
                # Loop completed without early stopping — save the final iteration,
                # not best_vol. The user ran all iterations intentionally.
                if best_iter < i_done:
                    print(f"\n  Note: peak SSIM={best_ssim:.6f} was at iter {best_iter}, "
                          f"not the final iteration. Saving final iteration ({i_done}).")
                best_vol = vol_tigre  # keep final iteration as the saved result
                best_iter = i_done

            vol_tigre = best_vol
            self.best_iter = best_iter
            self.crossval_metrics = {
                'iters': cv_iters,
                'ssim': cv_ssim,
                'psnr': cv_psnr,
                'mse': cv_mse,
                'best_iter': best_iter,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
                'stop_iter': i_done,
                'holdout_index': idx,
                'holdout_deg': holdout_deg,
            }
            print(f"  Saving volume from iter {best_iter} "
                  f"(SSIM={best_ssim:.6f}, PSNR={best_psnr:.4f} dB)\n")

            if self.checkpoint_dir is not None:
                metrics_path = Path(self.checkpoint_dir) / "crossval_metrics.json"
                metrics_path.write_text(json.dumps(self.crossval_metrics, indent=2))
                print(f"  Saved {metrics_path}")

        t_recon = time.time() - t_start
        del sinogram
        print(f"  Reconstruction took {t_recon / 60:.1f} min "
              f"({t_recon / self.iterations:.1f}s/iteration)")

        # Step 6: Transpose from TIGRE (z, y, x) to FDK convention (x, y, z)
        print(f"\n  TIGRE output shape: {vol_tigre.shape} (z, y, x)")
        print(f"  TIGRE output range: [{vol_tigre.min():.6f}, {vol_tigre.max():.6f}]")

        self.reconstructed_volume = vol_tigre.transpose(2, 1, 0).astype(np.float32)
        del vol_tigre

        # Crop from centered (possibly CUDA-hang-padded) volume to the ROI.
        self.reconstructed_volume = self._crop_to_roi(
            self.reconstructed_volume, Nx_orig, Ny_orig, Nz_orig, verbose=True)

        print(f"  Reordered to FDK convention: {self.reconstructed_volume.shape} (x, y, z)")

        # Step 7: Optional HU conversion (shared ct_core definition)
        if self.output_hu:
            print("\nConverting to Hounsfield Units...")
            self.reconstructed_volume = mu_to_hu(self.reconstructed_volume,
                                                 self.mu_water)

        print("\nReconstruction complete.")
        return self.reconstructed_volume

    def plot_crossval(self, save_prefix):
        """
        Save a publication-quality convergence figure to
        {save_prefix}_convergence.pdf and .png.

        Must be called after reconstruct() when crossval=True.
        """
        if self.crossval_metrics is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            print("  matplotlib not available — skipping convergence figure.")
            return

        m = self.crossval_metrics
        iters     = np.asarray(m['iters'])
        cv_ssim   = np.asarray(m['ssim'])
        cv_psnr   = np.asarray(m['psnr'])
        cv_mse    = np.asarray(m['mse'])
        best_iter = m['best_iter']
        stop_iter = m['stop_iter']

        # Guard against non-finite metrics (e.g. a divergent algorithm run,
        # as MLEM can do on ill-conditioned data): matplotlib's set_ylim
        # raises on NaN/Inf, and Inf samples in the plotted line blow up
        # autoscale even on panels without an explicit ylim. Replace
        # non-finite samples with NaN (matplotlib draws a gap there) and
        # compute axis limits from the finite subset only, so a partial
        # divergence still produces a readable figure instead of losing the
        # whole run to an unhandled exception during metric plotting.
        all_finite = np.isfinite(cv_ssim) & np.isfinite(cv_psnr) & np.isfinite(cv_mse)
        if not all_finite.all():
            first_bad_iter = int(iters[~all_finite][0])
            print(f"  WARNING: {int((~all_finite).sum())} non-finite crossval "
                  f"metric value(s) detected (first at iter {first_bad_iter}) "
                  f"— the run likely diverged numerically. Plotting finite "
                  f"values only; treat any 'best' iteration at/after this "
                  f"point with suspicion (SSIM/PSNR are unreliable once the "
                  f"reconstruction has overflowed).")
        cv_ssim_plot = np.where(np.isfinite(cv_ssim), cv_ssim, np.nan)
        cv_psnr_plot = np.where(np.isfinite(cv_psnr), cv_psnr, np.nan)
        cv_mse_plot = np.where(np.isfinite(cv_mse) & (cv_mse > 0), cv_mse, np.nan)

        def _safe_ylim(arr, pad_frac=0.12, pad_min=0.01):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None
            lo, hi = float(finite.min()), float(finite.max())
            pad = (hi - lo) * pad_frac or pad_min
            return lo - pad, hi + pad

        # ── Publication style ────────────────────────────────────────────────
        rc = {
            'font.family':        'sans-serif',
            'font.size':          9,
            'axes.labelsize':     9,
            'axes.titlesize':     9,
            'xtick.labelsize':    8,
            'ytick.labelsize':    8,
            'legend.fontsize':    8,
            'legend.framealpha':  0.85,
            'axes.linewidth':     0.75,
            'xtick.major.width':  0.75,
            'ytick.major.width':  0.75,
            'xtick.minor.width':  0.5,
            'ytick.minor.width':  0.5,
            'xtick.direction':    'out',
            'ytick.direction':    'out',
            'lines.linewidth':    1.5,
            'patch.linewidth':    0.75,
            'pdf.fonttype':       42,   # embed TrueType, not Type 3
            'ps.fonttype':        42,
        }

        blue  = '#2166ac'
        red   = '#d6604d'
        green = '#1a9641'
        grey  = '#555555'

        with plt.rc_context(rc):
            fig, axes = plt.subplots(3, 1, figsize=(3.5, 6.0), sharex=True)

            # Shade patience region (between best and stop) on all panels
            if stop_iter > best_iter:
                for ax in axes:
                    ax.axvspan(best_iter, stop_iter, color='#ffcccc',
                               alpha=0.45, zorder=0, linewidth=0)

            # Vertical line at peak
            for ax in axes:
                ax.axvline(best_iter, color=grey, linestyle='--',
                           linewidth=0.9, zorder=2)

            # ── Panel 1: SSIM ────────────────────────────────────────────────
            ax = axes[0]
            ax.plot(iters, cv_ssim_plot, color=blue, linewidth=1.5, zorder=3)
            best_idx  = int(np.where(iters == best_iter)[0][0])
            if np.isfinite(cv_ssim[best_idx]):
                ax.scatter([best_iter], [cv_ssim[best_idx]], color=blue,
                           s=60, zorder=4, marker='*', linewidths=0)
            ax.set_ylabel('SSIM')
            ylim = _safe_ylim(cv_ssim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], color=grey, linestyle='--',
                               linewidth=0.9, label=f'Peak iter {best_iter}'),
                    plt.Line2D([0], [0], color='#ffcccc', linewidth=6,
                               solid_capstyle='butt',
                               label=f'Patience window'),
                ],
                loc='lower right', handlelength=1.2,
            )

            # ── Panel 2: PSNR ────────────────────────────────────────────────
            ax = axes[1]
            ax.plot(iters, cv_psnr_plot, color=red, linewidth=1.5, zorder=3)
            if np.isfinite(cv_psnr[best_idx]):
                ax.scatter([best_iter], [cv_psnr[best_idx]], color=red,
                           s=60, zorder=4, marker='*', linewidths=0)
            ax.set_ylabel('PSNR (dB)')
            ylim = _safe_ylim(cv_psnr, pad_frac=0.12, pad_min=0.1)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            # ── Panel 3: Holdout MSE (log scale) ─────────────────────────────
            ax = axes[2]
            ax.semilogy(iters, cv_mse_plot, color=green, linewidth=1.5, zorder=3)
            if np.isfinite(cv_mse_plot[best_idx]):
                ax.scatter([best_iter], [cv_mse_plot[best_idx]], color=green,
                           s=60, zorder=4, marker='*', linewidths=0)
            ax.set_ylabel('Holdout MSE')
            ax.set_xlabel('Iteration')
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            ax.grid(True, which='minor', color='#e0e0e0',
                    linewidth=0.4, zorder=0)

            # ── Shared formatting ─────────────────────────────────────────────
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, which='major', color='#dddddd',
                        linewidth=0.5, zorder=0)
                ax.set_axisbelow(True)

            # Tighten x-axis to data range
            axes[0].set_xlim(iters[0] - self.eval_every * 0.5,
                             iters[-1] + self.eval_every * 0.5)

            plt.tight_layout(pad=0.6, h_pad=0.4)

            for ext in ('pdf', 'png'):
                path = f"{save_prefix}_convergence.{ext}"
                fig.savefig(path, dpi=300, bbox_inches='tight')
                print(f"  Saved convergence figure: {path}")

            plt.close(fig)
