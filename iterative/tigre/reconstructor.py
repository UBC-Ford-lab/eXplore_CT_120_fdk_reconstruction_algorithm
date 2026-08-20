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

from ...ct_core.early_stop import (STOP_METRICS, EarlyStopper, HoldoutScorer,
                                   LCurve, StoppingRules, metrics_dict,
                                   plot_convergence, resolve_holdout_index,
                                   resolve_min_iter, resolve_patience,
                                   solution_norm, write_metrics)
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


def build_tigre_geometry(geometry, N_b, N_a, detector_psi_deg=None,
                         cpa_raw=None, min_nxy=None):
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

    # Optional transaxial padding, off by default and unused by this backend.
    #
    # A constant `Nxy_min = max(1000, Nz)` was applied here unconditionally
    # until 2026-08-19, to dodge a CUDA hang. It cost 1.51x the voxels on the
    # production geometry, and it described the hang wrongly: the boundary is
    # a PHYSICAL size, not a voxel count (70.0 mm hangs and 75.0 mm passes at
    # 0.075, 0.1 AND 0.2 mm voxels), it does not depend on Nz, it moves with
    # detector width and angle count, and it is a non-monotone BAND — with a
    # half-width detector, 25 mm passes, 20 and 15 mm hang, 10 and 5 mm pass.
    #
    # None of that matters any more: the hang was traced to TIGRE's own
    # `set_w()` and is avoided at source by `geometric_row_weights`, which see.
    # This parameter remains only as an escape hatch for a hang elsewhere.
    if min_nxy is not None and min(Nx, Ny) < int(min_nxy):
        Nx_before_pad, Ny_before_pad = Nx, Ny
        Nx = max(Nx, int(min_nxy))
        Ny = max(Ny, int(min_nxy))
        print(f"  TIGRE hang workaround: padding volume from "
              f"({Nx_before_pad}, {Ny_before_pad}, {Nz}) to ({Nx}, {Ny}, {Nz}).")

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


def geometric_row_weights(geo, angles, gpuids=None,
                          projection_type="Siddon", grid=8):
    """R = diag(1/L_i): the reciprocal ray path length through the FOV box.

    This is what ``IterativeReconAlg.set_w`` computes internally, reproduced
    here for one reason: TIGRE's version pins ``nVoxel = [2, 2, 2]`` and that
    grid HANGS THE CUDA KERNEL for some field sizes.

    THE BUG, located 2026-08-19. ``set_w`` does not project the reconstruction
    volume — it builds a stand-in box (transaxial extent x1.1, z at least the
    detector height) and forces ``nVoxel`` to 2 per axis, so only the PHYSICAL
    extent ever reaches the kernel. In ``Siddon_projection.cu`` the ray walk is

        float imin,imax,jmin,jmax,kmin,kmax;          // <- float
        unsigned long Np = (imax-imin+1)+(jmax-jmin+1)+(kmax-kmin+1);
        for (unsigned long ii=0; ii<Np; ii++){ ... }

    For rays that graze or miss the box those bounds go degenerate, and the
    float->unsigned long conversion turns the intersection count into an
    astronomically large ``Np``, so the loop never ends. With only 2 voxels per
    axis the index arithmetic is coarse enough that whether any ray lands there
    flips with small changes in box size, detector width and angle set —
    MEASURED as a non-monotone hang BAND in physical millimetres, independent
    of the reconstruction's own voxel count (which never reaches the kernel).
    Signature: 100% GPU utilisation with ~325 MiB allocated, because the hang
    happens before anything large is allocated.

    THE FIX. A box of ones has line integral = chord length whatever the voxel
    count, so a finer grid is the same quantity computed on a grid that does
    not trip the degeneracy. MEASURED at 81.4 mm, where TIGRE's own grid still
    works: grid=8 and grid=2 agree to 0.000% median AND p99 relative difference
    over every ray with a real path (they differ only on grazing rays, which
    the mask below sends to inf anyway). Supplying ``W`` also means
    ``IterativeReconAlg.__init__`` never calls ``set_w`` at all — it is guarded
    by ``if not hasattr(self, "W")`` — so the hang cannot occur. A geometry
    that hung indefinitely reconstructs in 11.6 s this way.
    """
    geox = copy.deepcopy(geo)
    geox.sVoxel[1:] = geox.sVoxel[1:] * 1.1   # TIGRE's own margin, kept
    geox.sVoxel[0] = max(geox.sDetector[0], geox.sVoxel[0])
    n = int(grid)
    geox.nVoxel = np.array([n, n, n])
    geox.dVoxel = geox.sVoxel / geox.nVoxel
    W = tigre.Ax(np.ones(geox.nVoxel, dtype=np.float32), geox, angles,
                 projection_type, gpuids=gpuids)
    W[W <= min(geo.dVoxel / 2)] = np.inf
    return (1.0 / W).astype(np.float32)


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

    # Same geometric factor as plain SIRT, on a grid that cannot hang the
    # kernel (this function used to carry its own nVoxel=[2,2,2] copy).
    W_geom = geometric_row_weights(geo, angles, gpuids=gpuids,
                                   projection_type="interpolated")

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
                 blocksize=15, lmbda=1.0, lmbda_red=1.0,
                 set_w_grid=8,
                 nonneg=True, gpu_index=0,
                 bright_field=None, dark_field=None,
                 geometry_autocal=True, detector_psi_deg=None,
                 clamp_mode='none', soft_clip_transmission=True,
                 soft_clip_sharpness=200.0, upper_clamp=True,
                 upper_clamp_value=1.05,
                 ring_correction=False, ring_median_width=51,
                 air_normalization=True,
                 crossval=True, holdout_index=None,
                 withhold_eval=False,
                 eval_every=10, patience=None, min_stop_iter=None,
                 stop_metric='ssim', l_curve=False, l_curve_norm='l2',
                 stop_on=('holdout',),
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
            lmbda: Relaxation parameter (default 1.0, matching TIGRE's own).
                The step size in x <- x + lambda C A^T R (b - A x), applied at
                IterativeReconAlg as res += lmbda * (1/V) * Atb(W * residual).
                SIRT converges for any lambda in (0, 2) — the iteration matrix
                C A^T R A has spectral radius <= 1 under these row/column
                normalisations — so 1.0 is the unrelaxed textbook choice.
                Lower values smooth the trajectory and suppress early streaks
                at roughly proportional cost in iterations; they are worth
                considering for the ORDERED-SUBSET methods (ossart, sart),
                whose noisy per-block updates can orbit the solution at full
                relaxation, but not for full-batch sirt.
            lmbda_red: Relaxation reduction per iteration (default 1.0 = NO
                decay, matching TIGRE's own). Lambda decays as
                lmbda * lmbda_red^iter, so the TOTAL update budget is
                sum(lambda_k) = lmbda / (1 - lmbda_red) and is FINITE for any
                lmbda_red < 1: the iterate converges to wherever the shrinking
                step leaves it, not to the solution. The previous 0.5/0.97
                default gave a budget of ~17 against 100 for lambda=1 held over
                100 iterations, and was effectively frozen past ~150 — a run
                asked to "iterate until converged" could not. Semi-convergence
                belongs to the stopping rule, which finds the peak and restores
                that iterate; annealing the step only hides the turnover.
            nonneg: Enforce non-negativity constraint (default True)
            gpu_index: CUDA device index (default 0)
            bright_field: Unattenuated beam reference for flat-field correction
            dark_field: Electronic noise reference for flat-field correction
            clamp_mode: Line integral clamping mode ('none', 'soft', 'hard')
            soft_clip_transmission: Use soft clipping for transmission floor
            soft_clip_sharpness: Sharpness of soft clip transition
            upper_clamp: Clamp transmission from above
            upper_clamp_value: Maximum allowed transmission value
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
        self.set_w_grid = int(set_w_grid)
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

        # Ring correction
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
        # None = derive from the schedule (ct_core.early_stop).
        self.patience = patience
        self.min_stop_iter = min_stop_iter
        # Which held-out metric peaks. SSIM is the default here — the historical
        # choice for this backend, and the structural one, which turns over
        # earliest of the three.
        if stop_metric not in STOP_METRICS:
            raise ValueError(f"stop_metric must be one of {list(STOP_METRICS)}, "
                             f"got {stop_metric!r}")
        self.stop_metric = stop_metric
        self.l_curve = bool(l_curve)
        self.l_curve_norm = str(l_curve_norm)
        self.stop_on = tuple(stop_on)
        self.tv_lambda = tv_lambda
        self.tv_iters = tv_iters
        self.pwls = pwls
        self.best_iter = None       # set during reconstruct() when crossval is on
        self.crossval_metrics = None  # dict of lists; set after reconstruct()
        # The L-curve recorded during reconstruct(), kept so plot_crossval can
        # draw its panel afterwards.
        self._lcurve = None

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
          3. transpose + y-flip to on-disk VFF convention (z, y, x), matching
             ct_core/scan_setup.py's write step (vol.transpose(2,1,0)[:, ::-1, :])

        Shared by the end-of-reconstruct() final-volume path and the
        per-checkpoint export path so both are calibrated/oriented
        identically -- see checkpoint_dir docstring in __init__.

        Returns a float32 array in (z, y, x) on-disk convention, in μ (mm⁻¹)
        and unclipped, uncropped in xy/z (caller slices).

        Checkpoints stay in μ deliberately. HU calibration is fitted from the
        histogram of a whole volume, and a crossval checkpoint is a thin slab
        that need not contain either anchor population — calibrating each slab
        against its own histogram would make successive checkpoints drift on
        an inconsistent scale, which is the opposite of what a convergence
        series is for.
        """
        vol = vol_tigre.transpose(2, 1, 0).astype(np.float32)  # (x, y, z)
        vol = self._crop_to_roi(vol, Nx_orig, Ny_orig, Nz_orig)
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
        print(f"  [checkpoint] saved {out_path.name} {slab.shape} "
              f"[{slab.min():.6f}, {slab.max():.6f}] mm^-1 (uncalibrated μ)")

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
            idx = resolve_holdout_index(self.holdout_index, sinogram.shape[0])
            holdout_proj = sinogram[idx].copy()       # (N_b, N_a) TIGRE convention
            holdout_angle = tigre_angles[idx:idx + 1]
            holdout_deg = float(np.rad2deg(self.angles[idx]))
            if self.withhold_eval:
                sinogram = np.delete(sinogram, idx, axis=0)
                tigre_angles = np.delete(tigre_angles, idx, axis=0)
                print(f"\nCross-validation: WITHHOLDING projection {idx} "
                      f"({holdout_deg:.1f}° FDK) from the reconstruction; "
                      f"eval every {self.eval_every} iters, "
                      f"patience={resolve_patience(self.iterations, self.eval_every, self.patience)} "
                      f"checkpoints.")
            else:
                print(f"\nDiagnostics: evaluating against projection {idx} "
                      f"({holdout_deg:.1f}° FDK), which STAYS in the "
                      f"reconstruction (pass withhold_eval for true "
                      f"validation); eval every {self.eval_every} iters, "
                      f"patience={resolve_patience(self.iterations, self.eval_every, self.patience)} "
                      f"checkpoints.")

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

        if not self.pwls and self.algorithm != 'mlem':
            # Supply R = 1/L_i ourselves so IterativeReconAlg.set_w() — whose
            # nVoxel=[2,2,2] grid hangs the CUDA kernel at some field sizes —
            # is never called. Same quantity, non-degenerate grid; see
            # geometric_row_weights. MLEM has no relaxation weights.
            print("\nComputing row weights R = 1/L_i "
                  f"(grid {self.set_w_grid}^3; TIGRE's own 2^3 grid hangs at "
                  f"some field sizes)...")
            _t = time.time()
            kwargs['W'] = geometric_row_weights(
                geo, tigre_angles, gpuids=kwargs.get('gpuids'),
                grid=self.set_w_grid)
            print(f"  done in {time.time() - _t:.1f} s")

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
            # Cross-validation: chunked loop, stopping rules from ct_core.
            print(f"\n  {'Iter':>6}   {'PSNR (dB)':>10}   {'SSIM':>10}   "
                  f"{'Holdout MSE':>14}   {'':>4}")
            # The scorer owns the data range and the SSIM identity. Both used to
            # be local here: the range was recomputed from the held-out
            # projection (fine) but SSIM came from skimage's DEFAULT 7x7 UNIFORM
            # window, while the diag/ssim this same loop logged to W&B came from
            # ct_core's 11x11 Gaussian on the domain-covered crop. The number
            # that decided when to stop was therefore not the number you could
            # see, and neither was comparable with any other backend's.
            scorer = HoldoutScorer(holdout_proj, label=f"projection {idx}")
            _patience = resolve_patience(self.iterations, self.eval_every,
                                         self.patience)
            _min_iter = resolve_min_iter(self.iterations, self.min_stop_iter)
            stopper = EarlyStopper(patience=_patience, metric=self.stop_metric,
                                   min_iter=_min_iter)
            lcurve = None
            if self.l_curve:
                # A volume backend can afford the EXACT residual over the whole
                # sinogram — one forward projection per checkpoint — so this
                # L-curve is the real thing, not a subset proxy.
                lcurve = LCurve(patience=_patience, norm=self.l_curve_norm,
                                residual_kind="full sinogram")
            rules = StoppingRules(stopper=stopper, lcurve=lcurve,
                                  stop_on=self.stop_on)
            self._lcurve = lcurve
            print(f"  Stopping on {' + '.join(self.stop_on)}: held-out "
                  f"{self.stop_metric} (patience {_patience}, no stop "
                  f"before iteration {_min_iter})"
                  + (f", L-curve corner ({self.l_curve_norm} norm, exact residual)"
                     if lcurve is not None else ""))

            vol_tigre = None
            best_vol = None
            best_iter = 0
            lcurve_vols: dict = {}
            chunk_kwargs = {**kwargs, 'verbose': False}
            i_done = 0

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

                m = scorer.score(pred)
                mse, psnr, ssim = m["mse"], m["psnr"], m["ssim"]

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

                improved = stopper.update(i_done, m,
                                          snapshot_fn=lambda: vol_tigre.copy())
                if improved:
                    best_vol = stopper.best_state
                    best_iter = i_done
                    marker = '★'
                else:
                    marker = f'-{stopper.num_bad}'

                if lcurve is not None:
                    # Exact data residual: forward-project the current volume at
                    # EVERY angle actually used. One Ax, which is the same cost
                    # as one iteration of the algorithm itself.
                    full_pred = tigre.Ax(vol_tigre, geo, tigre_angles, 'interpolated')
                    resid = float(np.linalg.norm(full_pred - sinogram))
                    del full_pred
                    lcurve.add(i_done, resid, solution_norm(vol_tigre, self.l_curve_norm))
                    if 'lcurve' in self.stop_on:
                        lcurve_vols[i_done] = vol_tigre.copy()
                        keep_depth = (max(LCurve.MIN_POINTS, lcurve.smooth)
                                      + _patience + 1)
                        for old in sorted(lcurve_vols)[:-keep_depth]:
                            del lcurve_vols[old]
                    if self.log_fn is not None:
                        c_it, _ = lcurve.corner()
                        self.log_fn({"lcurve/residual": resid,
                                     "lcurve/solution": lcurve.solution[-1],
                                     **({"lcurve/corner_iter": c_it}
                                        if c_it is not None else {})}, i_done)

                print(f"  {i_done:>6}   {psnr:>10.4f}   {ssim:>10.6f}   "
                      f"{mse:>14.8e}   {marker}")

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(vol_tigre, i_done, Nx_orig, Ny_orig, Nz_orig)

                if rules.should_stop():
                    print(f"\n  Early stopping: {rules.reason()}.")
                    break

            else:
                # Loop completed without a rule firing — save the final
                # iteration, not the peak. The user ran all iterations
                # intentionally, so honour that and just say what was left on
                # the table.
                if stopper.best_iter is not None and stopper.best_iter < i_done:
                    print(f"\n  Note: peak {self.stop_metric}={stopper.best:.6g} "
                          f"was at iter {stopper.best_iter}, not the final "
                          f"iteration. Saving final iteration ({i_done}).")
                best_vol = vol_tigre
                best_iter = i_done
                rules.fired = None

            # Which iterate to keep. The held-out rule knows its own peak; the
            # L-curve's answer is its corner, which is why the recent volumes
            # were kept.
            if rules.fired == 'lcurve':
                keep = rules.best_iter()
                if keep in lcurve_vols:
                    best_vol, best_iter = lcurve_vols[keep], keep
                else:
                    print(f"  WARNING: the L-curve corner is at iter {keep} but "
                          f"that volume was not retained — saving iter {i_done}.")
                    best_vol, best_iter = vol_tigre, i_done
            lcurve_vols.clear()

            vol_tigre = best_vol
            self.best_iter = best_iter
            self.crossval_metrics = metrics_dict(
                stopper, lcurve, holdout_index=idx, holdout_deg=holdout_deg,
                stop_iter=i_done, fired=rules.fired,
                delivered_iter=best_iter)
            print(f"  Saving volume from iter {best_iter} "
                  f"(stopped by {rules.fired or 'iteration limit'})\n")

            if self.checkpoint_dir is not None:
                metrics_path = write_metrics(
                    Path(self.checkpoint_dir) / "crossval_metrics.json",
                    self.crossval_metrics)
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

        # No HU conversion: the volume stays in μ (mm⁻¹), unclipped, and is
        # calibrated once downstream (ct_core.hu_calibration) so every backend
        # lands on one scale fitted from the finished volume.
        print("\nReconstruction complete.")
        return self.reconstructed_volume
    def plot_crossval(self, save_prefix):
        """Save the convergence figure to {save_prefix}_convergence.{png,pdf}.

        The figure is `ct_core.early_stop.plot_convergence` — the same one the
        learned backends and muNeRF produce, so two runs of different backends
        are read off identically shaped plots. This method used to be 165 lines
        of its own matplotlib, which is how it came to plot a differently-defined
        SSIM from the one every other backend reported.

        Must be called after reconstruct() when crossval=True.
        """
        if self.crossval_metrics is None:
            return
        m = self.crossval_metrics
        history = {"iters": m["iters"], "ssim": m["ssim"],
                   "psnr": m["psnr"], "mse": m["mse"]}
        return plot_convergence(
            history, m.get("best_iter"), m.get("stop_iter"),
            self.eval_every, save_prefix,
            title=f"{self.algorithm.upper()} — held-out projection "
                  f"{m.get('holdout_index')}",
            lcurve=self._lcurve)
