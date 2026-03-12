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

import time

import numpy as np

try:
    import tigre
    from tigre.utilities.geometry import Geometry as TigreGeometry
    import tigre.algorithms as tigre_algs
except ImportError:
    tigre = None

from .ct_core.calibration import MU_WATER_80KV
from .ct_core.preprocessing import preprocess_sinogram

SUPPORTED_TIGRE_ALGORITHMS = ('ossart', 'sart', 'sirt')

_TIGRE_ALG_FUNCS = {
    'ossart': lambda: tigre_algs.ossart,
    'sart': lambda: tigre_algs.sart,
    'sirt': lambda: tigre_algs.sirt,
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


def build_tigre_geometry(geometry, N_b, N_a):
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

    # No detector rotation
    geo.rotDetector = np.array([0.0, 0.0, 0.0])

    # Accuracy for ray-tracing interpolation
    geo.accuracy = 0.5

    # Center of rotation correction
    geo.COR = 0.0

    geo.mode = 'cone'

    return geo


class TIGREReconstructor:
    """
    Iterative cone-beam CT reconstruction using the TIGRE toolbox.

    Supports OS-SART, SART, and SIRT algorithms. TIGRE handles GPU memory
    splitting internally, enabling reconstruction of volumes that exceed
    GPU VRAM.

    The output volume convention matches FDK: (Nx, Ny, Nz) with voxel coordinates.
    """

    def __init__(self, projections, angles, geometry,
                 algorithm='ossart', iterations=100,
                 blocksize=15, lmbda=0.5, lmbda_red=0.97,
                 nonneg=True, gpu_index=0,
                 bright_field=None, dark_field=None,
                 clamp_mode='none', soft_clip_transmission=True,
                 soft_clip_sharpness=50.0, upper_clamp=True,
                 upper_clamp_value=1.05,
                 mu_water=MU_WATER_80KV, output_hu=True):
        """
        Args:
            projections: Raw projections, shape (N_angles, N_b, N_a)
            angles: Projection angles in radians (FDK convention), shape (N_angles,)
            geometry: dict with R_s, R_d, da, db, vol_shape, vol_origin, dx, dz
            algorithm: TIGRE algorithm name ('ossart', 'sart', 'sirt')
            iterations: Number of iterations (default 100)
            blocksize: Number of projections per OS-SART block (default 15).
                Smaller = more subsets = faster convergence but noisier per-update.
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
        """
        _check_tigre_available()

        if algorithm not in SUPPORTED_TIGRE_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {SUPPORTED_TIGRE_ALGORITHMS}"
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
        self.clamp_mode = clamp_mode
        self.soft_clip_transmission = soft_clip_transmission
        self.soft_clip_sharpness = soft_clip_sharpness
        self.upper_clamp = upper_clamp
        self.upper_clamp_value = upper_clamp_value

        # HU conversion parameters
        self.mu_water = mu_water
        self.output_hu = output_hu

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

        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                 '--format=csv,noheader,nounits', f'--id={self.gpu_index}'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                name = parts[0]
                total = int(parts[1]) * 2**20
                free = int(parts[2]) * 2**20
                print(f"  GPU {self.gpu_index} ({name}): "
                      f"{total / 2**30:.2f} GiB total, {free / 2**30:.2f} GiB free")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("  (nvidia-smi not available — skipping GPU info)")

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
        )

        # Step 2: Build TIGRE geometry (may pad Nxy to avoid CUDA hang)
        Nx_orig, Ny_orig, Nz_orig = self.geometry['vol_shape']
        print("\nBuilding TIGRE geometry...")
        geo = build_tigre_geometry(self.geometry, self.N_b, self.N_a)
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

        print(f"\nRunning {self.algorithm.upper()} "
              f"({self.iterations} iterations, blocksize={self.blocksize}, "
              f"lambda={self.lmbda}, lambda_red={self.lmbda_red})...")

        t_start = time.time()
        vol_tigre = alg_func(sinogram, geo, tigre_angles, self.iterations, **kwargs)
        t_recon = time.time() - t_start
        del sinogram
        print(f"  Reconstruction took {t_recon / 60:.1f} min "
              f"({t_recon / self.iterations:.1f}s/iteration)")

        # Step 6: Transpose from TIGRE (z, y, x) to FDK convention (x, y, z)
        print(f"\n  TIGRE output shape: {vol_tigre.shape} (z, y, x)")
        print(f"  TIGRE output range: [{vol_tigre.min():.6f}, {vol_tigre.max():.6f}]")

        self.reconstructed_volume = vol_tigre.transpose(2, 1, 0).astype(np.float32)
        del vol_tigre

        # Crop from centered volume to original ROI dimensions.
        # The reconstructed volume is centered at isocenter (offOrigin=0).
        # For ROI reconstruction, the ROI may be off-center, so crop indices
        # account for the ROI offset: idx = (N_big - N_roi)/2 + offset/voxel_size
        # For full-FOV (vol_origin=0), this degenerates to a simple center-crop.
        Nx_pad, Ny_pad, Nz_pad = self.reconstructed_volume.shape
        if Nx_pad != Nx_orig or Ny_pad != Ny_orig or Nz_pad != Nz_orig:
            ox, oy, oz = self.geometry.get('vol_origin', (0, 0, 0))
            dx = self.geometry['dx']
            dz = self.geometry['dz']
            x0 = round((Nx_pad - Nx_orig) / 2 + ox / dx)
            y0 = round((Ny_pad - Ny_orig) / 2 + oy / dx)
            z0 = round((Nz_pad - Nz_orig) / 2 + oz / dz)
            self.reconstructed_volume = self.reconstructed_volume[
                x0:x0 + Nx_orig, y0:y0 + Ny_orig, z0:z0 + Nz_orig
            ]
            print(f"  Cropped ({Nx_pad}, {Ny_pad}, {Nz_pad}) → "
                  f"({Nx_orig}, {Ny_orig}, {Nz_orig}) "
                  f"[start: ({x0}, {y0}, {z0})]")

        print(f"  Reordered to FDK convention: {self.reconstructed_volume.shape} (x, y, z)")

        # Step 7: Optional HU conversion
        if self.output_hu:
            print("\nConverting to Hounsfield Units...")
            mu_water = self.mu_water
            print(f"  mu_water = {mu_water:.6f} mm^-1")

            p1 = float(np.percentile(self.reconstructed_volume, 1))
            p85 = float(np.percentile(self.reconstructed_volume, 85))
            print(f"  Observed: P1 (air) = {p1:.6f}, P85 (tissue) = {p85:.6f}")

            self.reconstructed_volume = (
                (self.reconstructed_volume - mu_water) / mu_water * 1000.0
            )
            self.reconstructed_volume = np.clip(
                self.reconstructed_volume, -1024, 4095
            ).astype(np.float32)

            hu_p1 = float(np.percentile(self.reconstructed_volume, 1))
            print(f"  Post-conversion P1 (expect ~-1000 for air): {hu_p1:.0f} HU")
            print(f"  Range: [{self.reconstructed_volume.min():.0f}, "
                  f"{self.reconstructed_volume.max():.0f}] HU")

        print("\nReconstruction complete.")
        return self.reconstructed_volume
