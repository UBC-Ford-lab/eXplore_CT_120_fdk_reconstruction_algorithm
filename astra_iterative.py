"""
ASTRA Toolbox-based iterative cone-beam CT reconstruction.

Provides SIRT, CGLS, SART, and FDK reconstruction using the ASTRA toolbox
as an alternative backend to the custom torch-based FDK pipeline.

The geometry mapping translates the FDK backprojection coordinate convention
(fdk.py lines 598-612) into ASTRA's cone_vec 12-element vectors.

Requires: astra-toolbox (optional dependency, imported conditionally)
"""

import numpy as np

try:
    import astra
except ImportError:
    astra = None

from .ct_core.calibration import MU_WATER_80KV
from .ct_core.preprocessing import preprocess_sinogram

SUPPORTED_ALGORITHMS = ('SIRT3D_CUDA', 'CGLS3D_CUDA', 'SART3D_CUDA', 'FDK_CUDA')


def _check_astra_available():
    """Raise ImportError if ASTRA toolbox is not installed."""
    if astra is None:
        raise ImportError(
            "ASTRA toolbox is required for iterative reconstruction. "
            "Install with: pip install astra-toolbox  "
            "(or conda install -c astra-toolbox astra-toolbox)"
        )


def geometry_to_astra_vectors(angles, geometry):
    """
    Convert FDK geometry convention to ASTRA cone_vec 12-element vectors.

    The FDK backprojection (fdk.py:598-612) defines:
        U = R_s + x*cos(beta) + y*sin(beta)
        a = (SDD/U) * (-x*sin(beta) + y*cos(beta))

    This means:
        - Source position: (-R_s*cos(beta), -R_s*sin(beta), 0)
        - Detector center: (R_d*cos(beta), R_d*sin(beta), 0)
        - Detector u-direction (horizontal): (-sin(beta)*da, cos(beta)*da, 0)
        - Detector v-direction (vertical): (0, 0, db)

    ASTRA cone_vec format (per row):
        [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

    Args:
        angles: 1D array of projection angles in radians, shape (N_angles,)
        geometry: dict with keys R_s, R_d, da, db

    Returns:
        np.ndarray of shape (N_angles, 12) — ASTRA cone_vec vectors
    """
    angles = np.asarray(angles, dtype=np.float64)
    R_s = geometry['R_s']
    R_d = geometry['R_d']
    da = geometry['da']
    db = geometry['db']

    cos_b = np.cos(angles)
    sin_b = np.sin(angles)
    zeros = np.zeros_like(angles)

    # Source position: (-R_s*cos(beta), -R_s*sin(beta), 0)
    srcX = -R_s * cos_b
    srcY = -R_s * sin_b
    srcZ = zeros

    # Detector center: (R_d*cos(beta), R_d*sin(beta), 0)
    dX = R_d * cos_b
    dY = R_d * sin_b
    dZ = zeros

    # Detector u-direction (horizontal, per pixel):
    # From FDK formula: a-coordinate corresponds to (-sin(beta), cos(beta))
    uX = -sin_b * da
    uY = cos_b * da
    uZ = zeros

    # Detector v-direction (vertical, per pixel): (0, 0, db)
    vX = zeros
    vY = zeros
    vZ = np.full_like(angles, db)

    vectors = np.column_stack([srcX, srcY, srcZ,
                               dX, dY, dZ,
                               uX, uY, uZ,
                               vX, vY, vZ])
    return vectors


def reorder_sinogram_to_astra(projections):
    """
    Reorder sinogram axes from FDK to ASTRA convention.

    FDK convention: (N_angles, N_b, N_a) — angles, detector rows, detector cols
    ASTRA convention: (N_b, N_angles, N_a) — detector rows, angles, detector cols

    Args:
        projections: np.ndarray of shape (N_angles, N_b, N_a)

    Returns:
        np.ndarray of shape (N_b, N_angles, N_a), C-contiguous
    """
    return np.ascontiguousarray(projections.transpose(1, 0, 2))


class ASTRAReconstructor:
    """
    Iterative cone-beam CT reconstruction using the ASTRA toolbox.

    Supports SIRT3D_CUDA, CGLS3D_CUDA, SART3D_CUDA, and FDK_CUDA algorithms.
    Handles preprocessing (flat-field correction, log transform) using the same
    ct_core functions as the FDK pipeline, but skips cone-beam weighting and
    ramp filtering (those are FDK-specific; iterative algorithms model the
    forward operator internally).

    The output volume convention matches FDK: (Nx, Ny, Nz) with voxel coordinates.
    """

    def __init__(self, projections, angles, geometry,
                 algorithm='SIRT3D_CUDA', iterations=100,
                 min_constraint=None, max_constraint=None,
                 gpu_index=0, super_sampling=1,
                 bright_field=None, dark_field=None,
                 clamp_mode='none', soft_clip_transmission=True,
                 soft_clip_sharpness=50.0, upper_clamp=True,
                 upper_clamp_value=1.05,
                 mu_water=MU_WATER_80KV, output_hu=True,
                 bhc_coeffs=None,
                 ring_correction=False, ring_median_width=51):
        """
        Args:
            projections: Raw projections, shape (N_angles, N_b, N_a)
            angles: Projection angles in radians, shape (N_angles,)
            geometry: dict with R_s, R_d, da, db, vol_shape, vol_origin, dx, dz,
                      central_pixel_a, central_pixel_b
            algorithm: ASTRA algorithm name (SIRT3D_CUDA, CGLS3D_CUDA, SART3D_CUDA, FDK_CUDA)
            iterations: Number of iterations (default 100)
            min_constraint: Minimum voxel value constraint (e.g. 0.0 for non-negativity)
            max_constraint: Maximum voxel value constraint
            gpu_index: CUDA device index (default 0)
            super_sampling: Detector/voxel super-sampling factor for forward/back-projection
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
        """
        _check_astra_available()

        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {SUPPORTED_ALGORITHMS}"
            )

        self.projections = projections
        self.angles = np.asarray(angles, dtype=np.float64)
        self.geometry = geometry
        self.algorithm = algorithm
        self.iterations = iterations
        self.min_constraint = min_constraint
        self.max_constraint = max_constraint
        self.gpu_index = gpu_index
        self.super_sampling = super_sampling

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

        # BHC and ring correction
        self.bhc_coeffs = bhc_coeffs
        self.ring_correction = ring_correction
        self.ring_median_width = ring_median_width

        self.reconstructed_volume = None

        # Detector dimensions
        self.N_angles, self.N_b, self.N_a = self.projections.shape

    def _preprocess(self, chunk_angles=20):
        """
        Apply flat-field correction, log transform, BHC, and ring correction.

        Delegates to the shared preprocess_sinogram() function in ct_core.
        """
        return preprocess_sinogram(
            projections=self.projections,
            bright_field=self.bright_field,
            dark_field=self.dark_field,
            clamp_mode=self.clamp_mode,
            soft_clip_transmission=self.soft_clip_transmission,
            soft_clip_sharpness=self.soft_clip_sharpness,
            upper_clamp=self.upper_clamp,
            upper_clamp_value=self.upper_clamp_value,
            chunk_angles=chunk_angles,
            bhc_coeffs=self.bhc_coeffs,
            ring_correction=self.ring_correction,
            ring_median_width=self.ring_median_width,
        )

    def _build_astra_geometries(self):
        """
        Create ASTRA projection and volume geometry objects.

        Returns:
            (proj_geom, vol_geom) — ASTRA geometry dicts
        """
        vectors = geometry_to_astra_vectors(self.angles, self.geometry)

        # ASTRA cone_vec projection geometry
        # detector grid: N_b rows × N_a columns
        proj_geom = astra.create_proj_geom('cone_vec', self.N_b, self.N_a, vectors)

        # Volume geometry with physical coordinates
        Nx, Ny, Nz = self.geometry['vol_shape']
        ox, oy, oz = self.geometry['vol_origin']
        dx = self.geometry['dx']
        dz = self.geometry['dz']

        # Volume bounds: centered at vol_origin (supports ROI reconstruction)
        x_min = ox - (Nx / 2) * dx
        x_max = ox + (Nx / 2) * dx
        y_min = oy - (Ny / 2) * dx
        y_max = oy + (Ny / 2) * dx
        z_min = oz - (Nz / 2) * dz
        z_max = oz + (Nz / 2) * dz

        vol_geom = astra.create_vol_geom(
            Ny, Nx, Nz,
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
        )

        return proj_geom, vol_geom

    def _check_gpu_memory(self):
        """
        Estimate GPU memory requirements and warn if insufficient.

        Checks volume + sinogram + algorithm workspace against available GPU memory.
        ASTRA SIRT/CGLS internally allocates ~2x sinogram + ~2x volume for the
        forward/back-projection buffers, so the workspace multiplier accounts for that.

        Uses pynvml directly to avoid initializing a torch CUDA context
        (which itself consumes ~300-500 MB of GPU memory).
        """
        Nx, Ny, Nz = self.geometry['vol_shape']
        f32 = 4  # bytes per float32

        vol_bytes = Nx * Ny * Nz * f32
        sino_bytes = self.N_angles * self.N_b * self.N_a * f32
        # ASTRA SIRT/CGLS needs: volume + sino + copies for forward/back-proj
        workspace_bytes = vol_bytes + sino_bytes
        total_estimate = vol_bytes + sino_bytes + workspace_bytes

        print(f"\nGPU memory estimate:")
        print(f"  Volume:    {vol_bytes / 2**30:.2f} GiB ({Nx}x{Ny}x{Nz} float32)")
        print(f"  Sinogram:  {sino_bytes / 2**30:.2f} GiB ({self.N_angles}x{self.N_b}x{self.N_a} float32)")
        print(f"  Workspace: {workspace_bytes / 2**30:.2f} GiB (estimate: ~1x vol + 1x sino)")
        print(f"  Total:     {total_estimate / 2**30:.2f} GiB")

        # Query GPU memory via nvidia-smi (avoids torch CUDA context overhead)
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
                total = int(parts[1]) * 2**20  # MiB to bytes
                free = int(parts[2]) * 2**20
                print(f"  GPU {self.gpu_index} ({name}): {total / 2**30:.2f} GiB total, {free / 2**30:.2f} GiB free")
                if total_estimate > free * 0.85:
                    raise MemoryError(
                        f"Estimated GPU memory need ({total_estimate / 2**30:.2f} GiB) "
                        f"exceeds available ({free / 2**30:.2f} GiB). "
                        f"Try: reduce FOV (--fov-xy, --fov-z), increase voxel size "
                        f"(--voxel-xy, --voxel-z), or use --downsample."
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("  (nvidia-smi not available — skipping GPU memory check)")

    def reconstruct(self):
        """
        Full reconstruction pipeline.

        Steps:
        1. Preprocess (flat-field + log transform)
        2. Build ASTRA geometries
        3. GPU memory pre-flight check
        4. Reorder sinogram to ASTRA convention
        5. Create ASTRA data objects
        6. Configure and run algorithm
        7. Retrieve result, cleanup
        8. Optional HU conversion

        Returns:
            self.reconstructed_volume — np.ndarray of shape (Nx, Ny, Nz)
        """
        _check_astra_available()

        print("=" * 60)
        print(f"ASTRA Iterative Reconstruction: {self.algorithm}")
        print("=" * 60)

        # Step 1: Preprocess
        sinogram = self._preprocess()

        # Step 2: Build geometries
        print("\nBuilding ASTRA geometries...")
        proj_geom, vol_geom = self._build_astra_geometries()

        # Step 3: GPU memory check
        self._check_gpu_memory()

        # Step 4: Reorder sinogram
        print("Reordering sinogram to ASTRA convention (N_b, N_angles, N_a)...")
        sino_astra = reorder_sinogram_to_astra(sinogram)
        del sinogram

        # Step 5-7: Create ASTRA objects, run algorithm, cleanup
        sino_id = None
        vol_id = None
        alg_id = None

        try:
            # Create sinogram data object
            sino_id = astra.data3d.create('-sino', proj_geom, sino_astra)
            del sino_astra

            # Create volume data object (initialized to zero)
            vol_id = astra.data3d.create('-vol', vol_geom, 0.0)

            # Configure algorithm
            cfg = astra.astra_dict(self.algorithm)
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = vol_id

            if self.algorithm != 'FDK_CUDA':
                # Iterative algorithm options
                if self.min_constraint is not None:
                    cfg['option'] = cfg.get('option', {})
                    cfg['option']['MinConstraint'] = self.min_constraint
                if self.max_constraint is not None:
                    cfg['option'] = cfg.get('option', {})
                    cfg['option']['MaxConstraint'] = self.max_constraint

            if self.gpu_index != 0:
                cfg['option'] = cfg.get('option', {})
                cfg['option']['GPUindex'] = self.gpu_index

            if self.super_sampling > 1:
                cfg['option'] = cfg.get('option', {})
                cfg['option']['DetectorSuperSampling'] = self.super_sampling
                cfg['option']['VoxelSuperSampling'] = self.super_sampling

            # Create and run algorithm
            alg_id = astra.algorithm.create(cfg)

            if self.algorithm == 'FDK_CUDA':
                print("Running ASTRA FDK...")
                astra.algorithm.run(alg_id)
            else:
                # For CGLS3D_CUDA: all iterations must be in a single run() call
                # because it resets internal state on each call.
                # For SIRT/SART: also fine as single call for simplicity.
                print(f"Running {self.algorithm} ({self.iterations} iterations)...")
                astra.algorithm.run(alg_id, self.iterations)

            # Retrieve result
            # ASTRA returns volume in (z, y, x) order
            vol_astra = astra.data3d.get(vol_id)
            print(f"  ASTRA output shape: {vol_astra.shape} (z, y, x)")
            print(f"  ASTRA output range: [{vol_astra.min():.6f}, {vol_astra.max():.6f}]")

            # Convert from ASTRA (z, y, x) to FDK convention (x, y, z)
            self.reconstructed_volume = vol_astra.transpose(2, 1, 0).astype(np.float32)
            print(f"  Reordered to FDK convention: {self.reconstructed_volume.shape} (x, y, z)")

        finally:
            # Cleanup ASTRA objects
            if alg_id is not None:
                astra.algorithm.delete(alg_id)
            if sino_id is not None:
                astra.data3d.delete(sino_id)
            if vol_id is not None:
                astra.data3d.delete(vol_id)

        # Step 8: Optional HU conversion
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
