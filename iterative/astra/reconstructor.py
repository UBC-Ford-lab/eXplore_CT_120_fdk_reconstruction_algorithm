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

from ...ct_core.early_stop import (STOP_METRICS, EarlyStopper, HoldoutScorer,
                                   LCurve, StoppingRules, metrics_dict,
                                   plot_convergence, resolve_holdout_index,
                                   resolve_min_iter, resolve_patience,
                                   solution_norm)
from ...ct_core.preprocessing import preprocess_sinogram
from ...ct_core.utils import query_gpu_memory

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
    # Measured detector in-plane rotation (about the detector normal). 0.0 is
    # a bit-exact no-op; the convention matches muNeRF's ray construction and
    # fdk.py (u' = c*u + s*v, v' = -s*u + c*v).
    psi = float(geometry.get('det_psi_rad', 0.0) or 0.0)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

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

    # Detector unit axes before psi: u_hat = (-sin b, cos b, 0), v_hat = z_hat.
    # psi rotates them in the detector plane; per-pixel vectors carry pitch.
    uX = c_psi * (-sin_b) * da
    uY = c_psi * cos_b * da
    uZ = np.full_like(angles, s_psi * da)

    vX = -s_psi * (-sin_b) * db
    vY = -s_psi * cos_b * db
    vZ = np.full_like(angles, c_psi * db)

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
                 soft_clip_sharpness=200.0, upper_clamp=True,
                 upper_clamp_value=1.05,
                 ring_correction=False, ring_median_width=51,
                 air_normalization=True,
                 crossval=True, holdout_index=None,
                 eval_every=10, patience=None, min_stop_iter=None,
                 stop_metric='ssim', l_curve=False, l_curve_norm='l2',
                 stop_on=('holdout',),
                 log_fn=None):
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

        # Ring correction
        self.ring_correction = ring_correction
        self.air_normalization = air_normalization
        self.ring_median_width = ring_median_width

        # Stopping rules (ct_core.early_stop). ASTRA had none: every requested
        # iteration ran in one call, so the iteration count was the whole
        # regularisation and it was picked by hand.
        self.crossval = bool(crossval)
        self.holdout_index = holdout_index
        self.eval_every = int(eval_every)
        # None = derive from the schedule (ct_core.early_stop).
        self.patience = None if patience is None else int(patience)
        self.min_stop_iter = (None if min_stop_iter is None
                              else int(min_stop_iter))
        if stop_metric not in STOP_METRICS:
            raise ValueError(f"stop_metric must be one of {list(STOP_METRICS)}, "
                             f"got {stop_metric!r}")
        self.stop_metric = stop_metric
        self.l_curve = bool(l_curve)
        self.l_curve_norm = str(l_curve_norm)
        self.stop_on = tuple(stop_on)
        self.log_fn = log_fn
        self.best_iter = None
        self.crossval_metrics = None
        self._lcurve = None

        self.reconstructed_volume = None

        # Detector dimensions
        self.N_angles, self.N_b, self.N_a = self.projections.shape

        if self.crossval and self.algorithm not in self.RESUMABLE \
                and self.algorithm != 'FDK_CUDA':
            print(f"  NOTE: {self.algorithm} rebuilds its internal state on each "
                  f"astra.algorithm.run() call, so it cannot be checkpointed "
                  f"without changing the iteration sequence. Running all "
                  f"{self.iterations} iterations in one call, with no stopping "
                  f"rule.")

    def _preprocess(self, chunk_angles=20):
        """
        Apply flat-field correction, log transform, and ring correction.

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
            ring_correction=self.ring_correction,
            air_normalization=self.air_normalization,
            ring_median_width=self.ring_median_width,
        )

    def _build_astra_geometries(self):
        """
        Create ASTRA projection and volume geometry objects.

        Returns:
            (proj_geom, vol_geom) — ASTRA geometry dicts
        """
        vectors = geometry_to_astra_vectors(self.angles, self.geometry)
        psi = float(self.geometry.get('det_psi_rad', 0.0) or 0.0)
        if psi:
            print(f"  Detector in-plane rotation: psi = "
                  f"{np.degrees(psi):+.4f} deg (rotated cone_vec axes)")

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
        gpu = query_gpu_memory(self.gpu_index)
        if gpu is None:
            print("  (nvidia-smi not available — skipping GPU memory check)")
            return
        print(f"  GPU {self.gpu_index} ({gpu['name']}): "
              f"{gpu['total_bytes'] / 2**30:.2f} GiB total, "
              f"{gpu['free_bytes'] / 2**30:.2f} GiB free")
        if total_estimate > gpu['free_bytes'] * 0.85:
            raise MemoryError(
                f"Estimated GPU memory need ({total_estimate / 2**30:.2f} GiB) "
                f"exceeds available ({gpu['free_bytes'] / 2**30:.2f} GiB). "
                f"Try: reduce FOV (--fov-xy, --fov-z), increase voxel size "
                f"(--voxel-xy, --voxel-z), or use --downsample."
            )

    # ---------------------------------------------------------------- stopping

    #: Algorithms whose internal state SURVIVES a partial `astra.algorithm.run`,
    #: which is what makes a checkpointed loop possible at all. CGLS is excluded
    #: deliberately: it rebuilds its Krylov state on each `run()` call, so
    #: chunking it does not continue the same iteration sequence — it restarts
    #: it, and the "iteration 30" of a chunked run is not the iteration 30 of a
    #: single call. Better to refuse the combination than to silently reconstruct
    #: something else.
    RESUMABLE = ('SIRT3D_CUDA', 'SART3D_CUDA')

    def _stopping_enabled(self) -> bool:
        return (self.crossval and self.algorithm in self.RESUMABLE
                and self.N_angles > 1)

    def _run_with_stopping(self, alg_id, vol_id, proj_geom, vol_geom, sino_astra):
        """Iterate in chunks, scoring at each checkpoint, and return the volume
        the stopping rules chose.

        ASTRA had NO stopping rule before this: it ran every requested iteration
        in a single `run()` call. For a semi-convergent method that means the
        iteration count was the only regularisation and it was chosen by hand.
        """
        idx = resolve_holdout_index(self.holdout_index, self.N_angles)
        holdout = sino_astra[:, idx, :].copy()          # (N_b, N_a)
        holdout_deg = float(np.rad2deg(self.angles[idx]))
        scorer = HoldoutScorer(holdout, label=f"projection {idx}")
        _patience = resolve_patience(self.iterations, self.eval_every,
                                     self.patience)
        _min_iter = resolve_min_iter(self.iterations, self.min_stop_iter)
        stopper = EarlyStopper(patience=_patience, metric=self.stop_metric,
                               min_iter=_min_iter)
        lcurve = (LCurve(patience=_patience, norm=self.l_curve_norm,
                         residual_kind="full sinogram") if self.l_curve else None)
        rules = StoppingRules(stopper=stopper, lcurve=lcurve, stop_on=self.stop_on)
        self._lcurve = lcurve

        print(f"\nRunning {self.algorithm} (up to {self.iterations} iterations, "
              f"checkpoint every {self.eval_every})")
        print(f"  Evaluating against projection {idx} ({holdout_deg:.1f}°), which "
              f"STAYS in the reconstruction — ASTRA's algorithms take the whole "
              f"sinogram, so a withheld angle would need a second geometry.")
        print(f"  Stopping on {' + '.join(self.stop_on)}: {self.stop_metric} "
              f"(patience {_patience}, no stop before {_min_iter})"
              + (f", L-curve corner ({self.l_curve_norm} norm, exact residual)"
                 if lcurve is not None else ""))
        print(f"\n  {'Iter':>6}   {'PSNR (dB)':>10}   {'SSIM':>10}   "
              f"{'Holdout MSE':>14}   {'':>4}")

        best_vol, best_iter, lcurve_vols = None, 0, {}
        i_done = 0
        while i_done < self.iterations:
            chunk = min(self.eval_every, self.iterations - i_done)
            astra.algorithm.run(alg_id, chunk)
            i_done += chunk
            vol = astra.data3d.get(vol_id)

            # Forward-project the current volume once; slice out the held-out
            # angle from it and use the whole thing for the residual, so the two
            # criteria cost one projection between them rather than one each.
            pid, pred_full = astra.create_sino3d_gpu(vol, proj_geom, vol_geom)
            try:
                m = scorer.score(pred_full[:, idx, :])
                improved = stopper.update(i_done, m, snapshot_fn=lambda: vol.copy())
                if lcurve is not None:
                    lcurve.add(i_done,
                               float(np.linalg.norm(pred_full - sino_astra)),
                               solution_norm(vol, self.l_curve_norm))
            finally:
                astra.data3d.delete(pid)
                del pred_full

            if improved:
                best_vol, best_iter = stopper.best_state, i_done
                marker = '*'
            else:
                marker = f'-{stopper.num_bad}'
            if lcurve is not None and 'lcurve' in self.stop_on:
                lcurve_vols[i_done] = vol.copy()
                depth = max(LCurve.MIN_POINTS, lcurve.smooth) + _patience + 1
                for old in sorted(lcurve_vols)[:-depth]:
                    del lcurve_vols[old]
            if self.log_fn is not None:
                self.log_fn({"diag/ssim": m["ssim"], "diag/psnr": m["psnr"],
                             "diag/mse": m["mse"]}, i_done)

            print(f"  {i_done:>6}   {m['psnr']:>10.4f}   {m['ssim']:>10.6f}   "
                  f"{m['mse']:>14.8e}   {marker}")

            if rules.should_stop():
                print(f"\n  Early stopping: {rules.reason()}.")
                break
        else:
            if stopper.best_iter is not None and stopper.best_iter < i_done:
                print(f"\n  Note: peak {self.stop_metric}={stopper.best:.6g} was "
                      f"at iter {stopper.best_iter}, not the final iteration. "
                      f"Saving final iteration ({i_done}).")
            best_vol, best_iter = vol, i_done
            rules.fired = None

        if rules.fired == 'lcurve':
            keep = rules.best_iter()
            if keep in lcurve_vols:
                best_vol, best_iter = lcurve_vols[keep], keep
            else:
                print(f"  WARNING: the L-curve corner is at iter {keep} but that "
                      f"volume was not retained — saving iter {i_done}.")
                best_vol, best_iter = vol, i_done
        lcurve_vols.clear()

        self.best_iter = best_iter
        self.crossval_metrics = metrics_dict(
            stopper, lcurve, holdout_index=idx, holdout_deg=holdout_deg,
            stop_iter=i_done, fired=rules.fired, delivered_iter=best_iter)
        print(f"  Saving volume from iter {best_iter} "
              f"(stopped by {rules.fired or 'iteration limit'})\n")
        return best_vol

    def plot_crossval(self, save_prefix):
        """The shared convergence figure — same one every other backend draws."""
        if self.crossval_metrics is None:
            return
        m = self.crossval_metrics
        history = {"iters": m["iters"], "ssim": m["ssim"],
                   "psnr": m["psnr"], "mse": m["mse"]}
        return plot_convergence(
            history, m.get("best_iter"), m.get("stop_iter"), self.eval_every,
            save_prefix,
            title=f"{self.algorithm} — evaluation projection "
                  f"{m.get('holdout_index')}",
            lcurve=self._lcurve)

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
        # Kept in ASTRA order for the residual and the held-out score, but only
        # when a stopping rule will actually read them — this is a full copy of
        # the sinogram and there is no reason to hold it otherwise.
        sinogram_for_resid = sino_astra.copy() if self._stopping_enabled() else None
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
                vol_astra = astra.data3d.get(vol_id)
            elif self._stopping_enabled():
                vol_astra = self._run_with_stopping(
                    alg_id, vol_id, proj_geom, vol_geom, sinogram_for_resid)
            else:
                print(f"Running {self.algorithm} ({self.iterations} iterations)...")
                astra.algorithm.run(alg_id, self.iterations)
                vol_astra = astra.data3d.get(vol_id)

            # ASTRA returns volume in (z, y, x) order
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

        # No HU conversion: the volume stays in μ (mm⁻¹), unclipped, and is
        # calibrated once downstream (ct_core.hu_calibration) so every backend
        # lands on one scale fitted from the finished volume.
        print("\nReconstruction complete.")
        return self.reconstructed_volume
