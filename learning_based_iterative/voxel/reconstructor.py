"""Learning-based iterative reconstruction with a dense voxel grid.

Reconstruction as optimization: the volume is a `VoxelGrid` (one free
parameter per voxel — SIRT's representation) fitted to the measured line
integrals by Adam through the differentiable renderer. This is the recipe
validated in muNeRF's ``configs/scan_1510_VOXEL_mse.yaml`` (run 9bn7j2ua),
ported to the ct_core backend contract so it is a drop-in peer of the FDK /
ASTRA / TIGRE backends:

  consume  raw-count projections (N_angles, N_b, N_a) + angles (radians, FDK
           convention) + ct_core ``build_geometry`` dict
  return   float32 (Nx, Ny, Nz) volume of linear attenuation mu (mm^-1)
           via ``reconstructed_volume`` -- NOT HU; calibration is a single
           downstream step shared by every backend (ct_core.hu_calibration)

Recipe notes carried over from the validated config:
  * plain MSE data term (SIRT's objective; no structural terms)
  * non-negativity as a PROJECTION after each step (SIRT's clip, not softplus)
  * init near air, not water — gradient pressure raises mu only where the
    projections demand it
  * short LR warmup (500) + cosine decay; the optimum sits EARLY (~16k steps
    on Scan_1510), so long schedules mostly fit noise
  * quadrature finer than the voxel (~0.55 voxel steps) to avoid aliasing
    against a critically-sampled grid
  * one held-out projection is excluded from training and scored for
    early stopping — the same SIRT-style holdout the TIGRE backend uses

Like the classical backends, the reconstruction FOV must cover the object
(truncated projections are unmodeled); the model domain here IS the export
FOV cylinder.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from ...ct_core.data_budget import RANDOM, data_budget, measurement_count
from ...ct_core.early_stop import (STOP_METRICS, EarlyStopper, HoldoutScorer,
                                   LCurve, StoppingRules, metrics_dict,
                                   resolve_holdout_index, solution_norm)
from ...ct_core.preprocessing import preprocess_sinogram
from ..scene import Scene, model_domain_from_geometry
from ..ray_sampler import (rays_from_indices, sample_projection_patch,
                           sample_random_rays, sample_random_rows)
from ..losses import DEFAULT_DATA_TERM, build_data_term
from ..renderer import render_compile_mode, render_rays, set_render_compile
from .model import VoxelGrid


class VoxelReconstructor:
    """Voxel-grid learning-based iterative reconstruction (backend contract)."""

    def __init__(self, projections, angles, geometry,
                 iterations: int = 20000,
                 rays_per_batch: int = 16384,
                 lr: float = 1e-4,
                 samples_per_ray: int | None = None,
                 lr_warmup_iters: int = 500,
                 init_density: float = 0.001,
                 gpu_index: int = 0,
                 seed: int = 0,
                 compile_mode: str = "off",
                 bright_field=None, dark_field=None,
                 ring_correction: bool = False,
                 air_normalization: bool = True,
                 soft_clip_sharpness: float = 200.0,
                 ring_median_width: int = 51,
                 crossval: bool = True,
                 holdout_index: int | None = None,
                 withhold_eval: bool = False,
                 eval_every: int = 250,
                 patience: int = 8,
                 stop_metric: str = "mse",
                 l_curve: bool = False,
                 l_curve_norm: str = "l2",
                 l_curve_snapshot_gib: float = 4.0,
                 stop_on: tuple = ("holdout",),
                 diag_downsample: int = 2,
                 figure_every_evals: int = 4,
                 log_every: int = 500,
                 loss: str = DEFAULT_DATA_TERM,
                 loss_options: dict | None = None,
                 emulate_sart: bool = False,
                 optimizer: str | None = None,
                 sart_outside_weight: float = 0.25,
                 sart_coverage_rays: int = 1_000_000,
                 log_fn=None, diag_fn=None):
        self.projections = projections
        self.angles = np.asarray(angles)
        self.geometry = dict(geometry)
        self.iterations = int(iterations)
        self.rays_per_batch = int(rays_per_batch)
        self.lr = float(lr)
        self.samples_per_ray = samples_per_ray
        self.lr_warmup_iters = int(lr_warmup_iters)
        self.init_density = float(init_density)
        self.gpu_index = int(gpu_index)
        self.seed = int(seed)
        # torch.compile fusion of the renderer's elementwise kernels. Off by
        # default: fusion reorders floating-point ops, so a compiled run is
        # not bit-comparable with an eager one and the mode has to travel
        # with the run's config. See renderer.set_render_compile.
        self.compile_mode = str(compile_mode)
        self.bright_field = bright_field
        self.dark_field = dark_field
        self.ring_correction = bool(ring_correction)
        self.air_normalization = bool(air_normalization)
        self.soft_clip_sharpness = float(soft_clip_sharpness)
        self.ring_median_width = int(ring_median_width)
        self.crossval = bool(crossval)
        self.holdout_index = holdout_index
        # Default False: the evaluation projection stays IN the training set
        # (diagnostic). True = the pre-2026-08-13 held-out behaviour.
        self.withhold_eval = bool(withhold_eval)
        self.eval_every = int(eval_every)
        self.patience = int(patience)
        # Which held-out metric decides the peak. They do not peak together:
        # MSE is (a subset of) the objective and turns over last, SSIM is
        # structural and turns over earliest. MSE is the default here because it
        # is what this backend minimises, so stopping on it is stopping on the
        # thing being optimised — but SSIM is the one to pick if the volume, not
        # the projection fit, is what matters.
        if stop_metric not in STOP_METRICS:
            raise ValueError(f"stop_metric must be one of {list(STOP_METRICS)}, "
                             f"got {stop_metric!r}")
        self.stop_metric = str(stop_metric)
        # The L-curve needs no held-out data, so it is the criterion that still
        # applies when the eval projection stays in training (the default here).
        self.l_curve = bool(l_curve)
        self.l_curve_norm = str(l_curve_norm)
        # Host-RAM budget for the iterate ring buffer that lets an L-curve
        # stop RETURN the corner's volume rather than just name it.
        self.l_curve_snapshot_gib = float(l_curve_snapshot_gib)
        self.stop_on = tuple(stop_on)
        # Evaluation renders the FULL eval projection at this detector
        # stride (SSIM needs a coherent 2-D image, not random rays).
        self.diag_downsample = max(1, int(diag_downsample))
        # Diagnostic figures (SSIM heatmap + power spectrum) are emitted on
        # every Nth eval — scalars stream at every eval regardless.
        self.figure_every_evals = max(1, int(figure_every_evals))
        self.log_every = int(log_every)
        # Data term, by name, from learning_based_iterative.losses. MSE is the
        # default because it is the objective classical SIRT descends, which is
        # what makes this backend a like-for-like comparison against it; any
        # other choice is an explicit departure and is printed at startup.
        self.loss = str(loss).strip().lower()
        self.loss_options = dict(loss_options or {})
        # --emulate-sart: the classical update, as far as this backend can go.
        # It is a PRESET, not a loss: R is the `sart` data term, C is a gradient
        # preconditioner through the renderer's grad_scale_fn, and the optimiser
        # becomes plain SGD because Adam's own per-parameter adaptive scaling
        # would compete with C for the same job. The dense voxel grid, the
        # non-negativity projection after each step, and the near-air init are
        # already the classical recipe.
        self.emulate_sart = bool(emulate_sart)
        self.sart_outside_weight = float(sart_outside_weight)
        self.sart_coverage_rays = int(sart_coverage_rays)
        # `optimizer=None` means "whatever suits the mode", which is how an
        # EXPLICIT choice stays distinguishable from the default. A caller that
        # passes 'adam' alongside emulate_sart gets Adam and a warning, not a
        # silent override — the point of the preset is to be legible, and
        # quietly ignoring an argument is the opposite of that.
        if optimizer is None:
            self.optimizer_name = "sgd" if self.emulate_sart else "adam"
        else:
            self.optimizer_name = str(optimizer).strip().lower()
        if self.emulate_sart:
            self.loss = "sart"
            if self.optimizer_name == "adam":
                print("  NOTE: --emulate-sart with Adam. Adam's second-moment "
                      "normalisation is itself a preconditioner and competes "
                      "with C, so this is not the classical update. Drop "
                      "--optimizer to get SGD.")
        if self.optimizer_name not in ("adam", "sgd"):
            raise ValueError(
                f"optimizer must be 'adam' or 'sgd', got "
                f"{self.optimizer_name!r}")
        # Optional live-metric sink: Callable[[dict, int], None] — called as
        # log_fn(metrics, step). Keeps the backend free of any wandb import;
        # the driver passes ReconLogger.log.
        self.log_fn = log_fn
        # Optional projection-diagnostics sink: Callable[(pred, target, step),
        # figures=bool] — the driver passes a wrapper around
        # ReconLogger.log_projection_diag. Also wandb-free here.
        self.diag_fn = diag_fn

        self.reconstructed_volume = None
        self.crossval_history: list[dict] = []
        # Filled in by reconstruct(): which rule ended the run, where it ended,
        # and the full convergence record (ct_core.early_stop.metrics_dict).
        self.stop_iter = 0
        self.delivered_iter = 0
        self.stopped_by = None
        self.convergence: dict | None = None
        # Filled in by reconstruct(): how much data the run actually consumed.
        # data_visits is the cross-algorithm unit — one classical iterative
        # iteration (SIRT, OS-SART) is exactly 1.00 visits per measurement,
        # so it, not the iteration count, is what compares between backends.
        self.n_measurements = 0
        self.iterations_run = 0
        self.data_visits = 0.0
        self.data_coverage = 0.0

    # ------------------------------------------------------------------ setup

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu_index}")
        print("  WARNING: CUDA not available — training the voxel grid on CPU "
              "will be orders of magnitude slower.")
        return torch.device("cpu")

    def _auto_samples_per_ray(self, domain) -> int:
        """~0.55-voxel quadrature step across the widest in-plane chord.

        The grid is critically sampled at the export voxel size; a coarser
        step aliases against it (see the VOXEL config's renderer notes).
        """
        dx = float(self.geometry["dx"])
        chord = (2.0 * float(domain.radius_xy) if domain.radius_xy is not None
                 else float((domain.aabb_max - domain.aabb_min).max()))
        return max(64, int(math.ceil(chord / (0.55 * dx))))

    def _preprocess(self) -> np.ndarray:
        return preprocess_sinogram(
            self.projections, self.bright_field, self.dark_field,
            ring_correction=self.ring_correction,
            air_normalization=self.air_normalization,
            soft_clip_sharpness=self.soft_clip_sharpness,
            ring_median_width=self.ring_median_width,
        )

    # ------------------------------------------------------------------- run

    def _build_loss(self, scene, gen, device):
        """(loss_fn, sample_batch, description) for ``self.loss``.

        The data term and the sampler are resolved together because the term
        dictates the shape of a batch:

          * per-ray terms (mse, weighted, huber) take any rays;
          * ramp-filtered terms (filtered, wiener) need COMPLETE detector rows —
            the filter is a convolution along the row, so a scattered subset of
            rays has no row to filter;
          * structural terms (ssim, msssim) need a contiguous 2-D PATCH, since
            SSIM is defined over a local window in both axes.

        `loss_options` is forwarded to the registry, which ignores keys that do
        not apply to the selected term.
        """
        opts = dict(self.loss_options)
        n_b, n_a = scene.sinogram.shape[1], scene.sinogram.shape[2]

        if self.loss == "sart":
            from ..sart import ray_support_lengths, roi_bounds
            rmin, rmax, radius, centre = roi_bounds(scene)
            spp = self._auto_samples_per_ray(scene.model_domain) \
                if self.samples_per_ray is None else int(self.samples_per_ray)
            # The loss reads this dict; the sampler refreshes it. L_i is a
            # property of the batch's rays, so it has to be recomputed per step.
            chord_state: dict = {}
            opts["chord_state"] = chord_state
            opts.setdefault("sart_clamp_lo", 0.25)
            opts.setdefault("sart_clamp_hi", 4.0)

            def sample_batch(*, exclude_angle=None):
                o, d, tgt = sample_random_rays(
                    scene, self.rays_per_batch, generator=gen, device=device,
                    exclude_angle=exclude_angle)
                chord_state["chord"] = ray_support_lengths(
                    o, d, scene, spp, roi_min=rmin, roi_max=rmax,
                    roi_radius=radius, roi_center_xy=centre,
                    outside_weight=self.sart_outside_weight)
                return o, d, tgt
            desc = (f"{self.rays_per_batch} random rays per step, row-weighted "
                    f"by 1/L over the object ROI (r={radius:.1f} mm, "
                    f"outside_weight={self.sart_outside_weight})")

        elif self.loss in ("filtered", "wiener"):
            n_rows = max(1, self.rays_per_batch // n_a)
            if "ramp_kernel" not in opts and "wiener_kernel" not in opts:
                from ..losses import _build_ramp_kernel
                key = "wiener_kernel" if self.loss == "wiener" else "ramp_kernel"
                opts[key] = _build_ramp_kernel(n_a, torch.float32, device)
                if self.loss == "wiener":
                    print("    no measured SNR gate supplied — falling back to "
                          "a plain ramp, i.e. equivalent to 'filtered'")

            def sample_batch(*, exclude_angle=None):
                return sample_random_rows(scene, n_rows, generator=gen,
                                          device=device,
                                          exclude_angle=exclude_angle)
            desc = f"{n_rows} complete detector rows per step ({n_a} cols)"

        elif self.loss in ("ssim", "msssim"):
            side = int(opts.pop("patch_size", 64))
            n_patches = int(opts.pop("num_patches",
                                     max(1, self.rays_per_batch // (side * side))))
            side = min(side, n_b, n_a)
            if "data_range" not in opts:
                # SSIM's stabilisers need a dynamic range. Taken from the
                # measured sinogram so it is a constant of the scan rather than
                # of whichever patch was drawn.
                sino = scene.sinogram
                opts["data_range"] = float(sino.max() - sino.min())

            def sample_batch(*, exclude_angle=None):
                return sample_projection_patch(scene, side, side,
                                               generator=gen, device=device,
                                               exclude_angle=exclude_angle,
                                               num_patches=n_patches)
            desc = (f"{n_patches} x {side}x{side} projection patches per step, "
                    f"data_range={opts['data_range']:.3f}")

        else:
            def sample_batch(*, exclude_angle=None):
                return sample_random_rays(scene, self.rays_per_batch,
                                          generator=gen, device=device,
                                          exclude_angle=exclude_angle)
            desc = f"{self.rays_per_batch} random rays per step"

        return build_data_term(self.loss, **opts), sample_batch, desc

    def reconstruct(self) -> np.ndarray:
        Nx, Ny, Nz = (int(v) for v in self.geometry["vol_shape"])
        n_params = Nx * Ny * Nz
        device = self._device()

        print(f"  Voxel grid: {Nx} x {Ny} x {Nz} = {n_params/1e6:.1f} M free "
              f"parameters ({n_params * 4 / 2**30:.2f} GiB fp32, ~4x with Adam)")

        sinogram = torch.from_numpy(np.ascontiguousarray(self._preprocess()))
        angles_t = torch.as_tensor(self.angles, dtype=torch.float32)

        domain = model_domain_from_geometry(self.geometry, shape="cylinder")
        scene = Scene(sinogram=sinogram.to(device), angles=angles_t.to(device),
                      geometry=self.geometry, scan_name="", model_domain=domain)

        spp = (int(self.samples_per_ray) if self.samples_per_ray
               else self._auto_samples_per_ray(domain))
        step_mm = (2.0 * float(domain.radius_xy)) / spp if domain.radius_xy else float("nan")
        print(f"  Quadrature: {spp} samples/ray (~{step_mm:.4f} mm; voxel "
              f"{float(self.geometry['dx']):.4f} mm)")

        # Grid shape == export volume shape: (Dz, Hy, Wx) tiling the FOV AABB.
        model = VoxelGrid((Nz, Ny, Nx), init_density=self.init_density).to(device)
        if self.optimizer_name == "sgd":
            # Plain steepest descent, which is what the classical update is.
            # Adam's second-moment normalisation is itself a per-parameter
            # preconditioner, so running it alongside C would give the volume two
            # competing ones and the result would not be the classical method.
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        gen = torch.Generator(device=device).manual_seed(self.seed)

        # ---- evaluation projection -----------------------------------------
        # Rendered as a coherent 2-D image (strided full detector grid, not
        # random rays) so SSIM/PSNR and the diagnostic figures are defined.
        # Withheld from training only when withhold_eval.
        holdout = None
        if self.crossval and scene.n_angles > 1:
            holdout = resolve_holdout_index(self.holdout_index, scene.n_angles)
            n_b, n_a = scene.detector_shape
            ds = self.diag_downsample
            # Only the detector window whose rays stay inside the domain —
            # outer rows exit the z-slab and outer columns miss the FOV
            # cylinder entirely, so both integrate through matter the volume
            # does not contain and would score FOV truncation, not the model
            # (same window as the noise ceiling and the classical backends'
            # final diag).
            from ...ct_core.projection_diag import covered_detector_window
            b0, b1, a0, a1 = covered_detector_window(self.geometry, n_b, n_a)
            hb_keep = torch.arange(b0, b1, ds, device=device)
            ha_keep = torch.arange(a0, a1, ds, device=device)
            hbb, haa = torch.meshgrid(hb_keep, ha_keep, indexing="ij")
            hb, ha = hbb.reshape(-1), haa.reshape(-1)
            hidx = torch.full_like(hb, holdout)
            h_o, h_d, h_t = rays_from_indices(scene, hidx, hb, ha)
            h_shape = (hb_keep.numel(), ha_keep.numel())
            h_target = h_t.reshape(h_shape)
            mode = ("WITHHELD from training" if self.withhold_eval
                    else "kept in training (diagnostic; pass withhold_eval "
                         "for true validation)")
            print(f"  Evaluation projection {holdout} — {mode}; rendered "
                  f"{h_shape[0]}x{h_shape[1]} (stride {ds}) from detector "
                  f"rows [{b0}, {b1}) of {n_b} and columns [{a0}, {a1}) of "
                  f"{n_a}, eval every {self.eval_every}, patience "
                  f"{self.patience}")
            # The window is already applied to the rays, so the scorer gets the
            # cropped target directly and needs no further window.
            scorer = HoldoutScorer(h_target.cpu().numpy(),
                                   label=f"projection {holdout}")

            def _render_eval() -> torch.Tensor:
                pred = torch.empty(h_o.shape[0], device=device)
                with torch.no_grad():
                    for i0 in range(0, h_o.shape[0], self.rays_per_batch):
                        i1 = min(i0 + self.rays_per_batch, h_o.shape[0])
                        pred[i0:i1] = render_rays(
                            h_o[i0:i1], h_d[i0:i1], model, scene,
                            num_samples=spp, stratified=False)
                return pred.reshape(h_shape)

        # ---- data budget ----------------------------------------------------
        # Rays are drawn uniformly WITH replacement over (angle, row, col), so
        # after k draws each measurement has been visited k/N times on average
        # and 1-(1-1/N)^k of them have been seen at least once. Iteration
        # counts are meaningless across batch sizes and backends; visits are
        # the common currency (a SIRT/OS-SART iteration = exactly 1.00).
        n_b_all, n_a_all = scene.detector_shape
        excluded = 1 if (holdout is not None and self.withhold_eval) else 0
        n_angles_pool = scene.n_angles - excluded
        self.n_measurements = measurement_count(
            scene.n_angles, n_b_all, n_a_all, excluded_angles=excluded)

        def budget_after(n_iters: int) -> dict:
            return data_budget(self.n_measurements,
                               rays_drawn=n_iters * self.rays_per_batch,
                               sampling=RANDOM)

        planned = budget_after(self.iterations)
        print(f"  Batch: {self.rays_per_batch} rays/iteration drawn from "
              f"{self.n_measurements / 1e6:.1f} M measurements "
              f"({n_angles_pool} angles x {n_b_all} x {n_a_all})")
        print(f"  Data budget: {self.iterations} iterations = "
              f"{planned['visits']:.2f} visits per measurement "
              f"({100 * planned['coverage']:.1f}% seen at least once). "
              f"1.00 visits = one full pass = one SIRT/OS-SART iteration.")

        def lr_at(it: int) -> float:
            if self.lr_warmup_iters and it < self.lr_warmup_iters:
                return self.lr * (it + 1) / self.lr_warmup_iters
            t = (it - self.lr_warmup_iters) / max(1, self.iterations - self.lr_warmup_iters)
            return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

        # ---- optional kernel fusion ----------------------------------------
        # Compilation happens on the first call, so do it BEFORE the timer:
        # otherwise the one-off cost lands in iteration 0 and poisons it/s.
        # The warmup draws its rays from a SEPARATE generator and never calls
        # optimizer.step(), so the model, the Adam state and the training RNG
        # stream are all exactly what an eager run would see — the only
        # difference between `--compile off` and `--compile on` is the kernels.
        set_render_compile(self.compile_mode)
        if render_compile_mode() != "off":
            t_c = time.time()
            warm_gen = torch.Generator(device=device).manual_seed(self.seed + 9973)
            w_o, w_d, w_t = sample_random_rays(
                scene, self.rays_per_batch, generator=warm_gen, device=device,
                exclude_angle=holdout if self.withhold_eval else None)
            w_pred = render_rays(w_o, w_d, model, scene, num_samples=spp,
                                 stratified=True, generator=warm_gen)
            torch.mean((w_pred - w_t) ** 2).backward()
            optimizer.zero_grad(set_to_none=True)
            del w_o, w_d, w_t, w_pred
            mode = render_compile_mode()
            if mode == "off":
                print("  torch.compile: unavailable — running eager (see warning)")
            else:
                print(f"  torch.compile ({mode}): quadrature + integration "
                      f"kernels fused in {time.time() - t_c:.1f} s. Numerics "
                      f"differ from eager in the last bits — compare only "
                      f"against other --compile {mode} runs.")

        # ---- data term and the sampler it requires ------------------------
        # The loss dictates the SHAPE of a batch, so the two are resolved
        # together. Sampling flat rays and then asking for SSIM would silently
        # score a 1-D strip as if it were an image, and a ramp filter applied
        # to a random scatter of rays is not a ramp filter at all.
        loss_fn, sample_batch, batch_desc = self._build_loss(scene, gen, device)

        # C = 1/coverage, applied to the BACKWARD pass only (the forward value is
        # untouched), which is what makes it a preconditioner rather than a term
        # in the objective. Geometry-only, so it is built once. Held neutral
        # outside the object ROI: the outer annulus is the least-determined part
        # of the volume and boosting its steps is the opposite of what C is for.
        grad_scale_fn = None
        if self.emulate_sart:
            from ..sart import CoveragePreconditioner, build_coverage_grid, roi_bounds
            rmin, rmax, _radius, _centre = roi_bounds(scene)
            t_cov = time.time()
            cov, g_min, g_max = build_coverage_grid(
                scene, grid_res=64, n_rays=self.sart_coverage_rays,
                aabb_min=rmin, aabb_max=rmax, device=device)
            grad_scale_fn = CoveragePreconditioner(
                scene, cov, g_min, g_max, outside_value=1.0, device=device)
            span = (grad_scale_fn.stats["C_max"]
                    / max(grad_scale_fn.stats["C_min"], 1e-12))
            print(f"  C preconditioner: coverage grid 64^3 over the ROI in "
                  f"{time.time() - t_cov:.1f} s, C in "
                  f"[{grad_scale_fn.stats['C_min']:.2f}, "
                  f"{grad_scale_fn.stats['C_max']:.2f}] (span {span:.1f}x)")
            if span < 1.2:
                print("    NOTE: C is nearly flat — coverage barely varies over "
                      "the ROI, so the preconditioner is close to a no-op here.")
        if self.emulate_sart:
            print(f"\n  --emulate-sart: R (row weighting) + C (coverage "
                  f"preconditioner) + SGD, on the dense voxel grid with the "
                  f"non-negativity projection.")
            print(f"    NOT strict SIRT: batches are random subsets rather than "
                  f"every ray per update, which is the SART/ordered-subset "
                  f"family. The batching is kept on purpose.")
        print(f"\n  Data term: {self.loss} — {batch_desc}")
        print(f"  Optimizer: {self.optimizer_name.upper()} (lr {self.lr:g})")
        if self.loss != "mse" and not self.emulate_sart:
            print(f"    (default is mse, the objective classical SIRT "
                  f"descends; this run departs from it)")

        # ---- stopping rules --------------------------------------------------
        # The shared implementation, so this backend, TIGRE, ASTRA and muNeRF all
        # stop on the same definitions. Notably the best-on-holdout STATE is now
        # captured: this loop used to detect the turnover and then return the
        # final grid anyway, i.e. the volume from after the peak.
        stopper = lcurve = rules = None
        if holdout is not None:
            stopper = EarlyStopper(patience=self.patience,
                                   metric=self.stop_metric)
            if self.l_curve:
                # A FIXED subset of rays, so this is a deterministic functional
                # of the volume and its curve is smooth — but it is not the
                # residual over the whole sinogram, which is why the kind is
                # recorded and plotted.
                kind = "holdout projection" if self.withhold_eval else "eval projection"
                lcurve = LCurve(patience=self.patience, norm=self.l_curve_norm,
                                residual_kind=kind)
            rules = StoppingRules(stopper=stopper, lcurve=lcurve,
                                  stop_on=self.stop_on)
            print(f"  Stopping on {' + '.join(self.stop_on)}: held-out "
                  f"{self.stop_metric} (patience {self.patience})"
                  + (f", L-curve corner ({self.l_curve_norm} norm)"
                     if lcurve is not None else ""))

        # How many past iterates to keep so the L-curve's corner can be
        # RETURNED and not merely reported. The worst-case lag is the whole
        # warm-up the rule waits through (`max(MIN_POINTS, smooth) + patience`
        # checkpoints), which for a large grid is more host RAM than is
        # reasonable — so the depth is capped by a budget and the shortfall is
        # announced if it ever bites.
        lcurve_depth = 0
        if lcurve is not None and "lcurve" in self.stop_on:
            want = max(LCurve.MIN_POINTS, lcurve.smooth) + self.patience + 1
            grid_bytes = n_params * 4
            afford = max(1, int(self.l_curve_snapshot_gib * 2**30) // max(1, grid_bytes))
            lcurve_depth = min(want, afford)
            print(f"  L-curve: keeping the last {lcurve_depth} iterates on the "
                  f"host ({lcurve_depth * grid_bytes / 2**30:.2f} GiB) so the "
                  f"corner's volume can be returned"
                  + ("" if lcurve_depth >= want else
                     f" — WANTED {want}; a corner further back than "
                     f"{lcurve_depth} checkpoints will be reported but not "
                     f"restored (raise l_curve_snapshot_gib)"))

        best_it = 0
        lcurve_snapshots: dict = {}
        t0 = time.time()
        stop_reason = "max iterations"

        for it in range(self.iterations):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(it)

            origins, directions, target = sample_batch(
                exclude_angle=holdout if self.withhold_eval else None)
            pred = render_rays(origins, directions, model, scene,
                               num_samples=spp, stratified=True, generator=gen,
                               grad_scale_fn=grad_scale_fn)
            # `target` carries the batch's shape (rays, rows or a patch); the
            # renderer returns a flat prediction, so restore it before scoring.
            loss = loss_fn(pred.reshape(target.shape), target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.clamp_nonneg()

            if self.log_every and (it % self.log_every == 0 or it == self.iterations - 1):
                rate = (it + 1) / max(1e-9, time.time() - t0)
                seen = budget_after(it + 1)['visits']
                print(f"  iter {it:6d}/{self.iterations}  loss {loss.item():.3e}  "
                      f"lr {lr_at(it):.2e}  {rate:.1f} it/s  "
                      f"{seen:.2f} visits/measurement", flush=True)
                if self.log_fn is not None:
                    self.log_fn({"train/loss": float(loss.item()),
                                 "train/lr": lr_at(it),
                                 "train/it_per_s": rate,
                                 # per step, so W&B can plot against data
                                 # visits — the batch-size-independent x-axis
                                 "train/data_visits": seen,
                                 "train/rays_per_batch": self.rays_per_batch},
                                it + 1)

            if holdout is not None and (it + 1) % self.eval_every == 0:
                h_pred = _render_eval()
                n_evals = len(self.crossval_history) + 1
                pred_np = h_pred.cpu().numpy()
                target_np = h_target.cpu().numpy()
                # One scorer, one SSIM identity, one fixed data range — so a
                # curve compared across iterations is comparing the model and
                # not a moving normalisation.
                m = scorer.score(pred_np)
                if self.diag_fn is not None:
                    self.diag_fn(pred_np, target_np, it + 1,
                                 figures=(n_evals % self.figure_every_evals == 0))
                elif self.log_fn is not None:
                    self.log_fn({"diag/ssim": m["ssim"], "diag/psnr": m["psnr"],
                                 "diag/mse": m["mse"]}, it + 1)
                self.crossval_history.append(
                    {"iter": it + 1, "mse": m["mse"],
                     "ssim": m["ssim"], "psnr": m["psnr"]})

                # Keep the BEST grid, not the last one. Copying 100+ M voxels is
                # only paid on an improvement, which is what snapshot_fn is for.
                stopper.update(it + 1, m,
                               snapshot_fn=lambda: model.mu.detach().clone())
                best_it = stopper.best_iter or best_it
                if lcurve is not None:
                    mu_cpu = model.mu.detach().cpu()
                    lcurve.add(it + 1, float(np.sqrt(m["mse"])),
                               solution_norm(mu_cpu.numpy(), self.l_curve_norm))
                    if "lcurve" in self.stop_on and lcurve_depth > 0:
                        # The corner is only identifiable in RETROSPECT, so
                        # "stop at the corner" can only mean "keep the corner's
                        # volume" if that iterate is still in hand. Held on the
                        # CPU — these are touched once, at the end, and would
                        # otherwise be a large GPU allocation. Depth is bounded
                        # by a RAM budget, and if the corner falls outside it the
                        # run says so rather than silently returning the wrong
                        # iterate.
                        lcurve_snapshots[it + 1] = mu_cpu.clone()
                        for old in sorted(lcurve_snapshots)[:-lcurve_depth]:
                            del lcurve_snapshots[old]
                    if self.log_fn is not None:
                        c_it, _ = lcurve.corner()
                        self.log_fn({"lcurve/residual": lcurve.residual[-1],
                                     "lcurve/solution": lcurve.solution[-1],
                                     **({"lcurve/corner_iter": c_it}
                                        if c_it is not None else {})}, it + 1)
                if rules.should_stop():
                    stop_reason = rules.reason()
                    break

        elapsed = time.time() - t0
        self.iterations_run = int(it + 1)
        final_budget = budget_after(self.iterations_run)
        self.data_visits = final_budget['visits']
        self.data_coverage = final_budget['coverage']
        print(f"  Training finished after {self.iterations_run} iterations "
              f"({elapsed/60:.1f} min): {stop_reason}")
        print(f"  Data consumed: {self.data_visits:.2f} visits per measurement "
              f"({self.iterations_run} x {self.rays_per_batch} rays = "
              f"{self.iterations_run * self.rays_per_batch / 1e6:.1f} M of "
              f"{self.n_measurements / 1e6:.1f} M measurements; "
              f"{100 * self.data_coverage:.1f}% seen at least once)")
        if holdout is not None and self.crossval_history:
            last = self.crossval_history[-1]
            print(f"  Eval projection: {self.stop_metric} "
                  f"{last[self.stop_metric]:.6g} at the end, best "
                  f"{stopper.best:.6g} at iter {stopper.best_iter}")
            if lcurve is not None:
                c_it, _ = lcurve.corner()
                print(f"  L-curve corner: {c_it if c_it is not None else 'none yet'}"
                      + ("" if c_it is None or stopper.best_iter is None
                         or c_it == stopper.best_iter
                         else f" — DISAGREES with the held-out peak "
                              f"({stopper.best_iter}); both are in the "
                              f"convergence figure"))
            # Guarantee final diagnostic figures even when the last eval
            # wasn't a figure checkpoint.
            if (self.diag_fn is not None
                    and len(self.crossval_history) % self.figure_every_evals != 0):
                h_pred = _render_eval()
                self.diag_fn(h_pred.cpu().numpy(), h_target.cpu().numpy(),
                             it + 1, figures=True, scalars=False)

        # ---- restore the best iterate ---------------------------------------
        # The run stopped BECAUSE the metric turned over, so the current grid is
        # from after the peak by construction. Restoring is the whole point of
        # having watched. Only skipped when the loop ran to its iteration limit,
        # where the user asked for every iteration and gets the last one.
        self.stop_iter = int(it + 1)
        self.stopped_by = None if rules is None else rules.fired
        # Which iteration the RETURNED grid is from. Reported as `best_iter` so
        # the data budget credits the delivered volume, not the run.
        self.delivered_iter = self.stop_iter
        if rules is not None and self.stopped_by is not None:
            keep = rules.best_iter()
            state = (stopper.best_state if self.stopped_by == "holdout"
                     else lcurve_snapshots.get(keep))
            if keep is not None and keep != self.stop_iter and state is not None:
                print(f"  Restoring the iterate from iteration {keep} "
                      f"(stopped at {self.stop_iter}, by {self.stopped_by})")
                with torch.no_grad():
                    model.mu.copy_(state.to(model.mu.device))
                self.delivered_iter = int(keep)
            elif keep is not None and keep != self.stop_iter:
                print(f"  WARNING: wanted the iterate from iteration {keep} but "
                      f"no snapshot was kept — returning iteration "
                      f"{self.stop_iter}.")
        lcurve_snapshots.clear()
        self.convergence = metrics_dict(
            stopper, lcurve, holdout_index=holdout,
            holdout_deg=(float(np.rad2deg(self.angles[holdout]))
                         if holdout is not None else None),
            stop_iter=self.stop_iter, fired=self.stopped_by,
            delivered_iter=self.delivered_iter)

        # ---- export: the parameter grid IS the volume ----------------------
        # mu[0, 0] is (Dz, Hy, Wx) with indices increasing along +z/+y/+x;
        # transpose to the FDK convention (Nx, Ny, Nz).
        vol = model.mu.detach()[0, 0].cpu().numpy().transpose(2, 1, 0)
        vol = np.ascontiguousarray(vol, dtype=np.float32)

        # No HU conversion: the parameter grid is μ (mm⁻¹) and stays that way,
        # unclipped. HU is fitted once downstream (ct_core.hu_calibration).
        self.reconstructed_volume = vol
        return vol
