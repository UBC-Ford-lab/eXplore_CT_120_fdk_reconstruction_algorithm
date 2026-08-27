"""The learning-based reconstruction loop, independent of the representation.

Reconstruction as optimization is the same loop whatever the volume is
parameterised by: draw rays, render them through the differentiable forward
operator, score against the measured line integrals, step, watch a held-out
projection, stop at the turnover, export. What changes between a dense voxel
grid and a coordinate network is only:

  1. **what the model is**            -> ``build_model`` / ``model_fn=``
  2. **what the integration domain is** -> ``build_domain`` / ``domain_fn=``
  3. **how a volume comes out of it**   -> ``export_volume`` / ``export_fn=``

Everything else — preprocessing, the loss/sampler pairing, the data budget,
the stopping rules, best-iterate restore, the logging hooks — is shared, and
that is the point of this file: before it, the loop lived inside
``voxel/reconstructor.py`` with ``VoxelGrid`` hardcoded in three places, so a
second representation could only be added by copying it.

The domain hook is the one that is easy to underestimate. The default is
``model_domain_from_geometry``, i.e. the integration domain IS the export FOV,
which is right for a backend compared against FDK/SIRT on the same grid. It is
NOT right when rays pass through matter outside the FOV (a scan bed, a holder,
the rest of the specimen in z): that attenuation has to be somewhere in the
model or it lands in the loss as an unmodelled residual. ``Scene`` has always
kept ``model_domain`` and ``export_aabb_*`` apart for exactly this reason; only
this loop had conflated them.

Each hook can be supplied EITHER by subclassing (override the method) OR by
passing a callable to the constructor. The callable wins. Subclass when the
representation is a permanent backend — ``voxel.VoxelReconstructor`` does —
and pass callables when a caller is injecting an experimental model, which is
what muNeRF does.

A CALLER MAY ALSO OWN MORE THAN THE REPRESENTATION. muNeRF owns its data (an
enlarged integration domain, a pooled sinogram, a detector-warp calibration),
its objective (a data term PLUS a self-calibrated structural term, which the
mutually-exclusive ``DATA_TERMS`` registry cannot express), its schedule (FDK
pretrain -> handoff warmup -> main) and its diagnostics (FDK agreement,
bone-band Dice, HU calibration — all of which need the MODEL, not just a
predicted projection). Each of those is a hook rather than a flag, so this
loop never grows a branch per caller:

| hook | signature | for |
|---|---|---|
| ``scene=`` | a ``Scene`` | caller-owned data; nothing is loaded or re-domained here |
| ``loss_fn=`` | ``(pred, target, it) -> Tensor`` | an objective the registry cannot name |
| ``sampler_fn=`` | ``(scene, gen, device) -> (batch_fn, desc)`` | required with ``loss_fn`` — the term dictates the batch SHAPE |
| ``stages=`` | ``[stage(model, scene, device)]`` | pre-main phases with their own objective and optimizer |
| ``extra_loss_fn=`` | ``(model, it, loss) -> Tensor`` | additive volume priors (``loss`` is what a self-calibrated weight balances against) |
| ``render_fn=`` | ``(o, d, model, scene, **kw) -> (M,)`` | a different forward operator (hierarchical, GH-NAF) |
| ``export=`` | ``bool`` | off when the caller exports through its own pipeline |
| ``on_iter=`` | ``(model, it, optimizer) -> dict?`` | per-step schedules; may return ``{"optimizer", "lr_schedule_fn"}`` to hand a staged run to a new network |
| ``on_step=`` | ``(model, it, loss)`` | per-step bookkeeping (checkpoints, a loss history) |
| ``lr_schedule_fn=`` | ``(it) -> multiplier`` | a caller-owned LR schedule; multiplier form keeps per-group ratios |
| ``on_eval=`` | ``(model, scene, it, metrics) -> dict`` | volume-domain diagnostics; returned scalars join the log stream |
| ``optimizer_fn=`` | ``(model) -> Optimizer`` | a caller-owned parameter grouping |
| ``early_stopper=`` | ``EarlyStopper`` | share ONE stopper when the caller reads it too |
| ``snapshot_fn=`` / ``restore_fn=`` | ``() -> state`` / ``(state)`` | when "the model" is more than one module |
| ``lr_plateau=`` | ``PlateauLRReducer`` | closed-loop LR; takes over at the end of warmup |
| ``on_lr_stage=`` | ``(volume_mu, it, stage, lr) -> None`` | the volume at every LR change, for a scrollable record of the stages |

``render_fn`` returns ONE prediction because that is what the loss is scored
on. An operator with more outputs — hierarchical's coarse AND fine, GH-NAF's
uncertainty and attention — returns the one that is scored and leaves the rest
in a closure the caller's ``loss_fn`` also sees. The two hooks are supplied by
the same caller, so sharing state between them needs no API for it.

``lr_plateau`` also changes when the run ENDS: with a reducer present, patience
running out buys a reduction instead of a stop, and training ends only once the
LR is at its floor and the metric still will not improve. Without it the
open-loop cosine owns the LR and the stopper ends the run on its own.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..ct_core.data_budget import RANDOM, data_budget, measurement_count
from ..ct_core.early_stop import (STOP_METRICS, EarlyStopper, HoldoutScorer,
                                  LCurve, StoppingRules, metrics_dict,
                                  solution_norm, resolve_holdout_index)
from ..ct_core.preprocessing import preprocess_sinogram
from .losses import DEFAULT_DATA_TERM, build_data_term
from .ray_sampler import (rays_from_indices, sample_projection_patch,
                          sample_random_rays, sample_random_rows)
from .renderer import (fusion_supported, ray_domain_intersect,
                       render_compile_mode, render_rays, set_render_compile)
from .scene import Scene, model_domain_from_geometry
from ..ct_core.early_stop import resolve_min_iter, resolve_patience
from .training import (OPTIMIZERS, autocast_ctx, build_optimizer,
                       clip_grad_norm,
                       lr_multiplier, maybe_compile_model, project_nonneg,
                       resolve_amp_dtype, unwrap_model)
from .volume import EXPORT_CHUNK, render_volume  # noqa: F401 (EXPORT_CHUNK re-exported)


class LearnedReconstructor:
    """Differentiable-rendering reconstruction, following the backend contract.

    consume  raw-count projections (N_angles, N_b, N_a) + angles (radians, FDK
             convention) + a ct_core ``build_geometry`` dict
    return   float32 (Nx, Ny, Nz) volume of linear attenuation mu (mm^-1) via
             ``reconstructed_volume`` — NOT HU; calibration is a single
             downstream step shared by every backend (ct_core.hu_calibration)
    """

    def __init__(self, projections, angles, geometry,
                 iterations: int = 20000,
                 rays_per_batch: int = 16384,
                 lr: float = 1e-4,
                 samples_per_ray: int | None = None,
                 subpixel_rays: bool = True,
                 lr_warmup_iters: int = 500,
                 gpu_index: int = 0,
                 seed: int = 0,
                 compile_mode: str = "on",
                 bright_field=None, dark_field=None,
                 ring_correction: bool = False,
                 air_normalization: bool = True,
                 soft_clip_sharpness: float = 200.0,
                 ring_median_width: int = 51,
                 crossval: bool = True,
                 holdout_index: int | None = None,
                 withhold_eval: bool = False,
                 eval_every: int = 250,
                 patience: int | None = None,
                 min_stop_iter: int | None = None,
                 stop_metric: str = "mse",
                 stop_min_delta: float = 0.0,
                 save_best: bool = True,
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
                 # ---- representation hooks (see the module docstring) ----
                 model_fn=None, domain_fn=None, export_fn=None,
                 # ---- caller-owned data, objective and schedule ----
                 scene=None, loss_fn=None, sampler_fn=None, render_fn=None,
                 stages=(), extra_loss_fn=None,
                 on_iter=None, on_step=None, on_eval=None,
                 lr_schedule_fn=None, lr_plateau=None,
                 on_lr_stage=None, export: bool = True,
                 optimizer_fn=None, early_stopper=None,
                 snapshot_fn=None, restore_fn=None,
                 # ---- training mechanics (see .training) ----
                 lr_groups: dict | None = None,
                 weight_decay: float = 0.0,
                 amp: str = "off",
                 compile_model: bool = False,
                 grad_clip_norm: float = 0.0,
                 log_fn=None, diag_fn=None):
        self.projections = projections
        # `angles` may be None when a prebuilt Scene is supplied — it carries
        # its own. Resolved in reconstruct() once the scene is known.
        self.angles = None if angles is None else np.asarray(angles)
        self.geometry = dict(geometry)
        self.iterations = int(iterations)
        self.rays_per_batch = int(rays_per_batch)
        self.lr = float(lr)
        self.samples_per_ray = samples_per_ray
        #: Place each ray uniformly inside its detector pixel rather than at
        #: the centre. ON by default: the pixel INTEGRATES the beam over its
        #: footprint, so the centre is an approximation the stochastic sampler
        #: removes for free (measured -14% volume noise, -19% edge fringes on
        #: Scan_1988). See ray_sampler.sample_random_rays.
        self.subpixel_rays = bool(subpixel_rays)
        self.lr_warmup_iters = int(lr_warmup_iters)
        self.gpu_index = int(gpu_index)
        self.seed = int(seed)
        # torch.compile fusion of the RENDERER's elementwise kernels. ON by
        # default: the forward is bandwidth-bound and fusion roughly halves
        # the traffic. It reorders floating-point ops, so a compiled run is
        # not bit-comparable with an eager one — which is why the mode has to
        # travel with the run's config, and why a run that must match an eager
        # baseline passes 'off' explicitly. Downgraded to eager, with a
        # printed reason, on hardware Triton cannot target (see
        # renderer.fusion_supported). The MODEL is
        # compiled separately (`compile_model`) because the two are worth
        # different amounts: a hash grid + MLP trunk benefits, a single
        # grid_sample does not.
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
        # None = derive from the schedule (ct_core.early_stop).
        self.patience = None if patience is None else int(patience)
        self.min_stop_iter = (None if min_stop_iter is None
                              else int(min_stop_iter))
        # Which held-out metric decides the peak. They do not peak together:
        # MSE is (a subset of) the objective and turns over last, SSIM is
        # structural and turns over earliest.
        if stop_metric not in STOP_METRICS:
            raise ValueError(f"stop_metric must be one of {list(STOP_METRICS)}, "
                             f"got {stop_metric!r}")
        self.stop_metric = str(stop_metric)
        # In the METRIC'S OWN UNITS, which differ by orders of magnitude
        # between ssim/psnr/mse — EarlyStopper warns when the threshold is
        # large enough to make every checkpoint a failure.
        self.stop_min_delta = float(stop_min_delta)
        # Keeping the best state costs a full model clone on every improvement.
        # Worth it almost always — it is the point of watching — but a large
        # dense grid with a monotone criterion makes "best" mean "last", and
        # then the clone buys nothing. Off => the last iterate is returned and
        # the periodic checkpoints are the way back.
        self.save_best = bool(save_best)
        # The L-curve needs no held-out data, so it is the criterion that still
        # applies when the eval projection stays in training (the default here).
        self.l_curve = bool(l_curve)
        self.l_curve_norm = str(l_curve_norm)
        self.l_curve_snapshot_gib = float(l_curve_snapshot_gib)
        self.stop_on = tuple(stop_on)
        self.diag_downsample = max(1, int(diag_downsample))
        self.figure_every_evals = max(1, int(figure_every_evals))
        self.log_every = int(log_every)
        self.loss = str(loss).strip().lower()
        self.loss_options = dict(loss_options or {})
        self.emulate_sart = bool(emulate_sart)
        self.sart_outside_weight = float(sart_outside_weight)
        self.sart_coverage_rays = int(sart_coverage_rays)

        # ---- representation hooks -------------------------------------------
        # A callable here OVERRIDES the corresponding method, so a caller can
        # inject a model without subclassing. None = use the method, which for
        # a subclass is its own and for this class is the documented default.
        self._model_fn = model_fn
        self._domain_fn = domain_fn
        self._export_fn = export_fn

        # ---- caller-owned data, objective and schedule ----------------------
        # These exist because a caller can legitimately own more than the
        # representation. muNeRF owns its Scene (an enlarged integration
        # domain, a pooled sinogram, a detector-warp calibration), its
        # objective (a data term PLUS a self-calibrated structural term, which
        # the mutually-exclusive DATA_TERMS registry cannot express), and its
        # schedule (FDK pretrain -> handoff warmup -> main). Each of those is a
        # hook rather than a flag, so this loop never grows a branch per
        # caller.
        #
        # `scene` REPLACES projections/preprocessing/build_domain entirely —
        # pass one and this class does no data loading at all.
        self._scene = scene
        # loss_fn(pred, target, it) -> Tensor. Overrides the DATA_TERMS
        # registry. `it` is passed because a self-calibrated weight is a
        # function of the iteration, not a constant.
        self._loss_fn = loss_fn
        # sampler_fn(scene, generator, device) -> (sample_batch, description).
        # The loss dictates the SHAPE of a batch, so a custom loss usually
        # needs a custom sampler; supplying one without the other is how a 1-D
        # strip of rays gets scored as if it were an image.
        self._sampler_fn = sampler_fn
        # render_fn(origins, directions, model, scene, *, num_samples,
        # stratified, generator, grad_scale_fn) -> (M,) predictions.
        # Defaults to `renderer.render_rays`. Overridden when the forward
        # operator itself differs rather than the model: NeRF-style
        # coarse+fine sampling (`render_rays_hierarchical`) and GH-NAF's
        # discrepancy/uncertainty heads both render differently, and neither
        # is expressible as a choice of model.
        self._render_fn = render_fn or render_rays
        # Pre-main phases, run in order after the model is built and before the
        # main loop: stage(model, scene, device). A stage owns its own
        # optimizer and schedule — it is a different objective, not a different
        # learning rate. FDK pretrain and the outside-ROI handoff warmup are
        # the two that exist.
        self.stages = tuple(stages or ())
        # extra_loss_fn(model, it, loss) -> Tensor added to the data loss.
        # Volume priors (TV, census, spectral, sharp-interface) are additive
        # and independent, which is exactly what the DATA_TERMS registry is
        # not. The current data loss is passed because a SELF-CALIBRATED prior
        # sets its weight from the ratio of its own gradient RMS to the data
        # term's — it has to differentiate the thing it is being balanced
        # against, so seeing the model alone is not enough.
        self._extra_loss_fn = extra_loss_fn
        # on_iter(model, it, optimizer): every iteration, before the batch.
        # For schedules that are a function of the step — a curriculum band
        # window, or a staged switch that hands training from one network to
        # another by zeroing a param group's LR, which is why the optimizer is
        # in the signature. Keep it cheap: it runs in the hot loop.
        self._on_iter = on_iter
        # on_eval(model, scene, it, metrics) -> dict | None: at each held-out
        # evaluation, with the MODEL in hand. This is the one `diag_fn` cannot
        # do: a diagnostic that renders a volume (FDK agreement, bone-band
        # Dice, HU calibration) needs the model, not just a predicted
        # projection. Returned scalars are merged into what log_fn receives.
        # on_step(model, it, loss): after the optimizer step, every iteration.
        # For per-iteration bookkeeping the loop does not do itself —
        # checkpoint cadences, a full loss history, a caller's own metric
        # stream.
        self._on_step = on_step
        self._on_eval = on_eval
        self._on_lr_stage = on_lr_stage
        # lr_schedule_fn(it) -> MULTIPLIER on each group's base LR. Defaults to
        # this loop's linear warmup + cosine. Supplied when the caller's
        # schedule is its own (constant / cosine / exponential, with its own
        # warmup); the multiplier form is what keeps per-group ratios intact,
        # since a scheduler that writes absolute LRs would flatten them.
        self._lr_schedule_fn = lr_schedule_fn
        # Whether reconstruct() produces a volume at all. Off for a caller that
        # wants the trained module and exports through its own pipeline (HU
        # calibration, VFF/TIFF writing, an ROI crop) — rendering a volume that
        # is then thrown away is not free.
        self.export = bool(export)
        # optimizer_fn(model) -> torch.optim.Optimizer. Default: Adam over
        # `build_param_groups`. Supplied when the caller's grouping is its own
        # — a dual-network run trains only one of its two networks at a time,
        # which is a fact about the run, not about the representation.
        self._optimizer_fn = optimizer_fn
        # An existing ct_core.early_stop.EarlyStopper. Supplied when the caller
        # also reads it (a convergence figure, a checkpoint decision, its
        # history), so there is ONE stopper and not two notions of "best".
        self._early_stopper = early_stopper
        # How the best iterate is captured and put back. Defaults to this
        # class's state-dict clone. Supplied when "the model" is more than one
        # module (a dual network, a compiled pair) and the caller's own
        # snapshot format is what its convergence bookkeeping reads back.
        self._snapshot_fn = snapshot_fn
        self._restore_fn = restore_fn
        # A ct_core.early_stop.PlateauLRReducer, or None for the open-loop
        # cosine. When present it takes over LR control at the end of warmup:
        # an open-loop schedule assumes a horizon that held-out stopping never
        # reaches, so the annealing phase would never happen.
        self.lr_plateau = lr_plateau

        # ---- training mechanics ---------------------------------------------
        # `lr_groups` maps a module CLASS NAME to its learning rate; everything
        # unmatched trains at `lr`. See training.build_param_groups for why
        # matching is by name (this package must not import muNeRF's encoders).
        self.lr_groups = dict(lr_groups or {})
        self.weight_decay = float(weight_decay)
        self.amp = str(amp)
        self.compile_model = bool(compile_model)
        self.grad_clip_norm = float(grad_clip_norm)

        if optimizer is None:
            self.optimizer_name = "sgd" if self.emulate_sart else "adam"
        else:
            self.optimizer_name = str(optimizer).strip().lower()
        if self.emulate_sart:
            self.loss = "sart"
            if self.optimizer_name.startswith("adam"):
                print("  NOTE: --emulate-sart with Adam. Adam's second-moment "
                      "normalisation is itself a preconditioner and competes "
                      "with C, so this is not the classical update. Drop "
                      "--optimizer to get SGD.")
        if self.optimizer_name not in OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {sorted(OPTIMIZERS)}, got "
                f"{self.optimizer_name!r}")

        # Optional live-metric sink: Callable[[dict, int], None]. Keeps this
        # package free of any wandb import; the driver passes ReconLogger.log.
        self.log_fn = log_fn
        # Optional projection-diagnostics sink: Callable[(pred, target, step),
        # figures=bool]. Also wandb-free here. muNeRF uses these two hooks to
        # attach its own diagnostics (FDK agreement, bone-band Dice, HU
        # calibration) without any of that living in this submodule.
        self.diag_fn = diag_fn

        self.reconstructed_volume = None
        self.crossval_history: list[dict] = []
        self.stop_iter = 0
        self.delivered_iter = 0
        self.stopped_by = None
        self.convergence: dict | None = None
        #: Resolved+measured ray quadrature; see _measure_quadrature.
        self.quadrature: dict = {}
        self.n_measurements = 0
        self.iterations_run = 0
        self.data_visits = 0.0
        self.data_coverage = 0.0
        #: Set by reconstruct(): the trained module, for a caller that wants
        #: the parameters rather than (or as well as) the exported volume.
        self.model = None

    # ------------------------------------------------- representation hooks --

    def build_domain(self):
        """The integration domain. DEFAULT: exactly the export FOV.

        Override (or pass ``domain_fn=``) when rays cross matter outside the
        reconstruction FOV — a scan bed, a holder, the specimen beyond the ROI
        in z. That attenuation is real and has to live somewhere in the model;
        left out, it arrives in the loss as a residual no volume can explain.
        """
        if self._domain_fn is not None:
            return self._domain_fn(self.geometry)
        return model_domain_from_geometry(self.geometry, shape="cylinder")

    def build_model(self, domain, device):
        """The model. No default — a representation is not a detail this loop
        can guess. Subclass and override, or pass ``model_fn=``.

        The contract is `CoordMLP`'s: ``(M, 3) in [-1,1]^3 -> (M,)`` (or
        ``(M, C)`` for a multi-channel/polychromatic head), where the
        normalisation spans the MODEL DOMAIN, not the export ROI.
        """
        if self._model_fn is not None:
            return self._model_fn(self.geometry, domain, device)
        raise NotImplementedError(
            f"{type(self).__name__} defines no model — override build_model() "
            f"or pass model_fn=(geometry, domain, device) -> nn.Module.")

    def export_volume(self, model, domain, device) -> np.ndarray:
        """Volume (Nx, Ny, Nz) of mu in mm^-1 over the EXPORT grid.

        DEFAULT: evaluate the model at every export voxel centre, in chunks —
        ``volume.render_volume``, the same call any diagnostic or export path
        makes, so a shipped volume and a mid-training render of it cannot
        disagree. A backend whose parameters ARE the export grid should
        override this and return them directly: resampling a grid onto itself
        is a needless trilinear pass.
        """
        if self._export_fn is not None:
            return self._export_fn(model, self.geometry, domain, device)
        return render_volume(model, self.geometry, domain, device=device)

    def _lr_stage(self, model, domain, device, it, stage, lr) -> None:
        """Hand the CURRENT volume to ``on_lr_stage``, and never let it matter.

        Fired at every LR change, including the reducer taking over — stage 0
        is the un-annealed reconstruction the later stages have to be read
        against, so a sequence that starts at the first reduction has no
        baseline in it.

        Reconstruction is the job and a figure is not worth ending a run over,
        so every failure here is reported and swallowed — the same contract
        ``log_fn`` and ``diag_fn`` already have.
        """
        if self._on_lr_stage is None:
            return
        try:
            self._on_lr_stage(self.export_volume(model, domain, device),
                              int(it), int(stage), float(lr))
        except Exception as e:
            print(f"  lr_plateau: stage views failed ({type(e).__name__}: {e})")

    # ------------------------------------------------------------------ setup

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu_index}")
        print("  WARNING: CUDA not available — differentiable-rendering "
              "reconstruction on CPU will be orders of magnitude slower.")
        return torch.device("cpu")

    def _auto_samples_per_ray(self, domain) -> int:
        """~0.55-voxel quadrature step across the widest in-plane chord.

        The export grid is critically sampled at the export voxel size; a
        coarser step aliases against it.
        """
        dx = float(self.geometry["dx"])
        chord = (2.0 * float(domain.radius_xy) if domain.radius_xy is not None
                 else float((domain.aabb_max - domain.aabb_min).max()))
        return max(64, int(math.ceil(chord / (0.55 * dx))))

    class _SkipProbeError(Exception):
        """n_probe=0 — report the analytic step only, draw no rays."""

    def _measure_quadrature(self, scene, domain, spp: int, device,
                            n_probe: int = 8192) -> dict:
        """What the ray step ACTUALLY is, measured rather than assumed.

        `_auto_samples_per_ray` sizes the step from the WIDEST chord, so that
        number is an upper bound on the step and every other ray is sampled
        finer than it. The average matters because it, not the bound, is what
        the quadrature error is a function of across the batch — and the gap
        between them is large whenever the model domain is much wider than the
        object (a short chord through the edge of the cylinder gets the same
        `spp` as one straight through the middle).

        Probe rays come from their own generator: this must not consume the
        training RNG stream, or turning the measurement on would change the
        reconstruction. No model is evaluated — it is a geometry intersection.
        """
        dx = float(self.geometry["dx"])
        widest = (2.0 * float(domain.radius_xy) if domain.radius_xy is not None
                  else float((domain.aabb_max - domain.aabb_min).max()))
        rec = {
            "samples_per_ray": int(spp),
            "mode": "pinned" if self.samples_per_ray else "auto",
            "chord_mm_widest": float(widest),
            "step_mm_widest": float(widest) / int(spp),
            "voxel_mm": dx,
        }
        if int(n_probe) <= 0:                  # probe off: analytic fields only
            rec["chord_mm_mean"] = float("nan")
            rec["ray_hit_fraction"] = float("nan")
            n_probe = 0
        try:
            if not n_probe:
                raise self._SkipProbeError
            gen = torch.Generator(device=device).manual_seed(self.seed + 7717)
            # Same exclusion the training sampler uses, so this describes the
            # rays the run actually integrates. Nothing is evaluated here, so
            # it leaks no held-out signal either way — but a probe that
            # measured a different ray population than it reports on would be
            # quietly wrong the moment an angle is withheld.
            exclude = None
            if self.withhold_eval and self.crossval and scene.n_angles > 1:
                exclude = resolve_holdout_index(self.holdout_index,
                                                scene.n_angles)
            o, d, _ = sample_random_rays(scene, int(n_probe), generator=gen,
                                         device=device, exclude_angle=exclude,
                                         subpixel=self.subpixel_rays)
            with torch.no_grad():
                t_near, t_far, hit = ray_domain_intersect(o, d, domain)
                # `hit` is the renderer's OWN validity mask, not a >0 test on
                # the segment: a ray that misses still returns a clamped
                # interval, and averaging those in would understate the step.
                seg = (t_far - t_near)[hit]
                chord_mean = float(seg.mean()) if seg.numel() else float("nan")
            rec["chord_mm_mean"] = chord_mean
            rec["ray_hit_fraction"] = float(hit.float().mean())
        except self._SkipProbeError:
            pass
        except Exception as e:                      # pragma: no cover - env dep
            print(f"  (quadrature probe unavailable: {type(e).__name__}: {e})")
            rec["chord_mm_mean"] = float("nan")
            rec["ray_hit_fraction"] = float("nan")

        rec["step_mm_mean"] = rec["chord_mm_mean"] / int(spp)
        rec["step_voxels_mean"] = rec["step_mm_mean"] / dx
        rec["step_voxels_widest"] = rec["step_mm_widest"] / dx
        # The oversampling factor the auto rule is written in terms of: 1.00
        # is one sample per voxel (Nyquist for a trilinear grid only if the
        # field were band-limited, which it is not), 1.82 is the 0.55 default.
        rec["samples_per_voxel_mean"] = (dx / rec["step_mm_mean"]
                                         if rec["step_mm_mean"] else float("nan"))
        rec["samples_per_voxel_widest"] = (dx / rec["step_mm_widest"]
                                           if rec["step_mm_widest"] else float("nan"))
        return rec

    def _preprocess(self) -> np.ndarray:
        return preprocess_sinogram(
            self.projections, self.bright_field, self.dark_field,
            ring_correction=self.ring_correction,
            air_normalization=self.air_normalization,
            soft_clip_sharpness=self.soft_clip_sharpness,
            ring_median_width=self.ring_median_width,
        )

    # ------------------------------------------------------- state snapshots

    @staticmethod
    def _snapshot(model) -> dict:
        """Clone the model state for best-iterate restore.

        A state dict rather than one named tensor, so this works for a voxel
        grid and a network alike. Kept on the parameters' own device: the
        stopper only holds ONE of these, and moving a large grid to the host on
        every improvement costs more than it saves.
        """
        return {k: v.detach().clone()
                for k, v in unwrap_model(model).state_dict().items()}

    @staticmethod
    def _restore(model, state: dict) -> None:
        unwrap_model(model).load_state_dict(state)

    @staticmethod
    def _state_cpu(model) -> dict:
        return {k: v.detach().to("cpu", copy=True)
                for k, v in unwrap_model(model).state_dict().items()}

    @staticmethod
    def _solution_vector(state_cpu: dict) -> np.ndarray:
        """Flat float vector of the solution, for the L-curve's ||x||."""
        parts = [v.reshape(-1).numpy() for v in state_cpu.values()
                 if v.is_floating_point()]
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

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
            from .sart import ray_support_lengths, roi_bounds
            rmin, rmax, radius, centre = roi_bounds(scene)
            spp = self._auto_samples_per_ray(scene.model_domain) \
                if self.samples_per_ray is None else int(self.samples_per_ray)
            # The loss reads this dict; the sampler refreshes it. L_i is a
            # property of the batch's rays, so it has to be recomputed per step.
            chord_state: dict = {}
            opts["chord_state"] = chord_state
            opts.setdefault("sart_clamp_lo", 0.25)
            opts.setdefault("sart_clamp_hi", 4.0)
            # --emulate-sart reproduces the classical update, which descends
            # the SUMMED misfit; C then supplies the per-voxel column-sum
            # normalisation. The two belong together — the sum without an
            # absolute C overshoots by the column sum, the mean with one
            # undershoots by the batch weight — so they are set from the same
            # flag and never independently.
            opts.setdefault("sart_reduction",
                            "sum" if self.emulate_sart else "mean")

            def sample_batch(*, exclude_angle=None):
                o, d, tgt = sample_random_rays(
                    scene, self.rays_per_batch, generator=gen, device=device,
                    exclude_angle=exclude_angle, subpixel=self.subpixel_rays)
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
                from .losses import _build_ramp_kernel
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
                                          exclude_angle=exclude_angle,
                                          subpixel=self.subpixel_rays)
            desc = f"{self.rays_per_batch} random rays per step"

        return build_data_term(self.loss, **opts), sample_batch, desc

    def reconstruct(self) -> np.ndarray:
        device = self._device()

        if self._scene is not None:
            # The caller owns the data. Nothing is loaded, preprocessed or
            # re-domained here — doing any of it would silently disagree with
            # the Scene that was handed over.
            scene = self._scene
            domain = scene.model_domain
            self.angles = scene.angles.detach().cpu().numpy()
            print(f"  Scene: supplied by the caller — "
                  f"{scene.n_angles} angles x {scene.detector_shape[0]} x "
                  f"{scene.detector_shape[1]}, domain "
                  f"{'cylinder r=%.1f' % domain.radius_xy if domain.radius_xy else domain.shape} mm")
        else:
            sinogram = torch.from_numpy(np.ascontiguousarray(self._preprocess()))
            angles_t = torch.as_tensor(self.angles, dtype=torch.float32)
            domain = self.build_domain()
            scene = Scene(sinogram=sinogram.to(device), angles=angles_t.to(device),
                          geometry=self.geometry, scan_name="",
                          model_domain=domain)

        spp = (int(self.samples_per_ray) if self.samples_per_ray
               else self._auto_samples_per_ray(domain))
        self.quadrature = self._measure_quadrature(scene, domain, spp, device)
        q = self.quadrature
        print(f"  Quadrature: {spp} samples/ray "
              f"({'auto' if not self.samples_per_ray else 'pinned'}) — "
              f"step {q['step_mm_mean']:.4f} mm mean / "
              f"{q['step_mm_widest']:.4f} mm at the widest chord, i.e. "
              f"{q['samples_per_voxel_mean']:.2f} samples per "
              f"{float(self.geometry['dx']):.4f} mm voxel")

        model = self.build_model(domain, device).to(device)
        n_params = sum(p.numel() for p in unwrap_model(model).parameters())
        print(f"  Model: {type(unwrap_model(model)).__name__}, "
              f"{n_params/1e6:.1f} M free parameters "
              f"({n_params * 4 / 2**30:.2f} GiB fp32, ~4x with Adam)")

        # ---- pre-main stages ------------------------------------------------
        # Run on the EAGER model and before the main optimizer exists: a stage
        # is a different objective with its own optimizer, and compiling for
        # the main loop's graph would be wasted on it. Adam state is empty
        # until the first main step, so a stage that rewrites the parameters
        # in place leaves nothing stale behind.
        for i, stage in enumerate(self.stages, 1):
            name = getattr(stage, "__name__", f"stage {i}")
            print(f"\n  --- stage {i}/{len(self.stages)}: {name} ---")
            stage(model, scene, device)

        model = maybe_compile_model(model, self.compile_model,
                                    label="torch.compile (model)")
        optimizer = (self._optimizer_fn(model) if self._optimizer_fn is not None
                     else build_optimizer(
                         model, lr=self.lr, weight_decay=self.weight_decay,
                         group_lrs=self.lr_groups,
                         optimizer=self.optimizer_name))
        amp_dtype = resolve_amp_dtype(self.amp, device)
        if amp_dtype is not None:
            print(f"  Precision: autocast {str(amp_dtype).split('.')[-1]} "
                  f"(amp={self.amp})")
        if self.grad_clip_norm > 0:
            print(f"  Gradient clipping: max_norm={self.grad_clip_norm:g}")
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
            from ..ct_core.projection_diag import covered_detector_window
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
                with torch.no_grad(), autocast_ctx(amp_dtype, device.type):
                    for i0 in range(0, h_o.shape[0], self.rays_per_batch):
                        i1 = min(i0 + self.rays_per_batch, h_o.shape[0])
                        pred[i0:i1] = self._render_fn(
                            h_o[i0:i1], h_d[i0:i1], model, scene,
                            num_samples=spp, stratified=False,
                            generator=None, grad_scale_fn=None).float()
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

        # Warmup + cosine unless the caller owns the schedule. One
        # implementation, in .training, shared with every caller.
        _schedule = self._lr_schedule_fn or lr_multiplier(
            "cosine", warmup=self.lr_warmup_iters, total=self.iterations)

        # Per-group LRs are RELATIVE to `self.lr`: the schedule scales each
        # group by the same factor, so a hash grid at 10x the MLP's rate stays
        # at 10x through warmup and decay instead of collapsing onto one curve.
        base_lrs = [g["lr"] for g in optimizer.param_groups]

        # ---- optional kernel fusion ----------------------------------------
        # Compilation happens on the first call, so do it BEFORE the timer:
        # otherwise the one-off cost lands in iteration 0 and poisons it/s.
        # The warmup draws its rays from a SEPARATE generator and never calls
        # optimizer.step(), so the model, the Adam state and the training RNG
        # stream are all exactly what an eager run would see — the only
        # difference between `--compile off` and `--compile on` is the kernels.
        set_render_compile(self.compile_mode)
        if self.compile_mode != "off":
            # Checked before compiling, not caught after: on a card Triton
            # cannot target this is the difference between one honest line and
            # a compile attempt that raises through _fused on the first render.
            _ok, _why = fusion_supported(device)
            if not _ok:
                set_render_compile("off")
                print(f"  torch.compile skipped: {_why} — running eager.")
        if render_compile_mode() != "off" or self.compile_model:
            t_c = time.time()
            warm_gen = torch.Generator(device=device).manual_seed(self.seed + 9973)
            w_o, w_d, w_t = sample_random_rays(
                scene, self.rays_per_batch, generator=warm_gen, device=device,
                exclude_angle=holdout if self.withhold_eval else None,
                subpixel=self.subpixel_rays)
            with autocast_ctx(amp_dtype, device.type):
                w_pred = self._render_fn(w_o, w_d, model, scene,
                                         num_samples=spp, stratified=True,
                                         generator=warm_gen,
                                         grad_scale_fn=None)
                w_loss = torch.mean((w_pred.float() - w_t) ** 2)
            w_loss.backward()
            optimizer.zero_grad(set_to_none=True)
            del w_o, w_d, w_t, w_pred, w_loss
            mode = render_compile_mode()
            if mode == "off" and self.compile_mode != "off":
                print("  torch.compile: unavailable — running eager (see warning)")
            elif mode != "off":
                print(f"  torch.compile ({mode}): quadrature + integration "
                      f"kernels fused in {time.time() - t_c:.1f} s. Numerics "
                      f"differ from eager in the last bits — compare only "
                      f"against other --compile {mode} runs.")
            else:
                print(f"  torch.compile (model): warmed up in "
                      f"{time.time() - t_c:.1f} s.")

        # ---- data term and the sampler it requires ------------------------
        # The loss dictates the SHAPE of a batch, so the two are resolved
        # together. Sampling flat rays and then asking for SSIM would silently
        # score a 1-D strip as if it were an image, and a ramp filter applied
        # to a random scatter of rays is not a ramp filter at all.
        if self._loss_fn is not None or self._sampler_fn is not None:
            if self._loss_fn is None or self._sampler_fn is None:
                raise ValueError(
                    "loss_fn and sampler_fn must be supplied together — the "
                    "data term dictates the SHAPE of a batch, so a custom "
                    "objective with the registry's sampler (or the reverse) "
                    "would score the wrong thing without erroring.")
            loss_fn = self._loss_fn
            sample_batch, batch_desc = self._sampler_fn(scene, gen, device)
        else:
            _registry_loss, sample_batch, batch_desc = self._build_loss(
                scene, gen, device)
            # The registry's terms are (pred, target); the loop's contract is
            # (pred, target, it) so a self-calibrated weight can see the step.
            def loss_fn(pred, target, it, _f=_registry_loss):
                return _f(pred, target)

        # C = 1/coverage, applied to the BACKWARD pass only (the forward value is
        # untouched), which is what makes it a preconditioner rather than a term
        # in the objective. Geometry-only, so it is built once. Held neutral
        # outside the object ROI: the outer annulus is the least-determined part
        # of the volume and boosting its steps is the opposite of what C is for.
        grad_scale_fn = None
        if self.emulate_sart:
            from .sart import CoveragePreconditioner, build_coverage_grid, roi_bounds
            rmin, rmax, _radius, _centre = roi_bounds(scene)
            t_cov = time.time()
            cov, g_min, g_max = build_coverage_grid(
                scene, grid_res=64, n_rays=self.sart_coverage_rays,
                aabb_min=rmin, aabb_max=rmax, device=device)
            geom = scene.geometry
            voxel_mm = (float(geom["dx"]), float(geom["dx"]),
                        float(geom["dz"]))
            grad_scale_fn = CoveragePreconditioner(
                scene, cov, g_min, g_max, outside_value=None, device=device,
                absolute=True, rays_per_batch=self.rays_per_batch,
                coverage_rays=self.sart_coverage_rays, voxel_mm=voxel_mm)
            span = (grad_scale_fn.stats["C_max"]
                    / max(grad_scale_fn.stats["C_min"], 1e-12))
            print(f"  C preconditioner: coverage grid 64^3 over the ROI in "
                  f"{time.time() - t_cov:.1f} s, C in "
                  f"[{grad_scale_fn.stats['C_min']:.3g}, "
                  f"{grad_scale_fn.stats['C_max']:.3g}] 1/mm "
                  f"(median {grad_scale_fn.stats['C_median']:.3g}, "
                  f"span {span:.1f}x)")
            print(f"    C is ABSOLUTE (1 / sum_i A_ij over the batch's rays), "
                  f"so --lr IS the classical relaxation lambda: lr=1.0 is one "
                  f"OS-SART update per step.")
            if span < 1.2:
                print("    NOTE: C is nearly flat — coverage barely varies over "
                      "the ROI, so the preconditioner is close to a no-op here.")
            print(f"\n  --emulate-sart: R (row weighting) + C (coverage "
                  f"preconditioner, absolute) + SGD on the SUMMED misfit, "
                  f"with the non-negativity projection.")
            print(f"    NOT strict SIRT: batches are random subsets rather than "
                  f"every ray per update, which is the SART/ordered-subset "
                  f"family. The batching is kept on purpose.")
        _term_name = ("caller-supplied" if self._loss_fn is not None
                      else self.loss)
        print(f"\n  Data term: {_term_name} — {batch_desc}")
        if (self._loss_fn is None and self.loss != "mse"
                and not self.emulate_sart):
            print(f"    (default is mse, the objective classical SIRT "
                  f"descends; this run departs from it)")

        # ---- stopping rules --------------------------------------------------
        # The shared implementation, so this backend, TIGRE, ASTRA and muNeRF all
        # stop on the same definitions. Notably the best-on-holdout STATE is now
        # captured: this loop used to detect the turnover and then return the
        # final grid anyway, i.e. the volume from after the peak.
        stopper = lcurve = rules = None
        # The reducer is DRIVEN by the held-out improvement flag. With no
        # held-out evaluation there is no signal, so it would pin the LR at its
        # post-warmup base for the whole run — strictly worse than the cosine
        # it displaced. Hand the LR back rather than silently doing that.
        if holdout is None and self.lr_plateau is not None:
            print("  lr_plateau: ignored — it is driven by the held-out "
                  "evaluation, and this run has none (--no-crossval). The "
                  "open-loop cosine keeps LR control.")
            self.lr_plateau = None
        if holdout is not None:
            # patience/min_iter default to fractions of THIS run's schedule
            # (see ct_core.early_stop), so the rule means the same thing at any
            # --iterations / --eval-every rather than silently becoming
            # stricter as runs get longer.
            _closed = self.lr_plateau is not None
            _patience = resolve_patience(self.iterations, self.eval_every,
                                         self.patience, closed_loop=_closed)
            _min_iter = resolve_min_iter(self.iterations, self.min_stop_iter,
                                         closed_loop=_closed)
            stopper = self._early_stopper or EarlyStopper(
                patience=_patience, min_delta=self.stop_min_delta,
                metric=self.stop_metric, min_iter=_min_iter)
            if self.patience is None or self.min_stop_iter is None:
                if _closed:
                    print(f"  Stopping guard: patience {_patience} evals "
                          f"({_patience * self.eval_every} iters); no iteration "
                          f"floor needed — the stop is gated on the LR reaching "
                          f"its floor (closed loop).")
                else:
                    print(f"  Stopping guard: patience {_patience} evals "
                          f"({_patience * self.eval_every} iters), no stop "
                          f"before iteration {_min_iter} of {self.iterations}")
            if self.l_curve:
                # A FIXED subset of rays, so this is a deterministic functional
                # of the volume and its curve is smooth — but it is not the
                # residual over the whole sinogram, which is why the kind is
                # recorded and plotted.
                kind = "holdout projection" if self.withhold_eval else "eval projection"
                lcurve = LCurve(patience=_patience, norm=self.l_curve_norm,
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
        # checkpoints), which for a large model is more host RAM than is
        # reasonable — so the depth is capped by a budget and the shortfall is
        # announced if it ever bites.
        lcurve_depth = 0
        if lcurve is not None and "lcurve" in self.stop_on:
            want = max(LCurve.MIN_POINTS, lcurve.smooth) + self.patience + 1
            state_bytes = n_params * 4
            afford = max(1, int(self.l_curve_snapshot_gib * 2**30) // max(1, state_bytes))
            lcurve_depth = min(want, afford)
            print(f"  L-curve: keeping the last {lcurve_depth} iterates on the "
                  f"host ({lcurve_depth * state_bytes / 2**30:.2f} GiB) so the "
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
            # LR ownership. Without a plateau reducer the open-loop cosine owns
            # the whole run. With one, the warmup ramp still owns the ramp, and
            # at the end of it the reducer takes over for good — which is the
            # point: a cosine scheduled over `iterations` never anneals when
            # held-out stopping fires at a fraction of them.
            if self.lr_plateau is None or it < self.lr_warmup_iters:
                scale = float(_schedule(it))
                for group, g0 in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = g0 * scale
            elif not self.lr_plateau.active:
                for group, g0 in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = g0
                self.lr_plateau.activate(optimizer)
                print(f"  lr_plateau: taking over LR control at iteration {it} "
                      f"(warmup done, base "
                      f"{', '.join(f'{g:.2e}' for g in base_lrs)})")
                self._lr_stage(model, domain, device, it, 0, base_lrs[0])

            if self._on_iter is not None:
                swap = self._on_iter(model, it, optimizer)
                if swap:
                    # A staged run hands training from one network to another,
                    # which means a NEW optimizer (fresh moments, different
                    # parameters) and usually a new schedule. Returning them is
                    # how a hook can do that without the loop knowing why.
                    optimizer = swap.get("optimizer", optimizer)
                    if "lr_schedule_fn" in swap:
                        _schedule = swap["lr_schedule_fn"]
                    base_lrs = [g["lr"] for g in optimizer.param_groups]
                    if self.lr_plateau is not None:
                        self.lr_plateau.activate(optimizer)

            origins, directions, target = sample_batch(
                exclude_angle=holdout if self.withhold_eval else None)
            with autocast_ctx(amp_dtype, device.type):
                pred = self._render_fn(origins, directions, model, scene,
                                       num_samples=spp, stratified=True,
                                       generator=gen,
                                       grad_scale_fn=grad_scale_fn)
                # `target` carries the batch's shape (rays, rows or a patch);
                # the renderer returns a flat prediction, so restore it before
                # scoring.
                loss = loss_fn(pred.reshape(target.shape), target, it)
                if self._extra_loss_fn is not None:
                    extra = self._extra_loss_fn(model, it, loss)
                    if extra is not None:
                        loss = loss + extra

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm(model, self.grad_clip_norm)
            optimizer.step()
            project_nonneg(model)
            if self._on_step is not None:
                self._on_step(model, it, loss)

            if self.log_every and (it % self.log_every == 0 or it == self.iterations - 1):
                rate = (it + 1) / max(1e-9, time.time() - t0)
                seen = budget_after(it + 1)['visits']
                _lr_now = float(optimizer.param_groups[0]["lr"])
                print(f"  iter {it:6d}/{self.iterations}  loss {loss.item():.3e}  "
                      f"lr {_lr_now:.2e}  {rate:.1f} it/s  "
                      f"{seen:.2f} visits/measurement", flush=True)
                if self.log_fn is not None:
                    self.log_fn({"train/loss": float(loss.item()),
                                 "train/lr": _lr_now,
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
                # The model is in hand here and nowhere else, which is what a
                # volume-domain diagnostic needs. Returned scalars ride along
                # with the loop's own so there is ONE logging stream and one
                # step axis.
                extra_scalars = None
                if self._on_eval is not None:
                    extra_scalars = self._on_eval(model, scene, it + 1, m)
                if self.diag_fn is not None:
                    self.diag_fn(pred_np, target_np, it + 1,
                                 figures=(n_evals % self.figure_every_evals == 0))
                elif self.log_fn is not None:
                    self.log_fn({"diag/ssim": m["ssim"], "diag/psnr": m["psnr"],
                                 "diag/mse": m["mse"]}, it + 1)
                if extra_scalars and self.log_fn is not None:
                    self.log_fn(dict(extra_scalars), it + 1)
                self.crossval_history.append(
                    {"iter": it + 1, "mse": m["mse"],
                     "ssim": m["ssim"], "psnr": m["psnr"]})

                # Keep the BEST state, not the last one. Copying the whole model
                # is only paid on an improvement, which is what snapshot_fn is for.
                improved = stopper.update(
                    it + 1, m,
                    snapshot_fn=((self._snapshot_fn
                                   or (lambda: self._snapshot(model)))
                                  if self.save_best else None))
                best_it = stopper.best_iter or best_it
                if self.lr_plateau is not None:
                    if self.lr_plateau.step(improved, optimizer):
                        # A reduction resets the stopper's patience on purpose:
                        # a noise dip should buy an LR cut, not end the run.
                        # The stop is then gated on at_floor below, so training
                        # ends only once the LR can no longer drop AND the
                        # metric still will not improve.
                        stopper.num_bad = 0
                        lrs = ", ".join(f"{g:.2e}"
                                        for g in self.lr_plateau.current_lrs(optimizer))
                        print(f"  lr_plateau: reduction "
                              f"{self.lr_plateau.num_reductions} -> {lrs}")
                        if self.log_fn is not None:
                            self.log_fn(
                                {"lr_plateau/reductions":
                                     self.lr_plateau.num_reductions,
                                 "lr_plateau/lr":
                                     self.lr_plateau.current_lrs(optimizer)[0]},
                                it + 1)
                        self._lr_stage(
                            model, domain, device, it + 1,
                            self.lr_plateau.num_reductions,
                            self.lr_plateau.current_lrs(optimizer)[0])
                if lcurve is not None:
                    state_cpu = self._state_cpu(model)
                    lcurve.add(it + 1, float(np.sqrt(m["mse"])),
                               solution_norm(self._solution_vector(state_cpu),
                                             self.l_curve_norm))
                    if "lcurve" in self.stop_on and lcurve_depth > 0:
                        # The corner is only identifiable in RETROSPECT, so
                        # "stop at the corner" can only mean "keep the corner's
                        # volume" if that iterate is still in hand. Held on the
                        # CPU — these are touched once, at the end, and would
                        # otherwise be a large GPU allocation. Depth is bounded
                        # by a RAM budget, and if the corner falls outside it the
                        # run says so rather than silently returning the wrong
                        # iterate.
                        lcurve_snapshots[it + 1] = state_cpu
                        for old in sorted(lcurve_snapshots)[:-lcurve_depth]:
                            del lcurve_snapshots[old]
                    if self.log_fn is not None:
                        c_it, _ = lcurve.corner()
                        self.log_fn({"lcurve/residual": lcurve.residual[-1],
                                     "lcurve/solution": lcurve.solution[-1],
                                     **({"lcurve/corner_iter": c_it}
                                        if c_it is not None else {})}, it + 1)
                if rules.should_stop():
                    if (self.lr_plateau is not None
                            and not self.lr_plateau.at_floor(optimizer)):
                        # Patience ran out but the LR has further to fall.
                        # Stopping here would deliver an un-annealed iterate.
                        stopper.num_bad = 0
                        rules.fired = None
                    else:
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
        # The run stopped BECAUSE the metric turned over, so the current state is
        # from after the peak by construction. Restoring is the whole point of
        # having watched. Only skipped when the loop ran to its iteration limit,
        # where the user asked for every iteration and gets the last one.
        self.stop_iter = int(it + 1)
        self.stopped_by = None if rules is None else rules.fired
        # Which iteration the RETURNED volume is from. Reported as `best_iter` so
        # the data budget credits the delivered volume, not the run.
        self.delivered_iter = self.stop_iter
        if rules is not None and self.stopped_by is not None:
            keep = rules.best_iter()
            state = (stopper.best_state if self.stopped_by == "holdout"
                     else lcurve_snapshots.get(keep))
            if keep is not None and keep != self.stop_iter and state is not None:
                print(f"  Restoring the iterate from iteration {keep} "
                      f"(stopped at {self.stop_iter}, by {self.stopped_by})")
                if self._restore_fn is not None:
                    self._restore_fn(state)
                else:
                    self._restore(model, state)
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

        # ---- export ---------------------------------------------------------
        # No HU conversion: the model holds mu (mm^-1) and it stays that way,
        # unclipped. HU is fitted once downstream (ct_core.hu_calibration).
        self.model = unwrap_model(model)
        if not self.export:
            print("  Export skipped (export=False) — the trained module is on "
                  "`.model`.")
            return None
        vol = self.export_volume(model, domain, device)
        self.reconstructed_volume = vol
        return vol
