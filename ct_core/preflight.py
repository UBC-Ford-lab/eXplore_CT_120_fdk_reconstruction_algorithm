"""Machine preflight: will this reconstruction fit on this machine?

Run by every driver after the scan is loaded (all shapes known) and BEFORE
any backend allocates GPU or large host buffers. Checks three things against
per-backend requirement estimates:

  1. Is a CUDA GPU present at all? (ASTRA and TIGRE algorithms are CUDA-only
     and would crash mid-run; FDK and the voxel trainer fall back to CPU,
     which is allowed but loudly warned — orders of magnitude slower.)
  2. Does the estimated peak VRAM fit in the GPU's free memory?
  3. Does the estimated peak host RAM fit in MemAvailable?

Each backend answers with a ``Footprint``, and the backends are a REGISTRY
(``FOOTPRINTS`` / ``register_footprint``) rather than a chain of
``if backend ==``. That matters for the learning-based family: a new
representation must be able to say how big it is without editing this file,
because its size is a fact about the representation and not about CT. The
dense voxel grid is the one learned backend registered here, and only because
its parameters ARE the exported volume — a shape this module already has.
Anything else (an INR's weight count, a splat count) arrives as
``footprint=(MachineRequest) -> Footprint`` from the algorithm itself; see
``learning_based_iterative.registry``.

A ``Footprint`` splits VRAM into what is RESIDENT for the whole run and what
scales with the ray batch. That split is not cosmetic: ``auto_rays_per_batch``
sizes the batch as "free VRAM minus resident, divided by per-ray", so a single
total would not answer the question. It is also where a network differs most
from a voxel grid — resident weights are megabytes, not gigabytes, so the same
card affords a far larger batch.

The estimates are deliberately simple, documented formulas with a safety
margin — they exist to catch the "this cannot possibly work" and "this will
be uncomfortably tight" cases up front, not to be byte-accurate:

  fdk    VRAM: processes in z-chunks sized to free VRAM at runtime, so the
         requirement is the sinogram + one modest volume slab.
         RAM: raw projections + float sinogram + full volume.
  astra  VRAM: 2x volume + 2x sinogram (SIRT/CGLS keep forward/back-projection
         buffers — same formula the backend itself uses).
         RAM: projections (float, x2 for the ASTRA reorder copy) + volume.
  tigre  VRAM: TIGRE splits internally, so the floor is the sinogram + one
         volume split; a note says larger free VRAM = fewer splits = faster.
         RAM: ~4x volume + ~3x sinogram (measured: 12 GB RSS at a 2.9 GiB
         volume — TIGRE and the crossval path keep several copies).
  learned  (voxel, and any representation registered beside it) VRAM:
         parameters x the optimizer's state count (4x for Adam: param + grad
         + m + v; 2x for plain SGD, which keeps no state) + sinogram +
         per-batch ray buffers. RAM: raw projections + float sinogram +
         exported volume. The shared shape is `learned_footprint`; only the
         PARAMETER COUNT differs per representation, and only the algorithm
         knows it.

Verdicts: ok / tight (>85% of free) / insufficient / no-gpu. `insufficient`
and `no-gpu`(when the backend requires one) abort with a clear message —
``--skip-preflight`` overrides, ``--preflight-only`` prints the report and
exits without reconstructing (for sizing jobs before submitting them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import PreflightAbort
from .utils import query_gpu_memory

GiB = float(2 ** 30)

# Resident fp32-EQUIVALENT copies of the parameters, per optimizer. Adam keeps
# the two moment buffers on top of the weights and their gradient; plain SGD
# keeps no state at all, so an --emulate-sart run needs HALF the VRAM an Adam
# run of the same grid does. Estimating every voxel run at Adam's 4x turned
# that into a spurious `insufficient` verdict and a floored ray batch.
# `adam_bf16` holds the same two moments at half width, so the pair costs one
# volume instead of two — the count is in fp32 units, hence 3 and not an
# integer number of buffers. Keys must stay in step with
# `learning_based_iterative.training.OPTIMIZERS`.
PARAM_COPIES = {"adam": 4, "adam_bf16": 3, "sgd": 2}
F32 = 4
SAFETY = 1.15          # multiplied onto every estimate
TIGHT_FRAC = 0.85      # need > 85% of free => "tight"

# Peak VRAM per quadrature sample in the differentiable renderer, in bytes.
# One ray-sample carries ~24 live fp32 values through the autograd graph
# (jitter, t-samples, mm points, normalized grid coords, grid_sample output,
# plus what backward retains). MEASURED on a P100: 16384 rays x 1091 spp
# added ~3.1 GiB over the persistent buffers = ~174 B/sample including
# caching-allocator slack; 96 B is the live-tensor estimate that the 0.85
# budget fraction and SAFETY are applied on top of.
BATCH_BYTES_PER_SAMPLE = 96


def host_mem_available_bytes():
    """MemAvailable from /proc/meminfo, or None off-Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@dataclass
class PreflightReport:
    backend: str
    gpu_required: bool
    gpu_name: str | None
    vram_free: int | None      # None = no GPU found
    vram_needed: int           # 0 = backend adapts / CPU
    ram_free: int | None
    ram_needed: int
    notes: list[str] = field(default_factory=list)
    # Set by run_preflight for a --preflight-only dry run: the machine-fit
    # question has been answered and the caller should return WITHOUT
    # reconstructing. A successful dry run is not an error, so it is a return
    # value rather than an exception (and never a sys.exit from a library).
    dry_run: bool = False

    @property
    def verdict(self) -> str:
        if self.vram_free is None:
            return "no-gpu"
        if (self.vram_needed > self.vram_free
                or (self.ram_free is not None and self.ram_needed > self.ram_free)):
            return "insufficient"
        if (self.vram_needed > TIGHT_FRAC * self.vram_free
                or (self.ram_free is not None
                    and self.ram_needed > TIGHT_FRAC * self.ram_free)):
            return "tight"
        return "ok"

    @property
    def fatal(self) -> bool:
        v = self.verdict
        return v == "insufficient" or (v == "no-gpu" and self.gpu_required)


# ------------------------------------------------------------- estimators --

def _sino_bytes(n_angles: int, n_b: int, n_a: int) -> int:
    return n_angles * n_b * n_a * F32


def _vol_bytes(vol_shape) -> int:
    nx, ny, nz = (int(v) for v in vol_shape)
    return nx * ny * nz * F32


@dataclass(frozen=True)
class MachineRequest:
    """Everything a footprint model is allowed to see about the job.

    Deliberately only SHAPES and the optimizer — no scan folder, no paths, no
    args namespace. A footprint is asked "how big is this job", and anything
    algorithm-specific it needs beyond these (a hash table size, a Gaussian
    count) is bound into the footprint callable by the algorithm itself,
    where that knowledge belongs.
    """
    n_angles: int
    n_b: int
    n_a: int
    vol_shape: tuple
    rays_per_batch: int = 0
    samples_per_ray: int = 0
    optimizer: str = "adam"

    @property
    def sino_bytes(self) -> int:
        return _sino_bytes(self.n_angles, self.n_b, self.n_a)

    @property
    def vol_bytes(self) -> int:
        return _vol_bytes(self.vol_shape)

    @property
    def param_copies(self) -> int:
        """Resident fp32-equivalent copies of the parameters, per optimizer."""
        return PARAM_COPIES.get(str(self.optimizer).strip().lower(), 4)


@dataclass(frozen=True)
class Footprint:
    """What one backend needs from the machine, split by what it scales with.

    The split is the whole point of the type. ``persistent_gpu_bytes`` is
    resident for the entire run and independent of the ray batch;
    ``bytes_per_ray_sample`` is the marginal cost of one more (ray x
    quadrature sample) in the differentiable renderer. ``auto_rays_per_batch``
    needs exactly that separation to answer "how many rays fit in what is
    left", which is why it cannot be derived from a single total.

    A backend with no ray batch (FDK, ASTRA, TIGRE) leaves
    ``bytes_per_ray_sample`` at zero and puts everything in the persistent
    term. That is not a special case — it falls out of the same formula.
    """
    persistent_gpu_bytes: int
    host_bytes: int
    gpu_required: bool = False
    bytes_per_ray_sample: int = 0
    notes: tuple = ()

    def gpu_bytes(self, rays_per_batch: int = 0, samples_per_ray: int = 0) -> int:
        """Peak VRAM including the batch, with the shared safety margin."""
        batch = (int(rays_per_batch) * max(1, int(samples_per_ray))
                 * int(self.bytes_per_ray_sample))
        return int((int(self.persistent_gpu_bytes) + batch) * SAFETY)

    @property
    def host_bytes_safe(self) -> int:
        return int(int(self.host_bytes) * SAFETY)


# ----------------------------------------------------------------------------
# The built-in backends. Each is a plain (MachineRequest) -> Footprint
# function, registered under its name; a new one is added by REGISTERING, not
# by extending a chain of `if backend ==`. Learning-based algorithms supply
# their own through `learning_based_iterative.registry` and never appear here,
# because their parameter count is a fact about the REPRESENTATION and this
# module has no business knowing it. The voxel entry stays only because the
# dense grid's parameters ARE the volume, which is a fact this module already
# has from `vol_shape`.
# ----------------------------------------------------------------------------

def _fdk_footprint(req: MachineRequest) -> Footprint:
    return Footprint(
        persistent_gpu_bytes=req.sino_bytes + req.vol_bytes // 8,
        host_bytes=2 * req.sino_bytes + req.vol_bytes,
        gpu_required=False,
        notes=("FDK z-chunks itself to free VRAM; runs on CPU if none "
               "(slow).",))


def _astra_footprint(req: MachineRequest) -> Footprint:
    return Footprint(
        persistent_gpu_bytes=2 * (req.sino_bytes + req.vol_bytes),
        host_bytes=2 * req.sino_bytes + req.vol_bytes,
        gpu_required=True,
        notes=("ASTRA CUDA algorithms cannot split the volume — the whole "
               "2x(vol+sino) workspace must fit at once.",))


def _tigre_footprint(req: MachineRequest) -> Footprint:
    return Footprint(
        persistent_gpu_bytes=req.sino_bytes + req.vol_bytes // 8,
        host_bytes=4 * req.vol_bytes + 3 * req.sino_bytes,
        gpu_required=True,
        notes=("TIGRE splits the volume across GPU passes; more free VRAM = "
               "fewer splits = faster. Host RAM is the binding constraint "
               "(~4x volume for TIGRE + crossval copies).",))


def learned_footprint(req: MachineRequest, *, param_bytes: int,
                      activation_bytes_per_sample: int = 0,
                      note: str = "") -> Footprint:
    """The shape EVERY learning-based algorithm's footprint has.

    Resident = the optimizer's copies of the parameters + the sinogram;
    marginal = the renderer's per-ray-sample traffic PLUS whatever the model
    itself retains per sample; host = projections + float sinogram + the
    exported volume. The shape is shared; the two numbers in it are not.

    ``param_bytes`` — for a dense grid this is ``req.vol_bytes`` (the
    parameters ARE the exported voxels); for a hash grid or an MLP it is the
    architecture's own weight count and has nothing to do with the export
    grid. Guessing either from the other is wrong by orders of magnitude, in
    the direction that reads as "fits comfortably" right up until the OOM.

    ``activation_bytes_per_sample`` — what the MODEL retains for backward, per
    quadrature sample, ON TOP of the renderer's ``BATCH_BYTES_PER_SAMPLE``.
    Zero for a voxel grid: a `grid_sample` keeps its output and nothing else,
    which is already inside the renderer's constant. NOT zero for a network,
    and not a detail — MEASURED on a 4x128 ReLU MLP it is 3,336 B, **35x the
    renderer's own 96 B**, so a batch sized as if the model were free is not
    merely tight, it cannot start. That is a real defect this argument exists
    to prevent, and it was found by running an MLP at a batch this preflight
    had approved: ~125,000 rays x 988 samples, which is 400 GB of activations
    on a 16 GB card.

    Callers are the algorithms themselves, via
    ``learning_based_iterative.registry``; this module never decides which
    number is right.
    """
    copies = req.param_copies
    extra = {4: "+Adam(m,v)", 3: "+Adam(m,v in bf16)"}.get(copies, "")
    base = (f"Trains param+grad{extra} = {copies}x "
            f"{int(param_bytes) / GiB:.2f} GiB of parameters on the GPU. "
            f"CPU fallback exists but is impractically slow.")
    per_sample = BATCH_BYTES_PER_SAMPLE + int(activation_bytes_per_sample)
    if activation_bytes_per_sample:
        base += (f" Activations add {int(activation_bytes_per_sample):,} B per "
                 f"ray-sample on top of the renderer's "
                 f"{BATCH_BYTES_PER_SAMPLE} B — {per_sample / BATCH_BYTES_PER_SAMPLE:.0f}x "
                 f"the traffic a voxel grid has — which is what sizes the batch.")
    return Footprint(
        persistent_gpu_bytes=copies * int(param_bytes) + req.sino_bytes,
        host_bytes=2 * req.sino_bytes + req.vol_bytes,
        gpu_required=False,
        bytes_per_ray_sample=per_sample,
        notes=((note,) if note else ()) + (base,))


#: The backends this module ships with — the CLASSICAL ones, which have no
#: algorithm descriptor and whose size is a fact about CT rather than about a
#: representation. Every learning-based algorithm registers itself instead (see
#: `learning_based_iterative.registry.register`), so importing that package is
#: what makes `voxel` resolvable here. Deliberately not pre-seeded with it:
#: an entry here would be a second place that knows how a voxel grid is
#: parameterised, and the whole point of the split is that there is one.
FOOTPRINTS: dict = {
    "fdk": _fdk_footprint,
    "astra": _astra_footprint,
    "tigre": _tigre_footprint,
}


def register_footprint(name: str, fn) -> None:
    """Teach the preflight about a backend it does not ship with.

    Idempotent for the SAME function (re-importing a module must not explode)
    and an error for a different one, so two algorithms cannot quietly claim
    one name and have the second silently win.
    """
    name = str(name).strip().lower()
    existing = FOOTPRINTS.get(name)
    # `!=`, not `is not`: a registry that wraps its algorithm in a fresh
    # callable each time (see `learning_based_iterative.registry`) would fail
    # an identity test on a harmless re-import while describing exactly the
    # same footprint. Equality is the question being asked.
    if existing is not None and existing != fn:
        raise ValueError(
            f"a different footprint is already registered for {name!r}; "
            f"pick another backend name rather than shadowing it")
    FOOTPRINTS[name] = fn


def resolve_footprint(backend, footprint=None):
    """The footprint callable for ``backend``, or the one passed in.

    ``footprint`` wins when given — that is how a caller supplies a model whose
    size it alone knows (an INR's parameter count, a splat count) without
    registering it globally first.
    """
    if footprint is not None:
        return footprint
    fn = FOOTPRINTS.get(str(backend).strip().lower())
    if fn is None:
        raise ValueError(
            f"unknown backend {backend!r} — no footprint registered. Known: "
            f"{sorted(FOOTPRINTS)}. A learning-based algorithm registers its "
            f"own via learning_based_iterative.registry, or a caller can pass "
            f"footprint=(MachineRequest) -> Footprint directly.")
    return fn


def estimate(backend: str, *, n_angles: int, n_b: int, n_a: int, vol_shape,
             rays_per_batch: int = 0, samples_per_ray: int = 0,
             optimizer: str = "adam", footprint=None):
    """(gpu_bytes, host_bytes, gpu_required, notes) for one backend.

    ``optimizer`` decides how many resident copies of the parameters the run
    needs (see ``PARAM_COPIES``); it is meaningful only to backends that train
    parameters, and ignored by the rest.

    ``footprint``: an explicit ``(MachineRequest) -> Footprint``, overriding
    the registry. Tuple return kept because every caller unpacks it.
    """
    fp = resolve_footprint(backend, footprint)(MachineRequest(
        n_angles=int(n_angles), n_b=int(n_b), n_a=int(n_a),
        vol_shape=tuple(vol_shape), rays_per_batch=int(rays_per_batch),
        samples_per_ray=int(samples_per_ray), optimizer=str(optimizer)))
    return (fp.gpu_bytes(rays_per_batch, samples_per_ray),
            fp.host_bytes_safe, bool(fp.gpu_required), list(fp.notes))


# --------------------------------------------------------- auto batch size --

AUTO_BATCH_FLOOR = 4096
AUTO_BATCH_CAP = 1 << 20        # 1M rays: throughput saturates well before this
# Fraction of free VRAM the auto batch may plan against. Deliberately below
# TIGHT_FRAC so a self-sized batch never lands on its own "tight fit" warning,
# and to leave room for allocator fragmentation over a long run.
AUTO_BATCH_FILL = 0.75


def auto_rays_per_batch(vram_free, *, n_angles: int, n_b: int, n_a: int,
                        vol_shape, samples_per_ray: int,
                        optimizer: str = "adam",
                        backend: str = "voxel", footprint=None,
                        floor: int = AUTO_BATCH_FLOOR,
                        cap: int = AUTO_BATCH_CAP,
                        fill: float = AUTO_BATCH_FILL) -> dict:
    """Largest ray batch that fits in free VRAM after the persistent buffers.

    Same pattern the FDK backend uses for its projection chunks: take the
    memory the card actually has free, subtract what must stay resident for
    the whole run (``PARAM_COPIES[optimizer]`` x parameters, plus the
    sinogram), and spend the rest on the per-step ray buffers. This is what
    lets one command saturate a 16 GB card and an 80 GB card without a
    hardware-specific flag — the batch is the only knob that scales with VRAM,
    and on a big GPU it is worth ~30x more rays per step (= more epochs over
    the sinogram in the same wall time).

    ``vram_free=None`` (no GPU) returns the floor. Returns a dict with the
    chosen ``rays`` and the terms behind it, so the driver can print the
    reasoning rather than an unexplained number.
    """
    spp = max(1, int(samples_per_ray))
    # Both terms come from the BACKEND'S OWN footprint, so the batch this
    # sizes and the VRAM the preflight reports can never disagree about what
    # stays resident. For the voxel grid that is `copies x volume + sinogram`
    # exactly as before; for a network it is `copies x weights + sinogram`,
    # which is smaller by orders of magnitude — the number that decides how
    # many rays a card can afford, and the reason this is not hardcoded.
    fp = resolve_footprint(backend, footprint)(MachineRequest(
        n_angles=int(n_angles), n_b=int(n_b), n_a=int(n_a),
        vol_shape=tuple(vol_shape), samples_per_ray=spp,
        optimizer=str(optimizer)))
    copies = PARAM_COPIES.get(str(optimizer).strip().lower(), 4)
    persistent = int(fp.persistent_gpu_bytes * SAFETY)
    per_ray = spp * int(fp.bytes_per_ray_sample)
    if per_ray <= 0:
        raise ValueError(
            f"backend {backend!r} declares no per-ray-sample cost, so there is "
            f"no ray batch to size. auto_rays_per_batch is for backends that "
            f"train through the differentiable renderer.")

    if vram_free is None:
        rays, budget, available = int(floor), 0, 0
    else:
        budget = int(vram_free * fill)
        available = max(0, budget - persistent)
        rays = int(available // per_ray)
        rays = (rays // 1024) * 1024                  # keep launches aligned
        rays = int(min(max(rays, floor), cap))

    return {
        'rays': rays,
        'samples_per_ray': spp,
        'param_copies': copies,
        'persistent_bytes': persistent,
        'budget_bytes': budget,
        'available_bytes': available,
        'bytes_per_ray': per_ray,
        'floored': vram_free is not None and available // per_ray < floor,
        'capped': vram_free is not None and available // per_ray > cap,
    }


# ------------------------------------------------------------------ check --

def run_preflight(backend: str, ctx, *, gpu_index: int = 0,
                  rays_per_batch: int = 0, samples_per_ray: int = 0,
                  optimizer: str = "adam", footprint=None,
                  skip: bool = False, only: bool = False,
                  logger=None) -> PreflightReport:
    """Print the machine-fit report; abort on a fatal verdict unless skipped.

    ``only=True``: print the report and return it with ``dry_run`` set — a
    dry run for sizing a job. The caller returns without reconstructing;
    exiting is the driver's decision, not this function's.
    ``skip=True``: print the report but never abort.

    Raises ``PreflightAbort`` when the machine cannot fit the job (and
    ``skip`` is not set), so a library caller can catch it and try a smaller
    grid instead of having its process killed.
    ``logger``: optional ReconLogger — the report is recorded on the W&B run
    (config + summary), and a fatal abort marks that run FAILED with the
    reason, so an auto-aborted job is visible in W&B rather than silent.

    ``footprint``: an explicit ``(MachineRequest) -> Footprint``, for a backend
    whose size only the caller knows. ``backend`` is then just the label the
    report is printed and logged under.
    """
    n_angles, n_b, n_a = (int(s) for s in ctx.projections.shape)
    gpu_b, host_b, gpu_req, notes = estimate(
        backend, n_angles=n_angles, n_b=n_b, n_a=n_a,
        vol_shape=ctx.geometry["vol_shape"],
        rays_per_batch=rays_per_batch, samples_per_ray=samples_per_ray,
        optimizer=optimizer, footprint=footprint)

    gpu = query_gpu_memory(gpu_index)
    report = PreflightReport(
        backend=backend, gpu_required=gpu_req,
        gpu_name=(gpu or {}).get("name"),
        vram_free=(gpu or {}).get("free_bytes"),
        vram_needed=gpu_b,
        ram_free=host_mem_available_bytes(),
        ram_needed=host_b,
        notes=notes,
    )

    nx, ny, nz = (int(v) for v in ctx.geometry["vol_shape"])
    print(f"\nPreflight ({backend}):")
    print(f"  Data:   {n_angles} x {n_b} x {n_a} projections -> "
          f"{nx} x {ny} x {nz} volume")
    if report.vram_free is None:
        print(f"  GPU:    NONE FOUND"
              + (" — this backend requires a CUDA GPU" if gpu_req
                 else " — will run on CPU (very slow)"))
    else:
        print(f"  GPU:    {report.gpu_name} — "
              f"need ~{gpu_b / GiB:.1f} GiB VRAM, "
              f"{report.vram_free / GiB:.1f} GiB free")
    if report.ram_free is not None:
        print(f"  RAM:    need ~{host_b / GiB:.1f} GiB, "
              f"{report.ram_free / GiB:.1f} GiB available")
    for n in notes:
        print(f"  Note:   {n}")
    print(f"  Verdict: {report.verdict.upper()}")

    if logger is not None:
        logger.log_preflight(report)

    if only:
        if logger is not None:
            logger.finish()
        report.dry_run = True
        return report
    if report.fatal and not skip:
        reason = (f"preflight {report.verdict}: "
                  f"need {report.vram_needed / GiB:.1f} GiB VRAM / "
                  f"{report.ram_needed / GiB:.1f} GiB RAM"
                  if report.verdict == "insufficient"
                  else "preflight: no CUDA GPU found and this backend "
                       "requires one")
        print("\nAborting before any large allocation. Reduce the load "
              "(--downsample, smaller --fov-xy/--fov-z, larger --voxel-xy) "
              "or run on a bigger machine. --skip-preflight overrides.")
        if ctx is not None and ctx.geometry.get('model_domain'):
            d = ctx.geometry['model_domain']
            print(f"  This grid is the MEASURED model domain "
                  f"({2 * d['x_max']:.0f} mm wide, z {d['z_min']:.0f} to "
                  f"{d['z_max']:.0f} mm) — the region the projections say has "
                  f"matter in it.\n"
                  f"  A larger --voxel-xy/--voxel-z keeps the domain and "
                  f"costs resolution; --model-domain off keeps the resolution "
                  f"and reintroduces the truncation bias the domain exists to "
                  f"remove (a smooth HU offset, not a visible artifact).")
        if logger is not None:
            logger.abort(reason)
        raise PreflightAbort(reason)
    if report.fatal and skip:
        print("  (--skip-preflight: continuing anyway — expect an OOM or "
              "a crash)")
    elif report.verdict == "tight":
        print("  (tight fit — expect heavy memory pressure; consider "
              "--downsample or a smaller FOV)")
    return report


def add_preflight_args(parser) -> None:
    parser.add_argument(
        '--skip-preflight', action='store_true', default=False,
        help='Run even when the preflight machine check says the job will '
             'not fit (expect OOM).')
    parser.add_argument(
        '--preflight-only', action='store_true', default=False,
        help='Load the scan, print the machine-fit report (GPU/VRAM/RAM vs '
             'this job), and exit without reconstructing. Use to size a job '
             'before submitting it.')
