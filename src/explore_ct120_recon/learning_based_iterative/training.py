"""Training mechanics shared by every learning-based backend.

Reconstruction-as-optimization needs four things that have nothing to do with
CT and everything to do with the hardware and the parameterisation:

  * **precision** — whether the forward pass runs under autocast, and in which
    dtype (`resolve_amp_dtype`, `autocast_ctx`);
  * **per-group learning rates** — a model whose parameters are not on one
    scale cannot be trained with one LR (`build_param_groups`,
    `build_optimizer`);
  * **kernel fusion of the MODEL** — distinct from
    ``renderer.set_render_compile``, which fuses the quadrature and
    integration. A hash-grid + MLP trunk is worth compiling on its own; a
    dense voxel grid is a single ``grid_sample`` and is not
    (`maybe_compile_model`);
  * **gradient clipping** (`clip_grad_norm`).

They live here rather than in a backend because they are the same decisions
for every representation, and because muNeRF had its own copy of all four —
which is one implementation too many under the rule that this submodule owns
anything both sides need.

NOTHING HERE KNOWS WHAT A MODEL IS. `build_param_groups` matches by module
CLASS NAME, `project_nonneg` duck-types on a method, and `maybe_compile_model`
looks only at where the parameters live. That is deliberate: the hash-grid
encoder is a tinycudann wrapper that exists only in muNeRF, and the submodule
must not import it to give it a learning rate.
"""

from __future__ import annotations

import contextlib
import math
from typing import Iterable, Mapping

import torch
import torch.nn as nn

from .adam_bf16 import AdamBF16

__all__ = [
    "AdamBF16", "AMP_MODES", "LR_SCHEDULES", "OPTIMIZERS", "autocast_ctx",
    "build_optimizer", "build_param_groups", "clip_grad_norm",
    "fused_adam_supported", "lr_multiplier", "maybe_compile_model",
    "project_nonneg", "resolve_amp_dtype", "unwrap_model",
]

#: Accepted `amp` settings. 'fp16' is recognised only so it can be rejected
#: with a reason rather than as an unknown string.
AMP_MODES = ("off", "auto", "bf16", "fp16")

#: Volta. Both bf16 and Triton (torch.compile's Inductor backend) need it, so
#: the two capability gates below are the same gate.
TENSOR_CORE_CAPABILITY = 7


def unwrap_model(model: nn.Module) -> nn.Module:
    """The eager module behind a `torch.compile` wrapper, if any.

    `compiled.state_dict()` prefixes every key with ``_orig_mod.``, which
    breaks a plain `load_state_dict` in inference, in tests, and in any run
    that did not compile. Going through this on the way to a checkpoint keeps
    the on-disk format identical either way.
    """
    return getattr(model, "_orig_mod", model)


# ------------------------------------------------------------------ compile --

def maybe_compile_model(model: nn.Module, enabled: bool = False, *,
                        label: str = "torch.compile") -> nn.Module:
    """`torch.compile(model)` when asked for and the hardware allows it.

    Skipped, with a printed reason, on: an older torch, a parameter-less
    module, a model on the CPU, and compute capability < 7.0 — Triton does not
    support Pascal and would otherwise raise on the first forward pass. Any
    other compile failure falls back to eager: training slowly beats not
    training.
    """
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print(f"  {label} unavailable on this torch version — running eager.")
        return model
    try:
        param = next(model.parameters())
    except StopIteration:
        return model                      # nothing to compile
    if param.device.type != "cuda":
        print(f"  {label} skipped: model is on CPU.")
        return model
    major, _ = torch.cuda.get_device_capability(param.device)
    if major < TENSOR_CORE_CAPABILITY:
        print(f"  {label} skipped: GPU compute capability {major}.x < "
              f"{TENSOR_CORE_CAPABILITY}.0 (Triton/Inductor unsupported on "
              f"Pascal).")
        return model
    try:
        compiled = torch.compile(model)
        print(f"  {label}: enabled")
        return compiled
    except Exception as e:                # pragma: no cover - fallback path
        print(f"  {label} disabled: {type(e).__name__}: {e}")
        return model


# ---------------------------------------------------------------- precision --

def resolve_amp_dtype(mode, device: torch.device):
    """Autocast dtype for the forward + loss, or None for fp32 everywhere.

      * ``'off'``  — no autocast.
      * ``'bf16'`` — bfloat16. No GradScaler needed: bf16 has fp32's exponent
        range, so gradients do not underflow the way fp16's do.
      * ``'auto'`` — bf16 on Volta and later, off below. Pascal has no Tensor
        Cores, where bf16 is emulated and SLOWER than fp32, so 'auto' is not
        "use it if it exists" but "use it if it pays".
      * ``'fp16'`` — rejected. It needs a GradScaler that no backend here
        wires up, and bf16 covers every card that could benefit.

    CPU always returns None. `mode` is lowercased, so a YAML ``amp: off``
    that parsed as the BOOLEAN False arrives as ``"false"`` and raises here
    rather than silently disabling autocast — quote it in the config.
    """
    m = str(mode).strip().lower()
    if m == "off" or device.type != "cuda":
        return None
    if m == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= TENSOR_CORE_CAPABILITY else None
    if m in ("bf16", "bfloat16"):
        return torch.bfloat16
    if m in ("fp16", "float16"):
        raise ValueError(
            "amp='fp16' is not supported — it needs a GradScaler to keep small "
            "gradients from underflowing, and bf16 covers the same hardware. "
            "Use 'bf16' or 'auto'.")
    raise ValueError(
        f"unknown amp={mode!r} (choices: {', '.join(AMP_MODES)}). Note that a "
        f"bare `off` in YAML 1.1 is the BOOLEAN False and arrives here as "
        f"'false' — quote it.")


#: Post-warmup shapes. Anything else is a caller-supplied `lr_schedule_fn`.
LR_SCHEDULES = ("constant", "cosine", "exponential")


def lr_multiplier(schedule: str = "cosine", *, warmup: int = 0,
                  total: int = 1, gamma: float = 0.999, offset: int = 0,
                  hold: float = 0.0, floor: float = 0.0):
    """`(iteration) -> multiplier` on each param group's OWN base LR.

    A multiplier, not a `torch.optim.lr_scheduler`, for two reasons that both
    bit during the muNeRF port:

    * a scheduler writes ABSOLUTE rates, which flattens per-group ratios — a
      hash grid at 10x the MLP's would silently collapse onto one curve;
    * constructing a warmup `LinearLR` applies its ``start_factor`` to the
      optimizer immediately, so a loop that reads the group LRs afterwards
      captures ``start_factor x base`` as the base (measured: 1e-10 instead of
      1e-2).

    Linear ramp from ~0 over `warmup` iterations, then `schedule` over the
    remaining ``total - warmup``. `offset` shifts the whole thing, for a staged
    run whose second phase starts partway through and wants its own ramp.

    ``hold`` (cosine only) is the fraction of the post-warmup run spent at
    the full rate before the decay begins, and ``floor`` the multiplier the
    decay ends on. Both default to the plain cosine (decay from the first
    post-warmup step, to zero). MEASURED on the Gaussian backend (run
    ufsqlhpn): the held-out fit was still improving at 80 % of the run,
    where the plain cosine had already cut the rate tenfold, and the last
    15 % of the iterations bought nothing. A late, floored anneal keeps the
    rate where the progress is being made.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr schedule must be one of {list(LR_SCHEDULES)}, "
                         f"got {schedule!r}")
    warmup, offset = int(warmup), int(offset)
    main = max(1, int(total) - warmup)
    hold, floor = float(hold), float(floor)
    if not 0.0 <= hold < 1.0:
        raise ValueError(f"lr hold fraction must be in [0, 1), got {hold}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"lr floor must be in [0, 1], got {floor}")
    held = int(round(hold * main))
    decay = max(1, main - held)

    def multiplier(iteration: int) -> float:
        k = int(iteration) - offset
        if warmup > 0 and k < warmup:
            # Floored rather than exactly 0 so a logged LR is never a bare zero.
            return max(1e-8, (k + 1) / warmup)
        if schedule == "cosine":
            frac = min(1.0, max(0.0, (k - warmup - held) / decay))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
        if schedule == "exponential":
            return float(gamma) ** max(0, k - warmup)
        return 1.0

    return multiplier


def autocast_ctx(amp_dtype, device_type: str = "cuda"):
    """`torch.amp.autocast` when a dtype was resolved, else a no-op."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=amp_dtype)


# ---------------------------------------------------------------- optimizer --

def build_param_groups(model: nn.Module, *, lr: float,
                       group_lrs: Mapping[str, float] | None = None,
                       verbose: bool = True) -> list[dict]:
    """Parameter groups, split by the CLASS NAME of the owning module.

    ``group_lrs={'HashGridEncoding': 1e-2}`` gives every parameter owned by a
    module of that class an LR of 1e-2 and leaves the rest on `lr`. Matching
    by name, not by class object, is what keeps this file free of any import
    from the models it serves — the hash-grid encoder is a tinycudann wrapper
    that lives in muNeRF and cannot be imported here.

    WHY THIS EXISTS AT ALL: a hash grid's feature tables and the MLP head that
    reads them are not on one scale (typically 1e-2 against 1e-3), and a dense
    voxel grid is on neither (the parameter IS mu in mm^-1, ~0.022 for water).
    One LR across the lot trains one of them wrong.

    A parameter owned by several matching modules takes the DEEPEST one's rate,
    so a specific inner module overrides a general outer one. A key that
    matches nothing is reported rather than ignored — silently is how a
    renamed class turns into a run at the wrong learning rate.
    """
    base = unwrap_model(model)
    group_lrs = dict(group_lrs or {})
    # id(param) -> (depth, class name). Depth breaks ties toward the inner
    # module; `named_modules` yields dotted paths, so depth is just the dots.
    owner: dict[int, tuple[int, str]] = {}
    seen: set[str] = set()
    for name, mod in base.named_modules():
        cls = type(mod).__name__
        if cls not in group_lrs:
            continue
        seen.add(cls)
        depth = name.count(".") + (1 if name else 0)
        for p in mod.parameters():
            prev = owner.get(id(p))
            if prev is None or depth > prev[0]:
                owner[id(p)] = (depth, cls)

    missing = sorted(set(group_lrs) - seen)
    if missing and verbose:
        print(f"    NOTE: no module of class {missing} in this model — "
              f"the learning rate(s) given for it are unused.")

    buckets: dict[str, list[nn.Parameter]] = {}
    for p in base.parameters():
        cls = owner.get(id(p), (0, None))[1]
        buckets.setdefault(cls, []).append(p)

    groups = []
    for cls, params in buckets.items():
        if not params:
            continue
        g_lr = group_lrs[cls] if cls is not None else lr
        groups.append({"params": params, "lr": float(g_lr)})
        if verbose:
            n = sum(p.numel() for p in params)
            print(f"    {cls or 'default'}: LR={float(g_lr):.2e} "
                  f"({n:,} parameters)")
    if not groups:                        # parameter-less model
        groups = [{"params": list(base.parameters()), "lr": float(lr)}]
    return groups


#: The optimizers a run may name. Adam and its bfloat16-state variant take the
#: same arguments and produce the same update; SGD is the classical one (see
#: the note in `build_optimizer`). Keys must stay in step with
#: `ct_core.preflight.PARAM_COPIES`, which sizes VRAM from the same names.
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adam_bf16": AdamBF16,
    "sgd": torch.optim.SGD,
}


def fused_adam_supported(param_groups) -> bool:
    """Whether torch's FUSED Adam kernel can take these parameters.

    Worth asking, because the answer is worth 2.5x the optimizer step and the
    optimizer step is not a rounding error here: on the voxel backend the
    parameters ARE the volume, so Adam moves ~0.5 GiB per iteration whatever
    the ray batch is. MEASURED on a P100 at 34 M parameters: 4.65 ms for the
    default foreach path (a chain of separate elementwise kernels, each one
    re-reading its operands) against 1.88 ms fused (one kernel, one pass).
    The two agree to 1e-6 over 200 steps — same arithmetic, fewer trips to
    memory.

    Fused Adam is CUDA-only and floating-point-only. Everything else — CPU
    runs, the test suite, an exotic dtype — has to take the ordinary path,
    so this is a question rather than an assumption.
    """
    params = [p for g in param_groups for p in g.get("params", ())]
    if not params or not torch.cuda.is_available():
        return False
    ok = (torch.float32, torch.float16, torch.bfloat16)
    return all(p.is_cuda and p.dtype in ok for p in params)


def build_optimizer(model: nn.Module, *, lr: float, weight_decay: float = 0.0,
                    group_lrs: Mapping[str, float] | None = None,
                    optimizer: str = "adam", verbose: bool = True):
    """Adam, Adam with bfloat16 moments, or SGD over `build_param_groups`.

    SGD is offered because it is what the classical update is: Adam's
    second-moment normalisation is itself a per-parameter preconditioner, so a
    run that also applies an explicit one (SART's C) would have two competing
    and would not be the classical method.

    ``adam_bf16`` is the same update as ``adam`` with the two moment buffers
    held in bfloat16 — on the voxel backend each of those buffers is a whole
    CT volume, so it returns one volume of VRAM to the ray batch. See
    `adam_bf16.AdamBF16` for why that needs stochastic rounding to be correct.
    """
    name = str(optimizer).strip().lower()
    if name not in OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(OPTIMIZERS)}, got {optimizer!r}")
    if verbose:
        note = (" — moments in bfloat16, 3x resident volumes instead of 4x"
                if name == "adam_bf16" else "")
        print(f"  Optimizer: {name.upper()} (weight_decay {weight_decay:g})"
              f"{note}")
    groups = build_param_groups(model, lr=lr, group_lrs=group_lrs,
                                verbose=verbose)
    kwargs = dict(weight_decay=float(weight_decay))
    if name == "adam" and fused_adam_supported(groups):
        kwargs["fused"] = True
    try:
        return OPTIMIZERS[name](groups, **kwargs)
    except (RuntimeError, TypeError) as exc:
        # A torch that does not accept `fused`, or accepts it and then refuses
        # these tensors. The unfused path is the same optimizer, so fall back
        # loudly rather than failing the run over a performance flag.
        if "fused" not in kwargs:
            raise
        if verbose:
            print(f"    (fused Adam unavailable: {exc} — using the standard "
                  f"kernel)")
        kwargs.pop("fused")
        return OPTIMIZERS[name](groups, **kwargs)


# ------------------------------------------------------------------- steps --

def clip_grad_norm(model_or_params, max_norm: float) -> float | None:
    """Global-norm gradient clipping. Returns the pre-clip norm, or None when
    clipping is off (``max_norm <= 0``), so a caller can log what it did.
    """
    if not max_norm or float(max_norm) <= 0:
        return None
    params = (unwrap_model(model_or_params).parameters()
              if isinstance(model_or_params, nn.Module)
              else model_or_params)
    total = torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_norm))
    return float(total)


def project_nonneg(model: nn.Module) -> bool:
    """Enforce mu >= 0 by PROJECTION after `optimizer.step()`, if the model
    supports it. Returns whether anything happened.

    This is TIGRE SIRT's ``res.clip(min=0)``, and it applies to a
    representation with one free parameter per voxel. It is a no-op for an INR
    whose head already guarantees mu >= 0 through the parameterisation (a
    softplus output), which is why the INR path needs no projection and why
    this duck-types on the method instead of testing for a class: a model that
    can be projected says so by defining ``clamp_nonneg``.
    """
    fn = getattr(unwrap_model(model), "clamp_nonneg", None)
    if callable(fn):
        fn()
        return True
    return False
