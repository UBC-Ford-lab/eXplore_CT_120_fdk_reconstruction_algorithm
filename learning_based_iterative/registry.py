"""The learned-algorithm registry: what ``--algorithm`` can name.

``LearnedReconstructor`` is already independent of the representation — it
touches a model only through ``nn.Module`` (``parameters``, ``state_dict``),
queries it as a callable, and duck-types the non-negativity projection. What
was NOT independent was the driver: ``run_learned_recon.py`` imported
``VoxelReconstructor`` directly and constructed it unconditionally, so
``--algorithm`` could only ever name the one class it already had. This module
is the missing half — the place a representation announces itself.

A ``LearnedAlgorithm`` is deliberately more than ``{name: class}``. A registry
of classes alone leaves three things stranded in the driver, and each of them
is where the next representation would otherwise have to add a branch:

* **its own CLI flags.** ``--init-density`` is a voxel-only knob (the
  parameter IS mu, so its starting value is a modelling choice); a network's
  output scale is set by its head instead. Left in the shared parser, every
  algorithm's ``--help`` and every run's config carries knobs that mean
  nothing to it.
* **its own constructor arguments.** Which is the same list, mapped.
* **its own machine footprint.** The voxel grid needs
  ``4 x one-fp32-per-exported-voxel`` of VRAM; a hash grid needs four copies
  of its hash tables, which do not scale with the export grid at all. Guessing
  either from the other is wrong by orders of magnitude in both directions,
  and the preflight is the one place a wrong guess turns into a spurious abort
  or an OOM ten minutes in.

So a new representation is: a ``LearnedReconstructor`` subclass answering the
three hooks, plus one ``LearnedAlgorithm`` describing it, plus a
``register()`` call. Nothing in the driver, nothing in ``ct_core``.

    from ..registry import LearnedAlgorithm, register

    def _add_args(parser):
        parser.add_argument('--hash-levels', type=int, default=16, ...)

    def _options(args):
        return {'hash_levels': args.hash_levels}

    def _footprint(args, req):
        n_params = ...                    # from args, not from req.vol_shape
        return Footprint(
            persistent_gpu_bytes=req.param_copies * n_params * 4 + req.sino_bytes,
            host_bytes=2 * req.sino_bytes + req.vol_bytes,
            bytes_per_ray_sample=BATCH_BYTES_PER_SAMPLE,
            notes=(f"Hash grid: {n_params/1e6:.1f}M parameters.",))

    register(LearnedAlgorithm(
        name='hashgrid', reconstructor=HashGridReconstructor,
        summary='multi-resolution hash-grid INR (Instant-NGP)',
        add_args=_add_args, options=_options, footprint=_footprint))

``options`` is used for BOTH the constructor kwargs and the run config, on
purpose: what the algorithm was given is exactly what should be recorded, and
two lists that must agree are two lists that drift. Keep the values scalars —
they are logged, and the run config is a whitelist of geometry and algorithm
numbers. Never put a filesystem path in there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..ct_core.preflight import Footprint, MachineRequest  # noqa: F401 (re-exported)


def _no_args(parser) -> None:
    """An algorithm with no flags of its own — the common case, eventually."""


def _no_options(args) -> dict:
    return {}


@dataclass(frozen=True)
class LearnedAlgorithm:
    """One selectable representation, and everything the driver needs from it.

    ``reconstructor`` is a ``LearnedReconstructor`` subclass. It receives the
    shared keyword arguments the driver builds for every algorithm, plus
    whatever ``options(args)`` returns — so a subclass declares its own
    constructor parameters normally and never has to accept the others by
    name.
    """
    name: str
    reconstructor: type
    summary: str
    #: ``(parser) -> None`` — flags only this algorithm understands.
    add_args: Callable[[Any], None] = _no_args
    #: ``(args) -> dict`` — its constructor kwargs, and its run-config entries.
    options: Callable[[Any], dict] = _no_options
    #: ``(args, MachineRequest) -> Footprint``. No default: a representation
    #: that will not say how big it is cannot be preflighted, and silently
    #: inheriting the voxel grid's `4 x volume` would be wrong for every
    #: network by orders of magnitude — in the direction that reads as "fits
    #: comfortably" right up until the OOM.
    footprint: Callable[[Any, MachineRequest], Footprint] = None
    #: Adam LR to use when ``--lr`` is not given. None = the driver's global
    #: default.
    #:
    #: THE SHARED DEFAULT IS NOT REPRESENTATION-NEUTRAL, which is easy to miss
    #: because it looks like one. 1e-4 is a step in units of the parameter, and
    #: for a dense grid the parameter IS mu (~0.022 for water), so 1e-4 is
    #: half a percent of water per step — a sensible size that was chosen with
    #: that in mind. A network's parameters are weights with no physical scale
    #: and their own conditioning, and the right rate is a property of the
    #: architecture: muNeRF's own config puts the Fourier+ReLU path at 5e-4
    #: and the hash-grid MLP head at 1e-3.
    #:
    #: MEASURED CONSEQUENCE: the first voxel-vs-MLP comparison ran the MLP at
    #: 1e-4 because that is what the driver hands out, i.e. at ONE FIFTH of the
    #: rate muNeRF's own config specifies for exactly that architecture, and
    #: the result was read as a fact about representations. "Everything else
    #: unchanged" is a trap whenever an unchanged DEFAULT was tuned for one of
    #: the things being compared: the flag is the same, the condition is not.
    #:
    #: A float, or ``(args) -> float`` when the rate depends on the
    #: algorithm's OWN flags — a hash-grid trunk and a Fourier trunk are
    #: different architectures under one ``--algorithm``, and muNeRF tunes
    #: them to 1e-3 and 5e-4 respectively.
    default_lr: float | Callable[[Any], float] | None = None

    def __post_init__(self):
        if self.footprint is None:
            raise ValueError(
                f"LearnedAlgorithm {self.name!r} declares no footprint. "
                f"Supply footprint=(args, MachineRequest) -> Footprint; see "
                f"`voxel.algorithm` for a worked example.")

    def bind_footprint(self, args) -> Callable[[MachineRequest], Footprint]:
        """The footprint with ``args`` bound — what the preflight wants.

        The two-stage shape is what lets an algorithm size itself from its OWN
        flags (a hash table's levels, a splat budget) while the preflight
        still only ever hands over shapes.
        """
        return lambda req: self.footprint(args, req)

    def config(self, args) -> dict:
        """``options(args)``, made safe to log.

        Scalars pass through; anything else becomes its ``repr``. A figure or a
        tensor in the run config is a crash at ``wandb.init`` time, an hour
        into a job, for a reason that has nothing to do with reconstruction.
        """
        out = {}
        for k, v in self.options(args).items():
            out[str(k)] = v if isinstance(v, (int, float, str, bool, type(None))) \
                else repr(v)
        return out


class _StaticFootprint:
    """The registered-by-name fallback: ``footprint(None, req)``.

    ``run_preflight`` is normally handed ``algorithm.bind_footprint(args)`` and
    never reaches this. It exists so ``estimate('<name>', ...)`` — the sizing
    path, which has shapes and no argparse namespace anywhere — still works.
    That is the reason an algorithm's footprint must tolerate ``args=None`` by
    falling back to its own defaults, and it is stated in
    ``LearnedAlgorithm.footprint``.

    A class rather than a closure so that re-registering the same algorithm
    compares equal instead of tripping the preflight's clash guard.
    """

    def __init__(self, algorithm: "LearnedAlgorithm"):
        self._algorithm = algorithm

    def __call__(self, req: MachineRequest) -> Footprint:
        return self._algorithm.footprint(None, req)

    def __eq__(self, other):
        return (isinstance(other, _StaticFootprint)
                and other._algorithm is self._algorithm)

    def __hash__(self):
        return hash(id(self._algorithm))


_REGISTRY: dict = {}


def register(algorithm: LearnedAlgorithm) -> LearnedAlgorithm:
    """Add an algorithm to the registry, and its footprint to the preflight.

    Both in one call, because a name the driver can select but the preflight
    cannot size is a name that aborts after the scan is already loaded.
    Idempotent for the same object (a module re-import must not explode) and an
    error for a different one under the same name.
    """
    from ..ct_core.preflight import register_footprint

    name = str(algorithm.name).strip().lower()
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not algorithm:
        raise ValueError(
            f"a different algorithm is already registered as {name!r} "
            f"({existing.reconstructor.__name__}); pick another name.")
    register_footprint(name, _StaticFootprint(algorithm))
    _REGISTRY[name] = algorithm
    return algorithm


def get(name: str) -> LearnedAlgorithm:
    """The algorithm registered as ``name``, or a ValueError naming the rest."""
    key = str(name).strip().lower()
    algo = _REGISTRY.get(key)
    if algo is None:
        raise ValueError(
            f"unknown learned algorithm {name!r}. Registered: {names()}")
    return algo


def names() -> tuple:
    """Registered names, sorted — what ``--algorithm`` accepts."""
    return tuple(sorted(_REGISTRY))


def algorithms() -> tuple:
    return tuple(_REGISTRY[n] for n in names())


def describe() -> str:
    """One line per algorithm, for ``--help``."""
    if not _REGISTRY:
        return "  (none registered)"
    width = max(len(n) for n in names())
    return "\n".join(f"  {a.name:<{width}}  {a.summary}" for a in algorithms())
