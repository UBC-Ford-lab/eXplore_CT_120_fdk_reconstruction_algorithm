"""
Run learning-based iterative cone-beam CT reconstruction.

The volume is a differentiable representation fitted to the measured line
integrals by gradient descent through a differentiable forward projector
(reconstruction as optimization — the same argmin classical iterative recon
solves, with autograd replacing the hand-derived update rule).

Algorithms (each in its own subfolder of ``learning_based_iterative/``, each
registering a ``LearnedAlgorithm`` descriptor — see that package's
``registry``):
  - voxel: dense voxel grid — SIRT's representation trained with Adam + MSE.

``--algorithm`` selects over the registry, and nothing in this file knows
which one it got. A representation contributes its own CLI flags, its own
constructor arguments and its own machine footprint through the descriptor,
so adding one touches ``learning_based_iterative`` and not this driver.

All algorithm-independent stages (scan loading, geometry build, detector
downsampling, detector-psi calibration, HU calibration + VFF export) live in
ct_core.pipeline and are shared with the FDK / iterative drivers.

Usage:
    python -m reconstruction.run_learned_recon data/scans/Scan_1510
    python -m reconstruction.run_learned_recon data/scans/Scan_1510 \\
        --iterations 40000 --downsample 3 --lr 1e-4
"""

import argparse
import os
import time

import numpy as np

from .learning_based_iterative.losses import (DATA_TERMS, DEFAULT_DATA_TERM,
                                              describe_data_terms)
from .learning_based_iterative import (algorithm_names, describe_algorithms,
                                       get_algorithm)
from .learning_based_iterative.detector_warp import resolve_detector_warp
from .ct_core.pipeline import (
    ReconLogger,
    add_common_args,
    add_model_domain_args,
    crop_to_export_roi,
    prepare_scan,
    resolve_or_measure_detector_psi,
    run_preflight,
    save_outputs,
)
from .ct_core.data_budget import RANDOM, data_budget
from .ct_core.early_stop import DEFAULT_MIN_LR_FRACTION
from .ct_core.hu_calibration import resolve_anchors
from .ct_core.errors import ConfigError, cli_main
from .ct_core.preflight import auto_rays_per_batch
from .ct_core.projection_diag import measure_noise_ceiling
from .ct_core.utils import query_gpu_memory
from .ct_core.early_stop import STOP_METRICS

#: What ``--algorithm`` defaults to. The dense grid, because it is the
#: like-for-like comparison against classical SIRT and therefore the
#: reference every other representation is read against.
DEFAULT_ALGORITHM = "voxel"


#: Env var naming modules to import before ``--algorithm`` is resolved, so a
#: shell or a job script can make an out-of-tree representation available once
#: instead of on every command line. Same meaning as ``--algorithm-module``,
#: separated by commas or the platform path separator.
ALGORITHM_MODULE_ENV = "CT_LEARNED_ALGORITHMS"


def _load_algorithm_modules(names) -> list:
    """Import modules whose import registers a representation.

    THIS IS WHAT MAKES AN OUT-OF-TREE MODEL A FIRST-CLASS ``--algorithm``.
    Registration happens as a side effect of importing the module that calls
    ``register_algorithm``, and a driver run as ``python -m
    reconstruction.run_learned_recon`` imports only the submodule — so a
    representation defined anywhere else (muNeRF's ``inr_pipeline``, a
    scratch experiment, another repo entirely) is invisible to it until
    something says which module to import. That is the whole mechanism:

        python -m reconstruction.run_learned_recon SCAN \
            --algorithm-module inr_pipeline.algorithms --algorithm parent_inr

    Deliberately a MODULE NAME and not a file path: the module has to be
    importable anyway for its reconstructor to be, so a name that resolves
    on ``sys.path`` is the honest unit, and it keeps a filesystem path out of
    the run's provenance.

    Errors name the module and the likely cause. An out-of-tree algorithm
    failing to import is the single most common way this goes wrong, and a
    bare ``ModuleNotFoundError`` from inside argparse would not say why.
    """
    import importlib

    out = []
    for name in names:
        for part in str(name).replace(os.pathsep, ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                importlib.import_module(part)
            except ImportError as e:
                raise ConfigError(
                    f"--algorithm-module {part!r} could not be imported "
                    f"({e}). It must be importable from where this driver "
                    f"runs — for a module in the parent repo, run from the "
                    f"repo root or set PYTHONPATH.") from None
            out.append(part)
    return out


def _resolve_algorithm(argv=None):
    """``--algorithm``, parsed before the parser whose shape it decides.

    A two-pass parse rather than one, for two reasons that both need an answer
    before the real parser exists. ``--algorithm-module`` decides WHICH
    algorithms exist at all (importing a module is what registers one), and
    ``--algorithm`` then decides which flags the parser must accept — the
    voxel grid contributes ``--init-density``, another representation
    contributes its own. So: import the plugins, resolve the name, and only
    then build the parser around the answer. ``--help`` shows the flags of the
    algorithm actually selected, and a run records only knobs that mean
    something to it.

    Returns ``(algorithm, modules)`` — the modules travel on so the run can
    record where its representation came from.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--algorithm', default=DEFAULT_ALGORITHM)
    pre.add_argument('--algorithm-module', action='append', default=[])
    known, _ = pre.parse_known_args(argv)
    modules = _load_algorithm_modules(
        list(known.algorithm_module)
        + ([os.environ[ALGORITHM_MODULE_ENV]]
           if os.environ.get(ALGORITHM_MODULE_ENV) else []))
    try:
        return get_algorithm(known.algorithm), modules
    except ValueError as e:
        hint = ("" if modules else
                f" If it is defined outside this package, name the module "
                f"that registers it with --algorithm-module (or "
                f"{ALGORITHM_MODULE_ENV}).")
        raise ConfigError(str(e) + hint) from None


def _resolve_optimizer(args):
    """Which optimizer this run will actually use.

    The trainer decides this (``--optimizer``, defaulting to SGD under
    ``--emulate-sart``), but the LEARNING RATE and the VRAM estimate are both
    needed before the trainer exists — and both differ by a factor that makes
    the wrong answer visible: 1e-4 vs 1.0 on the step size, 4x vs 2x on the
    resident parameter copies. Resolved once here so the two cannot disagree
    with each other or with ``trainer.py``.
    """
    if args.optimizer is not None:
        return str(args.optimizer).strip().lower()
    return "sgd" if args.emulate_sart else "adam"


#: Adam's LR for a representation that names none. A step in units of the
#: parameter — right for a dense grid, where the parameter IS mu.
DEFAULT_ADAM_LR = 1e-4


def _resolve_lr(args, algorithm=None):
    """The learning rate: explicit, else per REPRESENTATION, else per optimizer.

    Two defaults, and they answer different questions.

    SGD's 1.0 is the classical relaxation parameter lambda, not a tuned value —
    and it only means that because --emulate-sart descends the SUMMED misfit
    with an ABSOLUTE C (see learning_based_iterative.sart). Under any other
    combination the raw gradient carries an arbitrary scale and 1.0 means
    nothing in particular. It is a fact about the OPTIMIZER, so it stays here.

    Adam's is a fact about the REPRESENTATION and does not: 1e-4 is a step in
    units of the parameter, which is a considered size when the parameter is mu
    and an arbitrary one when it is a weight. So an algorithm may name its own
    (``LearnedAlgorithm.default_lr``), and the one that does not gets the value
    that was chosen for the grid. See that field for the measured reason this
    exists — a comparison run at one fifth of the rate the model's own config
    specifies, and read as a fact about representations.
    """
    if args.lr is not None:
        return float(args.lr)
    if _resolve_optimizer(args) == "sgd":
        return 1.0
    named = getattr(algorithm, "default_lr", None)
    if named is None:
        return DEFAULT_ADAM_LR
    return float(named(args) if callable(named) else named)


def _build_lr_plateau(args):
    """The closed-loop LR reducer, or None to leave the cosine in charge.

    Default ON: an open-loop cosine is scheduled over --iterations, but a
    held-out stopping rule normally fires at a fraction of them, so the
    annealing phase the schedule exists for never runs. Driving the LR from the
    stopper's own signal removes the horizon assumption entirely.
    """
    if not args.lr_plateau:
        return None
    from .ct_core.early_stop import PlateauLRReducer
    return PlateauLRReducer(
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr_fraction=args.lr_plateau_min_fraction,
        cooldown=args.lr_plateau_cooldown)


def _lr_stage_views_fn(ctx, args, logger):
    """``on_lr_stage`` for the trainer: the volume at every LR change, drawn
    as the three midplane views the finished run already gets.

    Lives in the driver because it needs the export crop, the HU map and the
    logger — none of which the trainer knows about, and none of which it
    should have to.

    The HU map is refitted on EVERY frame and the gain it found is written
    into the title and logged as a scalar. It was fitted ONCE, on stage 0,
    and reused for the whole sequence, so that a change in the picture could
    only ever be a change in the reconstruction. That argument was right
    about what it wanted and wrong about how to get it: stage 0 is the
    reducer TAKEOVER, a few hundred iterations in, and the volume there has
    not yet grown to its physical attenuation. The histogram fit is scale-
    EQUIVARIANT on purpose — it has to work on volumes in arbitrary units —
    so a volume whose mu is f times too small returns a gain 1/f times too
    large, and that gain then saturates every later frame to white.

    MEASURED on a finished Scan_1510 volume, shrinking its mu and refitting:

        mu x f    implied mu_water    real soft tissue (mu 0.0235) lands at
          1.0          0.02312                        +132 HU
          0.5          0.01156                       +1270 HU
          0.1          0.00231                      +10378 HU

    No quality gate catches this. Prominence, tissue mass and the block-
    average corroboration are all scale-invariant and read IDENTICALLY at
    every f (0.677 in all three rows above), because a shrunken volume is a
    perfectly well-formed histogram — just not one in mm^-1.

    Refitting per frame cannot fail that way: a bad fit stays inside the
    frame that produced it. The objection it raises — a drifting gain then
    looks like a change in contrast — is answered by REPORTING the gain
    instead of freezing it, so the two are told apart by reading a number.
    """
    if not getattr(args, 'lr_stage_views', True):
        return None
    if logger.run is None and not logger.plots_enabled:
        return None      # nothing would receive the figures

    def on_lr_stage(volume_mu, iteration, stage, lr):
        vol, geom = crop_to_export_roi(volume_mu, ctx.geometry)
        try:
            anchors = resolve_anchors(
                vol, getattr(args, 'hu_calibration', 'auto'),
                mu_water=getattr(args, 'mu_water', None),
                tissue_hu=getattr(args, 'tissue_hu', None), verbose=False)
        except ValueError as e:
            # An early frame can be too formless to have a histogram at all.
            # That is a fact about stage `stage`, not a reason to lose the
            # rest of the sequence.
            print(f"  lr_plateau: stage {stage} has no fittable histogram "
                  f"({e}) — views skipped for this stage")
            return
        gain = anchors.implied_mu_water
        flag = "" if anchors.gain_determined else " (UNDETERMINED)"
        logger.log_stage_views(
            anchors.apply(vol), geom, step=int(iteration),
            slug=f"lr_stage{int(stage):02d}",
            label=(f"LR stage {stage} \u00b7 lr {lr:.2e} \u00b7 "
                   f"iter {iteration} \u00b7 fitted mu_water "
                   f"{gain:.5f}/mm{flag}"),
            row={'stage': int(stage), 'iteration': int(iteration),
                 'lr': float(lr), 'mu_water': float(gain),
                 'gain_determined': bool(anchors.gain_determined)})
        # The gain as a curve, so a frame that looks brighter can be checked
        # against the calibration in one glance rather than by eye.
        logger.log({'lr_stage/implied_mu_water': float(gain),
                    'lr_stage/hu_scale': float(anchors.scale),
                    'lr_stage/gain_determined': int(bool(anchors.gain_determined)),
                    'lr_stage/lr': float(lr),
                    'lr_stage/stage': int(stage)}, step=int(iteration))

    return on_lr_stage


def _parse_loss_options(pairs):
    """``KEY=VALUE`` strings -> a dict, with numbers parsed as numbers.

    The registry's options are typed (patch counts are ints, weights are
    floats), and a string where a float belongs fails deep inside a loss rather
    than at the CLI. Anything that is not a number is passed through verbatim.
    """
    out = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--loss-option expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def parse_args(argv=None):
    """The parsed namespace AND the algorithm it was built for.

    Returned together because they are one answer: the namespace is only
    meaningful alongside the descriptor whose flags shaped it.
    """
    algorithm, algorithm_modules = _resolve_algorithm(argv)
    parser = argparse.ArgumentParser(
        description="Learning-based iterative reconstruction "
                    "(differentiable projector + gradient descent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms (--algorithm):
""" + describe_algorithms() + """

Examples:
  python -m reconstruction.run_learned_recon data/scans/Scan_1510
  python -m reconstruction.run_learned_recon data/scans/Scan_1510 --iterations 40000
  python -m reconstruction.run_learned_recon data/scans/Scan_1510 --downsample 3 --no-crossval
  python -m reconstruction.run_learned_recon data/scans/Scan_1510 --loss msssim
  python -m reconstruction.run_learned_recon data/scans/Scan_1510 --loss huber

Data terms (--loss):
""" + describe_data_terms() + """

The sampler follows the loss: per-ray terms draw random rays, the ramp-filtered
terms draw complete detector rows, and the structural terms draw 2-D patches
(--loss-option patch_size=N, num_patches=N).
        """
    )
    add_common_args(parser)
    add_model_domain_args(parser)

    # Declared on the real parser too, so it appears in --help and does not
    # come back as an unrecognized argument; the pre-parser already acted on it.
    parser.add_argument('--algorithm-module', action='append', default=[],
                        metavar='MODULE',
                        help='Import MODULE before resolving --algorithm, so a '
                             'representation defined OUTSIDE this package '
                             'becomes selectable (registration is a side '
                             'effect of the import). Repeatable; also settable '
                             'once via $' + ALGORITHM_MODULE_ENV + '. e.g. '
                             '--algorithm-module inr_pipeline.algorithms')
    parser.add_argument('--algorithm', default=DEFAULT_ALGORITHM,
                        choices=algorithm_names(),
                        help='Representation to optimize (default: '
                             '%(default)s). Listed under "Algorithms" below; '
                             'each brings its own options group. The list is '
                             'whatever is registered, so it grows with '
                             '--algorithm-module.')
    # Whatever the selected representation adds for itself. Grouped so
    # `--help` says which flags stop meaning anything under --algorithm X.
    algorithm.add_args(parser.add_argument_group(
        f'{algorithm.name} options',
        f'Flags specific to --algorithm {algorithm.name} '
        f'({algorithm.summary}).'))
    parser.add_argument('--loss', default=DEFAULT_DATA_TERM,
                        choices=sorted(DATA_TERMS),
                        help='Data term (default: %(default)s). MSE is the '
                             'objective classical SIRT descends, which is what '
                             'makes the default run a like-for-like comparison '
                             'against it. See the list below.')
    parser.add_argument('--emulate-sart', action='store_true',
                        help='Emulate the classical simultaneous update as far '
                             'as this backend goes: --loss sart SUMMED (row '
                             'weighting R = 1/L_i over the object ROI), the '
                             'coverage preconditioner C at its absolute '
                             '1/sum_i A_ij scale on the backward pass, and '
                             'plain SGD instead of Adam (Adam would be a '
                             'second, competing preconditioner). Together '
                             'those make --lr the classical relaxation lambda. '
                             'The dense voxel grid, the non-negativity '
                             'projection and the near-air init are already the '
                             'classical recipe. NOT strict SIRT: batches stay '
                             'random subsets, which is the SART/ordered-subset '
                             'family.')
    parser.add_argument('--optimizer', default=None,
                        choices=('adam', 'adam_bf16', 'sgd'),
                        help='Optimizer. Default: adam, or sgd under '
                             '--emulate-sart. Setting it explicitly is always '
                             'respected — pairing adam with --emulate-sart '
                             'warns rather than being overridden, since Adam is '
                             'itself a preconditioner competing with C. '
                             'adam_bf16 is the same update with the moment '
                             'buffers in bfloat16: on the voxel backend each '
                             'buffer is a whole volume, so it drops the '
                             'resident cost from 4x to 3x and buys back ~1.7x '
                             'the ray batch.')
    parser.add_argument('--sart-outside-weight', type=float, default=0.25,
                        metavar='W',
                        help='With --emulate-sart, the rate at which path '
                             'OUTSIDE the object ROI counts toward L_i '
                             '(default: %(default)s). Must be > 0: the bed and '
                             'holder attenuate, so those rays carry signal and '
                             'zero would drop them from the loss entirely.')
    parser.add_argument('--loss-option', action='append', default=[],
                        metavar='KEY=VALUE',
                        help='Extra option for the data term, repeatable: e.g. '
                             '--loss-option patch_size=96 '
                             '--loss-option ssim_weight=0.5. Options that do '
                             'not apply to the selected term are ignored.')
    parser.add_argument('--iterations', type=int, default=20000,
                        help='Optimizer steps (default: 20000). On Scan_1510 '
                             'the holdout optimum sat near 16k; crossval stops '
                             'earlier when the holdout MSE plateaus.')
    parser.add_argument('--rays-per-batch', default='auto', metavar='N|auto',
                        help='Rays per optimizer step (default: auto — the '
                             'largest batch that fits the free VRAM after the '
                             'grid, Adam state and sinogram, so the same '
                             'command uses a 16 GB and an 80 GB card fully). '
                             'The chosen value is printed and logged. Pin an '
                             'integer for run-to-run reproducibility across '
                             'different GPUs, since batch size changes the '
                             'optimization dynamics.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate on the voxel values. Default depends '
                             'on the optimizer, because the two are not on the '
                             'same scale at all: 1e-4 for Adam (~0.5%% of '
                             'mu_water per step, since Adam normalises the '
                             'gradient to ~unit magnitude) and 1.0 for SGD, '
                             'where it is the classical relaxation lambda: '
                             'with --emulate-sart one step IS one OS-SART '
                             'update, so lambda<1 under-relaxes and lambda>1 '
                             'over-relaxes exactly as in the textbook method. '
                             'Sharing one default would make an SGD run take '
                             'steps ~1e-4 of the intended size and appear to '
                             'do nothing.')
    parser.add_argument('--lr-warmup-iters', type=int, default=500,
                        help='Linear LR warmup steps before cosine decay '
                             '(default: 500)')
    parser.add_argument(
        '--subpixel-rays', action=argparse.BooleanOptionalAction, default=True,
        help='(default: on) Place each training ray uniformly inside its '
             'detector pixel '
             'instead of at the pixel centre. A detector pixel integrates the '
             'beam over its footprint (84.9 um at isocentre on Scan_1988 at '
             '--downsample 3, comparable to a 100 um voxel); sampling the '
             'centre models it as a delta function, making the forward model '
             'sharper than the instrument. Since rays are already drawn at '
             'random this costs nothing — over iterations the jitter becomes a '
             'Monte-Carlo integral over the pixel. Changes the forward '
             'operator, so runs with and without it are not bit-comparable.')
    parser.add_argument('--samples-per-ray', type=int, default=None,
                        help='Quadrature samples per ray (default: auto = '
                             '~0.55-voxel steps across the FOV chord, the '
                             'anti-aliasing rule from the validated recipe)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Sampling seed (default: 0)')
    parser.add_argument('--gpu-index', type=int, default=0,
                        help='CUDA device index (default: 0)')
    parser.add_argument('--compile', dest='compile_mode', default='on',
                        choices=('off', 'on', 'max-autotune'),
                        help='Fuse the renderer kernels with torch.compile '
                             '(default: on). The forward is bandwidth-bound, '
                             'so fusing the quadrature chain cuts the traffic '
                             'per step roughly in half. Needs Triton, i.e. '
                             'compute capability >= 7.0; on an older card the '
                             'run prints why and proceeds eager. Pass off to '
                             'make a run bit-comparable with an eager '
                             'baseline, since fusion reorders floating-point '
                             'ops. max-autotune benchmarks kernel variants '
                             '(minutes of extra compile).')
    parser.add_argument('--detector-warp', default='off',
                        choices=('off', 'auto', 'nonaffine', 'full'),
                        help='Per-pixel detector distortion correction applied '
                             'to ray geometry (default: off — matches the '
                             'classical backends, which have no equivalent). '
                             'auto/nonaffine/full use the serial-keyed '
                             'calibration in data/calibration/ when present.')

    # Cross-validation holdout (same knobs as the TIGRE backend)
    parser.add_argument('--no-crossval', action='store_true', default=False,
                        help='Disable the held-out projection + early stopping. '
                             'NOTE: this also returns the held-out angle to the '
                             'training pool.')
    parser.add_argument('--holdout-index', type=int, default=None, metavar='N',
                        help='Projection index to hold out (default: middle)')
    parser.add_argument('--eval-every', type=int, default=250, metavar='K',
                        help='Evaluate holdout MSE every K iterations '
                             '(default: 250)')
    parser.add_argument('--patience', type=int, default=None, metavar='P',
                        help='Stop after P holdout evals without improvement. '
                             'Default: 4 with --lr-plateau on (reaching the LR '
                             'floor already required six detected plateaus, so '
                             'the waiting is the reducer\'s job), else a '
                             'quarter of this run\'s evaluations with a floor '
                             'of 8, so the rule means the same thing at any '
                             '--iterations / --eval-every. Counted in '
                             'EVALUATIONS, so what it is worth in iterations '
                             'is P x --eval-every.')
    parser.add_argument(
        '--lr-plateau', action='store_true', default=True,
        help='Metric-driven LR decay (default: ON). Instead of an open-loop '
             'cosine scheduled over --iterations — a horizon a run that stops '
             'early never reaches, so the annealing never happens — the LR is '
             'halved whenever the SAME held-out metric the stopper watches '
             'stops improving, and the run ends only once the LR is at its '
             'floor and the metric still will not improve. Warmup is '
             'unaffected; the reducer takes over when it ends.')
    parser.add_argument(
        '--no-lr-plateau', dest='lr_plateau', action='store_false',
        help='Hand the LR back to the open-loop cosine schedule.')
    parser.add_argument(
        '--lr-plateau-factor', type=float, default=0.5, metavar='F',
        help='LR multiplier per reduction (default: %(default)s).')
    parser.add_argument(
        '--lr-plateau-patience', type=int, default=2, metavar='P',
        help='Non-improving evaluations that buy one LR reduction '
             '(default: %(default)s). This is the REDUCER\'s patience; '
             '--patience is the separate count that ends the run once the LR '
             'can no longer drop.')
    parser.add_argument(
        '--lr-plateau-min-fraction', type=float,
        default=DEFAULT_MIN_LR_FRACTION, metavar='F',
        help='LR floor as a fraction of the post-warmup base '
             '(default: %(default)s). With factor 0.5 that is 3 reductions '
             'before the floor, after which a plateau ends the run. It was '
             '0.02 (six reductions); reductions 4-6 were measured to change '
             'the volume by at most 2 grey levels of the 3000 HU display '
             'window, in the grain band, for 22 %% of the run. Lower it to '
             '0.02 to restore the old six if a run needs them.')
    parser.add_argument(
        '--lr-plateau-cooldown', type=int, default=1, metavar='K',
        help='Evaluations to wait after a reduction before counting again '
             '(default: %(default)s), so one plateau cannot cascade several '
             'cuts before the model has responded to the first.')
    parser.add_argument(
        '--lr-stage-views', action='store_true', default=True,
        help='Log the three midplane views at every LR change (default: ON) '
             'as plots/lr_stage/view_axial|coronal|sagittal — a slider over '
             'the LR stages, so how much of the late fitting is signal and '
             'how much is noise can be READ OFF the volume instead of '
             'inferred from the held-out curve. Stage 0 is the un-annealed '
             'reconstruction; one frame follows per reduction.')
    parser.add_argument(
        '--no-lr-stage-views', dest='lr_stage_views', action='store_false',
        help='Skip the per-stage views. Each one costs a full export of the '
             'model onto the export grid (cheap for a voxel grid, a render '
             'for anything else), paid at most --lr-plateau-min-fraction '
             'times over the run.')
    parser.add_argument(
        '--stop-min-gain', type=float, default=8e-4, metavar='R',
        help='The plateau definition: the relative improvement per SINOGRAM '
             'VISIT below which an evaluation does not count as progress '
             '(default: %(default)s, i.e. 0.08 %% per pass over the data). '
             'Both the LR reducer and the stop are driven by it. Per visit '
             'rather than per iteration or per evaluation because those two '
             'both move: the gain an evaluation can show scales with '
             '--eval-every, and the gain an iteration can show scales with '
             '--rays-per-batch. A visit is iterations x rays / measurements, '
             'so both cancel and one number transfers across cadences, batch '
             'sizes, cards and scans. Pass 0 for zero tolerance, where any '
             'decrease however small resets the patience — MEASURED on run '
             'ny96yzab that made the stop unreachable (longest non-improving '
             'streak 3 evaluations against a patience of 8) and the run ended '
             'on its iteration cap after 11.7 h; at this default the same run '
             'stops at 3.0 h having seen 122 of its 467 visits, for 0.008 of '
             'held-out SSIM measured on a projection that was already fitted '
             'below its own noise floor. Halve it toward 4e-4 or 2e-4 if the '
             'plots/lr_stage/view_* sequence shows the cut stages were still '
             'adding structure rather than grain.')
    parser.add_argument(
        '--stop-min-delta', type=float, default=0.0, metavar='D',
        help='Absolute improvement threshold in the metric\'s OWN units '
             '(default: %(default)s). Almost always the wrong knob — a value '
             'that suits SSIM is enormous next to an MSE — use '
             '--stop-min-gain instead.')
    parser.add_argument('--min-stop-iter', type=int, default=None, metavar='N',
                        help='No stopping rule may fire before iteration N '
                             '(default: half the scheduled iterations; pass 0 '
                             'to disable). Guards against ending a run during '
                             'the high-LR phase, where a plateau reflects the '
                             'step size the schedule was still going to shrink '
                             'rather than the reconstruction.')
    parser.add_argument(
        '--stop-metric',
        choices=STOP_METRICS,
        default='mse',
        help='Which held-out metric decides the peak. They do not peak '
             'together: mse is the objective and turns over last, ssim is '
             'structural and turns over earliest, psnr sits between. Default: '
             'ssim.'
    )
    parser.add_argument(
        '--l-curve',
        action='store_true',
        default=False,
        help='Also record the L-curve: the residual norm against the solution '
             'norm, in log-log, per checkpoint. Its corner is where further '
             'residual reduction starts buying disproportionate solution '
             'growth, i.e. where noise amplification takes over. Costs one '
             'extra forward projection per checkpoint and needs NO held-out '
             'data, so it is the criterion that still applies when an angle '
             'cannot be spared.'
    )
    parser.add_argument(
        '--l-curve-norm',
        choices=('l2', 'gradient'),
        default='l2',
        help="Solution norm for the L-curve: 'l2' (classical) or 'gradient' "
             "(the seminorm ||grad x||, more sensitive to the high-frequency "
             "noise that semi-convergence amplifies). Default: l2."
    )
    parser.add_argument(
        '--stop-on',
        nargs='+',
        choices=('holdout', 'lcurve'),
        default=['holdout'],
        metavar='RULE',
        help="Which rule(s) may END the run: 'holdout' (default) and/or "
             "'lcurve'. Whichever fires first wins. A rule that is recorded "
             "but not listed here is diagnostic only — the curve is still "
             "logged and plotted."
    )

    args = parser.parse_args(argv)
    # What was actually imported, not what was asked for: the env var
    # contributes too, and a run should record the modules its representation
    # really came from.
    args.algorithm_module = algorithm_modules
    return args, algorithm


def main(argv=None):
    args, algorithm = parse_args(argv)
    start = time.time()

    auto_batch = str(args.rays_per_batch).strip().lower() == 'auto'
    if not auto_batch:
        try:
            args.rays_per_batch = int(args.rays_per_batch)
        except ValueError:
            raise ConfigError(
                f"--rays-per-batch: expected an integer or 'auto', "
                f"got {args.rays_per_batch!r}") from None

    print("=" * 60)
    print(f"Learning-based Iterative Reconstruction "
          f"({algorithm.name}: {algorithm.summary})")
    print("=" * 60)
    print(f"Data folder: {args.data_folder}")
    print(f"Iterations: {args.iterations}  "
          f"rays/batch: {'auto' if auto_batch else args.rays_per_batch}  "
          f"lr: {args.lr}")

    # Shared front half: scan folder, projections, ROI, geometry, downsample.
    ctx = prepare_scan(args, fit_domain=True)

    if args.output:
        output_path = args.output
    else:
        output_path = ctx.default_output_path(
            f'_recon_{algorithm.name}_{args.iterations}it')
    print(f"\nOutput path: {output_path}")

    # Quadrature samples per ray — needed before the trainer exists, both to
    # size the ray batch and to estimate VRAM. Mirrors the trainer's auto rule.
    _spp = args.samples_per_ray or int(np.ceil(
        min(int(ctx.geometry['vol_shape'][0]), int(ctx.geometry['vol_shape'][1]))
        / 0.55))

    # ---- ray batch size: fill the card we actually got ---------------------
    # Same pattern as the FDK backend's chunk sizing: measure free VRAM, put
    # the persistent buffers aside, spend the rest per step. Pinning an
    # integer skips this entirely.
    if auto_batch:
        gpu = query_gpu_memory(args.gpu_index)
        n_ang, n_b, n_a = (int(s) for s in ctx.projections.shape)
        plan = auto_rays_per_batch(
            (gpu or {}).get('free_bytes'), n_angles=n_ang, n_b=n_b, n_a=n_a,
            vol_shape=ctx.geometry['vol_shape'], samples_per_ray=_spp,
            optimizer=_resolve_optimizer(args),
            # What stays resident is the REPRESENTATION'S, not the export
            # grid's: 4x the volume for a dense grid, 4x a few million weights
            # for a network. The batch is whatever the card has left, so the
            # difference is the difference between a floored batch and a
            # capped one.
            backend=algorithm.name, footprint=algorithm.bind_footprint(args))
        args.rays_per_batch = plan['rays']
        if gpu is None:
            print(f"\nRays/batch: auto -> {plan['rays']} (no GPU visible — "
                  f"floor)")
        else:
            limit = ('capped' if plan['capped'] else
                     'floored' if plan['floored'] else 'VRAM-limited')
            print(f"\nRays/batch: auto -> {plan['rays']} ({limit}; "
                  f"{plan['budget_bytes'] / 2**30:.1f} GiB budget - "
                  f"{plan['persistent_bytes'] / 2**30:.1f} GiB persistent, "
                  f"{_spp} samples/ray). Pin --rays-per-batch to reproduce "
                  f"this run on another GPU.")

    # Experiment logging: local PNGs next to the output, plus W&B unless
    # --no-wandb (on by default; inert until a project is configured).
    # Created BEFORE preflight (an auto-aborted job is recorded as a FAILED
    # run with the verdict); the trainer then logs LIVE (loss/lr/holdout-MSE
    # per step) through the log_fn callback.
    logger = ReconLogger(args, ctx, algorithm.name, output_path, params={
        # `algorithm.config(args)` is spliced in below: the representation's
        # own options, recorded under the same names its constructor takes,
        # so the run config says what the algorithm was actually given without
        # this driver listing knobs it does not understand.
        **algorithm.config(args),
        'iterations': args.iterations,
        'rays_per_batch': args.rays_per_batch,
        'rays_per_batch_mode': 'auto' if auto_batch else 'pinned',
        'lr': _resolve_lr(args, algorithm),
        'optimizer': _resolve_optimizer(args),
        'lr_plateau': bool(args.lr_plateau),
        'lr_plateau_factor': args.lr_plateau_factor if args.lr_plateau else None,
        'lr_plateau_patience': (args.lr_plateau_patience if args.lr_plateau
                                else None),
        'lr_stage_views': bool(args.lr_stage_views),
        'stop_min_gain': args.stop_min_gain,
        'stop_min_delta': args.stop_min_delta,
        'lr_warmup_iters': args.lr_warmup_iters,
        'samples_per_ray': args.samples_per_ray,
        'samples_per_ray_mode': 'pinned' if args.samples_per_ray else 'auto',
        'subpixel_rays': args.subpixel_rays,
        'seed': args.seed,
        'compile': args.compile_mode,
        'detector_warp': args.detector_warp,
        'crossval': not args.no_crossval,
        'withhold_eval': bool(args.withhold_eval),
        # Where the representation came from. A dotted module NAME (never a
        # filesystem path), so a run using an out-of-tree model still says
        # which module defined it; empty for the ones this package ships.
        'algorithm_modules': ','.join(args.algorithm_module),
    })

    # Machine fit check (GPU presence / VRAM / RAM) before any big allocation.
    # The size model comes from the ALGORITHM (`bind_footprint`), because the
    # resident parameter copies are the one term this driver cannot derive:
    # a dense grid's parameters are the exported volume, a network's have
    # nothing to do with it. Getting that wrong is a spurious abort in one
    # direction and an OOM ten minutes into the run in the other.
    if run_preflight(algorithm.name, ctx,
                     footprint=algorithm.bind_footprint(args),
                     gpu_index=args.gpu_index, logger=logger,
                     rays_per_batch=args.rays_per_batch, samples_per_ray=_spp,
                     optimizer=_resolve_optimizer(args),
                     skip=args.skip_preflight,
                     only=args.preflight_only).dry_run:
        return                      # --preflight-only: the question is answered

    # ---- geometry auto-calibration (detector in-plane rotation psi) --------
    # Cached scan-keyed JSON first; on a miss the half-scan-consistency
    # estimator measures psi from this scan's projections. rays_from_indices
    # applies geometry['det_psi_rad'] directly.
    if args.geometry_autocal:
        record = resolve_or_measure_detector_psi(
            ctx, fallback_note="using psi=0")
        if record is not None:
            ctx.geometry['det_psi_rad'] = float(np.radians(record['psi_deg']))

    # ---- optional detector unwarp (learning-based exclusive) ---------------
    # The differentiable projector builds every ray through one choke point,
    # so the per-pixel detector distortion calibration can be applied to ray
    # geometry — something the classical backends cannot do.
    if args.detector_warp != 'off':
        n_b, n_a = ctx.projections.shape[1], ctx.projections.shape[2]
        ds = int(ctx.downsample or 1)
        warp = resolve_detector_warp(
            {'detector_warp': {'mode': args.detector_warp}},
            ctx.scan_folder, (n_b * ds, n_a * ds), downsample=ds)
        if warp is not None:
            ctx.geometry['detector_warp'] = warp
            ctx.geometry['sinogram_downsample'] = ds

    # ---- projection diagnostics: noise ceiling at the eval angle -----------
    # Other acquisition phase when the scan has one, else the neighbouring
    # projection. The trainer streams diag/ssim|psnr|mse against this ceiling
    # at every eval checkpoint via diag_fn.
    eval_idx = (args.holdout_index if args.holdout_index is not None
                else int(ctx.projections.shape[0]) // 2)
    logger.set_noise_ceiling(measure_noise_ceiling(
        ctx, eval_idx, phase=args.phase))

    # `shared` is the LOOP's configuration — data, objective, schedule, the
    # stopping rules, the logging hooks — and every representation gets all of
    # it. The representation adds only `algorithm.options(args)` below, the
    # constructor arguments its own flags produced, and answers the three
    # hooks (model / domain / export) itself. Nothing in this call names a
    # representation.
    shared = dict(
        projections=ctx.projections,
        angles=ctx.angles,
        geometry=ctx.geometry,
        iterations=args.iterations,
        rays_per_batch=args.rays_per_batch,
        samples_per_ray=args.samples_per_ray,
        subpixel_rays=args.subpixel_rays,
        lr_warmup_iters=args.lr_warmup_iters,
        gpu_index=args.gpu_index,
        seed=args.seed,
        compile_mode=args.compile_mode,
        lr=_resolve_lr(args, algorithm),
        loss=args.loss,
        loss_options=_parse_loss_options(args.loss_option),
        emulate_sart=args.emulate_sart,
        optimizer=args.optimizer,
        sart_outside_weight=args.sart_outside_weight,
        bright_field=ctx.bright_field,
        dark_field=ctx.dark_field,
        ring_correction=args.ring_correction,
        air_normalization=args.air_normalization,
        soft_clip_sharpness=args.soft_clip_sharpness,
        ring_median_width=args.ring_median_width,
        crossval=not args.no_crossval,
        holdout_index=args.holdout_index,
        withhold_eval=args.withhold_eval,
        eval_every=args.eval_every,
        patience=args.patience,
        min_stop_iter=args.min_stop_iter,
        lr_plateau=_build_lr_plateau(args),
        stop_metric=args.stop_metric,
        stop_min_delta=args.stop_min_delta,
        stop_min_gain=args.stop_min_gain,
        l_curve=args.l_curve,
        l_curve_norm=args.l_curve_norm,
        stop_on=tuple(args.stop_on),
        log_fn=logger.log,
        # diag/* scalars every eval + SSIM-heatmap / power-spectrum figures
        # on a coarser cadence (figure_every_evals), all through the logger.
        diag_fn=logger.log_projection_diag,
        # plots/lr_stage/view_* — one frame per LR change, so the late
        # iterations can be judged on the volume rather than on the held-out
        # curve alone.
        on_lr_stage=_lr_stage_views_fn(ctx, args, logger),
    )

    # An option named like one of the shared arguments would silently decide
    # which of the two the reconstructor sees (or raise a bare TypeError about
    # duplicate keywords, deep in a constructor). Named here instead, where
    # the fix — rename the flag — is obvious.
    options = algorithm.options(args)
    clash = sorted(set(options) & set(shared))
    if clash:
        raise ConfigError(
            f"--algorithm {algorithm.name} declares option(s) {clash} that "
            f"collide with arguments every representation already gets. "
            f"Rename them in its LearnedAlgorithm.options.")
    reconstructor = algorithm.reconstructor(**shared, **options)

    reconstructor.reconstruct()

    # What the trainer ACTUALLY used, read back off the backend (not off args
    # — early stopping cuts the iteration count, and the batch may have been
    # auto-sized). Same data/* keys as the classical drivers; the difference
    # is the sampling mode, which is what makes coverage < 100% here.
    logger.set_data_budget(
        data_budget(reconstructor.n_measurements,
                    rays_drawn=(reconstructor.iterations_run
                                * reconstructor.rays_per_batch),
                    sampling=RANDOM),
        note=f"{reconstructor.iterations_run} iterations x "
             f"{reconstructor.rays_per_batch} rays, sampled with replacement",
        extra={'data/iterations_run': reconstructor.iterations_run,
               'data/rays_per_batch': reconstructor.rays_per_batch,
               'data/rays_per_batch_mode': 'auto' if auto_batch else 'pinned'})

    # The other half of that trade. Rays/batch is VRAM-limited by rays x
    # samples, so the ray step and the data budget above are bought with the
    # same memory: a coarser step buys more measurements per iteration and
    # pays in forward-model fidelity. Recording both makes the exchange rate
    # of a run legible after the fact instead of a thing to re-derive from
    # the geometry. Read off the trainer, which resolves 'auto' against the
    # MEASURED model domain — the driver's own estimate above only has to be
    # close enough to size VRAM.
    _q = getattr(reconstructor, 'quadrature', None) or {}
    if _q:
        logger.set_summary({
            'quadrature/samples_per_ray': int(_q['samples_per_ray']),
            'quadrature/mode': str(_q['mode']),
            'quadrature/step_mm_mean': float(_q['step_mm_mean']),
            'quadrature/step_mm_widest': float(_q['step_mm_widest']),
            'quadrature/step_voxels_mean': float(_q['step_voxels_mean']),
            'quadrature/step_voxels_widest': float(_q['step_voxels_widest']),
            'quadrature/samples_per_voxel_mean':
                float(_q['samples_per_voxel_mean']),
            'quadrature/samples_per_voxel_widest':
                float(_q['samples_per_voxel_widest']),
            'quadrature/chord_mm_mean': float(_q['chord_mm_mean']),
            'quadrature/chord_mm_widest': float(_q['chord_mm_widest']),
            'quadrature/voxel_mm': float(_q['voxel_mm']),
            'quadrature/ray_hit_fraction': float(_q['ray_hit_fraction']),
        })

    # Crop the reconstruction domain down to what is worth saving, THEN
    # calibrate and export. Everything downstream reports the delivered
    # volume, not the padded domain the optimizer needed.
    vol_export, ctx.geometry = crop_to_export_roi(
        reconstructor.reconstructed_volume, ctx.geometry)

    # Shared back half: HU calibration + bilateral filter + VFF export.
    _, _, volume_hu = save_outputs(vol_export, ctx, args, output_path,
                                   logger=logger, algorithm=algorithm.name)

    # replay_steps=False: the trainer already streamed these live via diag_fn.
    logger.log_convergence(reconstructor.crossval_history, replay_steps=False)
    logger.log_sinogram_preview(ctx.projections)
    logger.log_volume_summary(volume_hu, ctx)
    logger.log_recon_slices(volume_hu, geometry=ctx.geometry)
    logger.finish()

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    cli_main(main)
