"""
Run learning-based iterative cone-beam CT reconstruction.

The volume is a differentiable representation fitted to the measured line
integrals by gradient descent through a differentiable forward projector
(reconstruction as optimization — the same argmin classical iterative recon
solves, with autograd replacing the hand-derived update rule).

Algorithms (each in its own subfolder of ``learning_based_iterative/``):
  - voxel: dense voxel grid — SIRT's representation trained with Adam + MSE.

All algorithm-independent stages (scan loading, geometry build, detector
downsampling, detector-psi calibration, HU calibration + VFF export) live in
ct_core.pipeline and are shared with the FDK / iterative drivers.

Usage:
    python -m reconstruction.run_learned_recon data/scans/Scan_1510
    python -m reconstruction.run_learned_recon data/scans/Scan_1510 \\
        --iterations 40000 --downsample 3 --lr 1e-4
"""

import argparse
import time

import numpy as np

from .learning_based_iterative.losses import (DATA_TERMS, DEFAULT_DATA_TERM,
                                              describe_data_terms)
from .learning_based_iterative.voxel.reconstructor import VoxelReconstructor
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
from .ct_core.errors import ConfigError, cli_main
from .ct_core.preflight import auto_rays_per_batch
from .ct_core.projection_diag import measure_noise_ceiling
from .ct_core.utils import query_gpu_memory
from .ct_core.early_stop import STOP_METRICS

SUPPORTED_LEARNED_ALGORITHMS = ("voxel",)


def _resolve_lr(args):
    """The learning rate, defaulted per OPTIMIZER rather than globally.

    Adam and SGD differ by ~4-5 orders of magnitude here: Adam divides by the
    gradient's own running scale, so 1e-4 is a step in units of the parameter,
    while plain SGD steps by the raw gradient. 1.0 is the classical update's
    implicit step size, not a tuned value.
    """
    if args.lr is not None:
        return float(args.lr)
    optimizer = args.optimizer
    if optimizer is None:
        optimizer = "sgd" if args.emulate_sart else "adam"
    return 1.0 if optimizer == "sgd" else 1e-4


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learning-based iterative reconstruction "
                    "(differentiable projector + gradient descent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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

    parser.add_argument('--algorithm', default='voxel',
                        choices=SUPPORTED_LEARNED_ALGORITHMS,
                        help='Representation to optimize (default: voxel). '
                             'Future: nerf, hashgrid, gaussian_splatting.')
    parser.add_argument('--loss', default=DEFAULT_DATA_TERM,
                        choices=sorted(DATA_TERMS),
                        help='Data term (default: %(default)s). MSE is the '
                             'objective classical SIRT descends, which is what '
                             'makes the default run a like-for-like comparison '
                             'against it. See the list below.')
    parser.add_argument('--emulate-sart', action='store_true',
                        help='Emulate the classical simultaneous update as far '
                             'as this backend goes: --loss sart (row weighting '
                             'R = 1/L_i over the object ROI), the coverage '
                             'preconditioner C on the backward pass, and plain '
                             'SGD instead of Adam (Adam would be a second, '
                             'competing preconditioner). The dense voxel grid, '
                             'the non-negativity projection and the near-air '
                             'init are already the classical recipe. NOT strict '
                             'SIRT: batches stay random subsets, which is the '
                             'SART/ordered-subset family.')
    parser.add_argument('--optimizer', default=None, choices=('adam', 'sgd'),
                        help='Optimizer. Default: adam, or sgd under '
                             '--emulate-sart. Setting it explicitly is always '
                             'respected — pairing adam with --emulate-sart '
                             'warns rather than being overridden, since Adam is '
                             'itself a preconditioner competing with C.')
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
                             'gradient to ~unit magnitude) and 1.0 for SGD '
                             '(the classical update\'s implicit step is 1.0 with '
                             'the C A^T R structure). Sharing one default would '
                             'make an SGD run take steps ~1e-4 of the intended '
                             'size and appear to do nothing.')
    parser.add_argument('--lr-warmup-iters', type=int, default=500,
                        help='Linear LR warmup steps before cosine decay '
                             '(default: 500)')
    parser.add_argument('--samples-per-ray', type=int, default=None,
                        help='Quadrature samples per ray (default: auto = '
                             '~0.55-voxel steps across the FOV chord, the '
                             'anti-aliasing rule from the validated recipe)')
    parser.add_argument('--init-density', type=float, default=0.001,
                        help='Initial mu everywhere, in 1/mm (default: 0.001 '
                             '— near air, so gradients raise mu only where '
                             'the projections demand it)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Sampling seed (default: 0)')
    parser.add_argument('--gpu-index', type=int, default=0,
                        help='CUDA device index (default: 0)')
    parser.add_argument('--compile', dest='compile_mode', default='off',
                        choices=('off', 'on', 'max-autotune'),
                        help='Fuse the renderer kernels with torch.compile '
                             '(default: off). The forward is bandwidth-bound, '
                             'so fusing the quadrature chain cuts the traffic '
                             'per step roughly in half. Off by default because '
                             'fusion reorders floating-point ops: a compiled '
                             'run is comparable with other compiled runs, not '
                             'with an eager baseline. max-autotune benchmarks '
                             'kernel variants (minutes of extra compile).')
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
    parser.add_argument('--patience', type=int, default=8, metavar='P',
                        help='Stop after P holdout evals without improvement '
                             '(default: 8 = 2000 iters at eval-every=250)')
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

    return parser.parse_args()


def main():
    args = parse_args()
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
    print(f"Learning-based Iterative Reconstruction ({args.algorithm})")
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
            f'_recon_{args.algorithm}_{args.iterations}it')
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
            vol_shape=ctx.geometry['vol_shape'], samples_per_ray=_spp)
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
    logger = ReconLogger(args, ctx, args.algorithm, output_path, params={
        'iterations': args.iterations,
        'rays_per_batch': args.rays_per_batch,
        'rays_per_batch_mode': 'auto' if auto_batch else 'pinned',
        'lr': args.lr,
        'lr_warmup_iters': args.lr_warmup_iters,
        'samples_per_ray': args.samples_per_ray,
        'init_density': args.init_density,
        'seed': args.seed,
        'compile': args.compile_mode,
        'detector_warp': args.detector_warp,
        'crossval': not args.no_crossval,
        'withhold_eval': bool(args.withhold_eval),
    })

    # Machine fit check (GPU presence / VRAM / RAM) before any big allocation.
    # The voxel grid's VRAM need is dominated by 4x parameters (Adam), plus
    # the per-batch ray buffers sized just above.
    if run_preflight('voxel', ctx, gpu_index=args.gpu_index, logger=logger,
                     rays_per_batch=args.rays_per_batch, samples_per_ray=_spp,
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

    reconstructor = VoxelReconstructor(
        projections=ctx.projections,
        angles=ctx.angles,
        geometry=ctx.geometry,
        iterations=args.iterations,
        rays_per_batch=args.rays_per_batch,
        samples_per_ray=args.samples_per_ray,
        lr_warmup_iters=args.lr_warmup_iters,
        init_density=args.init_density,
        gpu_index=args.gpu_index,
        seed=args.seed,
        compile_mode=args.compile_mode,
        lr=_resolve_lr(args),
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
        stop_metric=args.stop_metric,
        l_curve=args.l_curve,
        l_curve_norm=args.l_curve_norm,
        stop_on=tuple(args.stop_on),
        log_fn=logger.log,
        # diag/* scalars every eval + SSIM-heatmap / power-spectrum figures
        # on a coarser cadence (figure_every_evals), all through the logger.
        diag_fn=logger.log_projection_diag,
    )

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

    # Crop the reconstruction domain down to what is worth saving, THEN
    # calibrate and export. Everything downstream reports the delivered
    # volume, not the padded domain the optimizer needed.
    vol_export, ctx.geometry = crop_to_export_roi(
        reconstructor.reconstructed_volume, ctx.geometry)

    # Shared back half: HU calibration + bilateral filter + VFF export.
    _, _, volume_hu = save_outputs(vol_export, ctx, args, output_path,
                                   logger=logger, algorithm=args.algorithm)

    # replay_steps=False: the trainer already streamed these live via diag_fn.
    logger.log_convergence(reconstructor.crossval_history, replay_steps=False)
    logger.log_sinogram_preview(ctx.projections)
    logger.log_volume_summary(volume_hu, ctx)
    logger.log_recon_slices(volume_hu)
    logger.finish()

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    cli_main(main)
