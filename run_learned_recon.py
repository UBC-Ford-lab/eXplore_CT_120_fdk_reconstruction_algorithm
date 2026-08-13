"""
Run learning-based iterative cone-beam CT reconstruction.

The volume is a differentiable representation fitted to the measured line
integrals by gradient descent through a differentiable forward projector
(reconstruction as optimization — the same argmin classical iterative recon
solves, with autograd replacing the hand-derived update rule).

Algorithms (each in its own subfolder of ``learning_based_iterative/``):
  - voxel: dense voxel grid — SIRT's representation trained with Adam + MSE.
           The recipe validated in muNeRF (configs/scan_1510_VOXEL_mse.yaml).

All algorithm-independent stages (scan loading, geometry build, detector
downsampling, detector-psi calibration, HU calibration + VFF export) live in
ct_core.pipeline and are shared with the FDK / iterative drivers.

Usage:
    python -m reconstruction.run_learned_recon data/scans/Scan_1510
    python -m reconstruction.run_learned_recon data/scans/Scan_1510 \\
        --iterations 40000 --downsample 3 --lr 1e-4
"""

import argparse
import sys
import time

import numpy as np

from .learning_based_iterative.voxel.reconstructor import VoxelReconstructor
from .learning_based_iterative.detector_warp import resolve_detector_warp
from .ct_core.pipeline import (
    ReconLogger,
    add_common_args,
    prepare_scan,
    resolve_or_measure_detector_psi,
    run_preflight,
    save_outputs,
)
from .ct_core.projection_diag import measure_noise_ceiling

SUPPORTED_LEARNED_ALGORITHMS = ("voxel",)


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
        """
    )
    add_common_args(parser)

    parser.add_argument('--algorithm', default='voxel',
                        choices=SUPPORTED_LEARNED_ALGORITHMS,
                        help='Representation to optimize (default: voxel). '
                             'Future: nerf, hashgrid, gaussian_splatting.')
    parser.add_argument('--iterations', type=int, default=20000,
                        help='Optimizer steps (default: 20000). On Scan_1510 '
                             'the holdout optimum sat near 16k; crossval stops '
                             'earlier when the holdout MSE plateaus.')
    parser.add_argument('--rays-per-batch', type=int, default=16384,
                        help='Rays per optimizer step (default: 16384)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Adam learning rate on the voxel values '
                             '(default: 1e-4 — ~0.5%% of mu_water per step). '
                             'First knob to sweep if training is slow/unstable.')
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

    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print(f"Learning-based Iterative Reconstruction ({args.algorithm})")
    print("=" * 60)
    print(f"Data folder: {args.data_folder}")
    print(f"Iterations: {args.iterations}  rays/batch: {args.rays_per_batch}  "
          f"lr: {args.lr}")

    # Shared front half: scan folder, projections, ROI, geometry, downsample.
    ctx = prepare_scan(args)

    if args.output:
        output_path = args.output
    else:
        output_path = ctx.default_output_path(
            f'_recon_{args.algorithm}_{args.iterations}it')
    print(f"\nOutput path: {output_path}")

    # Experiment logging: local PNGs next to the output, W&B when --wandb.
    # Created BEFORE preflight (an auto-aborted job is recorded as a FAILED
    # run with the verdict); the trainer then logs LIVE (loss/lr/holdout-MSE
    # per step) through the log_fn callback.
    logger = ReconLogger(args, ctx, args.algorithm, output_path, params={
        'iterations': args.iterations,
        'rays_per_batch': args.rays_per_batch,
        'lr': args.lr,
        'lr_warmup_iters': args.lr_warmup_iters,
        'samples_per_ray': args.samples_per_ray,
        'init_density': args.init_density,
        'seed': args.seed,
        'detector_warp': args.detector_warp,
        'crossval': not args.no_crossval,
        'withhold_eval': bool(args.withhold_eval),
    })

    # Machine fit check (GPU presence / VRAM / RAM) before any big allocation.
    # The voxel grid's VRAM need is dominated by 4x parameters (Adam), plus
    # the per-batch ray buffers — mirror the trainer's auto-spp rule here.
    _spp = args.samples_per_ray or int(np.ceil(
        min(int(ctx.geometry['vol_shape'][0]), int(ctx.geometry['vol_shape'][1]))
        / 0.55))
    run_preflight('voxel', ctx, gpu_index=args.gpu_index, logger=logger,
                  rays_per_batch=args.rays_per_batch, samples_per_ray=_spp,
                  skip=args.skip_preflight, only=args.preflight_only)

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
        ctx, eval_idx, phase=args.phase, bhc_coeffs=args.bhc_coeffs))

    reconstructor = VoxelReconstructor(
        projections=ctx.projections,
        angles=ctx.angles,
        geometry=ctx.geometry,
        iterations=args.iterations,
        rays_per_batch=args.rays_per_batch,
        lr=args.lr,
        samples_per_ray=args.samples_per_ray,
        lr_warmup_iters=args.lr_warmup_iters,
        init_density=args.init_density,
        gpu_index=args.gpu_index,
        seed=args.seed,
        bright_field=ctx.bright_field,
        dark_field=ctx.dark_field,
        output_hu=True,
        bhc_coeffs=args.bhc_coeffs,
        ring_correction=args.ring_correction,
        ring_median_width=args.ring_median_width,
        crossval=not args.no_crossval,
        holdout_index=args.holdout_index,
        withhold_eval=args.withhold_eval,
        eval_every=args.eval_every,
        patience=args.patience,
        log_fn=logger.log,
        # diag/* scalars every eval + SSIM-heatmap / power-spectrum figures
        # on a coarser cadence (figure_every_evals), all through the logger.
        diag_fn=logger.log_projection_diag,
    )

    reconstructor.reconstruct()

    # Shared back half: HU calibration + bilateral filter + VFF export.
    save_outputs(reconstructor.reconstructed_volume, ctx, args, output_path)

    # replay_steps=False: the trainer already streamed these live via diag_fn.
    logger.log_convergence(reconstructor.crossval_history, replay_steps=False)
    logger.log_sinogram_preview(ctx.projections)
    logger.log_volume_summary(reconstructor.reconstructed_volume, ctx)
    logger.log_recon_slices(reconstructor.reconstructed_volume)
    logger.finish()

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
