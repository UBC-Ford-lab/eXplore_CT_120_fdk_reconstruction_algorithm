"""
Run iterative cone-beam CT reconstruction using ASTRA or TIGRE toolbox.

Supports multiple backends:
  - astra: SIRT, CGLS, SART, FDK via ASTRA toolbox
  - tigre: OS-SART, SART, SIRT, MLEM via TIGRE (handles GPU memory splitting
    internally)

All algorithm-independent stages (scan loading, geometry build, detector
downsampling, detector-psi calibration, HU calibration + VFF export) live in
ct_core.pipeline and are shared with the FDK driver — this script only owns
backend/algorithm selection and the iterative-specific knobs.

Usage:
    python -m reconstruction.run_iterative_recon data/scans/Scan_1681
    python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --algorithm CGLS3D_CUDA --iterations 50
    python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --backend tigre --algorithm ossart --iterations 100
"""

import argparse
import time

import numpy as np

from .iterative.astra import ASTRAReconstructor, SUPPORTED_ALGORITHMS as ASTRA_ALGORITHMS
from .iterative.tigre import TIGREReconstructor, SUPPORTED_TIGRE_ALGORITHMS
from .ct_core.data_budget import classical_budget, measurement_count
from .ct_core.early_stop import STOP_METRICS
from .ct_core.errors import ConfigError, cli_main
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
from .ct_core.projection_diag import (
    level_to_air,
    measure_noise_ceiling,
    preprocess_frames,
    render_projection_from_volume,
)


def parse_args():
    """Parse command-line arguments (shared args + iterative-specific ones)."""
    parser = argparse.ArgumentParser(
        description='Run iterative reconstruction on VFF projections using ASTRA or TIGRE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m reconstruction.run_iterative_recon data/scans/Scan_1681
  python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --algorithm CGLS3D_CUDA --iterations 50
  python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --backend tigre --algorithm ossart --iterations 100
  python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --backend tigre --algorithm ossart --iterations 150 --lmbda 0.3
        """
    )
    add_common_args(parser)
    add_model_domain_args(parser)

    # Backend selection
    parser.add_argument(
        '--backend',
        default='astra',
        choices=('astra', 'tigre'),
        help='Reconstruction backend (default: astra)'
    )
    parser.add_argument(
        '--algorithm',
        default=None,
        help='Reconstruction algorithm. '
             'ASTRA: SIRT3D_CUDA, CGLS3D_CUDA, SART3D_CUDA, FDK_CUDA (default: SIRT3D_CUDA). '
             'TIGRE: ossart, sart, sirt, mlem (default: ossart). '
             'mlem is Maximum-Likelihood Expectation-Maximization under a Poisson '
             'noise model (full-batch, no ordered subsets; ignores --lmbda/--lmbda-red; '
             'incompatible with --pwls).'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for iterative algorithms (default: 100). '
             'Ignored for FDK_CUDA.'
    )
    parser.add_argument(
        '--min-constraint',
        type=float,
        default=None,
        help='Minimum voxel value constraint (e.g., 0.0 for non-negativity). '
             'Only for iterative algorithms.'
    )
    parser.add_argument(
        '--max-constraint',
        type=float,
        default=None,
        help='Maximum voxel value constraint. Only for iterative algorithms.'
    )
    parser.add_argument(
        '--gpu-index',
        type=int,
        default=0,
        help='CUDA device index (default: 0)'
    )
    parser.add_argument(
        '--super-sampling',
        type=int,
        default=1,
        help='Detector/voxel super-sampling factor (default: 1). '
             'Higher values improve accuracy at the cost of speed.'
    )
    # Cross-validation holdout (TIGRE backend only)
    parser.add_argument(
        '--no-crossval',
        action='store_true',
        default=False,
        help='Disable holdout cross-validation and run all iterations without '
             'early stopping. By default cross-val is on (TIGRE backend).'
    )
    parser.add_argument(
        '--holdout-index',
        type=int,
        default=None,
        metavar='N',
        help='Projection index to hold out (default: middle projection, N_angles//2). '
             'Ignored when --no-crossval is set or backend is astra.'
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=10,
        metavar='K',
        help='Evaluate holdout metrics every K iterations (default: 10).'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        metavar='P',
        help='Early stopping patience: stop if the metric does not improve for '
             'P consecutive eval checkpoints. Default: a quarter of this '
             "run's evaluations (floor 8), so the rule means the same thing "
             'at any --iterations / --eval-every instead of getting stricter '
             'as runs get longer. Use a large value to disable early stopping '
             'while keeping metric logging.'
    )
    parser.add_argument(
        '--min-stop-iter',
        type=int,
        default=None,
        metavar='N',
        help='No stopping rule may fire before iteration N (default: half the '
             'scheduled iterations; pass 0 to disable). Stopping late costs '
             'only wall-clock, since the best iterate is restored either way.'
    )
    parser.add_argument(
        '--stop-metric',
        choices=STOP_METRICS,
        default='ssim',
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
    parser.add_argument(
        '--checkpoint-dir',
        default=None,
        metavar='DIR',
        help='If set (and crossval is on), save a cropped copy of the '
             'reconstructed volume at every crossval eval checkpoint '
             '(every --eval-every iterations), in the same on-disk '
             'orientation/HU-calibration as the final saved volume. '
             'Written as DIR/iter{N:04d}.npy; DIR/crossval_metrics.json is '
             'also written once reconstruction finishes. Requires '
             '--checkpoint-z-range. Default: disabled.'
    )
    parser.add_argument(
        '--checkpoint-z-range',
        type=int,
        nargs=2,
        default=None,
        metavar=('Z0', 'Z1'),
        help='z-slice bounds [Z0, Z1) (on-disk VFF z-index convention) to '
             'slice out of each checkpoint volume before saving. Required '
             'if --checkpoint-dir is set.'
    )
    parser.add_argument(
        '--checkpoint-xy-range',
        type=int,
        nargs=4,
        default=None,
        metavar=('Y0', 'Y1', 'X0', 'X1'),
        help='In-plane crop bounds (on-disk VFF y/x convention) applied to '
             'each checkpoint volume before saving. Default: full xy plane.'
    )

    # TIGRE-specific arguments
    parser.add_argument(
        '--blocksize',
        type=int,
        default=15,
        help='Number of projections per OS-SART block (default: 15, TIGRE only). '
             'Smaller = more subsets = faster convergence but noisier per update.'
    )
    parser.add_argument(
        '--lmbda',
        type=float,
        default=1.0,
        help='Relaxation parameter lambda (default: 1.0, TIGRE only) — the '
             'step size in x <- x + lambda C A^T R (b - A x). SIRT converges '
             'for any lambda in (0, 2) because the iteration matrix C A^T R A '
             'has spectral radius <= 1, and 1.0 is the unrelaxed textbook '
             'choice (also TIGRE\'s own default). Lower values give a smoother '
             'trajectory and fewer early streaks in exchange for roughly '
             'proportionally more iterations. Consider lambda < 1 for the '
             'ORDERED-SUBSET methods (ossart, sart), whose per-block updates '
             'are noisy enough to orbit the solution at full relaxation; '
             'full-batch sirt has no such problem.'
    )
    parser.add_argument(
        '--lmbda-red',
        type=float,
        default=1.0,
        help='Relaxation reduction factor per iteration (default: 1.0 = NO '
             'decay, TIGRE only; also TIGRE\'s own default). Lambda decays as '
             'lmbda * lmbda_red^iter. Beware what a geometric decay does to a '
             'long run: at the old 0.97 default lambda is 0.05x its start by '
             'iteration 100 and 0.002x by 200, so the total update budget is '
             'sum(lambda_k) = lmbda/(1 - lmbda_red) and the iterate FREEZES '
             'wherever it happens to be rather than at the solution — extra '
             'iterations buy nothing. Semi-convergence is better handled by '
             'the held-out stopping rule, which detects the peak and restores '
             'that iterate, than by annealing the step so noise cannot be '
             'reached.'
    )
    parser.add_argument(
        '--tv-lambda',
        type=float,
        default=0.0,
        metavar='TV',
        help='TV regularization strength applied after each iteration chunk '
             '(default: 0, disabled). Acts on the normalised [0,1] image scale '
             'so the value is scan-independent. 10 is a good first guess for '
             'micro-CT: sharpens bone edges over plain SIRT without erasing fine '
             'trabecular detail. Useful range: 5–20. TIGRE only.'
    )
    parser.add_argument(
        '--tv-iters',
        type=int,
        default=50,
        metavar='N',
        help='Chambolle-Pock TV denoising iterations per application '
             '(default: 50, TIGRE only). Matches the TIGRE OSSART-TV default.'
    )
    parser.add_argument(
        '--pwls',
        action='store_true',
        default=False,
        help='Enable PWLS (penalized weighted least squares) data-fidelity '
             'weighting (default: off, i.e. plain geometric weighting). '
             'Down-weights rays with low estimated transmission (noisier '
             'measurements) instead of trusting every ray equally, '
             'approximating a maximum-likelihood weighting under a '
             'quantum-noise model. Composes with --tv-lambda and crossval. '
             'TIGRE only. Not supported with --algorithm mlem (MLEM already '
             'models per-ray photon statistics natively and would silently '
             'discard a PWLS weight array; this combination raises an error).'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set default algorithm based on backend
    if args.algorithm is None:
        args.algorithm = 'SIRT3D_CUDA' if args.backend == 'astra' else 'ossart'

    # Validate algorithm for chosen backend
    if args.backend == 'astra':
        if args.algorithm not in ASTRA_ALGORITHMS:
            raise ConfigError(
                f"algorithm '{args.algorithm}' is not supported by the ASTRA "
                f"backend. Supported: {ASTRA_ALGORITHMS}")
    elif args.backend == 'tigre':
        if args.algorithm not in SUPPORTED_TIGRE_ALGORITHMS:
            raise ConfigError(
                f"algorithm '{args.algorithm}' is not supported by the TIGRE "
                f"backend. Supported: {SUPPORTED_TIGRE_ALGORITHMS}")
        if args.algorithm == 'mlem' and args.pwls:
            raise ConfigError(
                "--pwls is not supported with --algorithm mlem. MLEM's "
                "constructor unconditionally overrides any custom W weight "
                "array with its own sensitivity map, so --pwls would be "
                "silently ignored rather than applied — and it is redundant "
                "anyway, since MLEM already models per-ray photon statistics "
                "natively through its Poisson likelihood.")

    if args.checkpoint_dir is not None and args.checkpoint_z_range is None:
        raise ConfigError(
            "--checkpoint-z-range is required when --checkpoint-dir is set.")

    start = time.time()

    print("=" * 60)
    print(f"Iterative Reconstruction Pipeline ({args.backend.upper()}: {args.algorithm})")
    print("=" * 60)
    print(f"Data folder: {args.data_folder}")
    print(f"Backend: {args.backend}")
    if args.algorithm != 'FDK_CUDA':
        # Every classical iteration sweeps the full sinogram exactly once, so
        # the iteration count IS the data-visit count — the unit the learned
        # backend reports (train/data_visits) and the only fair way to compare
        # the two families.
        print(f"Iterations: {args.iterations} "
              f"(= {float(args.iterations):.2f} visits per measurement; each "
              f"iteration uses every measurement exactly once)")
    if args.backend == 'astra':
        if args.min_constraint is not None:
            print(f"Min constraint: {args.min_constraint}")
        if args.max_constraint is not None:
            print(f"Max constraint: {args.max_constraint}")
    elif args.backend == 'tigre':
        if args.algorithm == 'mlem':
            print("MLEM: full-batch (no ordered subsets), no relaxation "
                  "parameter — --blocksize/--lmbda/--lmbda-red are ignored")
        else:
            print(f"Blocksize: {args.blocksize}")
            print(f"Lambda: {args.lmbda}, Lambda reduction: {args.lmbda_red}")
        if args.pwls:
            print(f"PWLS: enabled")

    # Shared front half: scan folder, projections, ROI, geometry, downsample.
    ctx = prepare_scan(args, fit_domain=True)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = f"_{args.algorithm.lower().replace('3d_cuda', '').replace('_cuda', '')}"
        if args.algorithm != 'FDK_CUDA':
            suffix += f"_{args.iterations}it"
        if getattr(args, 'pwls', False):
            suffix += "_pwls"
        output_path = ctx.default_output_path(f'_recon{suffix}')

    print(f"\nOutput path: {output_path}")

    # Experiment logging: local PNGs next to the output, plus W&B unless
    # --no-wandb (on by default; inert until a project is configured).
    # Created BEFORE preflight (an auto-aborted job is recorded as a FAILED
    # run with the verdict) and before the reconstructor (TIGRE streams its
    # crossval metrics live through log_fn while iterating).
    logger = ReconLogger(args, ctx, f'{args.backend}_{args.algorithm}',
                         output_path, params={
                             'iterations': args.iterations,
                             'blocksize': args.blocksize,
                             'lmbda': args.lmbda,
                             'lmbda_red': args.lmbda_red,
                             'tv_lambda': args.tv_lambda,
                             'pwls': bool(args.pwls),
                             'withhold_eval': bool(args.withhold_eval),
                         })

    # Machine fit check (GPU presence / VRAM / RAM) before any big allocation
    # — and before psi auto-measurement, which itself needs GPU minutes on a
    # cache miss. ASTRA/TIGRE are CUDA-only: no GPU is a hard abort here.
    if run_preflight(args.backend, ctx, gpu_index=args.gpu_index,
                     skip=args.skip_preflight, logger=logger,
                     only=args.preflight_only).dry_run:
        return                      # --preflight-only: the question is answered

    # ---- geometry auto-calibration (detector in-plane rotation psi) --------
    # Cached scan-keyed JSON first (written by any pipeline — muNeRF, FDK, or
    # this driver); on a miss, the half-scan-consistency estimator
    # (ct_core.geometry_selfcal, ported from muNeRF) measures psi from THIS
    # scan's projections and caches it. For TIGRE, absence additionally falls
    # back to its inline conjugate estimator (biased low ~-0.49 vs recon-true
    # ~-0.7 on Scan_1510, blind on symmetric objects, but reference-free).
    ext_psi = None
    if args.geometry_autocal:
        record = resolve_or_measure_detector_psi(
            ctx, fallback_note=("the inline conjugate estimator will run "
                                "instead (TIGRE) / psi=0 (ASTRA)"))
        if record is not None:
            ext_psi = float(record['psi_deg'])

    # ---- projection diagnostics: eval angle + noise ceiling ----------------
    # The evaluation projection is the central angle (the same one every
    # holdout scheme here has always used). The noise ceiling — the best
    # SSIM/PSNR any reconstruction can honestly reach against the noisy
    # measurement — comes from the other acquisition phase when the scan has
    # one, else the neighbouring projection.
    eval_idx = (args.holdout_index if args.holdout_index is not None
                else int(ctx.projections.shape[0]) // 2)
    logger.set_noise_ceiling(measure_noise_ceiling(
        ctx, eval_idx, phase=args.phase))

    # Initialize reconstructor based on backend
    if args.backend == 'astra':
        if ext_psi is not None:
            # applied inside geometry_to_astra_vectors (rotated u/v axes)
            ctx.geometry['det_psi_rad'] = float(np.radians(ext_psi))
        astra_proj, astra_angles = ctx.projections, ctx.angles
        if args.withhold_eval:
            print(f"  Withholding evaluation projection {eval_idx} from the "
                  f"reconstruction input (--withhold-eval).")
            astra_proj = np.delete(ctx.projections, eval_idx, axis=0)
            astra_angles = np.delete(ctx.angles, eval_idx, axis=0)
        reconstructor = ASTRAReconstructor(
            projections=astra_proj,
            angles=astra_angles,
            geometry=ctx.geometry,
            algorithm=args.algorithm,
            iterations=args.iterations,
            min_constraint=args.min_constraint,
            max_constraint=args.max_constraint,
            gpu_index=args.gpu_index,
            super_sampling=args.super_sampling,
            bright_field=ctx.bright_field,
            dark_field=ctx.dark_field,
            ring_correction=args.ring_correction,
            air_normalization=args.air_normalization,
            soft_clip_sharpness=args.soft_clip_sharpness,
            ring_median_width=args.ring_median_width,
            crossval=not args.no_crossval,
            holdout_index=args.holdout_index,
            eval_every=args.eval_every,
            patience=args.patience,
            min_stop_iter=args.min_stop_iter,
            stop_metric=args.stop_metric,
            l_curve=args.l_curve,
            l_curve_norm=args.l_curve_norm,
            stop_on=tuple(args.stop_on),
            log_fn=logger.log,
        )
    elif args.backend == 'tigre':
        reconstructor = TIGREReconstructor(
            detector_psi_deg=ext_psi,
            projections=ctx.projections,
            angles=ctx.angles,
            geometry=ctx.geometry,
            algorithm=args.algorithm,
            iterations=args.iterations,
            blocksize=args.blocksize,
            lmbda=args.lmbda,
            lmbda_red=args.lmbda_red,
            gpu_index=args.gpu_index,
            bright_field=ctx.bright_field,
            dark_field=ctx.dark_field,
            ring_correction=args.ring_correction,
            air_normalization=args.air_normalization,
            soft_clip_sharpness=args.soft_clip_sharpness,
            ring_median_width=args.ring_median_width,
            geometry_autocal=args.geometry_autocal,
            crossval=not args.no_crossval,
            holdout_index=args.holdout_index,
            withhold_eval=args.withhold_eval,
            eval_every=args.eval_every,
            patience=args.patience,
            min_stop_iter=args.min_stop_iter,
            stop_metric=args.stop_metric,
            l_curve=args.l_curve,
            l_curve_norm=args.l_curve_norm,
            stop_on=tuple(args.stop_on),
            tv_lambda=args.tv_lambda,
            tv_iters=args.tv_iters,
            pwls=args.pwls,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_z_range=(tuple(args.checkpoint_z_range)
                                 if args.checkpoint_z_range else None),
            checkpoint_xy_range=(tuple(args.checkpoint_xy_range)
                                  if args.checkpoint_xy_range else None),
            log_fn=logger.log,
            # diag/* scalars + SSIM-heatmap / power-spectrum figures at every
            # eval checkpoint, straight from TIGRE's own forward projection.
            diag_fn=logger.log_projection_diag,
        )

    # Run reconstruction
    reconstructor.reconstruct()

    # How much measured data the DELIVERED volume saw. Each classical
    # iteration sweeps the whole sinogram exactly once, so visits = the
    # iteration the saved volume comes from — which is best_iter, not the
    # iteration the run stopped at, when early stopping rolled back to the
    # peak-SSIM volume.
    n_b, n_a = int(ctx.projections.shape[1]), int(ctx.projections.shape[2])
    budget, note, extra = classical_budget(
        measurement_count(int(ctx.projections.shape[0]), n_b, n_a,
                          excluded_angles=1 if args.withhold_eval else 0),
        requested_iterations=args.iterations,
        crossval_metrics=getattr(reconstructor, 'crossval_metrics', None))
    logger.set_data_budget(budget, note=note, extra=extra)

    # Save convergence figure (no-op if crossval was off)
    if hasattr(reconstructor, 'plot_crossval'):
        reconstructor.plot_crossval(output_path)

    if getattr(reconstructor, 'crossval_metrics', None):
        # replay_steps=False: TIGRE already streamed these live via diag_fn;
        # this call only produces the local/uploaded convergence figure.
        logger.log_convergence(reconstructor.crossval_metrics,
                               replay_steps=False)
    else:
        # No per-iteration eval loop (ASTRA, or TIGRE with --no-crossval):
        # compute the projection diagnostics once, by forward-projecting the
        # FINAL volume at the evaluation angle through the canonical ray
        # tracer. Best-effort — a diagnostics failure never voids the recon.
        try:
            measured = preprocess_frames(
                ctx.projections[eval_idx:eval_idx + 1], ctx)[0]
            # The volume is mu now, so it goes to the forward model as-is
            # — no round trip through an assumed mu_water.
            # Air to zero first — see `level_to_air`. The anchors are
            # fitted here rather than reused because save_outputs runs AFTER
            # this block, and on the CROPPED volume; the level that belongs to
            # a forward projection is the one measured on the volume actually
            # being projected.
            mu_diag, _air = level_to_air(
                reconstructor.reconstructed_volume, verbose=True)
            pred, target = render_projection_from_volume(
                mu_diag, ctx, eval_idx, measured, volume_is_hu=False)
            logger.log_projection_diag(pred, target)
        except Exception as e:
            print(f"  Final projection diagnostics failed "
                  f"({type(e).__name__}: {e})")

    # Crop AFTER the diagnostics above: forward-projecting the volume has to
    # see the WHOLE domain — the matter outside the export ROI is precisely
    # what the domain exists to model, and dropping it would make the
    # predicted projection disagree with the measurement by construction.
    vol_export, ctx.geometry = crop_to_export_roi(
        reconstructor.reconstructed_volume, ctx.geometry)

    # Shared back half: HU calibration + bilateral filter + VFF export.
    _, _, volume_hu = save_outputs(vol_export, ctx, args, output_path,
                                   logger=logger,
                                   algorithm=f'{args.backend}_{args.algorithm}')

    logger.log_sinogram_preview(ctx.projections)
    logger.log_volume_summary(volume_hu, ctx)
    logger.log_recon_slices(volume_hu, geometry=ctx.geometry)
    logger.finish()

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    cli_main(main)
