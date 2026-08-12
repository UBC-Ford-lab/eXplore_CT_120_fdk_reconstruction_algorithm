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
import sys
import time

import numpy as np

from .iterative.astra import ASTRAReconstructor, SUPPORTED_ALGORITHMS as ASTRA_ALGORITHMS
from .iterative.tigre import TIGREReconstructor, SUPPORTED_TIGRE_ALGORITHMS
from .ct_core.pipeline import (
    ReconLogger,
    add_common_args,
    prepare_scan,
    resolve_or_measure_detector_psi,
    save_outputs,
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
    parser.add_argument(
        '--calibration-method',
        default='two_point',
        help='HU calibration method (default: "two_point"). '
             'Measures air/water from the volume and applies the standard '
             'CT HU formula. Self-calibrating, works with any config.'
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
        default=3,
        metavar='P',
        help='Early stopping patience: stop if SSIM does not improve for P '
             'consecutive eval checkpoints (default: 3, i.e. 30 iters at '
             'eval-every=10). Use a large value to disable early stopping '
             'while keeping metric logging.'
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
        default=0.5,
        help='Relaxation parameter (default: 0.5, TIGRE only). '
             'Lower values give smoother convergence and fewer streak artifacts.'
    )
    parser.add_argument(
        '--lmbda-red',
        type=float,
        default=0.97,
        help='Relaxation reduction factor per iteration (default: 0.97, TIGRE only). '
             'Lambda decays as lmbda * lmbda_red^iter, annealing toward zero.'
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
            print(f"Error: Algorithm '{args.algorithm}' not supported for ASTRA backend. "
                  f"Supported: {ASTRA_ALGORITHMS}")
            sys.exit(1)
    elif args.backend == 'tigre':
        if args.algorithm not in SUPPORTED_TIGRE_ALGORITHMS:
            print(f"Error: Algorithm '{args.algorithm}' not supported for TIGRE backend. "
                  f"Supported: {SUPPORTED_TIGRE_ALGORITHMS}")
            sys.exit(1)
        if args.algorithm == 'mlem' and args.pwls:
            print("Error: --pwls is not supported with --algorithm mlem. "
                  "MLEM's constructor unconditionally overrides any custom W "
                  "weight array with its own sensitivity map, so --pwls would "
                  "be silently ignored rather than applied — and it's "
                  "redundant anyway since MLEM already models per-ray photon "
                  "statistics natively through its Poisson likelihood.")
            sys.exit(1)

    if args.checkpoint_dir is not None and args.checkpoint_z_range is None:
        print("Error: --checkpoint-z-range is required when --checkpoint-dir is set.")
        sys.exit(1)

    start = time.time()

    print("=" * 60)
    print(f"Iterative Reconstruction Pipeline ({args.backend.upper()}: {args.algorithm})")
    print("=" * 60)
    print(f"Data folder: {args.data_folder}")
    print(f"Backend: {args.backend}")
    if args.algorithm != 'FDK_CUDA':
        print(f"Iterations: {args.iterations}")
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
    ctx = prepare_scan(args)

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

    # Initialize reconstructor based on backend
    if args.backend == 'astra':
        if not args.no_crossval:
            print("WARNING: cross-validation is only supported with --backend tigre. "
                  "Running ASTRA without holdout eval.")
        if ext_psi is not None:
            # applied inside geometry_to_astra_vectors (rotated u/v axes)
            ctx.geometry['det_psi_rad'] = float(np.radians(ext_psi))
        reconstructor = ASTRAReconstructor(
            projections=ctx.projections,
            angles=ctx.angles,
            geometry=ctx.geometry,
            algorithm=args.algorithm,
            iterations=args.iterations,
            min_constraint=args.min_constraint,
            max_constraint=args.max_constraint,
            gpu_index=args.gpu_index,
            super_sampling=args.super_sampling,
            bright_field=ctx.bright_field,
            dark_field=ctx.dark_field,
            output_hu=True,
            bhc_coeffs=args.bhc_coeffs,
            ring_correction=args.ring_correction,
            ring_median_width=args.ring_median_width,
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
            output_hu=True,
            bhc_coeffs=args.bhc_coeffs,
            ring_correction=args.ring_correction,
            ring_median_width=args.ring_median_width,
            geometry_autocal=args.geometry_autocal,
            crossval=not args.no_crossval,
            holdout_index=args.holdout_index,
            eval_every=args.eval_every,
            patience=args.patience,
            tv_lambda=args.tv_lambda,
            tv_iters=args.tv_iters,
            pwls=args.pwls,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_z_range=(tuple(args.checkpoint_z_range)
                                 if args.checkpoint_z_range else None),
            checkpoint_xy_range=(tuple(args.checkpoint_xy_range)
                                  if args.checkpoint_xy_range else None),
        )

    # Experiment logging: local PNGs next to the output, W&B when --wandb.
    logger = ReconLogger(args, ctx, f'{args.backend}_{args.algorithm}',
                         output_path, params={
                             'iterations': args.iterations,
                             'blocksize': args.blocksize,
                             'lmbda': args.lmbda,
                             'lmbda_red': args.lmbda_red,
                             'tv_lambda': args.tv_lambda,
                             'pwls': bool(args.pwls),
                         })

    # Run reconstruction
    reconstructor.reconstruct()

    # Save convergence figure (no-op if crossval was off)
    if hasattr(reconstructor, 'plot_crossval'):
        reconstructor.plot_crossval(output_path)

    # Shared back half: HU calibration + bilateral filter + VFF export.
    save_outputs(reconstructor.reconstructed_volume, ctx, args, output_path)

    if getattr(reconstructor, 'crossval_metrics', None):
        logger.log_convergence(reconstructor.crossval_metrics)
    logger.log_sinogram_preview(ctx.projections)
    logger.log_volume_summary(reconstructor.reconstructed_volume, ctx)
    logger.finish()

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
