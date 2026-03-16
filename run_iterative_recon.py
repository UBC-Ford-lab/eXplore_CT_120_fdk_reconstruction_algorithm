"""
Run iterative cone-beam CT reconstruction using ASTRA or TIGRE toolbox.

Supports multiple backends:
  - astra: SIRT, CGLS, SART, FDK via ASTRA toolbox
  - tigre: OS-SART, SART, SIRT via TIGRE (handles GPU memory splitting internally)

Uses the same data-loading and post-processing pipeline as the FDK script,
but replaces the reconstruction step with iterative algorithms.

Usage:
    python -m reconstruction.run_iterative_recon data/scans/Scan_1681
    python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --algorithm CGLS3D_CUDA --iterations 50
    python -m reconstruction.run_iterative_recon data/scans/Scan_1681 --backend tigre --algorithm ossart --iterations 100
"""

import argparse
import os
import sys
import time

import numpy as np

from .astra_iterative import ASTRAReconstructor, SUPPORTED_ALGORITHMS as ASTRA_ALGORITHMS
from .tigre_iterative import TIGREReconstructor, SUPPORTED_TIGRE_ALGORITHMS
from .ct_core.calibration import MU_WATER_80KV
from .ct_core.scan_setup import (
    auto_detect_scan_folder,
    load_scan_data,
    build_geometry,
    parse_crop_boundary,
    postprocess_and_save,
)


def parse_args():
    """Parse command-line arguments."""
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

    # Shared arguments (same as FDK)
    parser.add_argument(
        'data_folder',
        help='Path to folder containing projections and scan.xml'
    )
    parser.add_argument(
        '--scan-folder',
        help='Path to original scan folder with bright.vff/dark.vff (auto-detected if not specified)'
    )
    parser.add_argument(
        '--output',
        help='Output VFF filename (auto-generated from data_folder if not specified)'
    )
    parser.add_argument(
        '--total-angle',
        default='determined',
        help='Total angular coverage in degrees. Default: "determined" (reads IncrementAngle '
             'and ViewCount from scan.xml to compute total angle automatically). '
             'Specify a numeric value to override (e.g., --total-angle 360.0).'
    )
    parser.add_argument(
        '--projection-pattern',
        default=None,
        help='Glob pattern for projection files (default: auto-detect proj-* or acq*)'
    )
    parser.add_argument(
        '--voxel-xy',
        type=float,
        default=0.075,
        help='Reconstruction voxel size in the xy plane in mm (default: 0.075)'
    )
    parser.add_argument(
        '--voxel-z',
        type=float,
        default=0.075,
        help='Reconstruction voxel size in the z plane in mm (default: 0.075)'
    )
    parser.add_argument(
        '--fov-xy',
        type=float,
        default=45,
        help='Field of view in the xy plane in mm (default: 45)'
    )
    parser.add_argument(
        '--fov-z',
        type=float,
        default=120.0,
        help='Field of view in the z direction in mm (default: 120.0)'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Save reconstruction slice PNGs after completion'
    )
    parser.add_argument(
        '--bilateral-filter',
        action='store_true',
        help='Apply bilateral filter to calibrated volume (edge-preserving denoising)'
    )
    parser.add_argument(
        '--bilateral-sigma-spatial',
        type=float,
        default=1.5,
        help='Bilateral filter spatial sigma in mm (default: 1.5)'
    )
    parser.add_argument(
        '--bilateral-sigma-range',
        type=float,
        default=50.0,
        help='Bilateral filter intensity sigma in HU (default: 50.0)'
    )

    # Beam hardening correction
    parser.add_argument(
        '--bhc-coeffs',
        type=float,
        nargs='+',
        default=None,
        help='Sinogram-domain BHC polynomial coefficients (e.g., --bhc-coeffs 0.856 0.21). '
             'Applied after log transform: p_corrected = c1*p + c2*p^2 + ...'
    )
    parser.add_argument(
        '--ring-correction',
        action='store_true',
        default=True,
        help='Enable sinogram-space ring artifact correction (default: on)'
    )
    parser.add_argument(
        '--no-ring-correction',
        dest='ring_correction',
        action='store_false',
        help='Disable ring artifact correction'
    )
    parser.add_argument(
        '--ring-median-width',
        type=int,
        default=51,
        help='Median filter width for ring correction (odd int, default: 51)'
    )

    # Backend selection
    parser.add_argument(
        '--backend',
        default='astra',
        choices=('astra', 'tigre'),
        help='Reconstruction backend (default: astra)'
    )

    # Algorithm (choices depend on backend, validated in main())
    parser.add_argument(
        '--algorithm',
        default=None,
        help='Reconstruction algorithm. '
             'ASTRA: SIRT3D_CUDA, CGLS3D_CUDA, SART3D_CUDA, FDK_CUDA (default: SIRT3D_CUDA). '
             'TIGRE: ossart, sart, sirt (default: ossart).'
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
        '--downsample',
        type=int,
        default=1,
        help='Downsample projections by this factor before reconstruction (default: 1). '
             'Reduces GPU memory usage. Factor 2 halves each detector dimension.'
    )

    # Calibration arguments
    parser.add_argument(
        '--roi-config',
        default=None,
        help='JSON file with phantom insert ROI definitions for self-calibration. '
             'When provided (with --cal-z-range), the pipeline measures inserts in '
             'this volume and fits a per-method polynomial instead of using the '
             'hardcoded FDK calibration. Recommended for iterative methods.'
    )
    parser.add_argument(
        '--cal-z-range',
        type=int,
        nargs=2,
        metavar=('Z_START', 'Z_END'),
        default=None,
        help='Z-slice range for phantom insert measurements (required with --roi-config)'
    )
    parser.add_argument(
        '--cal-degree',
        type=int,
        default=2,
        help='Polynomial degree for self-calibration fit (default: 2)'
    )
    parser.add_argument(
        '--calibration-method',
        default='two_point',
        help='HU calibration method (default: "two_point"). '
             'Measures air/water from the volume and applies the standard '
             'CT HU formula. Self-calibrating, works with any config.'
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

    # ROI-based reconstruction
    parser.add_argument(
        '--roi',
        nargs='+',
        default=None,
        help='ROI-based reconstruction. Use "auto" to load from '
             'SubVolumeCoordinates.xml in the scan folder, or specify 6 values: '
             'x_min x_max y_min y_max z_min z_max (mm, isocenter-centered). '
             'When active, --fov-xy and --fov-z are ignored.'
    )

    return parser.parse_args()


def downsample_projections(projections, factor):
    """
    Downsample projections by averaging adjacent pixels.

    Args:
        projections: np.ndarray of shape (N_angles, N_b, N_a)
        factor: integer downsampling factor

    Returns:
        Downsampled projections
    """
    if factor <= 1:
        return projections

    N_angles, N_b, N_a = projections.shape

    # Trim to be divisible by factor
    N_b_new = (N_b // factor) * factor
    N_a_new = (N_a // factor) * factor
    trimmed = np.array(projections[:, :N_b_new, :N_a_new], dtype=np.float32)

    # Reshape and average
    downsampled = trimmed.reshape(
        N_angles, N_b_new // factor, factor, N_a_new // factor, factor
    ).mean(axis=(2, 4))

    return downsampled


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

    start = time.time()

    data_folder = args.data_folder

    print("=" * 60)
    print(f"Iterative Reconstruction Pipeline ({args.backend.upper()}: {args.algorithm})")
    print("=" * 60)
    print(f"Data folder: {data_folder}")
    print(f"Backend: {args.backend}")
    if args.algorithm != 'FDK_CUDA':
        print(f"Iterations: {args.iterations}")
    if args.backend == 'astra':
        if args.min_constraint is not None:
            print(f"Min constraint: {args.min_constraint}")
        if args.max_constraint is not None:
            print(f"Max constraint: {args.max_constraint}")
    elif args.backend == 'tigre':
        print(f"Blocksize: {args.blocksize}")
        print(f"Lambda: {args.lmbda}, Lambda reduction: {args.lmbda_red}")

    # Resolve scan folder
    if args.scan_folder:
        scan_folder = args.scan_folder
    else:
        try:
            scan_folder = auto_detect_scan_folder(data_folder)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Load scan data using shared utility
    scan_data = load_scan_data(
        data_folder, scan_folder,
        args.projection_pattern, args.total_angle,
    )
    projections = scan_data['projections']
    angles = scan_data['angles']
    bright_field = scan_data['bright_field']
    dark_field = scan_data['dark_field']

    # Parse ROI bounds if requested
    roi_bounds = None
    if args.roi is not None:
        if args.roi == ['auto']:
            roi_bounds = parse_crop_boundary(scan_folder, scan_data['xml_header'])
            if roi_bounds is None:
                print("Error: SubVolumeCoordinates.xml not found or invalid in scan folder.")
                print("  Looked in: " + os.path.join(scan_folder, 'Volumes', 'SubVolumeCoordinates.xml'))
                sys.exit(1)
        elif len(args.roi) == 6:
            vals = [float(v) for v in args.roi]
            roi_bounds = {
                'x_min': vals[0], 'x_max': vals[1],
                'y_min': vals[2], 'y_max': vals[3],
                'z_min': vals[4], 'z_max': vals[5],
            }
        else:
            print("Error: --roi requires 'auto' or exactly 6 values "
                  "(x_min x_max y_min y_max z_min z_max)")
            sys.exit(1)

    # Build geometry using shared utility
    geometry = build_geometry(
        scan_data['xml_header'],
        args.fov_xy, args.fov_z, args.voxel_xy, args.voxel_z,
        roi_bounds=roi_bounds,
    )

    # Downsample if requested
    if args.downsample > 1:
        print(f"\nDownsampling projections by factor {args.downsample}...")
        original_shape = projections.shape
        projections = downsample_projections(projections, args.downsample)
        print(f"  {original_shape} -> {projections.shape}")

        # Also downsample bright/dark fields
        N_b_new = projections.shape[1]
        N_a_new = projections.shape[2]
        factor = args.downsample
        N_b_trim = N_b_new * factor
        N_a_trim = N_a_new * factor
        bright_field = bright_field[:N_b_trim, :N_a_trim].reshape(
            N_b_new, factor, N_a_new, factor
        ).mean(axis=(1, 3))
        dark_field = dark_field[:N_b_trim, :N_a_trim].reshape(
            N_b_new, factor, N_a_new, factor
        ).mean(axis=(1, 3))

        # Update geometry: detector pixel size scales with downsample factor
        geometry['da'] *= args.downsample
        geometry['db'] *= args.downsample

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = f"_{args.algorithm.lower().replace('3d_cuda', '').replace('_cuda', '')}"
        if args.algorithm != 'FDK_CUDA':
            suffix += f"_{args.iterations}it"
        output_path = data_folder.rstrip('/') + f'_recon{suffix}'

    print(f"\nOutput path: {output_path}")

    # Convert angles to numpy
    angles_np = angles.numpy() if hasattr(angles, 'numpy') else np.asarray(angles)

    # Initialize reconstructor based on backend
    if args.backend == 'astra':
        reconstructor = ASTRAReconstructor(
            projections=projections,
            angles=angles_np,
            geometry=geometry,
            algorithm=args.algorithm,
            iterations=args.iterations,
            min_constraint=args.min_constraint,
            max_constraint=args.max_constraint,
            gpu_index=args.gpu_index,
            super_sampling=args.super_sampling,
            bright_field=bright_field,
            dark_field=dark_field,
            mu_water=MU_WATER_80KV,
            output_hu=True,
            bhc_coeffs=args.bhc_coeffs,
            ring_correction=args.ring_correction,
            ring_median_width=args.ring_median_width,
        )
    elif args.backend == 'tigre':
        reconstructor = TIGREReconstructor(
            projections=projections,
            angles=angles_np,
            geometry=geometry,
            algorithm=args.algorithm,
            iterations=args.iterations,
            blocksize=args.blocksize,
            lmbda=args.lmbda,
            lmbda_red=args.lmbda_red,
            gpu_index=args.gpu_index,
            bright_field=bright_field,
            dark_field=dark_field,
            mu_water=MU_WATER_80KV,
            output_hu=True,
            bhc_coeffs=args.bhc_coeffs,
            ring_correction=args.ring_correction,
            ring_median_width=args.ring_median_width,
        )

    # Run reconstruction
    reconstructor.reconstruct()

    # Post-process and save using shared utility
    cal_plot = output_path + '_calibration_diagnostic' if args.roi_config else None
    postprocess_and_save(
        volume=reconstructor.reconstructed_volume,
        geometry=geometry,
        output_path=output_path,
        bilateral_filter=args.bilateral_filter,
        bilateral_sigma_spatial=args.bilateral_sigma_spatial,
        bilateral_sigma_range=args.bilateral_sigma_range,
        voxel_xy=args.voxel_xy,
        roi_config=args.roi_config,
        cal_z_range=tuple(args.cal_z_range) if args.cal_z_range else None,
        cal_degree=args.cal_degree,
        cal_plot_path=cal_plot,
        calibration_method=args.calibration_method,
    )

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
