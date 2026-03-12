"""
Run FDK reconstruction on VFF projections with HU calibration.

Uses verified reconstruction settings: HU output, physical normalization,
soft-clip transmission, upper clamping, and polynomial calibration.

Defaults to Hamming window at matched cutoff (da/dx), which gives the
best noise-resolution trade-off per the filter kernel sweep verification.

Usage:
    python -m reconstruction.run_fdk_recon data/scans/Scan_1681
    python -m reconstruction.run_fdk_recon data/scans/Scan_1681 --filter-type ramp --filter-cutoff 1.0
    python -m reconstruction.run_fdk_recon data/scans/Scan_1681 --filter-type cosine --filter-cutoff 0.5
"""

import argparse
import os
import sys
import time

from .fdk import FDKReconstructor, SUPPORTED_FILTER_TYPES
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
        description='Run FDK reconstruction on VFF projections with verified HU pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m reconstruction.run_fdk_recon data/scans/Scan_1681
  python -m reconstruction.run_fdk_recon data/scans/Scan_1681 --filter-type ramp --filter-cutoff 1.0
  python -m reconstruction.run_fdk_recon data/scans/Scan_1681 --filter-type cosine --filter-cutoff 0.5
        """
    )

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
        '--filter-cutoff',
        default='match',
        help='Ramp filter bandwidth as fraction of Nyquist (0.0-1.0, default: match). '
             'Lower values reduce noise at the cost of spatial resolution. '
             'Use "match" to auto-compute da/dx (detector pixel / recon voxel), '
             'which limits the filter to frequencies the reconstruction grid can represent.'
    )
    parser.add_argument(
        '--filter-type',
        default='hamming',
        choices=SUPPORTED_FILTER_TYPES,
        help='Ramp filter window type (default: hamming). '
             'Hamming provides the best noise suppression at matched cutoff.'
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
        help='Field of view in the xy plane in mm (default: 45 (for most mice scans), select 94 for most phantom scanner studies)'
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
        help='Bilateral filter spatial sigma in mm (default: 1.5). '
             'Converted to voxels using --voxel-xy.'
    )
    parser.add_argument(
        '--bilateral-sigma-range',
        type=float,
        default=50.0,
        help='Bilateral filter intensity sigma in HU (default: 50.0). '
             'Controls edge-preservation threshold.'
    )
    parser.add_argument(
        '--metal-artifact-reduction',
        action='store_true',
        help='Enable sinogram-domain metal artifact reduction (LI-MAR). '
             'Detects and interpolates metal-corrupted pixels before '
             'cone-beam weighting and ramp filtering.'
    )
    parser.add_argument(
        '--mar-threshold',
        type=float,
        default=6.0,
        help='Line integral threshold for metal pixel detection (default: 6.0). '
             'Pixels with -log(T) > threshold are treated as metal-corrupted. '
             'Lower = more aggressive (4.0), higher = more conservative (8.0).'
    )
    # Calibration arguments
    parser.add_argument(
        '--roi-config',
        default=None,
        help='JSON file with phantom insert ROI definitions for self-calibration. '
             'When provided (with --cal-z-range), the pipeline measures inserts in '
             'this volume and fits a per-method polynomial instead of using the '
             'hardcoded FDK calibration.'
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
        default='fdk',
        help='Method key for stored calibration coefficients (default: "fdk"). '
             'Used for non-phantom scans to apply the correct polynomial.'
    )
    parser.add_argument(
        '--scan-type',
        choices=['half_scan', 'full_scan'],
        default=None,
        help='Scan type for selecting stored calibration coefficients. '
             'Half-scan and full-scan acquisitions have different uncalibrated '
             'value ranges. Default: half_scan.'
    )

    parser.add_argument(
        '--ring-correction',
        action='store_true',
        default=True,
        dest='ring_correction',
        help='Enable sinogram-space ring artifact correction (default: enabled). '
             'Removes fixed-pattern detector column offsets that cause '
             'concentric ring artifacts in the reconstruction.'
    )
    parser.add_argument(
        '--no-ring-correction',
        action='store_false',
        dest='ring_correction',
        help='Disable ring artifact correction.'
    )
    parser.add_argument(
        '--ring-median-width',
        type=int,
        default=51,
        help='Median filter width for ring correction (default: 51, must be odd). '
             'Controls the scale of features removed. Larger = more aggressive.'
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


def main():
    args = parse_args()

    start = time.time()

    data_folder = args.data_folder

    print("=" * 60)
    print("FDK Reconstruction Pipeline")
    print("=" * 60)
    print(f"Data folder: {data_folder}")
    print(f"HU mode: enabled (physical normalization, polynomial calibration)")

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

    # Resolve filter cutoff (may depend on geometry)
    if args.filter_cutoff.lower() == 'match':
        filter_cutoff = geometry['da'] / geometry['dx']
        print(f"\n  Filter cutoff 'match': da/dx = {geometry['da']:.4f}/{geometry['dx']:.4f} = {filter_cutoff:.4f}")
    else:
        filter_cutoff = float(args.filter_cutoff)
    if not 0.0 < filter_cutoff <= 1.0:
        print(f"Error: filter-cutoff must be in (0.0, 1.0], got {filter_cutoff:.4f}")
        sys.exit(1)
    print(f"  Filter type: {args.filter_type}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate from data_folder: data/scans/Scan_1681 -> data/scans/Scan_1681_recon
        output_path = data_folder.rstrip('/') + '_recon'

    print(f"\nOutput path: {output_path}")

    # Initialize reconstructor with verified settings
    reconstructor = FDKReconstructor(
        projections=projections,
        angles=angles,
        geometry=geometry,
        source_locations=None,
        folder_name=output_path,
        output_hu=True,
        bright_field=bright_field,
        dark_field=dark_field,
        mu_water=MU_WATER_80KV,
        clamp_mode="none",
        soft_clip_transmission=True,
        soft_clip_sharpness=50.0,
        upper_clamp=True,
        upper_clamp_value=1.05,
        physical_normalization=True,
        filter_cutoff=filter_cutoff,
        filter_type=args.filter_type,
        parker_weighting=False,
        metal_artifact_reduction=args.metal_artifact_reduction,
        mar_threshold=args.mar_threshold,
        ring_correction=args.ring_correction,
        ring_median_width=args.ring_median_width,
    )

    # Run full reconstruction
    reconstructor.reconstruct(display_volume=args.display)

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
        scan_type=args.scan_type,
    )

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
