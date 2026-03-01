"""
Run FDK reconstruction on VFF projections with HU calibration.

Uses verified reconstruction settings: HU output, physical normalization,
soft-clip transmission, upper clamping, and polynomial calibration.

Defaults to Hamming window at matched cutoff (da/dx), which gives the
best noise-resolution trade-off per the filter kernel sweep verification.

Usage:
    python reconstruction/run_recon_on_vff_file.py data/scans/Scan_1681
    python reconstruction/run_recon_on_vff_file.py data/scans/Scan_1681 --filter-type ramp --filter-cutoff 1.0
    python reconstruction/run_recon_on_vff_file.py data/scans/Scan_1681 --filter-type cosine --filter-cutoff 0.5
"""

import argparse
import re
import sys
import time
import os
from pathlib import Path

import cv2
import torch
import numpy as np
import xmltodict

from .fdk import FDKReconstructor, SUPPORTED_FILTER_TYPES
from .ct_core import tiff_converter
from .ct_core.vff_io import VFFDataset, write_vff
from .ct_core.calibration import (
    load_calibration_fields, MU_WATER_80KV,
    PHANTOM_CALIBRATION, fit_hu_calibration,
)


def auto_detect_scan_folder(data_folder: str) -> str:
    """
    Auto-detect the original scan folder from a results folder path.

    Naming convention:
        Results: data/results/Scan_XXXX_<suffix>/
        Scans:   data/scans/Scan_XXXX/

    Args:
        data_folder: Path to results folder (e.g., 'data/results/Scan_1681_with_pred')

    Returns:
        Path to scan folder (e.g., 'data/scans/Scan_1681')

    Raises:
        ValueError: If scan name cannot be extracted from folder name
    """
    folder_name = os.path.basename(data_folder.rstrip('/'))
    match = re.search(r'(Scan_\d+)', folder_name)

    if not match:
        raise ValueError(
            f"Could not extract scan name from '{folder_name}'. "
            f"Expected format: 'Scan_XXXX_<suffix>'. "
            f"Please specify --scan-folder explicitly."
        )

    scan_name = match.group(1)  # e.g., "Scan_1681"

    # Construct scan folder path relative to data_folder's parent structure
    # Assume standard structure: data/results/... and data/scans/...
    data_folder_path = Path(data_folder).resolve()

    # Try to find 'data' directory in the path
    for parent in data_folder_path.parents:
        potential_scan_folder = parent / 'scans' / scan_name
        if potential_scan_folder.exists():
            return str(potential_scan_folder)

    # Fallback: assume standard relative structure
    scan_folder = f"data/scans/{scan_name}"
    return scan_folder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run FDK reconstruction on VFF projections with verified HU pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reconstruction/run_recon_on_vff_file.py data/scans/Scan_1681
  python reconstruction/run_recon_on_vff_file.py data/scans/Scan_1681 --filter-type ramp --filter-cutoff 1.0
  python reconstruction/run_recon_on_vff_file.py data/scans/Scan_1681 --filter-type cosine --filter-cutoff 0.5
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

    return parser.parse_args()


def main():
    args = parse_args()

    start = time.time()

    data_folder = args.data_folder
    xml_file = os.path.join(data_folder, 'scan.xml')

    if not os.path.exists(xml_file):
        print(f"Error: scan.xml not found at {xml_file}")
        sys.exit(1)

    print("=" * 60)
    print("FDK Reconstruction Pipeline")
    print("=" * 60)
    print(f"Data folder: {data_folder}")
    print(f"HU mode: enabled (physical normalization, polynomial calibration)")

    # Load bright/dark fields (required for HU pipeline)
    if args.scan_folder:
        scan_folder = args.scan_folder
    else:
        try:
            scan_folder = auto_detect_scan_folder(data_folder)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"Scan folder: {scan_folder}")

    try:
        bright_field, dark_field = load_calibration_fields(scan_folder)
        print(f"Loaded bright field: shape {bright_field.shape}, mean {bright_field.mean():.0f}")
        print(f"Loaded dark field: shape {dark_field.shape}, mean {dark_field.mean():.0f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("HU calibration requires bright.vff and dark.vff in the scan folder.")
        sys.exit(1)

    # Auto-detect projection pattern if not specified
    if args.projection_pattern is None:
        if list(Path(data_folder).glob('proj-*.vff')):
            args.projection_pattern = 'proj-*.vff'
        else:
            args.projection_pattern = 'acq*'

    # Parse XML early (needed for total angle determination and geometry)
    header = xmltodict.parse(open(xml_file).read())

    # Count projection files
    print(f"\nLoading projections (pattern: {args.projection_pattern})...")
    proj_paths = sorted(Path(data_folder).glob(args.projection_pattern))
    # Apply phase filter only to acquisition files (not sequential proj-* files)
    proj_paths = [p for p in proj_paths if p.name.startswith('proj-') or '-00-' in str(p)]
    n_files = len(proj_paths)
    if n_files == 0:
        print(f"Error: No projections matching '{args.projection_pattern}' in {data_folder}")
        sys.exit(1)

    # Determine total angle
    if args.total_angle == 'determined':
        try:
            increment_angle = float(header['Series']['SeriesParams']['IncrementAngle'])
            xml_view_count = int(header['Series']['SeriesParams']['ViewCount'])
        except (KeyError, TypeError) as e:
            print(f"Error: Could not read IncrementAngle/ViewCount from scan.xml: {e}")
            print("Please specify --total-angle explicitly.")
            sys.exit(1)

        total_angle = increment_angle * n_files
        print(f"  Determined from scan.xml: IncrementAngle = {increment_angle:.6f} deg, "
              f"ViewCount = {xml_view_count}")
        if xml_view_count != n_files:
            print(f"  WARNING: ViewCount in scan.xml ({xml_view_count}) does not match "
                  f"number of projection files found ({n_files})")
        print(f"  Total angle = {increment_angle:.6f} x {n_files} projections = {total_angle:.4f} deg")
    else:
        total_angle = float(args.total_angle)
        print(f"  User-specified total angle: {total_angle:.4f} deg")

    projection_spacing = total_angle / n_files
    print(f"  {n_files} projections, {total_angle:.4f} deg total -> spacing = {projection_spacing:.6f} deg")

    dataset = VFFDataset(
        data_folder,
        xml_file,
        paths_str=args.projection_pattern,
        projection_spacing=projection_spacing,
    )
    projections = dataset.projections  # shape (N_angles, N_b, N_a)
    angles = dataset.angles_rad        # shape (N_angles,)

    print(f"Loaded {len(angles)} projections, shape: {projections.shape}")

    # Define geometry from XML
    source_to_isocenter = float(header['Series']['ObjectPosition'])
    detector_to_isocenter = float(header['Series']['DetectorPosition']) - source_to_isocenter

    # Compute volume shape from FOV and voxel size
    Nxy = round(args.fov_xy / args.voxel_xy)
    Nz = round(args.fov_z / args.voxel_z)

    geometry = {
        'R_d': detector_to_isocenter,
        'R_s': source_to_isocenter,
        'da': float(header['Series']['DetectorSpacing']),
        'db': float(header['Series']['DetectorSpacing']),
        'vol_shape': (Nxy, Nxy, Nz),
        'vol_origin': (0, 0, 0),
        'dx': args.voxel_xy,
        'dz': args.voxel_z,
        'central_pixel_a': float(header['Series']['CentreOfRotation']),
        'central_pixel_b': float(header['Series']['CentralSlice'])
    }

    print(f"\nGeometry:")
    print(f"  Source-to-isocenter: {geometry['R_s']:.2f} mm")
    print(f"  Detector-to-isocenter: {geometry['R_d']:.2f} mm")
    print(f"  Detector pixel size: {geometry['da']:.4f} mm")
    print(f"  Volume shape: {geometry['vol_shape']}")
    print(f"  Voxel size (xy): {geometry['dx']:.4f} mm")
    print(f"  Voxel size (z): {geometry['dz']:.4f} mm")

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
        parker_weighting=True,
        metal_artifact_reduction=args.metal_artifact_reduction,
        mar_threshold=args.mar_threshold,
    )

    # Run full reconstruction
    reconstructor.reconstruct(display_volume=args.display)

    # Apply polynomial HU calibration
    print("\n" + "=" * 60)
    print("Applying polynomial HU calibration (phantom insert data)")
    print("=" * 60)

    coeffs, rms, residuals = fit_hu_calibration(PHANTOM_CALIBRATION, degree=2)
    print(f"  Degree-2 polynomial coefficients: {coeffs}")
    print(f"  RMS residual: {rms:.1f} HU")

    # Extract volume as numpy (x, y, z)
    vol = reconstructor.reconstructed_volume
    vol_np = vol.cpu().numpy() if hasattr(vol, 'cpu') else vol

    # Apply calibration
    vol_calibrated = np.polyval(coeffs, vol_np).astype(np.float32)
    vol_calibrated = np.clip(vol_calibrated, -1024, 4095)

    # Optional bilateral filter (edge-preserving denoising)
    if args.bilateral_filter:
        print("\n" + "=" * 60)
        print("Applying bilateral filter (edge-preserving denoising)")
        print("=" * 60)

        sigma_spatial_vox = args.bilateral_sigma_spatial / args.voxel_xy
        print(f"  Spatial sigma: {args.bilateral_sigma_spatial:.2f} mm "
              f"= {sigma_spatial_vox:.1f} voxels")
        print(f"  Range sigma: {args.bilateral_sigma_range:.1f} HU")

        t_bf = time.time()
        Nz_slices = vol_calibrated.shape[2]
        for z in range(Nz_slices):
            vol_calibrated[:, :, z] = cv2.bilateralFilter(
                vol_calibrated[:, :, z],
                d=-1,
                sigmaColor=args.bilateral_sigma_range,
                sigmaSpace=sigma_spatial_vox,
            )
        t_bf_end = time.time()

        print(f"  Filtered range: [{vol_calibrated.min():.0f}, {vol_calibrated.max():.0f}] HU")
        print(f"  Bilateral filter applied in {t_bf_end - t_bf:.1f}s "
              f"({Nz_slices} slices)")

    # Save calibrated VFF using write_vff with transpose/y-flip (matching fdk.py:688)
    cal_path = output_path + ('_bilateral.vff' if args.bilateral_filter else '.vff')
    vol_vff = vol_calibrated.astype(np.int16).transpose(2, 1, 0)[:, ::-1, :]
    write_vff(cal_path, {
        'bits': 16,
        'spacing': f"{geometry['dx']} {geometry['dx']} {geometry['dz']}",
    }, vol_vff)
    print(f"Calibrated VFF saved to: {cal_path}")

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
