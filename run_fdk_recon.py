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

from pathlib import Path
import numpy as np

from .fdk import FDKReconstructor, SUPPORTED_FILTER_TYPES
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
        '--phase',
        default='00',
        help='Acquisition phase to reconstruct for multi-phase (gated) scans, '
             'e.g. 00 or 01. Selects projection files whose name contains '
             '"-<phase>-" (default: 00). Ignored for sequential proj-* scans.'
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
        '--cor',
        type=float,
        default=None,
        help='Override center-of-rotation detector pixel (central_pixel_a). '
             'Default: CentreOfRotation from scan.xml. Use to recalibrate COR '
             '(e.g. when an off-isocenter object reconstructs non-round).'
    )
    parser.add_argument(
        '--central-slice',
        type=float,
        default=None,
        help='Override central detector row (central_pixel_b). '
             'Default: CentralSlice from scan.xml.'
    )
    parser.add_argument(
        '--cor-offset-scale',
        type=float,
        default=1.0,
        help='Apply the COR/central-slice detector offset in backprojection. '
             '1.0 = apply verified-correct offset (default); 0.0 = off (legacy, '
             'COR at detector centre); -1.0 = flipped sign (diagnostic). Corrects '
             'non-round/mis-registered off-isocentre objects (projections are not '
             'pre-centred).'
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
    parser.add_argument(
        '--parker-weighting',
        action='store_true',
        default=True,
        dest='parker_weighting',
        help='Enable Parker (short-scan) redundancy weighting (default: enabled).'
    )
    parser.add_argument(
        '--no-parker',
        action='store_false',
        dest='parker_weighting',
        help='Disable Parker weighting (for comparison experiments).'
    )
    parser.add_argument(
        '--geometry-autocal',
        action='store_true',
        default=True,
        help="Measure detector in-plane rotation (psi) and the column centre "
             "of rotation from this scan's conjugate rays before "
             "reconstructing (reference-free, ~1 s). Default: on."
    )
    parser.add_argument(
        '--no-geometry-autocal',
        dest='geometry_autocal',
        action='store_false',
        help='Assume a perfectly square, centred detector — the '
             'pre-2026-08-11 behaviour (bit-exact legacy path).'
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

    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        default=True,
        help='Skip two-point auto-calibration; save physics-based HU directly '
             '(default: on). Recommended — auto-calibration maps the central '
             'ROI to 0 HU, which is unreliable when that ROI is not water.'
    )
    parser.add_argument(
        '--no-skip-calibration',
        dest='skip_calibration',
        action='store_false',
        help='Re-enable two-point auto-calibration (not recommended for mouse scans)'
    )

    # Beam hardening correction
    parser.add_argument(
        '--bhc-coeffs',
        nargs='+',
        type=float,
        default=None,
        help='BHC polynomial coefficients [c1, c2, ...] for sinogram-domain '
             'beam hardening correction: p_corrected = c1*p + c2*p^2 + ... '
             'Example: 0.856 0.21 (calibrated from water phantom at 80 kVp). '
             'Default: disabled (no BHC). Use --bhc-coeffs to enable.'
    )
    parser.add_argument(
        '--no-bhc',
        dest='bhc_coeffs',
        action='store_const',
        const=None,
        help='Disable sinogram-domain beam hardening correction'
    )

    # Bone beam hardening correction (Joseph & Spital two-pass)
    parser.add_argument(
        '--bone-bhc',
        action='store_true',
        help='Enable two-pass bone BHC (Joseph & Spital method). '
             'Requires physical_normalization (always on in this pipeline). '
             'Segments bone from pass-1 reconstruction, forward-projects bone '
             'contribution, and corrects sinogram before pass-2 backprojection.'
    )
    parser.add_argument(
        '--bone-bhc-threshold',
        type=float,
        default=1500,
        help='HU threshold for bone segmentation in pass-1 volume (default: 1500). '
             'Voxels above this threshold are classified as bone.'
    )
    parser.add_argument(
        '--bone-bhc-hu',
        type=float,
        default=3100,
        help='Monochromatic bone HU value for correction (default: 3100). '
             'Used to compute the ideal monochromatic bone attenuation.'
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
        sub_scan=f'-{args.phase}-',
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

    # Optional center-of-rotation / central-slice overrides (COR recalibration)
    if args.cor is not None:
        print(f"  COR override: central_pixel_a "
              f"{geometry['central_pixel_a']:.3f} -> {args.cor:.3f}")
        geometry['central_pixel_a'] = args.cor
    if args.central_slice is not None:
        print(f"  Central-slice override: central_pixel_b "
              f"{geometry['central_pixel_b']:.3f} -> {args.central_slice:.3f}")
        geometry['central_pixel_b'] = args.central_slice
    geometry['cor_offset_scale'] = args.cor_offset_scale

    # ---- geometry auto-calibration (psi + column CoR) -----------------------
    # FDK fuses flat-field + log + cone-weight + RAMP FILTER into one pass
    # (_preprocess_and_filter), so unlike TIGRE there is no point inside it that
    # holds unfiltered line integrals for the conjugate estimator. Rather than
    # duplicate a full 7 GB preprocessing pass, read the SAME scan-keyed
    # calibration muNeRF and the iterative pipeline write — it is the same
    # detector and the same projections, so re-measuring would only re-derive
    # the identical number.
    # An explicit --cor / --central-slice always wins; --no-geometry-autocal
    # restores the pre-2026-08-11 behaviour exactly (psi absent, CoR from XML).
    if args.geometry_autocal and args.cor is None:
        try:
            import json as _json
            from .ct_core.vff_io import detector_serial_from_scan
            _serial = detector_serial_from_scan(args.scan_folder)
            _tag = Path(args.scan_folder).name
            _cal = (Path(__file__).resolve().parents[1] / "data" / "calibration"
                    / f"detector_psi_{_serial}_{_tag}.json")
            if _serial and _cal.exists():
                _rec = _json.loads(_cal.read_text())
                geometry['det_psi_rad'] = np.radians(float(_rec["psi_deg"]))
                geometry['central_pixel_a'] = float(_rec["cpa0"])
                print(f"\n  Geometry auto-calibration from {_cal.name}:")
                print(f"    psi = {float(_rec['psi_deg']):+.4f} deg, "
                      f"central_pixel_a = {float(_rec['cpa0']):.3f} "
                      f"(measured {_rec.get('measured_on', '?')})")
            else:
                print(f"\n  Geometry auto-calibration: no cached measurement for "
                      f"this scan ({_cal.name}) — using psi=0 and the XML CoR. "
                      f"Run muNeRF or the iterative pipeline once on this scan, "
                      f"or scripts/detector_psi_from_conjugates.py, to populate it.")
        except Exception as _e:
            print(f"\n  Geometry auto-calibration skipped ({type(_e).__name__}: {_e})")

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

    # BHC coefficients (default: 80 kVp water BHC from calibration)
    bhc_coeffs = args.bhc_coeffs
    if bhc_coeffs is not None:
        print(f"\n  BHC coefficients: {bhc_coeffs}")
    else:
        print(f"\n  BHC: disabled (--no-bhc)")

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
        mu_water=None,
        clamp_mode="none",
        soft_clip_transmission=True,
        soft_clip_sharpness=50.0,
        upper_clamp=True,
        upper_clamp_value=1.05,
        physical_normalization=True,
        filter_cutoff=filter_cutoff,
        filter_type=args.filter_type,
        parker_weighting=args.parker_weighting,
        metal_artifact_reduction=args.metal_artifact_reduction,
        mar_threshold=args.mar_threshold,
        ring_correction=args.ring_correction,
        ring_median_width=args.ring_median_width,
        bhc_coeffs=bhc_coeffs,
        bone_bhc=args.bone_bhc,
        bone_bhc_threshold=args.bone_bhc_threshold,
        bone_bhc_hu=args.bone_bhc_hu,
    )

    # Run full reconstruction
    reconstructor.reconstruct(display_volume=args.display)

    # Post-process and save using shared utility
    postprocess_and_save(
        volume=reconstructor.reconstructed_volume,
        geometry=geometry,
        output_path=output_path,
        bilateral_filter=args.bilateral_filter,
        bilateral_sigma_spatial=args.bilateral_sigma_spatial,
        bilateral_sigma_range=args.bilateral_sigma_range,
        voxel_xy=args.voxel_xy,
        skip_calibration=args.skip_calibration,
    )

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
