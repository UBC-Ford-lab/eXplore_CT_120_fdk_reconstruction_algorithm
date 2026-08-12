"""
Run FDK reconstruction on VFF projections with HU calibration.

Uses verified reconstruction settings: HU output, physical normalization,
soft-clip transmission, upper clamping, and polynomial calibration.

Defaults to Hamming window at matched cutoff (da/dx), which gives the
best noise-resolution trade-off per the filter kernel sweep verification.

All algorithm-independent stages (scan loading, geometry build, detector-psi
calibration, HU calibration + VFF export) live in ct_core.pipeline and are
shared with the iterative drivers — this script only owns what is
FDK-specific: the filter settings, Parker weighting, MAR, bone BHC, and the
FDKReconstructor call.

Usage:
    python -m reconstruction.run_fdk_recon data/scans/Scan_1681
    python -m reconstruction.run_fdk_recon data/scans/Scan_1681 --filter-type ramp --filter-cutoff 1.0
    python -m reconstruction.run_fdk_recon data/scans/Scan_1681 --filter-type cosine --filter-cutoff 0.5
"""

import argparse
import sys
import time

import numpy as np
import torch

from .fdk import FDKReconstructor, SUPPORTED_FILTER_TYPES
from .ct_core.pipeline import (
    ReconLogger,
    add_common_args,
    prepare_scan,
    resolve_or_measure_detector_psi,
    save_outputs,
)


def parse_args():
    """Parse command-line arguments (shared args + FDK-specific ones)."""
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
    add_common_args(parser)

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

    return parser.parse_args()


def main():
    args = parse_args()

    start = time.time()

    print("=" * 60)
    print("FDK Reconstruction Pipeline")
    print("=" * 60)
    print(f"Data folder: {args.data_folder}")
    print(f"HU mode: enabled (physical normalization, polynomial calibration)")

    # Shared front half: scan folder, projections, ROI, geometry, downsample.
    ctx = prepare_scan(args)
    geometry = ctx.geometry

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

    # ---- geometry auto-calibration (detector in-plane rotation psi) --------
    # Cached scan-keyed JSON first (written by any pipeline); on a miss, the
    # half-scan-consistency estimator (ct_core.geometry_selfcal, ported from
    # muNeRF) measures psi from THIS scan's projections and caches it.
    # An explicit --cor always wins; --no-geometry-autocal restores the
    # pre-2026-08-11 behaviour exactly (psi absent, CoR from XML).
    if args.geometry_autocal and args.cor is None:
        record = resolve_or_measure_detector_psi(ctx)
        if record is not None:
            # psi ONLY — the fitted cpa0 intercept is estimator bias, not
            # geometry (see resolve_detector_psi docstring).
            geometry['det_psi_rad'] = np.radians(float(record['psi_deg']))

    # Resolve filter cutoff (may depend on geometry)
    if args.filter_cutoff.lower() == 'match':
        filter_cutoff = geometry['da'] / geometry['dx']
        print(f"\n  Filter cutoff 'match': da/dx = {geometry['da']:.4f}/"
              f"{geometry['dx']:.4f} = {filter_cutoff:.4f}")
    else:
        filter_cutoff = float(args.filter_cutoff)
    if not 0.0 < filter_cutoff <= 1.0:
        print(f"Error: filter-cutoff must be in (0.0, 1.0], got {filter_cutoff:.4f}")
        sys.exit(1)
    print(f"  Filter type: {args.filter_type}")

    output_path = args.output if args.output else ctx.default_output_path('_recon')
    print(f"\nOutput path: {output_path}")

    if args.bhc_coeffs is not None:
        print(f"\n  BHC coefficients: {args.bhc_coeffs}")
    else:
        print(f"\n  BHC: disabled (--no-bhc)")

    # Initialize reconstructor with verified settings
    reconstructor = FDKReconstructor(
        projections=ctx.projections,
        angles=torch.as_tensor(ctx.angles),
        geometry=geometry,
        source_locations=None,
        folder_name=output_path,
        output_hu=True,
        bright_field=ctx.bright_field,
        dark_field=ctx.dark_field,
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
        bhc_coeffs=args.bhc_coeffs,
        bone_bhc=args.bone_bhc,
        bone_bhc_threshold=args.bone_bhc_threshold,
        bone_bhc_hu=args.bone_bhc_hu,
    )

    # Experiment logging: local PNGs next to the output, W&B when --wandb.
    logger = ReconLogger(args, ctx, 'fdk', output_path, params={
        'filter_cutoff': filter_cutoff,
        'filter_type': args.filter_type,
        'parker_weighting': bool(args.parker_weighting),
        'bone_bhc': bool(args.bone_bhc),
    })

    reconstructor.reconstruct(display_volume=args.display)

    # Shared back half: HU calibration + bilateral filter + VFF export.
    save_outputs(reconstructor.reconstructed_volume, ctx, args, output_path)

    logger.log_sinogram_preview(ctx.projections)
    logger.log_volume_summary(reconstructor.reconstructed_volume, ctx)
    logger.finish()

    end = time.time()
    print(f"\nReconstruction finished in {(end - start)/60:.2f} minutes.")


if __name__ == '__main__':
    main()
