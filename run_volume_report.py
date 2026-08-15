"""
Report on an ALREADY-RECONSTRUCTED volume — no projections, no reconstruction.

The other drivers reconstruct a scan and then describe the result. This one
does only the describing, for a volume that already exists: the scanner
vendor's own .vff, a SIRT or FDK volume from an earlier run, a muNeRF export,
anything on the HU scale. It produces the volume-domain half of the standard
panel — the part that needs neither a sinogram nor a training loop — through
exactly the same code the reconstruction drivers call, so the figures are
directly comparable to any run in the same W&B project:

    plots/hu_histogram     HU distribution + bone-tail zoom
    plots/view_axial       central orthogonal slices on physical mm axes
    plots/view_coronal
    plots/view_sagittal
    recon_slices           the volume as a scrollable axial-slice sequence
    volume/*               shape, HU percentiles, mean/std/range, and the
                           measured air / tissue histogram peaks

Deliberately absent: the projection diagnostics (diag/ssim, the SSIM heatmap,
the power spectrum, the noise ceiling) and the convergence curve. Those are
not volume properties — they compare a forward projection against measured
data, which needs the scan this volume came from, its exact reconstruction
grid, and its geometry calibration. Reconstruct through run_fdk_recon /
run_iterative_recon / run_learned_recon to get them.

Unlike the reconstruction drivers, W&B logging is ON by default here (there
is no expensive compute to protect, and the point of the tool is to put an
external volume next to the runs) — pass --no-wandb to keep it local.

Usage:
    # vendor reconstruction, logged to W&B
    python -m reconstruction.run_volume_report \\
        data/scans/Scan_1988/Volumes/Half-scan-75um.vff \\
        --wandb-project my-ct-project --algorithm vendor_fdk

    # a volume this package produced (round-trips exactly), local plots only
    python -m reconstruction.run_volume_report Scan_1988_recon.vff --no-wandb

    # foreign volume: pin the geometry the header cannot carry
    python -m reconstruction.run_volume_report recon.vff \\
        --voxel-xy 0.075 --voxel-z 0.075 --origin 2.71 -5.53 0.19
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .ct_core.hu_calibration import (
    TISSUE_HU_DEFAULT,
    find_attenuation_anchors,
    format_calibration,
    landmark_check,
)
from .ct_core.pipeline import ScanContext
from .ct_core.volume_report import (
    format_statistics,
    hu_scale_warnings,
    load_reconstructed_volume,
    resolve_volume_geometry,
    volume_statistics,
)
from .ct_core.wandb_logging import HU_WINDOW, ReconLogger, add_wandb_args


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the volume-domain metrics and figures for an '
                    'already-reconstructed volume, and log them to W&B.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m reconstruction.run_volume_report vol.vff --wandb-project my-project
  python -m reconstruction.run_volume_report vol.vff --no-wandb
  python -m reconstruction.run_volume_report vol.vff --voxel-xy 0.075 --voxel-z 0.075
        """
    )
    parser.add_argument(
        'volume',
        help='Reconstructed volume to report on (.vff, .npy or .npz). VFF is '
             'read in the GE ncaa layout this package writes; .npy/.npz are '
             'assumed to be (x, y, z) in HU already.'
    )
    parser.add_argument(
        '--output',
        help='Base path for the local plot folder (default: '
             '<volume>_report, giving <volume>_report_plots/). Chosen so a '
             'report never overwrites the plots of the run that produced the '
             'volume.'
    )
    parser.add_argument(
        '--algorithm',
        default='external',
        help='Label for what produced this volume (default: external). Goes '
             'into the run config and the default run name, e.g. vendor_fdk, '
             'sirt, munerf.'
    )
    parser.add_argument(
        '--scan-folder',
        default=None,
        help='Scan folder this volume was reconstructed from. Optional and '
             'never read for data — only its BASENAME is recorded, so the '
             'report groups with the runs of the same scan. Defaults to the '
             'volume filename.'
    )
    parser.add_argument(
        '--voxel-xy',
        type=float,
        default=None,
        help='In-plane voxel size in mm (default: the VFF elementsize). Only '
             'sets the physical axes of the view figures.'
    )
    parser.add_argument(
        '--voxel-z',
        type=float,
        default=None,
        help='Slice thickness in mm (default: --voxel-xy, i.e. the VFF '
             'elementsize — GE headers carry one scalar voxel size, so an '
             'anisotropic grid must be given here).'
    )
    parser.add_argument(
        '--origin',
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=('X', 'Y', 'Z'),
        help='Volume CENTRE in isocentre-centred mm (default: 0 0 0, the '
             'full-FOV case). Set it for an ROI reconstruction so the view '
             'axes read in scanner coordinates.'
    )
    parser.add_argument(
        '--hu-window',
        nargs=2,
        type=float,
        default=HU_WINDOW,
        metavar=('LO', 'HI'),
        help=f'Display window for the view and slice images in HU '
             f'(default: {HU_WINDOW[0]:.0f} {HU_WINDOW[1]:.0f}). Does not '
             f'affect any metric.'
    )
    parser.add_argument(
        '--max-slices',
        type=int,
        default=240,
        help='Cap on the number of axial slices uploaded as recon_slices '
             '(default: 240; thicker volumes are strided down).'
    )
    parser.add_argument(
        '--no-slices',
        dest='slices',
        action='store_false',
        default=True,
        help='Skip the recon_slices sequence (the slow part of the upload).'
    )
    parser.add_argument(
        '--recalibrate',
        action='store_true',
        default=False,
        help='Fit the two HU anchors from this volume\'s own histogram (air '
             'to -1000, bulk soft tissue to --tissue-hu) and report the '
             'volume as recalibrated. Off by default, so the report describes '
             'the file as it stands. Use it to put a vendor volume or an old '
             'run on the same scale as current reconstructions, or to find '
             'out how far off a volume is: --diagnose-calibration reports the '
             'fit without applying it.'
    )
    parser.add_argument(
        '--diagnose-calibration',
        action='store_true',
        default=False,
        help='Fit and report the HU anchors but do NOT apply them. Cheap way '
             'to ask "is this volume on the HU scale, and by how much is it '
             'off?" without changing any figure.'
    )
    parser.add_argument(
        '--tissue-hu',
        type=float,
        default=None,
        help='Where the bulk soft-tissue anchor lands when recalibrating, in '
             'HU (default: 120, matching the vendor\'s scale for the same '
             'specimens). Sets the gain — see ct_core.hu_calibration. Pass 0 '
             'for a water phantom.'
    )
    parser.add_argument(
        '--hu-from-header',
        action='store_true',
        default=False,
        help='Apply the VFF header water/air anchors as '
             'HU = 1000 (v - water) / (water - air). OFF by default because '
             'both this package and the scanner vendor store values to which '
             'that mapping has ALREADY been applied — re-applying it rescales '
             'HU by a factor of ~254 on vendor files. Use only for a foreign '
             'volume that genuinely stores raw values.'
    )
    parser.add_argument(
        '--no-y-flip',
        dest='y_flip',
        action='store_false',
        default=True,
        help='Do not undo the y reversal in the VFF payload. The default '
             'inverts this package\'s own writer exactly; pass this for a '
             'foreign file that comes out mirrored.'
    )

    # Shared logging flags (--wandb-project / --wandb-entity / --wandb-run-name
    # / --wandb-mode / --no-plots), with --wandb ON: a report is cheap and
    # exists to be compared against the runs, so opting OUT is the exception.
    add_wandb_args(parser, wandb_default=True)
    parser.add_argument(
        '--no-wandb',
        dest='wandb',
        action='store_false',
        help='Do not log to Weights & Biases (default: logging is ON for this '
             'tool). Local PNGs are still written unless --no-plots.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    volume_path = Path(args.volume)
    print("=" * 60)
    print("Volume Report (already-reconstructed volume)")
    print("=" * 60)
    print(f"Volume: {volume_path}")

    try:
        volume, header = load_reconstructed_volume(
            volume_path, y_flip=args.y_flip,
            hu_from_header=args.hu_from_header)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    geometry = resolve_volume_geometry(
        volume, header,
        voxel_xy=args.voxel_xy, voxel_z=args.voxel_z, origin=args.origin)

    # The figure builders and the config whitelist read a ScanContext; this
    # one carries a volume and no measurements, which is exactly the case the
    # projection-side logging already guards against (n_angles 0, no sinogram
    # preview, no diagnostics). The scan identity is a BASENAME either way —
    # never a path (see wandb_logging's privacy notes).
    ctx = ScanContext(
        data_folder=str(volume_path),
        scan_folder=str(args.scan_folder or volume_path.name),
        projections=None,
        angles=np.zeros(0),
        bright_field=None,
        dark_field=None,
        xml_header={},
        geometry=geometry,
        downsample=1,
    )

    output_path = args.output or str(volume_path.with_suffix('')) + '_report'
    logger = ReconLogger(args, ctx, args.algorithm, output_path, params={
        'source_format': volume_path.suffix.lstrip('.').lower(),
        'hu_from_header': bool(args.hu_from_header),
        'y_flip': bool(args.y_flip),
        'voxel_xy_mm': geometry['dx'],
        'voxel_z_mm': geometry['dz'],
        'vol_origin_mm': list(geometry['vol_origin']),
    })

    # Calibration diagnosis / recalibration. The estimator is equivariant, so
    # it applies unchanged to a volume that is already in HU: it simply finds
    # the anchors where they currently sit and reports the map that would move
    # them onto -1000 / --tissue-hu. A map close to the identity means the
    # volume is already on the scale; anything else quantifies the offset and
    # gain error in one line.
    if args.recalibrate or args.diagnose_calibration:
        print("\nHU calibration fit:")
        anchors = find_attenuation_anchors(
            volume,
            tissue_hu=(TISSUE_HU_DEFAULT if args.tissue_hu is None
                       else float(args.tissue_hu)))
        print(format_calibration(anchors))
        logger.log_hu_calibration(anchors)
        if args.recalibrate:
            volume = anchors.apply(volume)
            print("  applied — every figure below shows the RECALIBRATED "
                  "volume")
        else:
            print("  not applied (--diagnose-calibration); the figures below "
                  "show the volume as stored")

    print("\nVolume statistics:")
    stats = volume_statistics(volume, geometry)
    print(format_statistics(stats))
    for warning in hu_scale_warnings(stats):
        print(f"  WARNING: {warning}")
    logger.set_summary(stats)
    logger.set_summary(landmark_check(volume))

    # The same figures every reconstruction produces, from the same code:
    # view_axial / view_coronal / view_sagittal + hu_histogram (+ the
    # volume/* percentiles, re-recorded here identically to a real run).
    hu_window = (float(args.hu_window[0]), float(args.hu_window[1]))
    logger.log_volume_summary(volume, ctx, hu_window=hu_window)
    if args.slices:
        logger.log_recon_slices(volume, hu_window=hu_window,
                                max_slices=args.max_slices)
    logger.finish()

    print(f"\nReport finished in {(time.time() - start) / 60:.2f} minutes.")


if __name__ == '__main__':
    main()
