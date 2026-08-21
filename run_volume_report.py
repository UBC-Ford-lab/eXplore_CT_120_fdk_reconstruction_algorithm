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

W&B logging is ON by default here, as it is on every driver — pass --no-wandb
to keep it local. (This tool was the first to default it on; the reconstruction
drivers followed on 2026-08-15.)

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
import time
from pathlib import Path

import numpy as np

from .ct_core.hu_calibration import (
    TISSUE_HU_DEFAULT,
    find_attenuation_anchors,
    format_calibration,
    landmark_check,
)
from .ct_core.errors import ScanDataError, cli_main
from .ct_core.pipeline import ScanContext
from .ct_core.volume_report import (
    format_statistics,
    hu_scale_warnings,
    load_reconstructed_volume,
    load_sidecar,
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
        default=None,
        metavar=('X', 'Y', 'Z'),
        help='Volume CENTRE in isocentre-centred mm (default: the '
             '<volume>.json sidecar if this package wrote the volume, else '
             '0 0 0 — the full-FOV case). Set it for a foreign ROI '
             'reconstruction so the view axes read in scanner coordinates.'
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
        '--vendor', action='store_true', default=False,
        help='This volume came from the GE scanner, not from this package. '
             'Applies everything that follows from that, so none of it has to '
             'be remembered per scan: --no-y-flip and --flip x (their axis '
             'conventions, MEASURED over all eight sign combinations), and the '
             'ROI centre DERIVED from the scan folder\'s '
             'Volumes/SubVolumeCoordinates.xml + scan.xml LandmarkOffsetVector '
             'rather than passed as numbers. Needs --scan-folder to point at '
             'the real scan directory. Still leaves --align-to to you: which '
             'reconstruction to register against is a genuine choice, not a '
             'convention.'
    )
    parser.add_argument(
        '--align-to', default=None, metavar='VOLUME',
        help='Register this volume onto VOLUME (a reconstruction this package '
             'produced — our own FDK of the same scan is the natural choice) '
             'and apply the measured sub-voxel offset before reporting. The '
             'GE vendor volume lands a few tenths of a mm off, differently per '
             'axis. It is MEASURED rather than read from the vendor XML '
             'because that does not work: correcting by the selection-vs-recon '
             'difference moves it the wrong way on all three axes, and the '
             'vendor records disagree with their own VFF header about voxel '
             'size by 0.5%% (0.28 mm over 763 slices). Measured once per scan '
             'and cached next to the detector-psi calibration.'
    )
    parser.add_argument(
        '--realign', action='store_true', default=False,
        help='Re-measure the alignment even if a cached one exists.')
    parser.add_argument(
        '--align-region', type=float, default=0.5, metavar='FRAC',
        help='Fit the alignment on the central FRAC of the reference '
             '(default: %(default)s). MEASURED on Scan_1510: the vendor volume '
             'matches ours neither by a translation nor by a scale — the '
             'best-fit z offset per slab runs +0.72/+0.81/+0.74/+0.68/+0.54/'
             '+0.32 mm bottom to top, a curved profile. So a whole-volume fit '
             'matches NOWHERE, least of all the midplane every figure shows. '
             'Pass 1.0 for the old whole-volume behaviour.'
    )
    parser.add_argument(
        '--align-stride', type=int, default=3, metavar='N',
        help='Sub-sample every Nth voxel per axis when registering '
             '(default: %(default)s). The offset is a whole-volume property, '
             'so this costs precision far more slowly than time.')
    parser.add_argument(
        '--flip', nargs='+', default=[], choices=['x', 'y', 'z'],
        metavar='AXIS',
        help='Reverse these axes after loading, in (x, y, z) order — for a '
             'FOREIGN file whose axis conventions differ from this package. '
             'Separate from --no-y-flip, which inverts our own writer. '
             'MEASURED: the GE vendor .vff needs "--flip x". A missed x mirror '
             'shows as a left-right flipped CORONAL view (coronal is (z, x)) '
             'while the axial view still looks plausible, so check all three '
             'axes against a trusted reconstruction, not just y.'
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

    # Shared logging flags (--wandb / --no-wandb / --wandb-project / --entity
    # / --run-name / --mode / --no-plots). W&B defaults ON here as it does
    # everywhere else now; this tool no longer needs to say so specially, and
    # its own --no-wandb has moved into add_wandb_args so every driver has it.
    add_wandb_args(parser)

    return parser.parse_args()


def _apply_vendor_conventions(args, volume_path):
    """Everything that follows from "this file came from the scanner".

    Kept in ONE place so a vendor volume needs no per-scan numbers. The axis
    conventions were measured (all eight sign combinations against a trusted
    reconstruction: -x+y+z at NCC 0.899 vs 0.788 for +x+y+z and 0.634 for
    +x-y+z), and the ROI centre is DERIVED from the scanner's own
    SubVolumeCoordinates.xml rather than typed in — the previous
    "--origin 2.715 -5.53 0.185" was correct for exactly one scan.
    """
    import xmltodict
    from .ct_core.scan_setup import parse_crop_boundary

    if args.y_flip:
        args.y_flip = False
    if 'x' not in args.flip:
        args.flip = list(args.flip) + ['x']
    print("  --vendor: GE axis conventions (no y-flip undo, x mirrored)")

    scan_folder = Path(args.scan_folder) if args.scan_folder else None
    if scan_folder is None or not scan_folder.is_dir():
        raise ScanDataError(
            "--vendor needs --scan-folder pointing at the real scan directory "
            "(it reads Volumes/SubVolumeCoordinates.xml and scan.xml to place "
            "the ROI). Pass --origin yourself to skip that.")
    xml_files = sorted(scan_folder.glob('scan.xml'))
    if not xml_files:
        raise ScanDataError(f"--vendor: no scan.xml in {scan_folder.name}")
    xml_header = xmltodict.parse(xml_files[0].read_text())
    bounds = parse_crop_boundary(str(scan_folder), xml_header)
    if bounds is None:
        raise ScanDataError(
            f"--vendor: no Volumes/SubVolumeCoordinates.xml in "
            f"{scan_folder.name}; pass --origin explicitly.")
    if args.origin is None:
        args.origin = [
            (float(bounds['x_min']) + float(bounds['x_max'])) / 2.0,
            (float(bounds['y_min']) + float(bounds['y_max'])) / 2.0,
            (float(bounds['z_min']) + float(bounds['z_max'])) / 2.0,
        ]
        print(f"    ROI centre from SubVolumeCoordinates.xml: "
              f"({args.origin[0]:+.3f}, {args.origin[1]:+.3f}, "
              f"{args.origin[2]:+.3f}) mm")


def main():
    args = parse_args()
    start = time.time()

    volume_path = Path(args.volume)
    print("=" * 60)
    print("Volume Report (already-reconstructed volume)")
    print("=" * 60)
    print(f"Volume: {volume_path}")

    if args.vendor:
        _apply_vendor_conventions(args, volume_path)

    try:
        volume, header = load_reconstructed_volume(
            volume_path, y_flip=args.y_flip, flip_axes=args.flip,
            hu_from_header=args.hu_from_header)
    except (FileNotFoundError, ValueError) as e:
        raise ScanDataError(str(e)) from e

    # Volumes this package wrote describe themselves; foreign ones do not, and
    # then the command line / header defaults apply exactly as before.
    sidecar = load_sidecar(volume_path)
    if args.algorithm == 'external' and sidecar.get('algorithm'):
        args.algorithm = str(sidecar['algorithm'])
    if args.scan_folder is None and sidecar.get('scan'):
        args.scan_folder = str(sidecar['scan'])

    geometry = resolve_volume_geometry(
        volume, header, sidecar=sidecar,
        voxel_xy=args.voxel_xy, voxel_z=args.voxel_z, origin=args.origin)

    # ---- alignment onto one of our own reconstructions ---------------------
    # Measured, cached, and applied to the CONTENT (a sub-voxel shift), so the
    # midplane views line up with every other run of the same scan.
    align_offset = None
    if args.align_to:
        from .ct_core.volume_align import (apply_volume_offset, load_alignment,
                                           measure_volume_offset,
                                           reference_fingerprint,
                                           save_alignment)
        ref_path = Path(args.align_to)
        scan_tag = str(args.scan_folder or volume_path.name)
        scan_tag = Path(scan_tag).name
        print(f"\nAligning to {ref_path.name}")
        try:
            ref_vol, ref_header = load_reconstructed_volume(
                ref_path, verbose=False)
        except (FileNotFoundError, ValueError) as e:
            raise ScanDataError(f"--align-to: {e}") from e
        ref_geom = resolve_volume_geometry(
            ref_vol, ref_header, sidecar=load_sidecar(ref_path), verbose=False)
        ref_grid = reference_fingerprint(ref_geom)
        align_offset = (None if args.realign
                        else load_alignment(scan_tag, volume_path.name,
                                            reference=ref_path.name,
                                            ref_grid=ref_grid))
        if align_offset is None:
            print(f"  reference {ref_vol.shape} at "
                  f"{ref_geom['dx']:.4f}/{ref_geom['dz']:.4f} mm; measuring "
                  f"(stride {args.align_stride})...")
            align_offset, ncc = measure_volume_offset(
                volume, geometry, ref_vol, ref_geom, stride=args.align_stride,
                region=args.align_region)
            save_alignment(scan_tag, volume_path.name, align_offset, ncc,
                           ref_path.name,
                           extra={'stride': int(args.align_stride),
                                  'region': float(args.align_region),
                                  'reference_grid': ref_grid})
        del ref_vol
        # Applied to the FIGURES only. The shift is sub-voxel, so it is a
        # linear interpolation, and interpolation smooths: on this volume it
        # moved the air peak -1016.6 -> -977.3 HU and the tissue peak
        # +121.8 -> +136.6. Alignment is a display and comparison concern, so
        # every HU measurement below stays on the volume AS STORED and only
        # the views and slice sequence see the aligned copy.
        volume_aligned = apply_volume_offset(volume, align_offset, geometry)
        print(f"  Applied ({align_offset[0]:+.3f}, {align_offset[1]:+.3f}, "
              f"{align_offset[2]:+.3f}) mm to the volume content "
              f"(figures only; HU statistics stay on the stored volume).")

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
        'flip_axes': list(args.flip),
        'align_to': (Path(args.align_to).name if args.align_to else None),
        'align_offset_mm': (list(map(float, align_offset))
                            if align_offset is not None else None),
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
    _figs = None if align_offset is None else volume_aligned
    logger.log_volume_summary(volume, ctx, hu_window=hu_window,
                              views_from=_figs)
    if args.slices:
        logger.log_recon_slices(volume if _figs is None else _figs,
                                hu_window=hu_window,
                                max_slices=args.max_slices,
                                geometry=geometry)
    logger.finish()

    print(f"\nReport finished in {(time.time() - start) / 60:.2f} minutes.")


if __name__ == '__main__':
    cli_main(main)
