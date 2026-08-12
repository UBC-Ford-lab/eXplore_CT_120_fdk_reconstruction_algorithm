"""
Measure the detector in-plane rotation (psi) for a scan, standalone.

Runs the half-scan-consistency estimator (ct_core.geometry_selfcal — the
validated method ported from muNeRF) on the scan's own projections and writes
the scan-keyed calibration JSON that every reconstruction pipeline (FDK,
ASTRA, TIGRE, muNeRF) reads:

    data/calibration/detector_psi_<serial>_<scanTag>.json

The reconstruction drivers do this automatically on a cache miss, so this
script is only needed to (re-)calibrate ahead of time — e.g. once per scan on
a cluster login/GPU node before submitting long jobs, or with --force after a
scanner recalibration.

Usage:
    python -m reconstruction.run_geometry_calibration data/scans/Scan_1510
    python -m reconstruction.run_geometry_calibration data/scans/Scan_1510 --force
"""

import argparse
import sys

from .ct_core.pipeline import (
    add_common_args,
    prepare_scan,
    resolve_detector_psi,
    measure_detector_psi,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure detector in-plane rotation (psi) from a scan\'s '
                    'own projections and cache it for all reconstruction '
                    'pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-measure even when a cached calibration exists for this scan '
             '(overwrites the JSON).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Detector Geometry Calibration (half-scan consistency)")
    print("=" * 60)

    if not args.force:
        # prepare_scan is the expensive step — check the cache with just the
        # scan folder first.
        from .ct_core.scan_setup import auto_detect_scan_folder
        scan_folder = args.scan_folder
        if not scan_folder:
            try:
                scan_folder = auto_detect_scan_folder(args.data_folder)
            except ValueError:
                scan_folder = None
        if scan_folder:
            record = resolve_detector_psi(scan_folder, verbose=False)
            if record is not None:
                print(f"Cached calibration already exists: psi = "
                      f"{float(record['psi_deg']):+.4f} deg (method "
                      f"{record.get('method', '?')}, measured "
                      f"{record.get('measured_on', '?')}).")
                print("Use --force to re-measure.")
                return

    ctx = prepare_scan(args)
    record = measure_detector_psi(ctx)
    if record is None:
        print("\nCalibration failed — see messages above.")
        sys.exit(1)
    print(f"\nDone: psi = {float(record['psi_deg']):+.4f} deg "
          f"(elapsed {float(record.get('elapsed_s', 0.0)):.0f}s).")


if __name__ == '__main__':
    main()
