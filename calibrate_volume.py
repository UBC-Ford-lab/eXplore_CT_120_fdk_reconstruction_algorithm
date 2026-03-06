"""
Calibrate a reconstructed CT volume to Hounsfield Units using phantom inserts.

Loads an uncalibrated VFF volume, measures mean values in circular ROIs
corresponding to known phantom inserts, fits a polynomial mapping
measured → true HU, applies it, and saves the calibrated volume.

Supports two modes:
  --visualize : Show insert ROIs overlaid on phantom slice (no calibration)
  (default)   : Measure inserts, fit polynomial, calibrate, save

The insert ROI positions are specified via a JSON config file:
  {
    "inserts": [
      {"name": "air",  "cy": 480, "cx": 420, "radius": 30, "true_hu": -940},
      {"name": "bone", "cy": 310, "cx": 750, "radius": 25, "true_hu": 1460},
      ...
    ]
  }

Usage:
    python -m reconstruction.calibrate_volume \\
        --input volume.vff --output volume_calibrated.vff \\
        --roi-config inserts.json --z-range 290 320 \\
        --plot calibration_diagnostic

    python -m reconstruction.calibrate_volume \\
        --input volume.vff --roi-config inserts.json \\
        --z-range 290 320 --visualize
"""

import argparse
import json
import os
import sys

import numpy as np

# Add project root to path
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from reconstruction.ct_core import vff_io as vff
from reconstruction.ct_core.calibration import (
    measure_insert_rois,
    calibrate_volume_polynomial,
    plot_calibration_diagnostic,
    fit_hu_calibration,
)


def load_roi_config(config_path: str) -> list:
    """Load insert ROI definitions from a JSON file."""
    with open(config_path) as f:
        data = json.load(f)
    inserts = data["inserts"]
    for ins in inserts:
        for key in ("name", "cy", "cx", "radius", "true_hu"):
            if key not in ins:
                raise ValueError(f"Insert missing required key '{key}': {ins}")
    return inserts


def visualize_rois(volume: np.ndarray, insert_rois: list, z_range: tuple,
                   output_path: str):
    """Show phantom slice with ROI overlays and measured values."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    sl = volume[z_range[0]:z_range[1]].mean(axis=0)
    measurements = measure_insert_rois(volume, insert_rois, z_range)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    vmin = float(np.percentile(sl, 1))
    vmax = float(np.percentile(sl, 99))
    im = ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Pixel value', shrink=0.8)

    for m in measurements:
        circle = Circle((m['cx'], m['cy']), m['radius'],
                         fill=False, edgecolor='cyan', linewidth=2)
        ax.add_patch(circle)
        ax.text(m['cx'] + m['radius'] + 8, m['cy'],
                f"{m['name']}\nmean={m['measured_mean']:.1f}\n"
                f"std={m['measured_std']:.1f}\ntrue={m['true_hu']}",
                fontsize=7, color='yellow', fontweight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='black', alpha=0.8))

    ax.set_title(f'Insert ROI verification (z={z_range[0]}-{z_range[1]} avg)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Visualization saved: {output_path}")
    plt.close(fig)

    # Print summary table
    print(f"\n  {'Name':<14s}  {'cy':>5s}  {'cx':>5s}  {'Measured':>10s}  "
          f"{'Std':>7s}  {'True HU':>8s}  {'Delta':>8s}")
    print("  " + "-" * 65)
    for m in measurements:
        delta = m['measured_mean'] - m['true_hu']
        print(f"  {m['name']:<14s}  {m['cy']:5d}  {m['cx']:5d}  "
              f"{m['measured_mean']:10.1f}  {m['measured_std']:7.1f}  "
              f"{m['true_hu']:8d}  {delta:+8.1f}")


def run_calibration(input_path: str, output_path: str, insert_rois: list,
                    z_range: tuple, degree: int, plot_path: str = None):
    """Full calibration pipeline: measure → fit → apply → save."""
    print(f"\n  Loading {input_path}...")
    header, vol = vff.read_vff(input_path, verbose=False)
    vol = vol.astype(np.float32)
    print(f"  Shape: {vol.shape}, range: [{vol.min():.0f}, {vol.max():.0f}]")

    # Measure inserts
    print(f"\n  Measuring {len(insert_rois)} insert ROIs (z={z_range[0]}-{z_range[1]})...")
    measurements = measure_insert_rois(vol, insert_rois, z_range)

    print(f"\n  {'Name':<14s}  {'Measured':>10s}  {'Std':>7s}  {'True HU':>8s}")
    print("  " + "-" * 45)
    for m in measurements:
        print(f"  {m['name']:<14s}  {m['measured_mean']:10.1f}  "
              f"{m['measured_std']:7.1f}  {m['true_hu']:8d}")

    # Build calibration pairs
    cal_data = np.array([[m['measured_mean'], m['true_hu']]
                         for m in measurements])

    # Fit and apply
    print(f"\n  Fitting degree-{degree} polynomial...")
    vol_cal, coeffs, rms = calibrate_volume_polynomial(
        vol, cal_data, degree=degree)
    print(f"  Coefficients: {coeffs}")
    print(f"  RMS residual: {rms:.1f} HU")
    print(f"  Calibrated range: [{vol_cal.min():.0f}, {vol_cal.max():.0f}]")

    # Verify calibration on the same ROIs
    print(f"\n  Verification (post-calibration):")
    cal_measurements = measure_insert_rois(vol_cal, insert_rois, z_range)
    print(f"  {'Name':<14s}  {'Calibrated':>10s}  {'True HU':>8s}  {'Error':>8s}")
    print("  " + "-" * 45)
    for m in cal_measurements:
        err = m['measured_mean'] - m['true_hu']
        print(f"  {m['name']:<14s}  {m['measured_mean']:10.1f}  "
              f"{m['true_hu']:8d}  {err:+8.1f}")

    # Diagnostic plot
    if plot_path:
        sl = vol[z_range[0]:z_range[1]].mean(axis=0)
        plot_calibration_diagnostic(measurements, coeffs, sl, plot_path)

    # Save
    print(f"\n  Writing calibrated volume to {output_path}...")
    vff.write_vff(output_path, header, vol_cal)
    print("  Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate a CT volume to HU using phantom insert measurements',
    )
    parser.add_argument('--input', required=True,
                        help='Path to uncalibrated VFF volume')
    parser.add_argument('--output',
                        help='Path for calibrated VFF output')
    parser.add_argument('--roi-config', required=True,
                        help='JSON file with insert ROI definitions')
    parser.add_argument('--z-range', type=int, nargs=2, required=True,
                        metavar=('Z_START', 'Z_END'),
                        help='Z-slice range for ROI measurements')
    parser.add_argument('--degree', type=int, default=2,
                        help='Polynomial degree (default: 2)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Save diagnostic plot to this path (without extension)')
    parser.add_argument('--visualize', action='store_true',
                        help='Only visualize ROIs on phantom slice (no calibration)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CT Volume HU Calibration")
    print("=" * 60)

    # Load ROI config
    insert_rois = load_roi_config(args.roi_config)
    print(f"  Loaded {len(insert_rois)} insert ROIs from {args.roi_config}")

    z_range = tuple(args.z_range)

    if args.visualize:
        print(f"\n  Loading {args.input}...")
        _, vol = vff.read_vff(args.input, verbose=False)
        vol = vol.astype(np.float32)
        print(f"  Shape: {vol.shape}")

        out_png = args.plot or os.path.splitext(args.input)[0] + '_roi_check.png'
        visualize_rois(vol, insert_rois, z_range, out_png)
        return

    if not args.output:
        print("ERROR: --output required when not using --visualize")
        sys.exit(1)

    run_calibration(
        input_path=args.input,
        output_path=args.output,
        insert_rois=insert_rois,
        z_range=z_range,
        degree=args.degree,
        plot_path=args.plot,
    )

    print("\nCalibration complete.")


if __name__ == '__main__':
    main()
