"""
Shared CLI utilities for CT reconstruction scripts.

Extracts common data-loading, geometry setup, and post-processing logic
used by both FDK and iterative reconstruction pipelines.
"""

import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import xmltodict

from .vff_io import VFFDataset, write_vff
from .calibration import (
    load_calibration_fields, MU_WATER_80KV,
    PHANTOM_CALIBRATION, fit_hu_calibration,
    measure_insert_rois, calibrate_volume_polynomial,
    plot_calibration_diagnostic, get_calibration_coefficients,
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


def load_scan_data(data_folder, scan_folder, projection_pattern, total_angle):
    """
    Load projections, angles, and calibration fields from a scan folder.

    Handles:
    - scan.xml parsing
    - Bright/dark field loading
    - Projection pattern auto-detection (proj-* vs acq*)
    - Total angle determination from XML or override
    - VFFDataset creation

    Args:
        data_folder: Path to folder containing projections and scan.xml
        scan_folder: Path to original scan folder with bright.vff/dark.vff
        projection_pattern: Glob pattern for projection files, or None to auto-detect
        total_angle: Total angular coverage string ('determined' or numeric degrees)

    Returns:
        dict with keys:
            'projections': np.ndarray (N_angles, N_b, N_a)
            'angles': torch.Tensor (N_angles,) in radians
            'bright_field': np.ndarray (N_b, N_a)
            'dark_field': np.ndarray (N_b, N_a)
            'xml_header': parsed XML dict
            'n_files': int, number of projection files
            'total_angle': float, total angular coverage in degrees
    """
    xml_file = os.path.join(data_folder, 'scan.xml')

    if not os.path.exists(xml_file):
        print(f"Error: scan.xml not found at {xml_file}")
        sys.exit(1)

    # Load bright/dark fields
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
    if projection_pattern is None:
        if list(Path(data_folder).glob('proj-*.vff')):
            projection_pattern = 'proj-*.vff'
        else:
            projection_pattern = 'acq*'

    # Parse XML
    header = xmltodict.parse(open(xml_file).read())

    # Count projection files
    print(f"\nLoading projections (pattern: {projection_pattern})...")
    proj_paths = sorted(Path(data_folder).glob(projection_pattern))
    # Apply phase filter only to acquisition files (not sequential proj-* files)
    proj_paths = [p for p in proj_paths if p.name.startswith('proj-') or '-00-' in str(p)]
    n_files = len(proj_paths)
    if n_files == 0:
        print(f"Error: No projections matching '{projection_pattern}' in {data_folder}")
        sys.exit(1)

    # Determine total angle
    if total_angle == 'determined':
        try:
            increment_angle = float(header['Series']['SeriesParams']['IncrementAngle'])
            xml_view_count = int(header['Series']['SeriesParams']['ViewCount'])
        except (KeyError, TypeError) as e:
            print(f"Error: Could not read IncrementAngle/ViewCount from scan.xml: {e}")
            print("Please specify --total-angle explicitly.")
            sys.exit(1)

        total_angle_deg = increment_angle * n_files
        print(f"  Determined from scan.xml: IncrementAngle = {increment_angle:.6f} deg, "
              f"ViewCount = {xml_view_count}")
        if xml_view_count != n_files:
            print(f"  WARNING: ViewCount in scan.xml ({xml_view_count}) does not match "
                  f"number of projection files found ({n_files})")
        print(f"  Total angle = {increment_angle:.6f} x {n_files} projections = {total_angle_deg:.4f} deg")
    else:
        total_angle_deg = float(total_angle)
        print(f"  User-specified total angle: {total_angle_deg:.4f} deg")

    projection_spacing = total_angle_deg / n_files
    print(f"  {n_files} projections, {total_angle_deg:.4f} deg total -> spacing = {projection_spacing:.6f} deg")

    dataset = VFFDataset(
        data_folder,
        xml_file,
        paths_str=projection_pattern,
        projection_spacing=projection_spacing,
    )
    projections = dataset.projections  # shape (N_angles, N_b, N_a)
    angles = dataset.angles_rad        # shape (N_angles,)

    print(f"Loaded {len(angles)} projections, shape: {projections.shape}")

    return {
        'projections': projections,
        'angles': angles,
        'bright_field': bright_field,
        'dark_field': dark_field,
        'xml_header': header,
        'n_files': n_files,
        'total_angle': total_angle_deg,
    }


def build_geometry(xml_header, fov_xy, fov_z, voxel_xy, voxel_z):
    """
    Construct geometry dict from XML header fields and FOV/voxel parameters.

    Args:
        xml_header: Parsed scan.xml dict (from xmltodict)
        fov_xy: Field of view in the xy plane in mm
        fov_z: Field of view in the z direction in mm
        voxel_xy: Voxel size in the xy plane in mm
        voxel_z: Voxel size in the z direction in mm

    Returns:
        geometry dict with keys: R_d, R_s, da, db, vol_shape, vol_origin,
        dx, dz, central_pixel_a, central_pixel_b
    """
    source_to_isocenter = float(xml_header['Series']['ObjectPosition'])
    detector_to_isocenter = float(xml_header['Series']['DetectorPosition']) - source_to_isocenter

    # Compute volume shape from FOV and voxel size
    Nxy = round(fov_xy / voxel_xy)
    Nz = round(fov_z / voxel_z)

    geometry = {
        'R_d': detector_to_isocenter,
        'R_s': source_to_isocenter,
        'da': float(xml_header['Series']['DetectorSpacing']),
        'db': float(xml_header['Series']['DetectorSpacing']),
        'vol_shape': (Nxy, Nxy, Nz),
        'vol_origin': (0, 0, 0),
        'dx': voxel_xy,
        'dz': voxel_z,
        'central_pixel_a': float(xml_header['Series']['CentreOfRotation']),
        'central_pixel_b': float(xml_header['Series']['CentralSlice'])
    }

    print(f"\nGeometry:")
    print(f"  Source-to-isocenter: {geometry['R_s']:.2f} mm")
    print(f"  Detector-to-isocenter: {geometry['R_d']:.2f} mm")
    print(f"  Detector pixel size: {geometry['da']:.4f} mm")
    print(f"  Volume shape: {geometry['vol_shape']}")
    print(f"  Voxel size (xy): {geometry['dx']:.4f} mm")
    print(f"  Voxel size (z): {geometry['dz']:.4f} mm")

    return geometry


def postprocess_and_save(volume, geometry, output_path, bilateral_filter=False,
                         bilateral_sigma_spatial=1.5, bilateral_sigma_range=50.0,
                         voxel_xy=0.075, roi_config=None, cal_z_range=None,
                         cal_degree=2, cal_plot_path=None,
                         calibration_method=None):
    """
    Apply polynomial HU calibration, optional bilateral filter, and save as VFF.

    Calibration modes (in priority order):
      1. Self-calibration (roi_config + cal_z_range provided):
         Measures phantom inserts in THIS volume and fits a polynomial to
         its own scale. Use for phantom scans to derive new coefficients.
      2. Stored per-method coefficients (calibration_method provided):
         Uses pre-fitted polynomial from CALIBRATION_COEFFICIENTS dict.
         Use for non-phantom scans (e.g., mouse data).
      3. Legacy fallback (neither provided):
         Uses PHANTOM_CALIBRATION fitted from FDK physical normalization.
         Only accurate for FDK with physical_normalization=True.

    Args:
        volume: Reconstructed volume as numpy array (x, y, z) in uncalibrated HU
        geometry: Geometry dict (needs 'dx', 'dz' for VFF spacing)
        output_path: Base output path (without .vff extension)
        bilateral_filter: Whether to apply bilateral filter
        bilateral_sigma_spatial: Bilateral filter spatial sigma in mm
        bilateral_sigma_range: Bilateral filter intensity sigma in HU
        voxel_xy: Voxel size in xy plane in mm (for bilateral filter conversion)
        roi_config: Path to JSON file with phantom insert ROI definitions,
                    or list of insert dicts. Enables self-calibration mode.
        cal_z_range: (z_start, z_end) slice range for phantom insert measurements.
                     Required when roi_config is provided.
        cal_degree: Polynomial degree for calibration fit (default: 2)
        cal_plot_path: If set, save calibration diagnostic plot to this path
                       (without extension). Only used in self-calibration mode.
        calibration_method: Method key for stored coefficients (e.g., 'fdk',
                           'astra_sirt', 'tigre_ossart'). Used when roi_config
                           is not provided to select the correct per-method
                           polynomial for non-phantom scans.

    Returns:
        Path to saved VFF file
    """
    import json

    # Extract volume as numpy (x, y, z)
    vol_np = volume.cpu().numpy() if hasattr(volume, 'cpu') else volume

    print("\n" + "=" * 60)
    print("Applying polynomial HU calibration")
    print("=" * 60)

    if roi_config is not None:
        # --- Mode 1: Self-calibration from phantom inserts ---
        if cal_z_range is None:
            raise ValueError("cal_z_range is required when roi_config is provided")

        # Load ROI config if it's a file path
        if isinstance(roi_config, (str, Path)):
            print(f"  Loading ROI config: {roi_config}")
            with open(roi_config) as f:
                insert_rois = json.load(f)["inserts"]
        else:
            insert_rois = roi_config

        # The volume is (x, y, z) but measure_insert_rois expects (z, y, x)
        vol_zyx = vol_np.transpose(2, 1, 0)  # (x, y, z) -> (z, y, x)

        # ROI coordinates are defined in VFF viewer space (y-flipped).
        # Create a y-flipped view for measurement and plotting.
        vol_zyx_vff = vol_zyx[:, ::-1, :]

        print(f"  Self-calibration: measuring {len(insert_rois)} inserts "
              f"at z={cal_z_range[0]}-{cal_z_range[1]}")
        measurements = measure_insert_rois(vol_zyx_vff, insert_rois, cal_z_range)

        print(f"\n  {'Name':<18s}  {'Measured':>10s}  {'Std':>7s}  {'True HU':>8s}")
        print("  " + "-" * 50)
        for m in measurements:
            print(f"  {m['name']:<18s}  {m['measured_mean']:10.1f}  "
                  f"{m['measured_std']:7.1f}  {m['true_hu']:8d}")

        # Build calibration pairs and fit — apply to original (un-flipped) volume
        cal_data = np.array([[m['measured_mean'], m['true_hu']]
                             for m in measurements])
        vol_calibrated, coeffs, rms = calibrate_volume_polynomial(
            vol_zyx, cal_data, degree=cal_degree)

        print(f"\n  Degree-{cal_degree} polynomial coefficients: {coeffs}")
        print(f"  RMS residual: {rms:.1f} HU")
        print(f"\n  To store these coefficients, add to CALIBRATION_COEFFICIENTS in calibration.py:")
        method_key = calibration_method or "<method>"
        print(f'    "{method_key}": {coeffs.tolist()},')

        # Diagnostic plot (use VFF-oriented slice so circles match inserts)
        if cal_plot_path:
            sl = vol_zyx_vff[cal_z_range[0]:cal_z_range[1]].mean(axis=0)
            plot_calibration_diagnostic(measurements, coeffs, sl, cal_plot_path)

        # Transpose back to (x, y, z)
        vol_calibrated = vol_calibrated.transpose(2, 1, 0)

    elif calibration_method is not None:
        # --- Mode 2: Stored per-method coefficients ---
        coeffs = get_calibration_coefficients(calibration_method)
        if coeffs is not None:
            print(f"  Mode: stored coefficients for '{calibration_method}'")
            print(f"  Polynomial coefficients: {coeffs}")
            vol_calibrated = np.polyval(coeffs, vol_np).astype(np.float32)
            vol_calibrated = np.clip(vol_calibrated, -1024, 4095)
        else:
            print(f"  WARNING: No stored coefficients for '{calibration_method}'.")
            print(f"  Run a phantom scan with --roi-config first to derive them.")
            print(f"  Falling back to legacy PHANTOM_CALIBRATION (FDK-only).")
            coeffs, rms, _ = fit_hu_calibration(PHANTOM_CALIBRATION, degree=2)
            print(f"  Degree-2 polynomial coefficients: {coeffs}")
            print(f"  RMS residual: {rms:.1f} HU")
            vol_calibrated = np.polyval(coeffs, vol_np).astype(np.float32)
            vol_calibrated = np.clip(vol_calibrated, -1024, 4095)

    else:
        # --- Mode 3: Legacy fallback (FDK-only) ---
        print("  Mode: legacy PHANTOM_CALIBRATION (FDK physical normalization)")
        coeffs, rms, _ = fit_hu_calibration(PHANTOM_CALIBRATION, degree=2)
        print(f"  Degree-2 polynomial coefficients: {coeffs}")
        print(f"  RMS residual: {rms:.1f} HU")
        vol_calibrated = np.polyval(coeffs, vol_np).astype(np.float32)
        vol_calibrated = np.clip(vol_calibrated, -1024, 4095)

    # Optional bilateral filter (edge-preserving denoising)
    if bilateral_filter:
        print("\n" + "=" * 60)
        print("Applying bilateral filter (edge-preserving denoising)")
        print("=" * 60)

        sigma_spatial_vox = bilateral_sigma_spatial / voxel_xy
        print(f"  Spatial sigma: {bilateral_sigma_spatial:.2f} mm "
              f"= {sigma_spatial_vox:.1f} voxels")
        print(f"  Range sigma: {bilateral_sigma_range:.1f} HU")

        t_bf = time.time()
        Nz_slices = vol_calibrated.shape[2]
        for z in range(Nz_slices):
            vol_calibrated[:, :, z] = cv2.bilateralFilter(
                vol_calibrated[:, :, z],
                d=-1,
                sigmaColor=bilateral_sigma_range,
                sigmaSpace=sigma_spatial_vox,
            )
        t_bf_end = time.time()

        print(f"  Filtered range: [{vol_calibrated.min():.0f}, {vol_calibrated.max():.0f}] HU")
        print(f"  Bilateral filter applied in {t_bf_end - t_bf:.1f}s "
              f"({Nz_slices} slices)")

    # Save uncalibrated VFF (physics-based HU only, no polynomial)
    vff_meta = {
        'bits': 16,
        'spacing': f"{geometry['dx']} {geometry['dx']} {geometry['dz']}",
    }
    uncal_path = output_path + '_uncalibrated.vff'
    vol_uncal_vff = vol_np.astype(np.int16).transpose(2, 1, 0)[:, ::-1, :]
    write_vff(uncal_path, vff_meta, vol_uncal_vff)
    print(f"Uncalibrated VFF saved to: {uncal_path}")

    # Save calibrated VFF using write_vff with transpose/y-flip (matching fdk.py:688)
    cal_path = output_path + ('_bilateral.vff' if bilateral_filter else '.vff')
    vol_vff = vol_calibrated.astype(np.int16).transpose(2, 1, 0)[:, ::-1, :]
    write_vff(cal_path, vff_meta, vol_vff)
    print(f"Calibrated VFF saved to: {cal_path}")

    return cal_path
