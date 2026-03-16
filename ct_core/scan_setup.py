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
from .calibration import load_calibration_fields, MU_WATER_80KV


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


def parse_crop_boundary(scan_folder, xml_header):
    """
    Parse GEHC SubVolumeCoordinates.xml and convert to isocenter-centered ROI bounds.

    The CropBoundary defines a physical bounding box in GEHC's scanner-absolute
    coordinate system. We convert to isocenter-centered coordinates by subtracting
    the LandmarkOffsetVector from scan.xml.

    Args:
        scan_folder: Path to scan folder containing Volumes/SubVolumeCoordinates.xml
        xml_header: Parsed scan.xml dict (for LandmarkOffsetVector)

    Returns:
        dict with keys: x_min, x_max, y_min, y_max, z_min, z_max (mm, isocenter-centered)
        or None if SubVolumeCoordinates.xml not found
    """
    xml_path = os.path.join(scan_folder, 'Volumes', 'SubVolumeCoordinates.xml')
    if not os.path.exists(xml_path):
        return None

    # Parse CropBoundary
    with open(xml_path) as f:
        crop_doc = xmltodict.parse(f.read())
    values = crop_doc['CropBoundary'].split()
    if len(values) != 6:
        print(f"  WARNING: CropBoundary has {len(values)} values, expected 6. Skipping ROI.")
        return None

    x_min, x_max, y_min, y_max, z_min, z_max = [float(v) for v in values]

    # Get LandmarkOffsetVector for coordinate conversion
    try:
        landmark_str = xml_header['Series']['LandmarkOffsetVector']
        lx, ly, lz = [float(v) for v in landmark_str.split()]
    except (KeyError, ValueError) as e:
        print(f"  WARNING: Could not parse LandmarkOffsetVector: {e}. Skipping ROI.")
        return None

    # Convert GEHC scanner coordinates to our reconstruction coordinates.
    # The CropBoundary x-axis is negated relative to our FDK/ASTRA/TIGRE
    # x-axis (confirmed by comparing tissue centering with GEHC output).
    # y and z are shifted by the LandmarkOffsetVector.
    roi = {
        'x_min': -(x_max - lx),
        'x_max': -(x_min - lx),
        'y_min': y_min - ly,
        'y_max': y_max - ly,
        'z_min': z_min - lz,
        'z_max': z_max - lz,
    }

    print(f"\nROI from SubVolumeCoordinates.xml:")
    print(f"  GEHC CropBoundary: x=[{x_min:.2f}, {x_max:.2f}], "
          f"y=[{y_min:.2f}, {y_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}] mm")
    print(f"  LandmarkOffsetVector: ({lx:.2f}, {ly:.2f}, {lz:.2f}) mm")
    print(f"  Isocenter-centered: x=[{roi['x_min']:.2f}, {roi['x_max']:.2f}], "
          f"y=[{roi['y_min']:.2f}, {roi['y_max']:.2f}], "
          f"z=[{roi['z_min']:.2f}, {roi['z_max']:.2f}] mm")
    print(f"  ROI size: {roi['x_max']-roi['x_min']:.1f} x "
          f"{roi['y_max']-roi['y_min']:.1f} x "
          f"{roi['z_max']-roi['z_min']:.1f} mm")

    return roi


def build_geometry(xml_header, fov_xy, fov_z, voxel_xy, voxel_z, roi_bounds=None):
    """
    Construct geometry dict from XML header fields and FOV/voxel parameters.

    Args:
        xml_header: Parsed scan.xml dict (from xmltodict)
        fov_xy: Field of view in the xy plane in mm
        fov_z: Field of view in the z direction in mm
        voxel_xy: Voxel size in the xy plane in mm
        voxel_z: Voxel size in the z direction in mm
        roi_bounds: Optional dict with x_min/x_max/y_min/y_max/z_min/z_max (mm,
                    isocenter-centered) from parse_crop_boundary(). When provided,
                    fov_xy and fov_z are ignored and the volume is sized/positioned
                    to match the ROI.

    Returns:
        geometry dict with keys: R_d, R_s, da, db, vol_shape, vol_origin,
        dx, dz, central_pixel_a, central_pixel_b
    """
    source_to_isocenter = float(xml_header['Series']['ObjectPosition'])
    detector_to_isocenter = float(xml_header['Series']['DetectorPosition']) - source_to_isocenter

    if roi_bounds is not None:
        # ROI-based reconstruction: volume sized and positioned from ROI bounds
        Nx = round((roi_bounds['x_max'] - roi_bounds['x_min']) / voxel_xy)
        Ny = round((roi_bounds['y_max'] - roi_bounds['y_min']) / voxel_xy)
        Nz = round((roi_bounds['z_max'] - roi_bounds['z_min']) / voxel_z)
        vol_shape = (Nx, Ny, Nz)
        vol_origin = (
            (roi_bounds['x_min'] + roi_bounds['x_max']) / 2,
            (roi_bounds['y_min'] + roi_bounds['y_max']) / 2,
            (roi_bounds['z_min'] + roi_bounds['z_max']) / 2,
        )
    else:
        # Standard full-FOV reconstruction centered at isocenter
        Nxy = round(fov_xy / voxel_xy)
        Nz = round(fov_z / voxel_z)
        vol_shape = (Nxy, Nxy, Nz)
        vol_origin = (0, 0, 0)

    geometry = {
        'R_d': detector_to_isocenter,
        'R_s': source_to_isocenter,
        'da': float(xml_header['Series']['DetectorSpacing']),
        'db': float(xml_header['Series']['DetectorSpacing']),
        'vol_shape': vol_shape,
        'vol_origin': vol_origin,
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
    print(f"  Volume origin: ({vol_origin[0]:.2f}, {vol_origin[1]:.2f}, {vol_origin[2]:.2f}) mm")
    print(f"  Voxel size (xy): {geometry['dx']:.4f} mm")
    print(f"  Voxel size (z): {geometry['dz']:.4f} mm")

    return geometry


def postprocess_and_save(volume, geometry, output_path, bilateral_filter=False,
                         bilateral_sigma_spatial=1.5, bilateral_sigma_range=50.0,
                         voxel_xy=0.075):
    """
    Apply two-point HU calibration, optional bilateral filter, and save as VFF.

    Two-point calibration measures air and water/tissue from the volume itself,
    then applies the standard CT HU formula:
        HU = (raw - water) / (water - air) × 1000
    Self-calibrating — works regardless of BHC, filter, or normalization.

    Args:
        volume: Reconstructed volume as numpy array (x, y, z) in uncalibrated HU
        geometry: Geometry dict (needs 'dx', 'dz' for VFF spacing)
        output_path: Base output path (without .vff extension)
        bilateral_filter: Whether to apply bilateral filter
        bilateral_sigma_spatial: Bilateral filter spatial sigma in mm
        bilateral_sigma_range: Bilateral filter intensity sigma in HU
        voxel_xy: Voxel size in xy plane in mm (for bilateral filter conversion)

    Returns:
        Path to saved VFF file
    """
    # Extract volume as numpy (x, y, z)
    vol_np = volume.cpu().numpy() if hasattr(volume, 'cpu') else volume

    print("\n" + "=" * 60)
    print("Applying two-point HU calibration")
    print("=" * 60)

    if True:
        # --- Mode 2: Two-point linear calibration (standard CT formula) ---
        # Measures air and water/tissue peaks from the volume histogram,
        # then maps: HU = (raw - water_peak) / (water_peak - air_peak) * 1000
        # This is the standard CT HU definition and works regardless of
        # BHC, filter, or normalization settings.
        print("  Mode: two-point linear calibration (air/water from histogram)")

        # --- Measure air value: median of voxels below -500 HU ---
        air_voxels = vol_np[vol_np < -500.0]
        if len(air_voxels) > 0:
            hu_air = float(np.median(air_voxels))
        else:
            print("  WARNING: no air voxels found — using -1000")
            hu_air = -1000.0

        # --- Measure water/tissue value from central ROI ---
        # Small central ROI (30×30 voxels ≈ 2.25 mm), only z-slices
        # where the center is inside the object (not air).
        # Volume is (x, y, z).
        Nx, Ny, Nz = vol_np.shape
        cx, cy = Nx // 2, Ny // 2
        h = min(15, Nx // 10, Ny // 10)
        center_line = vol_np[cx - h:cx + h, cy - h:cy + h, :]
        z_means = center_line.mean(axis=(0, 1))
        inside_mask = z_means > -500.0  # z-slices where center is object
        n_inside = int(inside_mask.sum())

        if n_inside > 10:
            center_inside = center_line[:, :, inside_mask]
            p25, p75 = np.percentile(center_inside, [25, 75])
            water_voxels = center_inside[
                (center_inside > p25) & (center_inside < p75)]
            hu_water = float(np.median(water_voxels))
            print(f"  Water/tissue ROI: {n_inside} z-slices inside object")
        else:
            print("  WARNING: could not segment inside/outside — using 0")
            hu_water = 0.0

        print(f"  Air value:   {hu_air:.1f} HU")
        print(f"  Water value: {hu_water:.1f} HU")

        if abs(hu_water - hu_air) < 1.0:
            print("  WARNING: air and water peaks are too close — "
                  "falling back to literature values")
            hu_air = -1000.0
            hu_water = 0.0

        # Standard CT HU formula: water → 0, air → -1000
        scale = 1000.0 / (hu_water - hu_air)
        vol_calibrated = ((vol_np - hu_water) * scale).astype(np.float32)
        vol_calibrated = np.clip(vol_calibrated, -1024, 4095)

        # Verify calibration
        print(f"  Scale factor: {scale:.4f}")
        print(f"  Verification: air ({hu_air:.1f}) → "
              f"{(hu_air - hu_water) * scale:.0f} HU")
        print(f"  Verification: water ({hu_water:.1f}) → "
              f"{(hu_water - hu_water) * scale:.0f} HU")

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
