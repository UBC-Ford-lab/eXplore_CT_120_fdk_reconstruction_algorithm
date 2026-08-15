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
from .calibration import load_calibration_fields
from .hu_calibration import (
    TISSUE_HU_DEFAULT,
    find_attenuation_anchors,
    fixed_anchors,
    format_calibration,
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


def load_scan_data(data_folder, scan_folder, projection_pattern, total_angle,
                   sub_scan='-00-'):
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
    if sub_scan:
        proj_paths = [p for p in proj_paths if p.name.startswith('proj-') or sub_scan in str(p)]
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
        sub_scan=sub_scan or '-00-',
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


def build_geometry(xml_header, fov_xy, fov_z, voxel_xy, voxel_z, roi_bounds=None,
                   verbose=True):
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
        'central_pixel_b': float(xml_header['Series']['CentralSlice']),
        # Sub-box of vol_shape to write out, or None for the whole grid. Set
        # by the drivers that separate the RECONSTRUCTION domain (which must
        # cover every attenuating thing the rays cross — see ct_core.support)
        # from the region worth saving. FDK leaves it None because its ROI
        # already IS its grid.
        'export_roi': None,
    }

    if not verbose:
        return geometry

    print(f"\nGeometry:")
    print(f"  Source-to-isocenter: {geometry['R_s']:.2f} mm")
    print(f"  Detector-to-isocenter: {geometry['R_d']:.2f} mm")
    print(f"  Detector pixel size: {geometry['da']:.4f} mm")
    print(f"  Volume shape: {geometry['vol_shape']}")
    print(f"  Volume origin: ({vol_origin[0]:.2f}, {vol_origin[1]:.2f}, {vol_origin[2]:.2f}) mm")
    print(f"  Voxel size (xy): {geometry['dx']:.4f} mm")
    print(f"  Voxel size (z): {geometry['dz']:.4f} mm")

    return geometry


def _hist_mode(values, bins=256, clip_pct=(0.5, 99.5)):
    """Robust mode (histogram peak) of a 1-D population.

    Histograms over the central `clip_pct` percentile range so a handful of
    extreme voxels can't create a spurious peak, takes the tallest bin, and
    refines to sub-bin precision with a parabolic fit to the peak and its two
    neighbours. Returns None on empty input; falls back to the median if the
    range is degenerate.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, clip_pct)
    if not (hi > lo):
        return float(np.median(v))
    hist, edges = np.histogram(v, bins=bins, range=(float(lo), float(hi)))
    if hist.max() == 0:
        return float(np.median(v))
    k = int(np.argmax(hist))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if 0 < k < len(hist) - 1:
        y0, y1, y2 = float(hist[k - 1]), float(hist[k]), float(hist[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = 0.5 * (y0 - y2) / denom
            delta = max(-0.5, min(0.5, delta))
            return float(centers[k] + delta * (centers[1] - centers[0]))
    return float(centers[k])


def postprocess_and_save(volume_mu, geometry, output_path,
                         bilateral_filter=False,
                         bilateral_sigma_spatial=1.5, bilateral_sigma_range=50.0,
                         voxel_xy=0.075,
                         hu_calibration='auto', mu_water=None,
                         tissue_hu=None, save_mu=False, anchors=None):
    """Calibrate a reconstructed volume to HU, optionally filter it, write VFF.

    ``volume_mu`` is linear attenuation (mm^-1) straight out of a backend:
    unclipped, unconverted. Every backend now returns exactly that, so this is
    the single place in the package where the HU scale is decided.

    Calibration fits BOTH degrees of freedom of the affine error a
    reconstruction carries (gain and offset) against two facts that hold for
    every scan — air is zero attenuation, and the bulk of what we image is
    soft tissue. See ct_core.hu_calibration for why that is a two-anchor
    problem and why the anchors are found from the histogram's shape rather
    than from thresholds or a fixed central box.

    What this replaced, and why:
      * a one-point map with a hardcoded scanner constant (mu_water = 0.0219),
        empirically back-fitted to a single 2022 scan. It fits the gain only,
        assumes zero offset, and does not transfer: measured on finished
        volumes, the same constant put soft tissue at -69 HU on one scan and
        -297 HU on another.
      * a clip to [-1024, 4095] applied before anything could measure the
        volume. On real reconstructions that pinned 19-44 % of all voxels onto
        exactly the floor, destroying the air peak — the one anchor that is
        physically exact — and saturating the bone tail at the top.
      * a two-point fallback that read its water anchor from a fixed 30x30
        central box, which in a mouse is lung. It was disabled by default,
        so in practice nothing was ever calibrated.

    Args:
        volume_mu: (x, y, z) attenuation array (numpy or torch).
        geometry: geometry dict; 'dx'/'dz' set the VFF voxel size.
        output_path: base path, no extension.
        bilateral_filter: apply an edge-preserving denoise after calibration.
        bilateral_sigma_spatial: bilateral spatial sigma in mm.
        bilateral_sigma_range: bilateral intensity sigma in HU.
        voxel_xy: voxel size in mm, for the bilateral sigma conversion.
        hu_calibration: 'auto' fits both anchors from this volume's histogram;
            'fixed' pins the gain to ``mu_water`` and air to zero attenuation,
            reproducing the classical one-point map through the same code path.
        mu_water: attenuation of water (mm^-1); required by 'fixed'.
        tissue_hu: where the bulk-tissue anchor should land. Default 120 HU,
            matching the vendor's scale for the same specimens; pass 0 for a
            water phantom. This single number IS the gain assumption — see
            hu_calibration.
        save_mu: also write the uncalibrated attenuation as <output>_mu.npy.
            This replaces the old '_uncalibrated.vff', which was written in the
            same already-converted, already-clipped units as the calibrated
            file and was therefore byte-identical to it.
        anchors: a pre-fitted HUAnchors to apply instead of fitting. Use it to
            put several volumes on one common scale.

    Returns:
        (path to the written VFF, the HUAnchors that were applied, the
        calibrated HU volume). The volume comes back so callers report and
        plot the same numbers they shipped, rather than re-deriving them or —
        as the drivers previously did — plotting the pre-calibration array.
    """
    vol_np = volume_mu.cpu().numpy() if hasattr(volume_mu, 'cpu') else volume_mu
    vol_np = np.asarray(vol_np, dtype=np.float32)

    print("\n" + "=" * 60)
    print("HU calibration")
    print("=" * 60)

    if anchors is None:
        if hu_calibration == 'fixed':
            if mu_water is None:
                raise ValueError(
                    "hu_calibration='fixed' needs an explicit mu_water")
            # tissue_hu deliberately does NOT apply here: this mode's second
            # anchor is water at 0 HU by definition, not a measured tissue
            # peak. Say so rather than ignoring the flag in silence.
            if tissue_hu is not None:
                print(f"  NOTE: --tissue-hu {float(tissue_hu):.0f} ignored — "
                      f"it places the measured bulk-tissue peak, and "
                      f"hu_calibration='fixed' anchors WATER (0 HU) via "
                      f"mu_water instead.")
            anchors = fixed_anchors(float(mu_water))
        elif hu_calibration == 'auto':
            anchors = find_attenuation_anchors(
                vol_np,
                tissue_hu=(TISSUE_HU_DEFAULT if tissue_hu is None
                           else float(tissue_hu)))
        else:
            raise ValueError(
                f"unknown hu_calibration mode {hu_calibration!r} "
                f"(expected 'auto' or 'fixed')")

    print(format_calibration(anchors))
    vol_calibrated = anchors.apply(vol_np)

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

    # elementsize is the GE `ncaa` scalar voxel size (mm); spacing is kept for
    # the anisotropy warning in write_vff (dz != dx cannot be expressed).
    vff_meta = {
        'bits': 16,
        'elementsize': geometry['dx'],
        'spacing': f"{geometry['dx']} {geometry['dx']} {geometry['dz']}",
    }

    # The uncalibrated companion is the ATTENUATION, as float32. Writing it as
    # a VFF was pointless: int16 cannot represent mu ~ 0.02 mm^-1 at all, and
    # because the old code had already converted and clipped before this
    # point, the "_uncalibrated.vff" it produced was byte-identical to the
    # calibrated one whenever calibration was skipped — which was the default.
    if save_mu:
        mu_path = output_path + '_mu.npy'
        np.save(mu_path, vol_np)
        print(f"Uncalibrated attenuation (float32 mu, mm^-1) saved to: {mu_path}")

    # Hand write_vff the float HU volume and let it round and clip: it rounds
    # to nearest (astype truncates toward zero, a systematic +0.5 HU bias on
    # the air floor) and clips to the full int16 range with a warning, rather
    # than wrapping. Pre-casting here with .astype(np.int16), as this used to,
    # bypassed both protections.
    cal_path = output_path + ('_bilateral.vff' if bilateral_filter else '.vff')
    vol_vff = vol_calibrated.transpose(2, 1, 0)[:, ::-1, :]
    write_vff(cal_path, vff_meta, vol_vff)
    print(f"Calibrated VFF saved to: {cal_path}")

    return cal_path, anchors, vol_calibrated
