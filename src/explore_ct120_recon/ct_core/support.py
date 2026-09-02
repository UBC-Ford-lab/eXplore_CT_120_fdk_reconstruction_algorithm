"""Measure the region of space that actually attenuates, from the projections.

WHY THIS EXISTS. A detector pixel measures the line integral of mu along the
WHOLE ray — through the animal, the bed, the cage, everything. A method that
FITS that measurement (SIRT, OS-SART, the learned backends) can only reproduce
it if its reconstruction volume covers every attenuating thing the ray crossed.
Reconstruct a smaller box and the missing attenuation does not disappear: the
solver is forced to explain it with the voxels it does have, i.e.

    measured   p = A_dom x_dom + A_out x_out
    the model      A_dom x
    least squares  x = x_true + A_dom^+ (A_out x_out)

and that second term — "reconstruct the cage, but you may only put it inside
the box" — is a smooth, low-frequency cup across the ENTIRE reconstruction,
not a boundary artifact. Measured on Scan_1510 it is ~96 HU of DC bias, with
individual rays demanding 400-480 HU. Edges survive it; the HU scale does not.

FDK is exempt: it filters the full-width projections and then backprojects
into whatever grid you ask for, so its ROI is purely an output crop and never
enters a forward model. That asymmetry is why the auto domain is on by default
for the iterative and learned backends only.

WHAT IT MEASURES. The outermost detector channel that sees anything above the
air noise floor, converted to millimetres at isocentre. That bounds the object
support: matter further out than that would have cast a shadow there. Two
independent bounds are applied on top:

  * the DETECTOR REACH — nothing outside the fan can ever be measured, so the
    domain is clamped to what the detector sees;
  * the CONE REACH in z — no ray reaches beyond the cone, so voxels past it
    can never receive a gradient. They are pure dead weight (on Scan_1510's
    default 120 mm fov_z, 347 M of 576 M parameters).

The domain is centred on the rotation axis in xy (the object turns about it,
so its shadow is symmetric there) but is allowed to be asymmetric in z, where
nothing forces symmetry and the saving is real.
"""

from __future__ import annotations

import numpy as np

# Absolute floor on the detection threshold, in line-integral units. Below
# this we are inside the flat-field's own residual structure (Scan_1510's
# ~2% gain drift shows up as |L| ~ 0.01-0.03 on genuinely empty channels).
MIN_THRESHOLD = 0.02
# Noise multiple above the air baseline, on top of the floor.
NOISE_SIGMAS = 6.0
# Detector channels (as a fraction of the width) assumed to be air, used to
# set the baseline and its scatter.
AIR_EDGE_FRAC = 0.02
# Median-filter width (channels) applied to the profile before thresholding,
# so a single hot pixel cannot define the support.
SMOOTH_CHANNELS = 9
# Percentile taken across the other detector axis. High enough to catch a
# small dense object anywhere in the column, low enough to reject outliers.
PROFILE_PERCENTILE = 98.0


def _line_integrals(frame, bright, dark):
    """-log transmission for one frame, or the frame itself if pre-processed."""
    frame = np.asarray(frame, dtype=np.float32)
    if bright is None or dark is None:
        # Already line integrals (the drivers print the same assumption).
        return frame
    num = np.clip(frame - dark, 1e-3, None)
    den = np.clip(bright - dark, 1e-3, None)
    return -np.log(np.clip(num / den, 1e-6, None))


def _smooth(profile, width=SMOOTH_CHANNELS):
    """Running median — kills hot pixels without moving a real edge."""
    w = int(max(1, width) | 1)                     # force odd
    if w == 1 or profile.size < w:
        return profile
    pad = w // 2
    padded = np.pad(profile, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, w)
    return np.median(view, axis=-1)


def _support_extent(profile):
    """(lo_index, hi_index, threshold) of the channels above the air floor.

    Returns (None, None, threshold) when the profile is entirely air.
    """
    n = profile.size
    edge = max(3, int(round(n * AIR_EDGE_FRAC)))
    air = np.concatenate([profile[:edge], profile[-edge:]])
    base = float(np.median(air))
    mad = float(np.median(np.abs(air - base)))
    sigma = 1.4826 * mad
    threshold = base + max(MIN_THRESHOLD, NOISE_SIGMAS * sigma)

    hit = np.flatnonzero(profile > threshold)
    if hit.size == 0:
        return None, None, threshold
    return int(hit[0]), int(hit[-1]), threshold


def measure_attenuating_support(projections, geometry, *, bright_field=None,
                                dark_field=None, n_angles=16, margin_mm=1.0,
                                verbose=True) -> dict:
    """Bound the attenuating object from the projections themselves.

    Args:
        projections: (N_angles, N_b, N_a) raw counts, or line integrals when
            bright_field/dark_field are None. Must already be downsampled and
            geometry must already carry the matching da/db/central pixels.
        geometry: ct_core build_geometry dict (R_s, R_d, da, db, central_pixel_*).
        n_angles: how many views to sample. The support is the union over them,
            so a handful is plenty — the object is rigid.
        margin_mm: safety margin added to every measured extent, at isocentre.

    Returns a dict of millimetre extents at isocentre plus the diagnostics
    needed to explain the choice in a log line.
    """
    proj = np.asarray(projections)
    n_ang, n_b, n_a = proj.shape

    R_s = float(geometry["R_s"])
    SDD = R_s + float(geometry["R_d"])
    da, db = float(geometry["da"]), float(geometry["db"])
    ca, cb = float(geometry["central_pixel_a"]), float(geometry["central_pixel_b"])

    # Millimetres at the DETECTOR for each channel index.
    a_det = (np.arange(n_a) - ca) * da
    b_det = (np.arange(n_b) - cb) * db

    # A ray through a point at axial distance u from the source hits the
    # detector magnified by SDD/u. Undoing that at isocentre (u = R_s) is the
    # transaxial map; for z we want the WORST case over the domain, applied
    # after the radius is known.
    to_iso_xy = R_s / SDD

    idx = np.unique(np.linspace(0, n_ang - 1, min(int(n_angles), n_ang)).astype(int))
    a_lo_mm, a_hi_mm, b_lo_mm, b_hi_mm = [], [], [], []
    # Kept per axis: when one axis is object-saturated its baseline is drawn
    # from object, not air, and its threshold is meaningless — merging the two
    # would report that nonsense as if it governed both.
    thr_xy, thr_z = [], []

    for i in idx:
        L = _line_integrals(proj[i], bright_field, dark_field)
        col = _smooth(np.percentile(L, PROFILE_PERCENTILE, axis=0))
        row = _smooth(np.percentile(L, PROFILE_PERCENTILE, axis=1))

        c0, c1, t_c = _support_extent(col)
        r0, r1, t_r = _support_extent(row)
        thr_xy.append(t_c); thr_z.append(t_r)
        if c0 is not None:
            a_lo_mm.append(a_det[c0]); a_hi_mm.append(a_det[c1])
        if r0 is not None:
            b_lo_mm.append(b_det[r0]); b_hi_mm.append(b_det[r1])

    # Hard limits from the hardware, independent of what is in the field.
    detector_reach_xy = float(max(abs(a_det[0]), abs(a_det[-1])) * to_iso_xy)
    b_half_det = float(max(abs(b_det[0]), abs(b_det[-1])))

    # Each axis falls back independently to the FULL detector extent. An axis
    # with no detected edge is ambiguous — either nothing is there, or the
    # object runs off both ends and the "air" channels used for the baseline
    # are themselves object (which is the normal case in z: the bed and cage
    # are longer than the cone). Both readings are safe to answer the same
    # way, since a domain is only ever wrong for being too SMALL.
    saturated_xy = not a_lo_mm
    saturated_z = not b_lo_mm

    if saturated_xy:
        radius_xy = detector_reach_xy
    else:
        # Symmetric in xy about the rotation axis: the object turns, so its
        # shadow sweeps both sides regardless of which side it sits on.
        a_max_det = max(max(abs(v) for v in a_lo_mm), max(abs(v) for v in a_hi_mm))
        radius_xy = min(a_max_det * to_iso_xy + margin_mm, detector_reach_xy)

    z_lo_det, z_hi_det = ((-b_half_det, b_half_det) if saturated_z
                          else (min(b_lo_mm), max(b_hi_mm)))

    # z magnification is worst at the FAR edge of the domain, not at
    # isocentre — a voxel there is closer to the detector and so is reached by
    # rays from further up the cone.
    to_iso_z = (R_s + radius_xy) / SDD
    cone_reach_z = b_half_det * to_iso_z
    z_min = max(-cone_reach_z, z_lo_det * to_iso_z - margin_mm)
    z_max = min(cone_reach_z, z_hi_det * to_iso_z + margin_mm)

    support = {
        "radius_xy": float(radius_xy),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "detector_reach_xy": detector_reach_xy,
        "cone_reach_z": float(cone_reach_z),
        "threshold_xy": float(np.median(thr_xy)) if thr_xy else 0.0,
        "threshold_z": float(np.median(thr_z)) if thr_z else 0.0,
        "angles_sampled": int(idx.size),
        "margin_mm": float(margin_mm),
        "saturated_xy": bool(saturated_xy),
        "saturated_z": bool(saturated_z),
    }

    if verbose:
        print("\nAttenuating support measured from the projections:")
        print(f"  {support['angles_sampled']} views sampled")
        why_xy = ("no air channel at either edge — using the full fan"
                  if saturated_xy else
                  f"outermost shadow {radius_xy - margin_mm:.2f} mm "
                  f"+ {margin_mm:.1f} mm margin, threshold "
                  f"{support['threshold_xy']:.4f}")
        print(f"  transaxial: radius {radius_xy:.2f} mm at isocentre "
              f"({why_xy}; detector reaches {detector_reach_xy:.2f} mm)")
        why_z = ("object runs past both ends — cone-limited"
                 if saturated_z else "object-limited")
        print(f"  axial:      z in [{z_min:.2f}, {z_max:.2f}] mm "
              f"({why_z}; cone reaches +-{cone_reach_z:.2f} mm)")
    return support


def support_to_domain_bounds(support: dict) -> dict:
    """Support -> the bounds dict `build_geometry` takes for `roi_bounds`."""
    r = float(support["radius_xy"])
    return {"x_min": -r, "x_max": r, "y_min": -r, "y_max": r,
            "z_min": float(support["z_min"]), "z_max": float(support["z_max"])}


def explicit_domain_bounds(extent_xy: float, half_z: float) -> dict:
    """Bounds for a hand-specified domain, in muNeRF's config units.

    `extent_xy` is the full width (muNeRF's `model_domain.extent_xy`), and
    `half_z` its `half_extent_z`. Scan_1510's validated pair is 88.0 / 29.0.
    """
    r = float(extent_xy) / 2.0
    h = float(half_z)
    return {"x_min": -r, "x_max": r, "y_min": -r, "y_max": r,
            "z_min": -h, "z_max": h}


def crop_bounds_to_indices(bounds: dict, geometry: dict):
    """Index slices of `geometry`'s grid covering `bounds`.

    Voxel i spans [edge + i*d, edge + (i+1)*d] with edge = origin - N*d/2, so
    the covering index range is floor/ceil of the bound offsets. Returns
    ((i0, i1), (j0, j1), (k0, k1)) clamped to the grid, each a half-open range.
    """
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    ox, oy, oz = (float(v) for v in geometry["vol_origin"])
    dx, dz = float(geometry["dx"]), float(geometry["dz"])

    def span(lo, hi, n, origin, d):
        edge = origin - n * d / 2.0
        i0 = int(np.floor((lo - edge) / d))
        i1 = int(np.ceil((hi - edge) / d))
        i0 = max(0, min(n, i0))
        i1 = max(i0 + 1, min(n, i1))
        return i0, i1

    return (span(bounds["x_min"], bounds["x_max"], Nx, ox, dx),
            span(bounds["y_min"], bounds["y_max"], Ny, oy, dx),
            span(bounds["z_min"], bounds["z_max"], Nz, oz, dz))


def export_grid_geometry(geometry: dict, verbose: bool = False) -> dict:
    """The geometry of `geometry['export_roi']` as a sub-grid — no array needed.

    The lattice arithmetic of the export crop, separated from the cropping, so
    that a caller which can EVALUATE its volume at arbitrary points (a learned
    representation) can render the exported voxels directly instead of
    materialising the whole reconstruction domain and throwing most of it away.
    The two must agree voxel for voxel, which is why they share this function
    rather than each deriving a grid from the ROI bounds: a grid built from the
    bounds alone is centred on the ROI, while the crop lands on the DOMAIN
    lattice, and the two are a sub-voxel shift apart.

    `export_roi` is cleared on the way out, exactly as `crop_to_export_roi`
    clears it: the crop is consumed, so a later call is a no-op.
    """
    bounds = geometry.get("export_roi")
    if not bounds:
        return geometry

    (i0, i1), (j0, j1), (k0, k1) = crop_bounds_to_indices(bounds, geometry)
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    ox, oy, oz = (float(v) for v in geometry["vol_origin"])
    dx, dz = float(geometry["dx"]), float(geometry["dz"])

    out = dict(geometry)
    out["export_roi"] = None
    out["vol_shape"] = (i1 - i0, j1 - j0, k1 - k0)
    out["vol_origin"] = (
        (ox - Nx * dx / 2.0) + (i0 + (i1 - i0) / 2.0) * dx,
        (oy - Ny * dx / 2.0) + (j0 + (j1 - j0) / 2.0) * dx,
        (oz - Nz * dz / 2.0) + (k0 + (k1 - k0) / 2.0) * dz,
    )
    if verbose:
        print(f"\nExport crop: {Nx} x {Ny} x {Nz} -> "
              f"{out['vol_shape'][0]} x {out['vol_shape'][1]} x "
              f"{out['vol_shape'][2]}"
              f"  ({np.prod(out['vol_shape']) / max(1, Nx * Ny * Nz) * 100:.1f}% "
              f"of the reconstruction domain)")
    return out


def crop_to_export_roi(volume, geometry: dict):
    """Crop an (Nx, Ny, Nz) volume to `geometry['export_roi']`.

    Returns `(volume, geometry)` — the inputs untouched when no export ROI is
    set, otherwise the cropped array (contiguous) and a geometry copy whose
    vol_shape/vol_origin describe it, so the VFF header and every downstream
    plot agree with the pixels.
    """
    bounds = geometry.get("export_roi")
    if not bounds:
        return volume, geometry

    (i0, i1), (j0, j1), (k0, k1) = crop_bounds_to_indices(bounds, geometry)
    vol = np.ascontiguousarray(volume[i0:i1, j0:j1, k0:k1])
    return vol, export_grid_geometry(geometry, verbose=True)
