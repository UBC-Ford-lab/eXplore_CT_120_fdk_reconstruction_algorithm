"""Read an ALREADY-RECONSTRUCTED volume back into the pipeline's conventions.

The reconstruction drivers all end at ``save_outputs`` -> a VFF on disk, and
everything downstream of that point (the HU histogram, the three midplane
views, the scrollable slice sequence, the ``volume/*`` summary scalars) is
computed from the volume alone — no projections, no iterations. This module
is the missing inverse: it takes a finished volume from ANY source (this
package's own output, the scanner vendor's reconstruction, a SIRT volume, a
muNeRF export) and hands back exactly what those figure builders expect, so
``run_volume_report`` can produce the same panel for a volume it did not
reconstruct.

Three things have to be undone or supplied, and each is a place to get it
silently wrong:

  * AXIS ORDER — ``scan_setup.postprocess_and_save`` writes
    ``volume.transpose(2, 1, 0)[:, ::-1, :]``, i.e. (x, y, z) -> (z, -y, x).
    ``load_reconstructed_volume`` applies the exact inverse, so a round trip
    of this package's own output is the identity.
  * HU SCALE — the stored integers are HU already. The GE ``ncaa`` header
    also carries ``water``/``air`` anchors and the MicroView display formula
    ``HU = 1000 (v - water) / (water - air)``, but on both this package's
    files (water=0, air=-1000, i.e. the identity) and the vendor's
    (water=4.75, air=0.82) that mapping has ALREADY been applied before
    quantization — on Scan_1988's Half-scan-75um.vff the header's float
    ``min=-34.353283`` maps through it to exactly the stored integer minimum
    -9949. Applying it a second time would scale HU by 254x, so it is off by
    default and available as ``hu_from_header`` for foreign files that really
    do store raw values.
  * GEOMETRY — the GE header keeps one scalar ``elementsize``, so the z voxel
    size and the volume's position in the scanner frame do not survive the
    write. For volumes THIS package wrote, both are read back from the
    ``<volume>.json`` sidecar (``load_sidecar``); for a foreign volume they
    can be supplied by hand, defaulting to the header voxel size and an
    isocentre origin. The header's own ``origin`` field is NOT used — the
    vendor writes detector-frame numbers there
    (``-581.6533 3066.9634 46.4056``), not an isocentre-relative volume
    centre in mm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scan_setup import _hist_mode
from .vff_io import read_vff


def load_sidecar(path, *, verbose: bool = True) -> dict:
    """The ``<volume>.json`` companion written next to a volume, or ``{}``.

    Absent for any volume this package did not write (the vendor's, an older
    run), which is not an error — it just means the geometry has to come from
    the header or the command line.
    """
    side = Path(path).with_suffix('.json')
    if not side.exists():
        return {}
    try:
        record = json.loads(side.read_text())
    except (OSError, ValueError) as e:
        if verbose:
            print(f"  Sidecar {side.name} unreadable ({type(e).__name__}: "
                  f"{e}) — falling back to the header.")
        return {}
    if verbose:
        made_by = record.get('algorithm', 'unknown')
        print(f"  Sidecar {side.name}: {made_by}"
              + (f", scan {record['scan']}" if 'scan' in record else "")
              + (f", {record['created']}" if 'created' in record else ""))
    return record


# --------------------------------------------------------------------- load --

def load_reconstructed_volume(path, *, y_flip: bool = True,
                              flip_axes=(),
                              hu_from_header: bool = False,
                              verbose: bool = True):
    """Load a finished volume as float32 HU in the pipeline's (x, y, z) order.

    Supports ``.vff`` (GE ncaa, the format every driver writes) and ``.npy`` /
    ``.npz`` (assumed to be (x, y, z) already, in HU, with no flip applied).

    ``y_flip`` undoes the y reversal this package's VFF writer applies; turn
    it off for a foreign file that is mirrored the other way. ``hu_from_header``
    applies the header's water/air anchors — see the module docstring for why
    that is normally wrong.

    ``flip_axes`` reverses whole axes AFTER the load, in the final (x, y, z)
    order — for a foreign file whose axis conventions differ from this
    package's. It is separate from ``y_flip`` on purpose: ``y_flip`` inverts
    OUR OWN writer, while this describes THEIR convention.

    MEASURED 2026-08-19 on Scan_1510's vendor volume: it needs ``('x',)``.
    Scoring all eight sign combinations against a trusted reconstruction
    resampled onto the same lattice gave -x+y+z at NCC 0.899, against 0.788 for
    +x+y+z and 0.634 for +x-y+z. Because a coronal view is (z, x), an
    unaccounted x mirror shows up as a left-right flipped coronal image while
    leaving the axial view looking plausible — check all three axes, not just
    the one there is a flag for.

    Returns ``(volume, header)`` where volume is float32 (Nx, Ny, Nz).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"volume not found: {path}")

    suffix = path.suffix.lower()
    if suffix in ('.npy', '.npz'):
        arr = np.load(path)
        if suffix == '.npz':
            key = 'volume' if 'volume' in arr else list(arr.keys())[0]
            arr = arr[key]
        volume = np.asarray(arr, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"expected a 3-D volume, got shape {volume.shape}")
        header: dict = {}
        if verbose:
            print(f"  Loaded {path.name}: {volume.shape} (x, y, z), "
                  f"assumed already in HU")
        return volume, header

    header, data = read_vff(str(path), verbose=False)
    # VFF payload is (z, y, x); this package wrote it as (x, y, z) ->
    # transpose(2, 1, 0) then a y flip, so invert in the opposite order.
    data = np.asarray(data)
    if y_flip:
        data = data[:, ::-1, :]
    volume = np.ascontiguousarray(data.transpose(2, 1, 0), dtype=np.float32)

    for _ax in (flip_axes or ()):
        _i = {'x': 0, 'y': 1, 'z': 2}.get(str(_ax).lower())
        if _i is None:
            raise ValueError(f"flip_axes entries must be x, y or z, got {_ax!r}")
        volume = np.ascontiguousarray(np.flip(volume, axis=_i))
    if verbose and flip_axes:
        print(f"  Flipped axes {tuple(flip_axes)} (foreign axis convention)")

    if hu_from_header:
        water = float(header.get('water', 0.0))
        air = float(header.get('air', -1000.0))
        if abs(water - air) < 1e-9:
            raise ValueError(f"degenerate header calibration: water={water}, "
                             f"air={air}")
        volume = (1000.0 * (volume - water) / (water - air)).astype(np.float32)
        if verbose:
            print(f"  Applied header HU calibration: water={water:g}, "
                  f"air={air:g}")

    if verbose:
        print(f"  Loaded {path.name}: {volume.shape} (x, y, z), "
              f"{volume.nbytes / 2**30:.2f} GiB as float32")
        if not hu_from_header and 'water' in header:
            print(f"    header water={float(header['water']):g}, "
                  f"air={float(header['air']):g} — NOT re-applied (the stored "
                  f"values are already HU; --hu-from-header overrides)")
    return volume, header


# ----------------------------------------------------------------- geometry --

def resolve_volume_geometry(volume, header: dict, *, voxel_xy=None,
                            voxel_z=None, origin=None, sidecar=None,
                            verbose: bool = True) -> dict:
    """Build the ``geometry`` dict the figure builders read.

    Only ``dx``/``dz``/``vol_shape``/``vol_origin`` matter here (the midplane
    views put physical mm on the axes); the projection-geometry entries are
    absent because no projections are involved.

    Precedence, most specific first: an explicit argument, then the
    ``<volume>.json`` sidecar (written by every driver in this package, and
    the only place an anisotropic grid or an ROI's position survives), then
    the GE ``elementsize``/``spacing`` header for the voxel size and the
    isocentre for the origin. A last-resort 1.0 mm is used with a warning —
    an unlabelled axis is better than a wrong one.
    """
    side = sidecar or {}
    side_voxel = side.get('voxel_size_mm') or {}
    if voxel_xy is None and 'xy' in side_voxel:
        voxel_xy = float(side_voxel['xy'])
    if voxel_z is None and 'z' in side_voxel:
        voxel_z = float(side_voxel['z'])
    if origin is None and side.get('vol_origin_mm') is not None:
        origin = tuple(float(v) for v in side['vol_origin_mm'])
    if origin is None:
        origin = (0.0, 0.0, 0.0)

    header_size = None
    if 'elementsize' in header:
        header_size = float(header['elementsize'])
    elif 'spacing' in header:
        try:
            header_size = float(str(header['spacing']).split()[0])
        except (ValueError, IndexError):
            header_size = None

    dx = float(voxel_xy) if voxel_xy is not None else header_size
    if dx is None:
        dx = 1.0
        if verbose:
            print("  WARNING: no voxel size in the header and none given — "
                  "the view axes are in voxels, not mm (pass --voxel-xy).")
    dz = float(voxel_z) if voxel_z is not None else dx

    geometry = {
        'vol_shape': tuple(int(n) for n in volume.shape),
        'vol_origin': tuple(float(v) for v in origin),
        'dx': float(dx),
        'dz': float(dz),
    }
    if verbose:
        Nx, Ny, Nz = geometry['vol_shape']
        ox, oy, oz = geometry['vol_origin']
        print(f"  Grid: {Nx} x {Ny} x {Nz} voxels @ {dx:.4f} mm (xy) / "
              f"{dz:.4f} mm (z)")
        print(f"        extent {Nx * dx:.2f} x {Ny * dx:.2f} x {Nz * dz:.2f} mm"
              f", centred at ({ox:.2f}, {oy:.2f}, {oz:.2f}) mm")
    return geometry


# --------------------------------------------------------------- statistics --

AIR_TISSUE_SPLIT_HU = -500.0


def volume_statistics(volume, geometry: dict | None = None) -> dict:
    """Volume-domain scalars: distribution + the two HU landmarks.

    The percentiles are the same ones ``ReconLogger.log_volume_summary``
    records for every reconstruction. The air and tissue peaks are the
    histogram modes of the sub- and super--500 HU populations, measured over
    the WHOLE array rather than a central box — a central box is what made
    the two-point self-calibration lock onto lung instead of soft tissue.
    They are diagnostics here: nothing is rescaled, so a volume whose air
    peak is not near -1000 HU is reported, not corrected.
    """
    vol = np.asarray(volume)
    flat = vol.reshape(-1)
    p1, p50, p99, p999 = (float(v) for v in
                          np.percentile(flat, (1, 50, 99, 99.9)))
    air = _hist_mode(flat[flat < AIR_TISSUE_SPLIT_HU])
    tissue = _hist_mode(flat[flat >= AIR_TISSUE_SPLIT_HU])

    stats = {
        'volume/hu_p1': p1,
        'volume/hu_p50': p50,
        'volume/hu_p99': p99,
        'volume/hu_p99.9': p999,
        'volume/hu_mean': float(flat.mean()),
        'volume/hu_std': float(flat.std()),
        'volume/hu_min': float(flat.min()),
        'volume/hu_max': float(flat.max()),
        'volume/shape': list(vol.shape),
        'volume/voxels': int(vol.size),
    }
    if air is not None:
        stats['volume/hu_air_peak'] = float(air)
    if tissue is not None:
        stats['volume/hu_tissue_peak'] = float(tissue)
    if geometry:
        dx, dz = float(geometry['dx']), float(geometry['dz'])
        Nx, Ny, Nz = geometry['vol_shape']
        stats.update({
            'volume/voxel_xy_mm': dx,
            'volume/voxel_z_mm': dz,
            'volume/extent_xy_mm': float(Nx * dx),
            'volume/extent_z_mm': float(Nz * dz),
        })
    return stats


def format_statistics(stats: dict) -> str:
    """The printed report — the same numbers that go to W&B."""
    def g(key, default=float('nan')):
        return stats.get(key, default)

    lines = [
        f"  shape          {tuple(stats['volume/shape'])}  "
        f"({stats['volume/voxels'] / 1e6:.1f} M voxels)",
        f"  HU range       [{g('volume/hu_min'):.0f}, {g('volume/hu_max'):.0f}]",
        f"  HU mean/std    {g('volume/hu_mean'):.1f} +/- {g('volume/hu_std'):.1f}",
        f"  HU p1/p50      {g('volume/hu_p1'):.1f} / {g('volume/hu_p50'):.1f}",
        f"  HU p99/p99.9   {g('volume/hu_p99'):.1f} / {g('volume/hu_p99.9'):.1f}",
    ]
    if 'volume/hu_air_peak' in stats:
        lines.append(f"  air peak       {g('volume/hu_air_peak'):.1f} HU "
                     f"(expected ~ -1000)")
    if 'volume/hu_tissue_peak' in stats:
        lines.append(f"  tissue peak    {g('volume/hu_tissue_peak'):.1f} HU "
                     f"(expected ~ 0 for water/soft tissue)")
    return "\n".join(lines)


def hu_scale_warnings(stats: dict, air_tol: float = 100.0,
                      tissue_tol: float = 150.0,
                      min_span: float = 200.0) -> list[str]:
    """Flag a volume whose HU landmarks are off — it is almost always the
    file that is on a different scale, not the anatomy.

    The span check is not redundant with the air check: a volume still on an
    attenuation scale has NO voxel below -500, so the air population is empty
    and there is no peak to be off. What gives it away is the total spread —
    air to soft tissue alone is 1000 HU, so anything under a couple of hundred
    is not Hounsfield units.
    """
    warnings = []
    air = stats.get('volume/hu_air_peak')
    tissue = stats.get('volume/hu_tissue_peak')
    span = stats.get('volume/hu_p99.9', 0.0) - stats.get('volume/hu_p1', 0.0)
    if air is None and span < min_span:
        return [f"no voxel below {AIR_TISSUE_SPLIT_HU:.0f} HU and the p1-p99.9 "
                f"spread is only {span:.1f} — this volume is not on the HU "
                f"scale (raw attenuation, or a normalized export). Rescale it, "
                f"or read the figures as relative values."]
    if air is not None and abs(air + 1000.0) > air_tol:
        warnings.append(
            f"air peak is {air:.0f} HU, {abs(air + 1000.0):.0f} HU off the "
            f"-1000 anchor — this volume may not be on the HU scale "
            f"(try --hu-from-header, or treat the numbers as relative).")
    if tissue is not None and abs(tissue) > tissue_tol:
        warnings.append(
            f"tissue peak is {tissue:+.0f} HU rather than ~0 — either a "
            f"calibration offset or a specimen with no water-like tissue "
            f"(a phantom of one material will do this legitimately).")
    return warnings
