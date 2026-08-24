"""Rigid alignment of a FOREIGN volume onto this package's reconstructions.

WHY THIS IS MEASURED AND NOT READ. The GE vendor volume for a scan does not
land on the same physical grid as a reconstruction this package produces: it is
offset by a few tenths of a millimetre, differently on each axis. The obvious
fix — read the offset out of the vendor's own metadata — was tried and does not
work:

  * ``Volumes/SubVolumeCoordinates`` records the SELECTION (origin 1155/1020/0,
    size 995/1040/2295 unbinned) while ``recon-<task>.xml`` records what was
    actually reconstructed (origin 1155/1019/10, size 989/1034/2289). The
    second is authoritative — 989/3, 1034/3, 2289/3 is exactly the vendor
    volume's (329, 344, 763) — but correcting for the difference moves the
    volume the WRONG WAY on all three axes and makes the fit worse (measured
    residual 0.32 / -0.22 / 0.84 mm, against 0.25 / -0.13 / 0.67 uncorrected).
  * the records are not even self-consistent about scale: CropBoundary divided
    by the selection size gives 0.074703 / 0.074719 / 0.074917 mm per binned
    voxel on the three axes, while the VFF header says 0.075080. Over 763 z
    voxels that 0.5% accumulates to 0.28 mm — the same size as the offset being
    chased. No metadata-derived placement can beat that.

So the offset is MEASURED, once per scan, and cached beside the detector-psi
calibration it deliberately resembles: same directory, same "measure on a miss,
reuse thereafter" contract, same JSON-with-provenance shape. Nothing is
hardcoded and nothing is scan-specific in the code.

WHAT IT DOES NOT FIX. This is a translation only. The vendor volume also
carries a scale discrepancy of order a percent (see the memory on the vendor
.vff not being a metric reference), and no rigid shift addresses that. The
reported NCC is the honest ceiling: if it is far below what a good alignment
should give, the volume differs from the reference by more than a translation
and the number says so.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .paths import calibration_dir

#: Sub-sampling stride for the correlation. The offset is a whole-volume rigid
#: property, so a third of the voxels per axis (1/27 of the data) estimates it
#: to well under a voxel while keeping a measurement to ~2 minutes.
DEFAULT_STRIDE = 3

#: Coarse-to-fine search, in mm: (half-span, step) per round. The first round
#: has to cover more than any plausible offset; the last sets the precision.
DEFAULT_SCHEDULE = ((1.0, 0.10), (0.15, 0.02), (0.03, 0.005))


def _grid_mm(shape, origin, dx, dz):
    """Voxel-centre coordinates (x, y, z) in mm, as three 1-D arrays."""
    nx, ny, nz = (int(v) for v in shape)
    ox, oy, oz = (float(v) for v in origin)
    return (
        (np.arange(nx) - (nx - 1) / 2.0) * float(dx) + ox,
        (np.arange(ny) - (ny - 1) / 2.0) * float(dx) + oy,
        (np.arange(nz) - (nz - 1) / 2.0) * float(dz) + oz,
    )


def measure_volume_alignment(volume, geometry, reference, ref_geometry, *,
                             stride: int = DEFAULT_STRIDE,
                             schedule=DEFAULT_SCHEDULE, fit_scale: bool = True,
                             region: float = 1.0, verbose: bool = True):
    """Translation AND per-axis scale relating ``volume`` to ``reference``.

    A TRANSLATION ALONE IS NOT ENOUGH, and fitting one to a scaled volume is
    actively misleading. MEASURED on Scan_1510's vendor volume against our own
    75 um FDK: the z offset that best matches a slab falls monotonically from
    +0.72 mm at z = -23.7 to +0.32 mm at z = +24.0, i.e. -0.86% per mm — a
    z-SCALE error of ~0.49 mm across the 57 mm extent, not an offset. The
    whole-volume translation fit lands on a compromise that matches NOWHERE:
    the midplane views (which is what anyone actually looks at) need the
    CENTRE's value, ~0.74 mm, while the whole-volume fit gives 0.60-0.67.

    Returns ``(offset_mm, scale, ncc)``. The model is

        vendor_position = origin + scale * (reference_position + offset - origin)

    so ``scale`` multiplies the voxel pitch and ``offset`` shifts the centre —
    both expressible as METADATA, with no resampling. See
    ``aligned_geometry``.
    """
    """Offset in mm that ``volume`` is displaced by, relative to ``reference``.

    Both are ``(Nx, Ny, Nz)`` in this package's order, each with its own
    ``geometry`` (``vol_shape``/``vol_origin``/``dx``/``dz``) — they do NOT
    need to share a lattice, which is the point: the vendor volume is 0.07508
    mm and a reconstruction here is typically 0.1 mm.

    Returns ``(offset_mm, ncc)``. ``offset_mm`` is what must be SUBTRACTED from
    the volume's assumed positions to put its content where it belongs, i.e.
    the corrected centre is ``vol_origin - offset``. The NCC is computed over
    the OVERLAP only, so it is comparable whether the reference is an ROI or a
    full-FOV volume.

    Coordinate descent over a coarse-to-fine schedule rather than a full 3-D
    grid: the axes are near-independent for a translation this small, and two
    refinement rounds recover the coupling.
    """
    from scipy.ndimage import map_coordinates

    ref = np.asarray(reference, dtype=np.float32)
    vol = np.asarray(volume, dtype=np.float32)
    rx, ry, rz = _grid_mm(ref_geometry['vol_shape'], ref_geometry['vol_origin'],
                          ref_geometry['dx'], ref_geometry['dz'])
    # Restrict the fit to the CENTRAL `region` fraction of the reference.
    #
    # WHY THIS EXISTS. On Scan_1510 the vendor volume's z mapping matches ours
    # neither by a translation nor by a scale: the best-fit z offset per slab
    # runs +0.72, +0.81, +0.74, +0.68, +0.54, +0.32 mm from bottom to top — a
    # CURVED profile, flat through the middle and collapsing at the top. No
    # global model matches everywhere, so a whole-volume fit lands on a
    # compromise that matches NOWHERE, and in particular not at the midplane,
    # which is the one slice every figure shows. Fitting the central region
    # instead makes the views agree with the other reconstructions exactly,
    # at the price of a known, reported mismatch toward the ends.
    if region < 1.0:
        keep = []
        for axis, coord in enumerate((rx, ry, rz)):
            n = len(coord)
            half = max(1, int(round(n * float(region) / 2.0)))
            mid = n // 2
            keep.append(slice(max(0, mid - half), min(n, mid + half)))
        ref = ref[keep[0], keep[1], keep[2]]
        rx, ry, rz = rx[keep[0]], ry[keep[1]], rz[keep[2]]
        if verbose:
            print(f"    fitting the central {region:.0%} of the reference "
                  f"({ref.shape}) — a whole-volume fit matches nowhere when "
                  f"the two disagree by more than a rigid transform")
    s = max(1, int(stride))
    gx, gy, gz = np.meshgrid(rx[::s], ry[::s], rz[::s], indexing='ij')
    target = ref[::s, ::s, ::s].ravel().astype(np.float64)

    vshape = [int(v) for v in geometry['vol_shape']]
    vorig = [float(v) for v in geometry['vol_origin']]
    vd = [float(geometry['dx']), float(geometry['dx']), float(geometry['dz'])]
    grids = (gx, gy, gz)

    def score(off, scale=None):
        sc = (1.0, 1.0, 1.0) if scale is None else scale
        idx = [sc[k] * (grids[k] + off[k] - vorig[k]) / vd[k]
               + (vshape[k] - 1) / 2.0 for k in range(3)]
        flat = [i.ravel() for i in idx]
        # Score ONLY the overlap. The reference is routinely much larger than
        # the foreign volume — a full-FOV reconstruction against a vendor ROI
        # is the normal case — and every reference voxel outside the foreign
        # extent would otherwise contribute an edge-extended constant, diluting
        # the correlation and dragging the fit toward whatever the border does.
        inside = np.ones(flat[0].shape, dtype=bool)
        for k in range(3):
            inside &= (flat[k] >= 0) & (flat[k] <= vshape[k] - 1)
        if inside.sum() < 1000:
            return -1.0
        got = map_coordinates(vol, np.array([f[inside] for f in flat]),
                              order=1, mode='nearest').astype(np.float64)
        got = (got - got.mean()) / max(got.std(), 1e-9)
        ref_in = target[inside]
        ref_in = (ref_in - ref_in.mean()) / max(ref_in.std(), 1e-9)
        return float((ref_in * got).mean())

    off = np.zeros(3)
    scale = np.ones(3)
    # Scale steps are chosen so the induced displacement at the volume EDGE is
    # comparable to the translation step of the same round — otherwise the
    # scale search is either meaningless or hopelessly slow.
    half_extent = [max(1e-6, vshape[k] * vd[k] / 2.0) for k in range(3)]
    for span, step in schedule:
        for k in range(3):
            cands = np.arange(off[k] - span, off[k] + span + 1e-9, step)
            vals = [score(np.where(np.arange(3) == k, c, off), scale)
                    for c in cands]
            off[k] = float(cands[int(np.argmax(vals))])
        if fit_scale:
            for k in range(3):
                s_span = span / half_extent[k]
                s_step = max(step / half_extent[k], 1e-5)
                cands = np.arange(scale[k] - s_span, scale[k] + s_span + 1e-12,
                                  s_step)
                vals = [score(off, np.where(np.arange(3) == k, c, scale))
                        for c in cands]
                scale[k] = float(cands[int(np.argmax(vals))])
        if verbose:
            print(f"    offset ({off[0]:+.3f}, {off[1]:+.3f}, {off[2]:+.3f}) mm"
                  f"   scale ({scale[0]:.5f}, {scale[1]:.5f}, {scale[2]:.5f})"
                  f"   NCC {score(off, scale):.5f}")
    return off, scale, score(off, scale)


def measure_volume_offset(volume, geometry, reference, ref_geometry, **kw):
    """Translation only. Kept for callers that want a rigid answer; prefer
    ``measure_volume_alignment``, which also reports the scale that a rigid
    fit would silently absorb."""
    off, _scale, ncc = measure_volume_alignment(
        volume, geometry, reference, ref_geometry, fit_scale=False, **kw)
    return off, ncc


def aligned_geometry(geometry, offset_mm, scale):
    """The volume's geometry with the measured alignment folded in.

    METADATA ONLY — no resampling, so nothing is interpolated, the histogram is
    untouched and the HU statistics stay exact. A scale multiplies the voxel
    pitch; a translation moves the origin.
    """
    off = np.asarray(offset_mm, dtype=float)
    sc = np.asarray(scale, dtype=float)
    out = dict(geometry)
    out['dx'] = float(geometry['dx']) / float(np.mean(sc[:2]))
    out['dz'] = float(geometry['dz']) / float(sc[2])
    out['vol_origin'] = tuple(float(o) - float(d)
                              for o, d in zip(geometry['vol_origin'], off))
    return out


def apply_volume_offset(volume, offset_mm, geometry):
    """Move ``volume``'s content so it sits where ``offset_mm`` says it does.

    Content believed to be at position p is really at ``p - offset``, so the
    array is shifted by ``-offset/voxel`` in index units. Linear interpolation,
    because the offset is deliberately sub-voxel — rounding it to whole voxels
    would throw away most of what was measured.
    """
    from scipy.ndimage import shift as ndshift

    off = np.asarray(offset_mm, dtype=np.float64)
    if not np.any(off):
        return volume
    vd = np.array([float(geometry['dx']), float(geometry['dx']),
                   float(geometry['dz'])])
    return ndshift(np.asarray(volume, dtype=np.float32), -off / vd,
                   order=1, mode='nearest').astype(np.float32)


def resample_to_geometry(volume, geometry, target_geometry, *,
                         fill: float = 0.0, chunk: int = 32):
    """``volume`` resampled onto ``target_geometry``'s lattice, trilinearly.

    Both arguments are the ``vol_shape``/``vol_origin``/``dx``/``dz`` dicts
    this package passes around, in mm about the isocentre, so the two need
    share neither a shape, a voxel size nor an origin. That is the whole
    point: the vendor volume is 0.074795 mm on its own 1166 x 1165 x 765
    grid, a reconstruction here is 0.075 or 0.1 mm on another, and anything
    that wants to put the two through the SAME forward model or the same ROI
    has to move one onto the other's lattice first.

    Positions the source does not cover are filled with ``fill`` rather than
    edge-extended. An out-of-extent voxel is MISSING data, and smearing the
    border value into it would invent attenuating material outside the
    reconstructed FOV — which a forward model would then integrate through,
    turning a metadata mismatch into a fake projection.

    Resampling is a LAST RESORT for a comparison: it interpolates, so it
    softens the volume slightly and it perturbs the histogram. Prefer
    ``aligned_geometry``, which expresses an alignment as metadata and
    touches no voxel. Use this only where a common lattice is genuinely
    required.

    The transform is axis-aligned and separable, so it is done as a pair of
    in-plane warps per output slice blended along z. That keeps the working
    set at two source planes instead of the three full-size coordinate
    arrays a direct ``map_coordinates`` would need — 27 GB for a 1155^2 x 846
    target, which is why the obvious implementation is not the one here.
    """
    from scipy.ndimage import map_coordinates

    vol = np.asarray(volume, dtype=np.float32)
    src_shape = [int(v) for v in geometry['vol_shape']]
    if tuple(vol.shape) != tuple(src_shape):
        raise ValueError(
            f"volume shape {tuple(vol.shape)} does not match its geometry's "
            f"vol_shape {tuple(src_shape)}")
    so = [float(v) for v in geometry['vol_origin']]
    sd = [float(geometry['dx']), float(geometry['dx']), float(geometry['dz'])]

    tx, ty, tz = _grid_mm(target_geometry['vol_shape'],
                          target_geometry['vol_origin'],
                          target_geometry['dx'], target_geometry['dz'])
    nx, ny, nz = len(tx), len(ty), len(tz)

    # Target positions in SOURCE index units, per axis (the separability).
    #
    # Snapped to the valid range within a tolerance, because a lattice that
    # coincides with the source's own lands on index 0 as -8.9e-16 and on
    # N-1 as N-1+eps. Without the snap the outermost row of every axis falls
    # the wrong side of the extent test and comes back as `fill`, so
    # resampling a volume onto ITS OWN geometry would blank its six faces —
    # an identity that is not the identity.
    def snap(idx, n, tol=1e-6):
        idx = np.asarray(idx, dtype=np.float64).copy()
        idx[(idx < 0) & (idx > -tol)] = 0.0
        idx[(idx > n - 1) & (idx < n - 1 + tol)] = n - 1.0
        return idx

    ix = snap((tx - so[0]) / sd[0] + (src_shape[0] - 1) / 2.0, src_shape[0])
    iy = snap((ty - so[1]) / sd[1] + (src_shape[1] - 1) / 2.0, src_shape[1])
    iz = snap((tz - so[2]) / sd[2] + (src_shape[2] - 1) / 2.0, src_shape[2])

    gx, gy = np.meshgrid(ix, iy, indexing='ij')
    plane_coords = np.array([gx.ravel(), gy.ravel()])
    in_plane = ((gx >= 0) & (gx <= src_shape[0] - 1)
                & (gy >= 0) & (gy <= src_shape[1] - 1))

    out = np.full((nx, ny, nz), float(fill), dtype=np.float32)

    def warp(k_src):
        """Source plane ``k_src`` warped onto the target's in-plane lattice."""
        got = map_coordinates(vol[:, :, k_src], plane_coords, order=1,
                              mode='nearest').reshape(nx, ny)
        return np.where(in_plane, got, fill)

    # Walk the output in z chunks so each source plane is warped once even
    # when several output slices land between the same pair.
    cache: dict = {}
    for z0 in range(0, nz, max(1, int(chunk))):
        z1 = min(nz, z0 + max(1, int(chunk)))
        zc = iz[z0:z1]
        lo = np.floor(zc).astype(int)
        frac = zc - lo
        need = set(lo[(lo >= 0) & (lo <= src_shape[2] - 1)].tolist())
        need |= set((lo + 1)[((lo + 1) >= 0)
                             & ((lo + 1) <= src_shape[2] - 1)].tolist())
        cache = {k: cache.get(k) if cache.get(k) is not None else warp(k)
                 for k in need}
        for j, (k, f) in enumerate(zip(lo, frac)):
            if k < 0 or k > src_shape[2] - 1:
                continue          # below or above the source stack
            if f <= 0:
                out[:, :, z0 + j] = cache[k]
            elif k + 1 <= src_shape[2] - 1:
                out[:, :, z0 + j] = (1.0 - f) * cache[k] + f * cache[k + 1]
            # else: strictly above the last source plane -> outside, keep fill
    return out

# ------------------------------------------------------------------ cache ---

def alignment_path(scan_tag: str, volume_name: str) -> Path:
    """Where the measured offset for one foreign volume lives."""
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in volume_name)
    return calibration_dir() / f"volume_align_{scan_tag}_{safe}.json"


def load_alignment(scan_tag: str, volume_name: str, *, reference=None,
                   ref_grid=None, verbose: bool = True):
    """The cached offset, or None on a miss.

    ``reference`` invalidates the entry when it names a DIFFERENT reference
    than the one the offset was measured against. The offset is only
    meaningful relative to what it was registered to, so silently reusing it
    after the reference changes would be worse than not caching at all.
    """
    path = alignment_path(scan_tag, volume_name)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
        off = np.array([float(v) for v in rec['offset_mm']])
    except (ValueError, KeyError, TypeError) as exc:
        print(f"  alignment cache {path.name} unreadable ({exc}) — remeasuring.")
        return None
    if reference is not None and str(rec.get('reference')) != str(reference):
        print(f"  alignment cache was measured against "
              f"{rec.get('reference')!r}, not {reference!r} — remeasuring.")
        return None
    if ref_grid is not None and rec.get('reference_grid') is not None:
        if list(rec['reference_grid']) != list(ref_grid):
            # Basenames are not unique: nearly every driver here writes
            # `recon.vff`, so two different references collide on name alone.
            # The grid fingerprint separates them without recording a path.
            print(f"  alignment cache was measured against a DIFFERENT grid "
                  f"({rec['reference_grid']} vs {list(ref_grid)}) — "
                  f"remeasuring.")
            return None
    if verbose:
        print(f"  Alignment (cached {path.name}): "
              f"({off[0]:+.3f}, {off[1]:+.3f}, {off[2]:+.3f}) mm, "
              f"NCC {rec.get('ncc', float('nan')):.4f}, "
              f"vs {rec.get('reference', '?')}, measured {rec.get('measured_on', '?')}")
    return off


def reference_fingerprint(geometry) -> list:
    """Shape + pitch + origin of a reference, as plain numbers.

    The cache key needs to tell two references apart, and a BASENAME cannot:
    almost every driver in this package writes its output as ``recon.vff``.
    This is data rather than a path, so it stays inside the public-repo rule
    that no filesystem location is ever recorded.
    """
    return [int(v) for v in geometry['vol_shape']] + [
        round(float(geometry['dx']), 6), round(float(geometry['dz']), 6),
        *(round(float(v), 4) for v in geometry['vol_origin'])]


def save_alignment(scan_tag: str, volume_name: str, offset_mm, ncc,
                   reference_name: str, extra: dict | None = None) -> Path:
    """Record a measured offset. Only BASENAMES are stored — never a path."""
    path = alignment_path(scan_tag, volume_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        'offset_mm': [float(v) for v in offset_mm],
        'ncc': float(ncc),
        'volume': str(volume_name),
        'reference': str(reference_name),
        'scan': str(scan_tag),
        'measured_on': date.today().isoformat(),
        'method': 'ncc-coordinate-descent',
        **(extra or {}),
    }
    path.write_text(json.dumps(rec, indent=2))
    print(f"  Saved alignment to {path.name} — future reports on this volume "
          f"reuse it.")
    return path
