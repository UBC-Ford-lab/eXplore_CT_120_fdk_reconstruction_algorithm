"""Score ALREADY-RECONSTRUCTED volumes against the measured projections.

``run_volume_report`` answers "what does this volume look like"; this answers
"does this volume explain the data it was reconstructed from", for a volume
that is already on disk. Every reconstruction driver already computes the
diag/* bundle at the end of its own run, but only for the volume it just
made, only at ONE angle, and only inside that run. A comparison table wants
the same numbers for volumes made months apart by different algorithms, over
the same set of angles, scored the same way — which means recomputing them
from the files.

Three things have to be held fixed or the columns are not comparable, and
each is a place a table quietly stops meaning anything:

  * THE ANGLES. Scored angles are chosen by a deterministic rule (evenly
    spaced from the middle outwards) so two invocations on the same scan pick
    the same ones. ``--angle-indices`` pins them explicitly.
  * THE DETECTOR WINDOW. ``covered_detector_window`` shrinks the scored
    rectangle to the rays that stay inside the reconstruction domain, so a
    volume with a smaller extent would be scored on less of the detector —
    and score better for it. Every volume in one invocation is scored on the
    INTERSECTION of the windows, reported as ``proj/window``.
  * THE UNITS. The forward model integrates mu, and the volumes on disk are
    HU. Each volume is converted back through ITS OWN calibration (the
    ``<volume>.json`` sidecar's two anchors), not through a shared constant —
    the whole point of the reference-free calibration is that the scale is a
    property of the run.

Volumes are forward-projected at their NATIVE grid: the geometry handed to
the ray tracer is the scan's projection geometry with the volume's own
vol_shape/vol_origin/dx/dz. Nothing is resampled, because resampling a 75 um
volume onto a 100 um lattice to compare it against a 100 um one would soften
the very thing being measured.

A volume with no sidecar (the vendor's) has no anchors to invert, so its mu
scale is unknown. Rather than invent one, every row also reports the metrics
after the single least-squares gain that best matches the measured
projection. The forward model is linear in mu, so that costs no second
render: scaling mu by g scales the prediction by g exactly. ``proj/gain_fit``
is 1.0 for a volume whose calibration already agrees with the data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .ct_core.errors import ConfigError, cli_main
from .ct_core.hu_calibration import HUAnchors, fixed_anchors
from .ct_core.pipeline import (
    add_common_args,
    add_model_domain_args,
    prepare_scan,
)
from .ct_core.projection_diag import (
    covered_detector_window,
    evaluate_projection,
    measure_noise_ceiling,
    preprocess_frames,
    render_projection_from_volume,
    ssim_heatmap_figure,
)
from .ct_core.volume_report import (
    load_reconstructed_volume,
    load_sidecar,
    resolve_volume_geometry,
)
from .ct_core.wandb_logging import ReconLogger


# --------------------------------------------------------------------------
# Angle selection
# --------------------------------------------------------------------------

def choose_angles(n_angles: int, count: int) -> list:
    """``count`` angle indices, evenly spread, centred on the middle one.

    Centred on the middle deliberately: that is the index every backend's own
    diag/* uses, so a one-angle report here reproduces the number the
    reconstruction logged, and a wider report brackets it symmetrically
    instead of drifting toward one end of a short scan.
    """
    n, k = int(n_angles), int(count)
    if n <= 0:
        return []
    k = max(1, min(k, n))
    if k == 1:
        return [n // 2]
    # Evenly spaced over the full sweep, then shifted so the set is centred
    # on n//2 — exact when k is odd, off by half a step when it is even.
    pos = np.linspace(0, n - 1, k)
    pos = pos + (n // 2 - pos.mean())
    idx = sorted({int(round(min(max(p, 0), n - 1))) for p in pos})
    # Rounding can collide near the ends; top up from the unused indices so
    # the caller always gets the count they asked for when the scan allows.
    if len(idx) < k:
        for j in range(n):
            if j not in idx:
                idx.append(j)
                if len(idx) == k:
                    break
        idx = sorted(idx)
    return idx


# --------------------------------------------------------------------------
# Volume loading
# --------------------------------------------------------------------------

def anchors_for(sidecar: dict, mu_water: float | None, name: str):
    """The HU map to invert, and a one-line account of where it came from.

    Prefers the volume's OWN fitted anchors. The fallback is the classical
    one-point map, which is a real assumption and is reported as one — a
    volume scored through a borrowed mu_water is telling you about that
    constant as much as about the reconstruction.
    """
    hu = (sidecar or {}).get('hu_calibration') or {}
    if {'anchor_air_value', 'anchor_tissue_value'} <= set(hu):
        anchors = HUAnchors(
            value_air=float(hu['anchor_air_value']),
            value_tissue=float(hu['anchor_tissue_value']),
            air_hu=float(hu.get('target_air', -1000.0)),
            tissue_hu=float(hu.get('target_tissue', 120.0)),
        )
        return anchors, (f"own two-anchor calibration "
                         f"(scale {anchors.scale:.1f} HU per mm^-1)")
    from .ct_core.calibration import MU_WATER_80KV
    mu_w = MU_WATER_80KV if mu_water is None else float(mu_water)
    print(f"  {name}: no sidecar anchors — inverting HU with the one-point "
          f"map at mu_water = {mu_w:.5f} mm^-1. The absolute level of this "
          f"row rests on that constant; read proj/gain_fit alongside it.")
    return fixed_anchors(mu_w), f"assumed mu_water {mu_w:.5f} mm^-1"


def peek_geometry(path, args, *, verbose: bool = True):
    """This volume's grid, from its HEADER and sidecar — no voxels loaded.

    Used for the pre-flight pass that resolves the common detector window.
    Loading every volume just to read its shape would hold 23 GB for
    Scan_1988's seven, when the pass needs four numbers from each.
    """
    from .ct_core.vff_io import read_vff

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"volume not found: {path}")
    sidecar = load_sidecar(path, verbose=verbose)
    if path.suffix.lower() == '.vff':
        header, _data = read_vff(str(path), verbose=False)
        shape = tuple(int(v) for v in str(header['size']).split())
    else:
        header = {}
        shape = np.load(path, mmap_mode='r').shape
    # A shape is all resolve_volume_geometry wants from the array.
    class _Shaped:
        pass
    stub = _Shaped()
    stub.shape = shape
    return resolve_volume_geometry(
        stub, header, voxel_xy=args.volume_voxel_xy,
        voxel_z=args.volume_voxel_z, origin=args.volume_origin,
        sidecar=sidecar, verbose=verbose), sidecar


def load_volume_as_mu(path, args, *, vendor: bool = False,
                      verbose: bool = True):
    """Load a finished volume and hand back (mu, geometry, provenance)."""
    path = Path(path)
    sidecar = load_sidecar(path, verbose=verbose)
    y_flip = not (args.no_y_flip or vendor)
    flips = tuple(set(tuple(args.flip or ()) + (('x',) if vendor else ())))
    volume_hu, header = load_reconstructed_volume(
        path,
        y_flip=y_flip,
        flip_axes=flips,
        verbose=verbose,
    )
    geometry = resolve_volume_geometry(
        volume_hu, header,
        voxel_xy=args.volume_voxel_xy, voxel_z=args.volume_voxel_z,
        origin=args.volume_origin, sidecar=sidecar, verbose=verbose)
    anchors, how = anchors_for(sidecar, args.mu_water, path.name)
    mu = np.clip(anchors.invert(volume_hu), 0.0, None)
    return mu, geometry, {
        'calibration': how,
        'algorithm': (sidecar or {}).get('algorithm', 'external'),
        'hu_scale': float(anchors.scale),
    }


def scan_geometry_for(ctx, volume_geometry: dict) -> dict:
    """The scan's geometry with THIS volume's grid substituted in.

    The forward model needs both halves: the projection geometry (source and
    detector distances, pitch, centre of rotation, detector psi) which belongs
    to the scan and is identical for every row, and the grid
    (vol_shape/vol_origin/dx/dz) which belongs to the file. Passing only the
    grid — what ``resolve_volume_geometry`` returns — leaves
    ``covered_detector_window`` with no ``R_s`` to work from.
    """
    out = dict(ctx.geometry)
    for key in ('vol_shape', 'vol_origin', 'dx', 'dz'):
        out[key] = volume_geometry[key]
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def gain_fit(pred: np.ndarray, target: np.ndarray) -> float:
    """The least-squares scalar g minimising ||g*pred - target||.

    Exact for the forward model, which is linear in mu: rendering g*mu gives
    g*pred, so the gain-corrected metrics need no second render.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    denom = float((p * p).sum())
    return float((p * t).sum() / denom) if denom > 0 else 1.0


def score_volume(mu, geometry, ctx, indices, measured, window, *,
                 render_downsample: int, verbose: bool = True):
    """Per-angle metrics for one volume, as-rendered and after the gain fit."""
    rows = []
    first_pair = None
    for i, idx in enumerate(indices):
        pred, target = render_projection_from_volume(
            mu, ctx, idx, measured[idx], volume_is_hu=False,
            downsample=render_downsample, geometry=geometry, window=window)
        g = gain_fit(pred, target)
        m = evaluate_projection(pred, target)
        mg = evaluate_projection(pred * g, target)
        rows.append({'angle_index': int(idx), 'gain_fit': g,
                     'ssim': m['ssim'], 'psnr': m['psnr'], 'mse': m['mse'],
                     'ssim_gainfit': mg['ssim'], 'psnr_gainfit': mg['psnr'],
                     'mse_gainfit': mg['mse']})
        if verbose:
            print(f"    angle {idx:4d}   ssim {m['ssim']:.4f}   "
                  f"psnr {m['psnr']:6.2f} dB   mse {m['mse']:.4e}   "
                  f"gain {g:.4f}")
        if first_pair is None:
            first_pair = (pred, target)
    return rows, first_pair


def aggregate(rows: list) -> dict:
    """Mean over the scored angles, plus the spread — a single angle is a
    sample, and two volumes half a standard deviation apart over three angles
    have not been separated by this measurement."""
    out = {}
    for key in ('ssim', 'psnr', 'mse', 'ssim_gainfit', 'psnr_gainfit',
                'mse_gainfit', 'gain_fit'):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f'proj/{key}'] = float(vals.mean())
        out[f'proj/{key}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out['proj/n_angles'] = len(rows)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    add_model_domain_args(parser)
    parser.add_argument(
        '--volume', action='append', required=True, metavar='PATH',
        help='Reconstructed volume to score (.vff/.npy/.npz). Repeat the flag '
             'to score several against the same angles and the same detector '
             'window — which is the only way the rows compare.')
    parser.add_argument(
        '--label', action='append', default=None, metavar='NAME',
        help='Row label for the matching --volume (default: its stem). '
             'Repeat once per volume, in the same order.')
    parser.add_argument(
        '--angles', type=int, default=5, metavar='N',
        help='How many projections to score (default: 5), evenly spread and '
             'centred on the middle angle — the one every backend\'s own '
             'diag/* uses, so --angles 1 reproduces what the run logged.')
    parser.add_argument(
        '--angle-indices', default=None, metavar='I,J,K',
        help='Score exactly these projection indices instead, comma '
             'separated. Overrides --angles.')
    parser.add_argument(
        '--render-downsample', type=int, default=2, metavar='N',
        help='Detector stride for the rendered comparison (default: 2, what '
             'the drivers use for diag/*, so the numbers line up with the '
             'ones already in W&B).')
    parser.add_argument(
        '--no-ceiling', action='store_true',
        help='Skip the noise-ceiling measurement. It needs a second '
             'acquisition phase to be meaningful and falls back to the '
             'neighbouring projection, which biases the ceiling low.')
    parser.add_argument(
        '--report', default=None, metavar='PATH',
        help='Where to write the JSON record of every row (default: '
             '<first volume>_projection_report.json).')
    # Volume-side conventions, mirroring run_volume_report so a vendor file
    # is described the same way in both tools.
    parser.add_argument('--volume-voxel-xy', type=float, default=None,
                        help='In-plane voxel size in mm of the VOLUMES '
                             '(default: the sidecar, else the VFF '
                             'elementsize). Separate from --voxel-xy, which '
                             'describes the reconstruction grid.')
    parser.add_argument('--volume-voxel-z', type=float, default=None,
                        help='Slice thickness in mm of the volumes.')
    parser.add_argument('--volume-origin', nargs=3, type=float, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Volume CENTRE in isocentre-centred mm (default: '
                             'the sidecar, else 0 0 0).')
    parser.add_argument('--flip', nargs='+', default=(), metavar='AXIS',
                        choices=('x', 'y', 'z'),
                        help='Reverse these axes after loading — for a '
                             'foreign file whose conventions differ.')
    parser.add_argument('--no-y-flip', action='store_true',
                        help='Do not undo this package\'s own y reversal.')
    parser.add_argument('--vendor', action='store_true',
                        help='Every --volume is a GE vendor file: load them '
                             'all with --no-y-flip --flip x.')
    parser.add_argument('--vendor-last', action='store_true',
                        help='Only the LAST --volume is a vendor file. Use '
                             'this to score a vendor reconstruction '
                             'alongside this package\'s own, which are '
                             'mirrored the other way. NOTE: a vendor volume '
                             'has no sidecar, so it is placed by its header '
                             'and header alone — measure its offset with '
                             'run_volume_report --align-to first and pass the '
                             'result as --volume-origin, or its rows report '
                             'a misalignment as if it were a reconstruction '
                             'difference.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.vendor:
        args.no_y_flip = True
        args.flip = tuple(set(tuple(args.flip or ()) + ('x',)))

    labels = list(args.label or [])
    if labels and len(labels) != len(args.volume):
        raise ConfigError(
            f"--label given {len(labels)} times for {len(args.volume)} "
            f"--volume arguments; pass one label per volume or none at all.")
    if not labels:
        labels = [Path(v).parent.name or Path(v).stem for v in args.volume]

    start = time.time()
    print("=" * 60)
    print("Projection-domain report")
    print("=" * 60)

    ctx = prepare_scan(args, fit_domain=True)
    n_angles = int(ctx.projections.shape[0])

    if args.angle_indices:
        try:
            indices = [int(v) for v in args.angle_indices.split(',') if v.strip()]
        except ValueError as e:
            raise ConfigError(f"--angle-indices must be integers: {e}")
        bad = [i for i in indices if not 0 <= i < n_angles]
        if bad:
            raise ConfigError(
                f"--angle-indices {bad} outside the scan's 0..{n_angles - 1}")
    else:
        indices = choose_angles(n_angles, args.angles)
    print(f"\nScoring {len(indices)} of {n_angles} projections: {indices}")

    # ---- pre-flight: geometry only, no voxels -------------------------
    # Every volume's grid is read from its header first, so a bad path fails
    # in a second and the common detector window is known before anything is
    # loaded. The volumes themselves are then read ONE AT A TIME below:
    # holding all seven of Scan_1988's at once is 23 GB, and only one is ever
    # being forward-projected.
    plan = []
    for i, (path, label) in enumerate(zip(args.volume, labels)):
        vendor = bool(args.vendor_last and i == len(args.volume) - 1)
        print(f"\n[{label}] {path}" + ("   (vendor conventions)" if vendor
                                       else ""))
        vol_geometry, sidecar = peek_geometry(path, args)
        geometry = scan_geometry_for(ctx, vol_geometry)
        if vendor or not sidecar:
            print("  NOTE: no sidecar — this volume is placed by its header "
                  "alone. If it is displaced from the scan frame, the "
                  "misalignment lands in these numbers.")
        plan.append((label, Path(path), geometry, vendor))

    # ---- the common detector window ----------------------------------
    n_b, n_a = int(ctx.projections.shape[1]), int(ctx.projections.shape[2])
    windows = [covered_detector_window(g, n_b, n_a) for _, _, g, _ in plan]
    window = (max(w[0] for w in windows), min(w[1] for w in windows),
              max(w[2] for w in windows), min(w[3] for w in windows))
    if window[1] <= window[0] or window[3] <= window[2]:
        raise ConfigError(
            "the volumes' reconstruction domains do not share any detector "
            "window — they cannot be scored against each other. Windows: "
            + "; ".join(str(w) for w in windows))
    print(f"\nCommon detector window: rows [{window[0]}, {window[1]}) of "
          f"{n_b}, columns [{window[2]}, {window[3]}) of {n_a}")
    if len(set(windows)) > 1:
        print("  (the volumes' own windows differ — every row is scored on "
              "the intersection so the columns compare)")

    # ---- the measured projections ------------------------------------
    measured = {}
    for idx in indices:
        measured[idx] = preprocess_frames(
            ctx.projections[idx:idx + 1], ctx)[0]

    ceiling = None
    if not args.no_ceiling:
        ceiling = measure_noise_ceiling(ctx, indices[len(indices) // 2],
                                        phase=args.phase)

    # ---- score, one volume in memory at a time ------------------------
    results = []
    for label, path, geometry, vendor in plan:
        print(f"\n[{label}] forward-projecting "
              f"{geometry['vol_shape']} @ {geometry['dx']:.4f} mm")
        mu, _vol_geometry, prov = load_volume_as_mu(path, args, vendor=vendor,
                                                    verbose=False)
        print(f"  HU inverted through its {prov['calibration']}")
        rows, pair = score_volume(mu, geometry, ctx, indices, measured, window,
                                  render_downsample=args.render_downsample)
        del mu
        summary = aggregate(rows)
        results.append({'label': label, 'volume': path.name,
                        'algorithm': prov['algorithm'],
                        'calibration': prov['calibration'],
                        'hu_scale': prov['hu_scale'],
                        'vol_shape': list(geometry['vol_shape']),
                        'voxel_xy_mm': float(geometry['dx']),
                        'voxel_z_mm': float(geometry['dz']),
                        'per_angle': rows, **summary, '_pair': pair})

    # ---- report -------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'volume':28s} {'ssim':>7s} {'psnr':>8s} {'mse':>11s} "
          f"{'gain':>7s} {'ssim*':>7s} {'psnr*':>8s}")
    print("-" * 78)
    for r in results:
        print(f"{r['label'][:28]:28s} {r['proj/ssim']:7.4f} "
              f"{r['proj/psnr']:8.2f} {r['proj/mse']:11.4e} "
              f"{r['proj/gain_fit']:7.4f} {r['proj/ssim_gainfit']:7.4f} "
              f"{r['proj/psnr_gainfit']:8.2f}")
    if ceiling is not None:
        print("-" * 78)
        print(f"{'noise ceiling':28s} {ceiling['ssim']:7.4f} "
              f"{ceiling['psnr']:8.2f} {ceiling['mse']:11.4e}"
              f"   ({ceiling['source']})")
    print("=" * 78)
    print("* = after the least-squares gain; see proj/gain_fit.")

    report_path = Path(args.report) if args.report else (
        Path(args.volume[0]).with_name(
            Path(args.volume[0]).stem + '_projection_report.json'))
    record = {
        'scan': Path(ctx.scan_folder).name,
        'n_angles': n_angles,
        'scored_angles': indices,
        'detector_window': list(window),
        'render_downsample': int(args.render_downsample),
        'downsample': int(ctx.downsample),
        'noise_ceiling': ({k: v for k, v in ceiling.items() if k != 'pair'}
                          if ceiling else None),
        'volumes': [{k: v for k, v in r.items() if k != '_pair'}
                    for r in results],
    }
    report_path.write_text(json.dumps(record, indent=2))
    print(f"\nReport written to {report_path}")

    # ---- logging ------------------------------------------------------
    # One W&B run per volume: these are separate reconstructions being
    # compared, and folding them into a single run would put six different
    # volumes' scalars under the same keys.
    for r in results:
        logger = ReconLogger(
            args, ctx, f"projreport_{r['algorithm']}",
            str(report_path.with_suffix('')) + f"_{r['label']}",
            params={'volume': r['volume'], 'scored_angles': indices,
                    'detector_window': list(window),
                    'render_downsample': int(args.render_downsample),
                    'vol_shape': r['vol_shape'],
                    'voxel_xy_mm': r['voxel_xy_mm']})
        if ceiling is not None:
            logger.set_noise_ceiling(ceiling)
        logger.set_summary({k: v for k, v in r.items()
                            if k.startswith('proj/')})
        if r['_pair'] is not None:
            pred, target = r['_pair']
            try:
                logger._emit('proj_ssim_heatmap',
                             ssim_heatmap_figure(
                                 pred, target,
                                 noise_pair=(ceiling or {}).get('pair'),
                                 title=f" ({r['label']})"))
            except Exception as e:
                print(f"  heatmap failed ({type(e).__name__}: {e})")
        logger.finish()

    print(f"\nFinished in {(time.time() - start) / 60:.2f} minutes.")


if __name__ == '__main__':
    cli_main(main)
