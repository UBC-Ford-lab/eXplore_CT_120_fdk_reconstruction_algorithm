"""Detector in-plane rotation (psi) + column centre of rotation, measured from
the PROJECTIONS ALONE.

SHARED by muNeRF, the FDK pipeline and the TIGRE iterative pipeline — it lives
in ct_core precisely so all three reconstruct on the SAME measured geometry
instead of three different assumed ones. Before 2026-08-11 all three assumed a
perfectly square detector (psi = 0) and a centred COR.

See the module-level notes below for the method, the cone-beam caveat and the
fail-safe bounds.
"""


from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

_CAL_DIRNAME = "calibration"

# Fail-safe bounds for `auto` (see resolve_detector_psi). Calibrated on
# Scan_1510: a good fit has residual ~0.25-0.29 columns; a knowingly-bad input
# (transferred detector warp) pushes it to 0.62-2.06 and psi past -1.1 deg.
MAX_PSI_DEG = 2.0
MAX_RESID_COLS = 0.45
# Measured joint-cost depths: 3.7-7.2 (Scan_1510), 3.1-4.8 (Scan_1988),
# 2.7-3.3 (Scan_1989). Below ~1.5 there is no minimum worth trusting.
MIN_JOINT_DEPTH = 1.5


# --------------------------------------------------------------------------
# core estimator
# --------------------------------------------------------------------------

def _build_pairs(betas, gamma, cpa, n_a, dev):
    """Conjugate index/weight arrays for one candidate `cpa`.

    Depend only on (betas, cpa) — NOT on the detector row — so they are built
    once per candidate and reused for every band. That is what makes the sweep
    cost ~1 s instead of minutes.
    """
    bmin, bmax = float(betas.min()), float(betas.max())
    dbeta = float(np.median(np.diff(betas.detach().cpu().numpy())))
    j = torch.arange(n_a, device=dev, dtype=torch.float64)
    J2 = 2.0 * cpa - j                                       # exact mirror
    B2 = betas[:, None] + math.pi + 2.0 * gamma[None, :]
    # FULL scan: conjugates WRAP past 360 deg. Without this the wrapped pairs
    # are silently discarded, which both halves the statistics and makes the
    # surviving set asymmetric in beta — a short scan has no wrap to handle, a
    # full one is mostly wrap.
    if (bmax - bmin) > math.radians(350.0):
        period = bmax - bmin + dbeta
        B2 = bmin + torch.remainder(B2 - bmin, period)
    ok = ((B2 >= bmin) & (B2 <= bmax)
          & (J2 >= 0)[None, :] & (J2 <= n_a - 1)[None, :])
    if not bool(ok.any()):
        return None
    bi, ci = torch.nonzero(ok, as_tuple=True)
    fb = (B2[bi, ci] - bmin) / dbeta
    ib = fb.long().clamp(0, betas.numel() - 2)
    wb = (fb - ib).clamp(0, 1)
    j2 = J2[ci]
    ij = j2.long().clamp(0, n_a - 2)
    wj = (j2 - ij).clamp(0, 1)
    return bi, ci, ib, wb, ij, wj


def _band_cost(band, pk, thresh):
    """Conjugate disagreement for one prepared row band.

    VARIANCE of the pair difference, not mean-square: a per-projection gain
    offset (Scan_1510 drifts ~2%) inflates a mean-square but leaves a variance
    untouched. Object rays only — air rays agree for every cpa and would just
    dilute the minimum.
    """
    bi, ci, ib, wb, ij, wj = pk
    v1 = band[bi, :, ci]
    v2 = ((1 - wb)[:, None] * ((1 - wj)[:, None] * band[ib, :, ij]
                               + wj[:, None] * band[ib, :, ij + 1])
          + wb[:, None] * ((1 - wj)[:, None] * band[ib + 1, :, ij]
                           + wj[:, None] * band[ib + 1, :, ij + 1]))
    m = (v1.abs() > thresh) & (v2.abs() > thresh)
    if int(m.sum()) < 500:
        return float("nan"), 0
    return float((v1 - v2)[m].var()), int(m.sum())


def _to_ideal_grid(band, rows, warp, downsample, dev):
    """Resample a row band from pipeline columns onto the IDEAL detector grid.

    `warp.ideal_indices` maps pipeline -> ideal; we need the inverse to know
    which pipeline column supplies each ideal column, so invert the (monotone)
    map by 1-D interpolation, per row. Returns (band_ideal, ideal_row_centres).
    """
    n_beta, n_rows, n_a = band.shape
    a_pipe = torch.arange(n_a, device=dev, dtype=torch.float64)
    out = torch.empty_like(band)
    b_ideal = []
    for k, b in enumerate(rows.tolist()):
        bb = torch.full((n_a,), float(b), device=dev, dtype=torch.float64)
        b_id, a_id = warp.ideal_indices(bb, a_pipe, downsample=downsample)
        b_ideal.append(float(b_id.mean()))
        # invert a_id(a): where does ideal column k come from?
        src = torch.from_numpy(
            np.interp(a_pipe.cpu().numpy(), a_id.cpu().numpy(),
                      a_pipe.cpu().numpy())).to(dev)
        i0 = src.long().clamp(0, n_a - 2)
        w = (src - i0).clamp(0, 1)
        out[:, k, :] = (1 - w) * band[:, k, :].gather(
            1, i0.expand(n_beta, -1)) + w * band[:, k, :].gather(
            1, (i0 + 1).expand(n_beta, -1))
    return out, float(np.mean(b_ideal))


def _refine(f, x0, h, n=7):
    xs = x0 + h * np.arange(-(n // 2), n // 2 + 1)
    ys = np.array([f(float(x)) for x in xs])
    if not np.all(np.isfinite(ys)):
        return float("nan")
    i = int(ys.argmin())
    if 0 < i < len(xs) - 1:
        den = ys[i - 1] - 2 * ys[i] + ys[i + 1]
        if abs(den) > 1e-20:
            return float(np.clip(xs[i] + 0.5 * (ys[i - 1] - ys[i + 1]) / den * h,
                                 xs[0], xs[-1]))
    return float(xs[i])


def estimate_psi_joint(sinogram, angles, geometry, *, warp=None, downsample=1,
                      bands=25, row_frac=0.90, coarse_half=8.0,
                      coarse_step=0.25, thresh=0.15, device=None, verbose=True):
    """Fit (cpa0, psi) JOINTLY to every row band at once.

    WHY THIS IS MORE ROBUST THAN THE PER-BAND FIT. The two-stage estimator
    (argmin cpa in each band, then regress cpa vs row) takes each band's answer
    at FACE VALUE, so a band whose cost curve is nearly flat contributes a
    garbage cpa that the regression weights like any other. On a smooth phantom
    most bands are like that, which is why Scan_1988/1989 produced 1-2 column
    fit residuals.

    Minimising the SUMMED cost instead is self-weighting: a flat band adds a
    near-constant to the total and therefore cannot move the minimum, while a
    sharp band dominates it. No band ever has to be identified as bad.

    The conditioning limit itself is NOT removed — it is set by
    Var[dp/da] / (conjugate floor), and the phantoms' floor is 12-28x Scan_1510's
    because a thick scattering object genuinely disagrees with its own conjugate
    rays. This uses the available information properly; it does not create more.

    Cheap because the cost table factorises: pair indices depend only on
    (betas, cpa), so one build per cpa candidate serves EVERY band, and the
    (cpa0, slope) search is then pure arithmetic on the precomputed table.
    """
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    S = sinogram.to(dev).double()
    betas = angles.to(dev).double()
    if not bool((betas[1:] > betas[:-1]).all()):
        order = torch.argsort(betas)
        betas, S = betas[order], S[order]
        if verbose:
            print("    angles were not monotone — sorted before pairing")
    da, db = float(geometry["da"]), float(geometry["db"])
    SDD = float(geometry["R_s"]) + float(geometry["R_d"])
    cpa_c = float(geometry["central_pixel_a"])
    cpb = float(geometry["central_pixel_b"])
    n_beta, n_b, n_a = S.shape
    j = torch.arange(n_a, device=dev, dtype=torch.float64)

    edges = np.linspace(cpb - row_frac / 2 * n_b, cpb + row_frac / 2 * n_b,
                        bands + 1)
    groups, centres = [], []
    for k in range(bands):
        r0, r1 = int(round(edges[k])), int(round(edges[k + 1]))
        r0, r1 = max(0, r0), min(n_b, r1)
        if r1 - r0 >= 4:
            groups.append(torch.arange(r0, r1, device=dev))
            centres.append(0.5 * (r0 + r1 - 1))
    if len(groups) < 4:
        raise RuntimeError("too few usable row bands for a joint psi fit")
    bandsS = [S[:, g, :] for g in groups]
    if warp is not None:
        out = [_to_ideal_grid(b, g, warp, downsample, dev)
               for b, g in zip(bandsS, groups)]
        bandsS = [o[0] for o in out]
        centres = [o[1] for o in out]
    centres = np.asarray(centres, dtype=float)

    grid = np.arange(-coarse_half, coarse_half + 1e-9, coarse_step)
    table = np.full((len(bandsS), len(grid)), np.nan)
    for gi, off in enumerate(grid):
        cpa = cpa_c + off
        gam = torch.atan((j - cpa) * da / SDD)
        pk = _build_pairs(betas, gam, cpa, n_a, dev)
        if pk is None:
            continue
        for bi, bnd in enumerate(bandsS):
            table[bi, gi] = _band_cost(bnd, pk, thresh)[0]
    ok = np.all(np.isfinite(table), axis=1)
    table, centres = table[ok], centres[ok]
    if table.shape[0] < 4:
        raise RuntimeError("too few bands with a finite cost curve")
    # normalise each band by its own median so bands of different absolute
    # attenuation contribute comparably; a flat band stays flat either way.
    table = table / np.median(table, axis=1, keepdims=True)

    def total(cpa0, slope):
        want = (cpa0 - cpa_c) + slope * (centres - cpb)
        if want.min() < grid[0] or want.max() > grid[-1]:
            return np.inf
        idx = np.interp(want, grid, np.arange(len(grid)))
        i0 = np.clip(idx.astype(int), 0, len(grid) - 2)
        w = idx - i0
        return float(np.sum((1 - w) * table[np.arange(len(centres)), i0]
                            + w * table[np.arange(len(centres)), i0 + 1]))

    c_grid = cpa_c + np.arange(-6.0, 6.01, 0.1)
    s_grid = np.arange(-0.030, 0.0301, 0.0005)
    Z = np.array([[total(c, s) for s in s_grid] for c in c_grid])
    Zf = Z[np.isfinite(Z)]                       # inf = outside the cost table
    ic, is_ = np.unravel_index(np.nanargmin(Z), Z.shape)
    c0, s0 = c_grid[ic], s_grid[is_]
    for hc, hs in ((0.05, 0.0002), (0.02, 0.00008)):
        cs = c0 + hc * np.arange(-2, 3)
        ss = s0 + hs * np.arange(-2, 3)
        Zl = np.array([[total(c, s) for s in ss] for c in cs])
        a, b = np.unravel_index(np.nanargmin(Zl), Zl.shape)
        c0, s0 = cs[a], ss[b]

    psi = math.degrees(math.asin(max(-1.0, min(1.0, s0 * da / db))))
    # uncertainty from the curvature of the summed cost along slope
    hs = 0.0005
    f0, fm, fp = total(c0, s0), total(c0, s0 - hs), total(c0, s0 + hs)
    curv = (fm - 2 * f0 + fp) / hs ** 2
    n_eff = table.shape[0]
    se = float(np.sqrt(2.0 * f0 / max(curv, 1e-12) / max(n_eff, 1))) if curv > 0 \
        else float("nan")
    bandf = lambda x: math.degrees(math.asin(max(-1.0, min(1.0, x * da / db))))  # noqa: E731
    return {
        "psi_deg": psi,
        "psi_deg_lo": bandf(s0 - se) if np.isfinite(se) else psi,
        "psi_deg_hi": bandf(s0 + se) if np.isfinite(se) else psi,
        "cpa0": float(c0),
        "slope_col_per_row": float(s0),
        "geom_centre": (n_a - 1) / 2.0,
        "n_bands": int(table.shape[0]),
        "method": "joint",
        "warp_applied": warp is not None,
        "thresh": float(thresh),
        # depth of the joint minimum: how much the summed cost rises over the
        # search box. <~1 means there is no minimum to find.
        # depth of the joint minimum over the FINITE part of the search box.
        # A real minimum needs depth >> 0; a flat surface means the data does
        # not determine (cpa0, psi) however stable the argmin looks.
        "joint_depth": float((Zf.max() - f0) / max(f0, 1e-12)),
        "fit_resid_rms_cols": float(
            0.30 / max((Zf.max() - f0) / max(f0, 1e-12), 1e-9)),
        "bands": [],
    }


def estimate_psi(sinogram, angles, geometry, *, warp=None, downsample=1,
                 bands=9, band_rows=45, row_frac=0.80, coarse_half=8.0,
                 coarse_step=0.5, thresh=0.15, device=None, verbose=True):
    """Measure (psi, cpa0) from the sinogram. Returns a result dict."""
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    S = sinogram.to(dev).double()
    betas = angles.to(dev).double()
    # SORT BY ANGLE. `_build_pairs` locates a conjugate by turning an angle into
    # an INDEX (`(b2-bmin)/dbeta`), which silently assumes the angles are sorted
    # and uniformly spaced. A full scan that starts mid-circle violates that:
    # Scan_1989 runs 68.4 -> 359.6 deg then WRAPS to 0.4 -> 68.2 (discontinuity
    # at index 357). Unsorted, every conjugate lookup fetches the wrong
    # projection — measured cost 0.04911 vs 0.00222 on a monotone subset, i.e.
    # 22x inflated, which is what the "160x cost floor anomaly" actually was.
    # (The RENDERER is unaffected: it indexes scene.angles directly and never
    # assumes an ordering.)
    if not bool((betas[1:] > betas[:-1]).all()):
        order = torch.argsort(betas)
        betas = betas[order]
        S = S[order]
        if verbose:
            print("    angles were not monotone (full scan wrapping past 360) "
                  "— sorted before pairing")
    _steps = torch.diff(betas)
    _jit = float((_steps.max() - _steps.min()) / _steps.median().clamp_min(1e-12))
    if _jit > 0.25 and verbose:
        print(f"    WARNING: angle steps vary by {100*_jit:.0f}% of the median "
              f"even after sorting; the uniform-grid index assumption is weak")
    da, db = float(geometry["da"]), float(geometry["db"])
    SDD = float(geometry["R_s"]) + float(geometry["R_d"])
    cpa0 = float(geometry["central_pixel_a"])
    cpb = float(geometry["central_pixel_b"])
    n_beta, n_b, n_a = S.shape
    j = torch.arange(n_a, device=dev, dtype=torch.float64)

    centres = np.linspace(-row_frac / 2, row_frac / 2, bands) * n_b + cpb
    half = band_rows // 2
    grid = np.arange(-coarse_half, coarse_half + 1e-9, coarse_step)

    rows_out, cpa_out, pairs_out = [], [], []
    for c in centres:
        r0, r1 = int(round(c)) - half, int(round(c)) + half + 1
        if r0 < 0 or r1 > n_b:
            continue
        rows = torch.arange(r0, r1, device=dev)
        band = S[:, rows, :]
        row_abscissa = float(rows.double().mean())
        if warp is not None:
            band, row_abscissa = _to_ideal_grid(band, rows, warp, downsample, dev)

        def f(cpa, _band=band):
            gam = torch.atan((j - cpa) * da / SDD)
            pk = _build_pairs(betas, gam, cpa, n_a, dev)
            return float("nan") if pk is None else _band_cost(_band, pk, thresh)[0]

        ys = np.array([f(cpa0 + o) for o in grid])
        if not np.all(np.isfinite(ys)):
            continue
        best = _refine(f, cpa0 + grid[int(ys.argmin())], coarse_step / 2.0)
        if not np.isfinite(best):
            continue
        gam = torch.atan((j - best) * da / SDD)
        pk = _build_pairs(betas, gam, best, n_a, dev)
        npair = 0 if pk is None else _band_cost(band, pk, thresh)[1]
        rows_out.append(row_abscissa); cpa_out.append(best); pairs_out.append(npair)
        if verbose:
            print(f"    band row {row_abscissa:7.1f}  cpa {best:8.3f} "
                  f"({best - cpa0:+.3f})  pairs {npair:,}")

    if len(rows_out) < 3:
        raise RuntimeError("too few usable row bands for a psi fit "
                           "(widen row_frac or lower thresh)")

    B = np.asarray(rows_out); C = np.asarray(cpa_out)
    A = np.stack([np.ones_like(B), B - cpb], 1)
    coef, *_ = np.linalg.lstsq(A, C, rcond=None)
    icept, slope = float(coef[0]), float(coef[1])
    resid = C - A @ coef
    psi = math.degrees(math.asin(max(-1.0, min(1.0, slope * da / db))))
    sx2 = float(((B - cpb) ** 2).sum())
    se = float(np.sqrt((resid ** 2).sum() / max(len(B) - 2, 1) / max(sx2, 1e-12)))
    band = lambda s: math.degrees(math.asin(max(-1.0, min(1.0, s * da / db))))  # noqa: E731
    return {
        "psi_deg": psi,
        "psi_deg_lo": band(slope - se),
        "psi_deg_hi": band(slope + se),
        "cpa0": icept,
        "slope_col_per_row": slope,
        "geom_centre": (n_a - 1) / 2.0,
        "fit_resid_rms_cols": float(np.sqrt((resid ** 2).mean())),
        "n_bands": len(B),
        "warp_applied": warp is not None,
        "thresh": float(thresh),
        "bands": [{"row": float(b), "cpa": float(c), "pairs": int(p)}
                  for b, c, p in zip(B, C, pairs_out)],
    }


# --------------------------------------------------------------------------
# config resolution + serial-keyed cache
# --------------------------------------------------------------------------

def _cache_path(calib_dir, serial, scan_folder=None):
    """Cached PER SCAN, not per detector.

    psi was originally cached serial-keyed on the theory that a mounting angle
    is a property of the detector. MEASURED 2026-08-11, that is wrong across
    EPOCHS: Scan_1510 (2022) gives psi = -0.51..-0.62 while Scan_1988 and
    Scan_1989 (2026, same serial CAC91105CP) both give ~0.00..+0.10, with the
    joint fit well conditioned in all three (depth 2.7-7.2). Sharing one value
    across them would have handed the 2026 phantom scans a 2022 mouse scan's
    calibration. The measurement costs ~1 s, so per-scan is the safe default and
    the detector-level file is only a fallback.
    """
    if scan_folder is not None:
        tag = Path(scan_folder).name
        return Path(calib_dir) / f"detector_psi_{serial}_{tag}.json"
    return Path(calib_dir) / f"detector_psi_{serial}.json"


def _record(geometry, rec, source):
    """Stash the fit diagnostics on the geometry dict so the trainer can push
    them to W&B. psi was silently 0 in every historical run precisely because
    nothing surfaced it; the fit residual travels with the value so a marginal
    measurement is visible in the run record, not just in stdout."""
    if not isinstance(geometry, dict):
        return
    geometry["det_psi_source"] = source
    for k in ("psi_deg", "psi_deg_lo", "psi_deg_hi", "cpa0", "geom_centre",
              "fit_resid_rms_cols", "n_bands", "slope_col_per_row"):
        if k in rec:
            geometry[f"det_psi_{k}"] = rec[k]


def resolve_detector_psi(geom_cfg, geometry, *, sinogram=None, angles=None,
                         scan_folder=None, warp=None, downsample=1,
                         calib_dir=None, verbose=True):
    """Resolve `geometry.detector_psi_deg` to a number, in degrees.

    Modes
    -----
    ``off`` / ``0``   disabled (exact no-op in ray_sampler).
    a number          use it verbatim.
    ``auto``          use the SERIAL-KEYED cache if present; otherwise measure
                      it from this scan's projections and write the cache.

    psi is a MOUNTING ANGLE — a property of the detector, fixed within a
    calibration epoch — so it is cached per detector serial exactly like the
    warp, not recomputed per scan. That is what makes it free for every
    subsequent scan on the same hardware.
    """
    raw = geom_cfg.get("detector_psi_deg", 0.0)
    if raw is None:
        return 0.0
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("off", "none", "false", "disabled"):
            return 0.0
        if s != "auto":
            return float(raw)
    elif not isinstance(raw, bool):
        return float(raw)
    else:                                    # YAML `off` -> False
        return 0.0

    # ---- auto ----
    from .vff_io import detector_serial_from_scan
    if calib_dir is None:
        import paths as _p
        calib_dir = Path(_p.DATA_DIR) / _CAL_DIRNAME
    calib_dir = Path(calib_dir)
    serial = detector_serial_from_scan(scan_folder) if scan_folder else None
    cache = _cache_path(calib_dir, serial, scan_folder) if serial else None

    if cache is not None and cache.exists():
        rec = json.loads(cache.read_text())
        if verbose:
            print(f"  Detector psi: {rec['psi_deg']:+.4f} deg from cache "
                  f"{cache.name} (measured {rec.get('measured_on', '?')})")
        _record(geometry, rec, source="cache")
        return float(rec["psi_deg"])

    if sinogram is None or angles is None:
        if verbose:
            print("  Detector psi: auto requested but no sinogram available "
                  "(geometry-only path) — using 0.0")
        return 0.0

    if verbose:
        print("  Detector psi: auto — measuring from conjugate rays "
              f"(warp {'ON' if warp is not None else 'OFF'}) ...")
    try:
        res = estimate_psi_joint(sinogram, angles, geometry, warp=warp,
                                 downsample=downsample, verbose=verbose)
    except Exception as e:
        print(f"  Detector psi: auto FAILED ({type(e).__name__}: {e}) — using 0.0")
        return 0.0
    if verbose:
        print(f"  Detector psi: {res['psi_deg']:+.4f} deg "
              f"[{res['psi_deg_lo']:+.4f}, {res['psi_deg_hi']:+.4f}], "
              f"cpa0 {res['cpa0']:.3f} (geom centre {res['geom_centre']:.2f}), "
              f"fit resid {res['fit_resid_rms_cols']:.3f} col")

    # FAIL SAFE. This runs unattended before a multi-hour reconstruction, so a
    # bad fit must NOT silently become the geometry. Both guards are calibrated
    # on Scan_1510: a good fit there has residual ~0.25-0.29 columns, while
    # feeding it a warp known to be transferred from another epoch blows the
    # residual out to 0.62-2.06 and swings psi past -1.1 deg.
    bad = []
    if not np.isfinite(res["psi_deg"]) or abs(res["psi_deg"]) > MAX_PSI_DEG:
        bad.append(f"|psi| > {MAX_PSI_DEG} deg (got {res['psi_deg']:+.3f}) — a "
                   f"detector is not mounted that crooked; the fit is unreliable")
    if res.get("joint_depth", 0.0) < MIN_JOINT_DEPTH:
        bad.append(f"joint-cost depth {res.get('joint_depth', 0.0):.2f} < "
                   f"{MIN_JOINT_DEPTH} — the (cpa0, psi) surface is flat, so the "
                   f"data does not determine them however stable the argmin looks")
    if bad:
        print("  Detector psi: REJECTED, falling back to 0.0")
        for b in bad:
            print(f"                {b}")
        return 0.0

    _record(geometry, res, source="measured")
    if cache is not None:
        import datetime
        rec = dict(res)
        rec["detector_serial"] = serial
        rec["measured_on"] = datetime.date.today().isoformat()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec, indent=2))
        if verbose:
            print(f"                cached to {cache}")
    return float(res["psi_deg"])
