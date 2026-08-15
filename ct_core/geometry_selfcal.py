"""Half-scan geometry self-calibration — detector in-plane rotation (psi).

PORT of muNeRF's validated Tier-2 estimator (inr_pipeline/geometry_selfcal.py,
2026-08-12) into the standalone reconstruction package, so that EVERY
reconstruction algorithm (FDK, ASTRA, TIGRE, and future backends) can
self-calibrate without muNeRF in the loop. The scoring machinery
(ramp filter, gradient-NCC, two-stage grid search with edge extension,
guards) is copied verbatim; only the FBP building block differs: muNeRF
backprojects through its differentiable renderer (autograd adjoint), here a
direct voxel-driven torch backprojector implements the SAME geometry:

    source  = -R_s * (cos t, sin t, 0)
    u_hat   = (-sin t, cos t, 0),  v_hat = (0, 0, 1)
    psi     : rotates (u_hat, v_hat) about the detector normal,
              centred on (central_pixel_a, central_pixel_b)
    a_idx   = cpa + ( cos(psi)*p_u + sin(psi)*p_v) / da
    b_idx   = cpb + (-sin(psi)*p_u + cos(psi)*p_v) / db
    with p_u = (SDD/U)(-x sin t + y cos t),  p_v = (SDD/U) z,
         U   = R_s + x cos t + y sin t

which is exactly muNeRF's rays_from_indices convention, so the psi_deg this
module measures and the psi_deg muNeRF measures mean the same rotation and
share one calibration JSON.

PRINCIPLE. The two halves of the view range see the object from disjoint
directions; their FBP reconstructions agree about the volume IF AND ONLY IF
the assumed geometry is consistent with the data. Score = zero-mean NCC of
gradient-magnitude images of the two half-reconstructions, at two axial
planes (psi error is antisymmetric in z, column-CoR error symmetric).
No registration, no windows, no feature detection.

VALIDATED (muNeRF side, Scan_1510): full-resolution peak -0.775 deg
(independent sweep: -0.795); works on the rotationally symmetric Scan_1988
phantom (+0.19 deg, sharp) where conjugate rays are structurally blind.
This port is additionally validated to reproduce the Scan_1510 measurement
from raw projections.

POLICY. psi_deg is applied; the fitted cpa0 intercept is DIAGNOSTIC ONLY
(known estimator bias — applying it split run zsu85kc6's off-midplane tube
into two overlapped half-discs; the applied column CoR stays at the
geometric centre).
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .paths import calibration_dir

# Guards for unattended use ahead of multi-hour jobs (identical to muNeRF).
MAX_PSI_DEG = 2.0            # a detector is not mounted that crooked
MIN_PROMINENCE = 0.05        # (max-min)/|min| of the coarse curve; a flat
                             # curve = degenerate data (do not trust the argmax)
MIN_ANGLES = 60              # need two meaningful half-scans
MAX_GRID_EXTENSIONS = 2      # widen an edge-peaked grid at most this often


# --------------------------------------------------------------------------
# FBP building blocks
# --------------------------------------------------------------------------

def ramp_filter(sino: torch.Tensor, window: str = "hann") -> torch.Tensor:
    """Ram-Lak ramp along the detector COLUMN axis, optionally windowed.

    Applied to the measured line integrals once — it depends only on the
    data, not on cpa/cpb/psi, so all candidate geometries share it.
    """
    n_a = sino.shape[-1]
    n_pad = 1 << int(np.ceil(np.log2(2 * n_a)))
    pad = torch.zeros(*sino.shape[:-1], n_pad, dtype=sino.dtype,
                      device=sino.device)
    pad[..., :n_a] = sino
    freq = torch.fft.rfftfreq(n_pad, device=sino.device, dtype=sino.dtype)
    ramp = 2.0 * freq
    if window == "hann":
        ramp = ramp * (0.5 + 0.5 * torch.cos(
            np.pi * freq / freq.max().clamp_min(1e-12)))
    out = torch.fft.irfft(torch.fft.rfft(pad, dim=-1) * ramp, n=n_pad, dim=-1)
    return out[..., :n_a].contiguous()


@torch.no_grad()
def backproject_slab(sino_f: torch.Tensor, angles: torch.Tensor,
                     geometry: dict, z_centre: float, z_half: float,
                     vox_xy: float, vox_z: float, device) -> torch.Tensor:
    """FBP a thin axial slab under the muNeRF ray convention (see module doc).

    Voxel-driven: for every slab voxel and angle, project to the detector
    (including psi about the (cpa, cpb) point) and bilinear-sample the
    filtered sinogram; rays that miss the detector contribute zero. The
    absolute scale is irrelevant — the consistency score is NCC.

    Returns the slab volume as a (nz, ny, nx) tensor on the CPU.
    """
    g = geometry
    R_s, R_d = float(g["R_s"]), float(g["R_d"])
    SDD = R_s + R_d
    da, db = float(g["da"]), float(g["db"])
    cpa, cpb = float(g["central_pixel_a"]), float(g["central_pixel_b"])
    psi = float(g.get("det_psi_rad", 0.0) or 0.0)
    c_psi, s_psi = float(np.cos(psi)), float(np.sin(psi))

    ox, oy, _ = g["vol_origin"]
    nx_full, ny_full, _ = g["vol_shape"]
    dx = float(g["dx"])
    hx, hy = nx_full * dx / 2.0, ny_full * dx / 2.0

    nx = max(2, int(round(2 * hx / vox_xy)))
    ny = max(2, int(round(2 * hy / vox_xy)))
    nz = max(2, int(round(2 * z_half / vox_z)))

    xs = ox - hx + (torch.arange(nx, device=device, dtype=torch.float32)
                    + 0.5) * (2 * hx / nx)
    ys = oy - hy + (torch.arange(ny, device=device, dtype=torch.float32)
                    + 0.5) * (2 * hy / ny)
    zs = z_centre - z_half + (torch.arange(nz, device=device,
                                           dtype=torch.float32)
                              + 0.5) * (2 * z_half / nz)

    X = xs.view(1, 1, nx)            # broadcast over (nz, ny, nx)
    Y = ys.view(1, ny, 1)
    Z = zs.view(nz, 1, 1)

    n_b, n_a = sino_f.shape[-2], sino_f.shape[-1]
    vol = torch.zeros((nz, ny, nx), device=device, dtype=torch.float32)

    for i in range(int(angles.numel())):
        t = float(angles[i])
        ct, st = float(np.cos(t)), float(np.sin(t))

        U = R_s + X * ct + Y * st                      # (1, ny, nx)
        ratio = SDD / U.clamp_min(1e-6)
        p_u = ratio * (-X * st + Y * ct)               # (1, ny, nx)
        p_v = ratio * Z                                # (nz, ny, nx)

        a_idx = cpa + (c_psi * p_u + s_psi * p_v) / da
        b_idx = cpb + (-s_psi * p_u + c_psi * p_v) / db

        # normalized grid_sample coords over the (n_b, n_a) projection
        gx = (2.0 * a_idx / (n_a - 1) - 1.0).expand(nz, ny, nx)
        gy = 2.0 * b_idx / (n_b - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(1, nz, ny * nx, 2)

        proj = sino_f[i].reshape(1, 1, n_b, n_a)
        sampled = F.grid_sample(proj, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        vol += sampled.reshape(nz, ny, nx)

    return vol.cpu()


# --------------------------------------------------------------------------
# half-scan consistency scoring (verbatim from muNeRF)
# --------------------------------------------------------------------------

def grad_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean NCC of gradient-magnitude images — registration-free."""
    ga = np.hypot(*np.gradient(a))
    gb = np.hypot(*np.gradient(b))
    ga -= ga.mean()
    gb -= gb.mean()
    denom = np.sqrt((ga ** 2).sum() * (gb ** 2).sum())
    return float((ga * gb).sum() / max(denom, 1e-30))


def pool_sinogram(sino: torch.Tensor, geometry: dict, f: int):
    """Average-pool the detector a further f x and rescale the geometry.

    Index conversion: output pixel j pools raw [j*f, j*f+f-1], centroid
    j*f+(f-1)/2, so idx_pooled = (idx-(f-1)/2)/f. Pitch scales by f.
    Angles untouched. Returns (pooled sinogram, adjusted geometry copy).
    """
    if f == 1:
        return sino, dict(geometry)
    nb = (sino.shape[1] // f) * f
    na = (sino.shape[2] // f) * f
    pooled = F.avg_pool2d(sino[:, :nb, :na].unsqueeze(1),
                          kernel_size=f).squeeze(1).contiguous()
    g = dict(geometry)
    g["da"] = float(g["da"]) * f
    g["db"] = float(g["db"]) * f
    to_p = lambda idx: (idx - (f - 1) / 2.0) / f                # noqa: E731
    g["central_pixel_a"] = to_p(float(g["central_pixel_a"]))
    g["central_pixel_b"] = to_p(float(g["central_pixel_b"]))
    g["sinogram_downsample"] = int(g.get("sinogram_downsample", 1)) * f
    return pooled, g


def _half_pair(sino_f, angles, geometry, z, vox_mm, device):
    """FBP slabs from the two halves of the view range, current geometry."""
    n_half = int(angles.numel()) // 2
    out = []
    for sl in (slice(0, n_half), slice(n_half, None)):
        vol = backproject_slab(sino_f[sl], angles[sl], geometry,
                               z, 1.0, vox_mm, 0.20, device)
        out.append(vol[vol.shape[0] // 2].numpy())
    return out


def scan_with_extension(score_fn, grid, step, max_ext=MAX_GRID_EXTENSIONS,
                        lo_bound=-MAX_PSI_DEG, hi_bound=MAX_PSI_DEG,
                        verbose=True):
    """Evaluate score_fn over `grid`; if the argmax lands on an edge, extend
    the grid 3 steps past that edge (bounded), up to `max_ext` times. Returns
    (xs, ys, edge_peaked) with xs sorted ascending."""
    xs = list(grid)
    ys = [score_fn(x) for x in xs]
    for _ in range(max_ext):
        i = int(np.argmax(ys))
        if 0 < i < len(xs) - 1:
            return np.array(xs), np.array(ys), False
        if i == 0:
            new = [xs[0] - step * k for k in range(1, 4)
                   if xs[0] - step * k >= lo_bound]
        else:
            new = [xs[-1] + step * k for k in range(1, 4)
                   if xs[-1] + step * k <= hi_bound]
        if not new:
            return np.array(xs), np.array(ys), True
        if verbose:
            print(f"    peak at grid edge — extending by {len(new)} steps")
        for x in sorted(new):
            j = int(np.searchsorted(xs, x))
            xs.insert(j, x)
            ys.insert(j, score_fn(x))
    i = int(np.argmax(ys))
    return np.array(xs), np.array(ys), (i in (0, len(xs) - 1))


def _parabolic_max(xs, ys):
    i = int(np.argmax(ys))
    if 0 < i < len(xs) - 1:
        x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom < 0:
            return float(x1 + 0.5 * (x0 - x2) * (y0 - y2) / (2 * denom))
    return float(xs[i])


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

def estimate_psi_halfncc(sinogram: torch.Tensor, angles: torch.Tensor,
                         geometry: dict, *, downsample: int = 1,
                         device=None, verbose: bool = True) -> dict[str, Any]:
    """Two-stage half-scan-consistency estimate of (psi, cpa0-diagnostic).

    Args mirror the muNeRF original: `sinogram` = LINE INTEGRALS (flat-fielded
    + log), shape (N_angles, N_b, N_a), at detector downsample `downsample`
    relative to raw; `geometry` = the build_geometry dict.

    Stage 1 (coarse): detector pooled to ~raw-ds-9, 0.15 mm voxels, psi grid
    +-1.2 deg step 0.15 (extended if the peak sits on an edge).
    Stage 2 (refine): ~raw-ds-3, 0.05 mm voxels, step 0.05 around the coarse
    peak, parabolic vertex.
    Then a cpa scan at the fitted psi — DIAGNOSTIC ONLY (the applied CoR
    policy is the geometric centre; a large fitted offset is printed loudly).

    Raises RuntimeError when the data cannot support the estimate (too few
    angles, flat curve, edge-locked peak, |psi| out of bounds) so the caller
    can fall back — this runs unattended ahead of multi-hour jobs.
    """
    t00 = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_ang = int(sinogram.shape[0])
    if n_ang < MIN_ANGLES:
        raise RuntimeError(f"only {n_ang} projections — half-scan "
                           f"consistency needs >= {MIN_ANGLES}")

    # sort by angle (defensive; a wrapped/unsorted array breaks the halves)
    order = torch.argsort(angles)
    angles = angles[order].to(torch.float32)
    sinogram = sinogram[order]

    g0 = dict(geometry)
    g0.setdefault("sinogram_downsample", int(downsample))
    ozn = float(g0["vol_origin"][2])
    nz = int(g0["vol_shape"][2])
    dz = float(g0["dz"])

    # Planes at the z extremes: psi error is antisymmetric in z, cpa0 error
    # symmetric, so the pair constrains both. DEVIATION from the muNeRF
    # original (which always ran on a scanner-ROI-sized volume): the recon
    # drivers may request a z-FOV taller than the detector actually covers,
    # so clamp the planes to the detector's axial coverage at isocentre —
    # a plab outside coverage backprojects to zeros and scores nothing.
    n_b_in = int(sinogram.shape[1])
    z_cov_half = (n_b_in * float(g0["db"]) / 2.0) \
        * float(g0["R_s"]) / (float(g0["R_s"]) + float(g0["R_d"]))
    z_hi = min(0.45 * nz * dz, 0.45 * 2 * z_cov_half)
    z_planes = (ozn + z_hi, ozn - z_hi)

    def make_stage(target_raw_ds, vox_mm):
        f = max(1, int(round(target_raw_ds / max(1, int(downsample)))))
        sino_p, g_p = pool_sinogram(sinogram, g0, f)
        sino_p = sino_p.to(device, torch.float32)
        sf = ramp_filter(sino_p)
        centre = (sino_p.shape[2] - 1) / 2.0

        def score(psi_deg, cpa_offset=0.0):
            g_p["det_psi_rad"] = float(np.radians(psi_deg))
            g_p["central_pixel_a"] = centre + float(cpa_offset)
            s = 0.0
            for z in z_planes:
                a, b = _half_pair(sf, angles, g_p, z, vox_mm, device)
                s += grad_ncc(a, b)
            return s / len(z_planes)

        return score, f, centre

    # ---- stage 1: coarse -------------------------------------------------
    score1, f1, _ = make_stage(9, 0.15)
    grid1 = np.arange(-1.2, 1.2 + 1e-9, 0.15)
    xs, ys, edge = scan_with_extension(score1, grid1, 0.15, verbose=verbose)
    if edge:
        raise RuntimeError("coarse peak locked to the grid edge even after "
                           "extension — curve untrustworthy")
    prominence = float((ys.max() - ys.min()) / max(abs(ys.min()), 1e-30))
    psi_coarse = _parabolic_max(xs, ys)
    if verbose:
        print(f"  half-NCC stage 1 (pool {f1}x): peak {psi_coarse:+.3f} deg, "
              f"prominence {prominence:.3f}, {time.time()-t00:.0f}s")
    if prominence < MIN_PROMINENCE:
        raise RuntimeError(f"coarse curve nearly flat (prominence "
                           f"{prominence:.3f} < {MIN_PROMINENCE}) — the data "
                           f"does not determine psi")

    # ---- stage 2: refine -------------------------------------------------
    # Same edge-extension as stage 1: the coarse peak location carries a
    # resolution bias of up to ~0.15 deg (measured on Scan_1510: coarse -0.61
    # vs fine -0.78), so the fine peak can fall outside a fixed +-0.15 window
    # — clipping there would silently return the window edge.
    t1 = time.time()
    score2, f2, centre2 = make_stage(3, 0.05)
    grid2 = psi_coarse + np.arange(-3, 4) * 0.05
    xs2, ys2, edge2 = scan_with_extension(score2, grid2, 0.05,
                                          verbose=verbose)
    if edge2:
        raise RuntimeError("refined peak locked to the grid edge even after "
                           "extension")
    psi = _parabolic_max(xs2, ys2)
    if not np.isfinite(psi) or abs(psi) > MAX_PSI_DEG:
        raise RuntimeError(f"refined psi {psi:+.3f} outside +-{MAX_PSI_DEG}")
    if verbose:
        print(f"  half-NCC stage 2 (pool {f2}x): psi {psi:+.4f} deg, "
              f"{time.time()-t1:.0f}s")

    # ---- cpa diagnostic --------------------------------------------------
    t2 = time.time()
    cpa_offsets = (-1.0, 0.0, 1.0)
    cpa_ys = np.array([score2(psi, o) for o in cpa_offsets])
    cpa_off = _parabolic_max(np.array(cpa_offsets), cpa_ys)
    cpa_off_input_px = float(cpa_off) * f2      # back to input-ds columns
    if verbose:
        print(f"  half-NCC cpa diagnostic: centre {cpa_off:+.3f} pooled px "
              f"(= {cpa_off_input_px:+.3f} input px), {time.time()-t2:.0f}s")
        if abs(cpa_off) > 0.75:
            print("  WARNING: fitted column-CoR is far from the geometric "
                  "centre — the applied policy (geom_center) may be wrong "
                  "for this scan; investigate before trusting the recon")

    n_a_in = int(sinogram.shape[2])
    centre_in = (n_a_in - 1) / 2.0
    return {
        "method": "halfncc",
        "psi_deg": float(psi),
        "cpa0": centre_in + cpa_off_input_px,   # input-ds detector index
        "geom_centre": centre_in,
        "cpa0_offset_px": cpa_off_input_px,
        "prominence": prominence,
        "coarse_grid": xs.tolist(), "coarse_ncc": ys.tolist(),
        "refine_grid": xs2.tolist(), "refine_ncc": ys2.tolist(),
        "cpa_offsets_pooled": list(cpa_offsets),
        "cpa_ncc": cpa_ys.tolist(),
        "planes_mm": [float(z) for z in z_planes],
        "elapsed_s": time.time() - t00,
    }


# --------------------------------------------------------------------------
# raw-projection front end + calibration JSON writer
# --------------------------------------------------------------------------

def prepare_estimation_sinogram(projections, bright_field, dark_field,
                                factor: int = 3, chunk_angles: int = 20,
                                device=None) -> torch.Tensor:
    """Chunked flat-field + log + detector pooling, for the estimator only.

    Avoids materializing the full-resolution float32 sinogram (~7 GB on a
    2296x3500 detector): each chunk of angles is flat-fielded, logged, and
    average-pooled by `factor` before the next chunk loads. The estimator's
    stages never need more than raw-ds-3. Simple hard clipping is used (the
    NCC score is insensitive to the clamp flavour — both halves share it).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_ang, n_b, n_a = projections.shape
    denom = torch.from_numpy(
        np.clip(np.asarray(bright_field, dtype=np.float32)
                - np.asarray(dark_field, dtype=np.float32), 1.0, None)
    ).to(device)
    dark = torch.from_numpy(
        np.asarray(dark_field, dtype=np.float32)).to(device)

    out = []
    for start in range(0, n_ang, chunk_angles):
        end = min(start + chunk_angles, n_ang)
        chunk = torch.from_numpy(
            np.array(projections[start:end], dtype=np.float32)).to(device)
        T = ((chunk - dark) / denom).clamp(1e-6, 1.05)
        p = -torch.log(T)
        if factor > 1:
            nb = (n_b // factor) * factor
            na = (n_a // factor) * factor
            p = F.avg_pool2d(p[:, :nb, :na].unsqueeze(1),
                             kernel_size=factor).squeeze(1)
        out.append(p.cpu())
    return torch.cat(out, dim=0).contiguous()


def calibration_json_path(repo_root, detector_serial: str,
                          scan_tag: str) -> Path:
    """The scan-keyed calibration file shared with muNeRF.

    ``repo_root=None`` resolves the shared calibration directory through
    ``ct_core.paths`` (env override, then an existing data/calibration in any
    ancestor, then the project root) instead of assuming a checkout layout.
    """
    base = (calibration_dir() if repo_root is None
            else Path(repo_root) / "data" / "calibration")
    return base / f"detector_psi_{detector_serial}_{scan_tag}.json"


def write_calibration_json(path: Path, result: dict, *, downsample: int,
                           detector_serial: Optional[str]) -> None:
    """Persist an estimate in the schema muNeRF reads (and upgrades)."""
    ds = int(downsample)
    record = {
        "psi_deg": float(result["psi_deg"]),
        "cpa0": float(result["cpa0"]),
        # raw-detector column index: idx_raw = idx_ds * ds + (ds-1)/2
        "cpa0_raw": float(result["cpa0"]) * ds + (ds - 1) / 2.0,
        "cpa0_offset_px": float(result.get("cpa0_offset_px", 0.0)),
        "prominence": float(result.get("prominence", 0.0)),
        "method": result.get("method", "halfncc"),
        "downsample": ds,
        "detector_serial": detector_serial,
        "measured_on": date.today().isoformat(),
        "elapsed_s": float(result.get("elapsed_s", 0.0)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
