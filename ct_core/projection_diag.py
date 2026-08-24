"""Projection-domain diagnostics shared by every reconstruction backend.

Ported from muNeRF's ``inr_pipeline/metrics.py`` + ``train.py`` diagnostics
(2026-08-13) so ALL backends — FDK, ASTRA, TIGRE, voxel — report the same
projection-quality evidence:

  * ``ssim_2d`` / ``psnr_2d``   — canonical pure-torch implementations
    (muNeRF's ``inr_pipeline.metrics`` re-exports these; single identity).
  * ``evaluate_projection``     — SSIM / PSNR / MSE of a predicted vs
    measured projection (numpy in, plain floats out).
  * ``measure_noise_ceiling``   — the noise-limited SSIM/PSNR ceiling and
    MSE floor from two independent measurements of the same line integrals:
    the OTHER acquisition phase when the scan has one (e.g. Scan_1510's
    acq-01 frames), else the neighbouring projection (conservative — it
    also carries ~one angular step of real signal change).
  * ``render_projection_from_volume`` — forward-project a finished volume
    at one angle through the canonical ray tracer, so non-streaming
    backends (FDK, ASTRA) get the same diagnostics as iterative ones.
  * ``ssim_heatmap_figure`` / ``power_spectrum_figure`` — the muNeRF
    diagnostic figures, without their FDK-baseline panels.

The evaluation projection is by convention the CENTRAL angle
(``n_angles // 2``) — the same projection every holdout/crossval scheme in
this repo has always used — and by default it stays IN the reconstruction
(diagnostic, not validation). Drivers expose ``--withhold-eval`` to turn it
into a true held-out validation projection.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from matplotlib.figure import Figure


# --------------------------------------------------------------------------
# SSIM / PSNR (canonical home — muNeRF's inr_pipeline.metrics re-exports)
# --------------------------------------------------------------------------

def ssim_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
    mask: torch.Tensor | None = None,
    return_map: bool = False,
) -> torch.Tensor:
    """Single-channel SSIM on two 2-D tensors. Returns a scalar.

    Follows Wang et al. 2004: separable Gaussian window, K1=0.01, K2=0.03.

    `data_range` is the dynamic range used in the C1/C2 stabilizers; if None,
    we use `target.max() - target.min()` (matches skimage's default behavior).
    Pass an explicit value when you want SSIM to be comparable across iters.

    `mask` (optional, bool, same shape as pred): True = valid pixel to include.
    The SSIM map is cropped by (window_size-1)/2 on each side due to 'valid'
    convolution, so the mask is center-cropped to match before averaging.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.ndim != 2:
        raise ValueError(f"ssim_2d expects 2-D inputs, got {pred.ndim}-D")

    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range == 0:
            return torch.tensor(1.0 if torch.equal(pred, target) else 0.0,
                                dtype=pred.dtype, device=pred.device)

    # The window, the stabiliser constants and the variance arithmetic are
    # `ct_core.ssim`'s — the same code the learned backends' structural LOSS
    # runs, so a metric and a loss on the same pair can never disagree about
    # what SSIM means. Only the envelope below (mask, map-vs-scalar) is local.
    from .ssim import ssim_components, to_bchw

    p = to_bchw(pred)
    t = to_bchw(target)
    ssim_map, _ = ssim_components(p, t, data_range=data_range,
                                  window_size=window_size, sigma=sigma)
    if return_map:
        return ssim_map.squeeze(0).squeeze(0)  # (H', W')

    if mask is not None:
        pad = (window_size - 1) // 2
        mask_crop = mask[pad:-pad, pad:-pad] if pad > 0 else mask
        mask_crop = mask_crop.to(ssim_map.device).reshape(ssim_map.shape)
        return ssim_map[mask_crop].mean()
    return ssim_map.mean()


def psnr_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PSNR in dB, scalar tensor.

    `mask` (optional, bool, same shape as pred): True = valid pixel.
    MSE is computed only over masked pixels.
    """
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range == 0:
            data_range = 1.0
    if mask is not None:
        diff = (pred[mask] - target[mask])
        mse = (diff * diff).mean()
    else:
        mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return torch.tensor(float("inf"), dtype=pred.dtype, device=pred.device)
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, dtype=pred.dtype, device=pred.device) / mse)


def _to_tensor(a) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


def evaluate_projection(pred, target) -> dict:
    """SSIM / PSNR / MSE of a predicted vs measured 2-D projection.

    Accepts numpy arrays (or tensors); returns plain floats. `data_range`
    comes from the target so values are comparable across iterations.
    """
    p, t = _to_tensor(pred), _to_tensor(target)
    dr = float(t.max() - t.min())
    if dr <= 0:
        dr = 1.0
    # SSIM's 11x11 window needs the image to be at least that big; on tiny
    # projections (small test scans / aggressive strides) shrink it to the
    # largest odd size that fits, floored at 3.
    win = min(11, int(min(p.shape)))
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    return {
        "ssim": float(ssim_2d(p, t, data_range=dr, window_size=win)),
        "psnr": float(psnr_2d(p, t, data_range=dr)),
        "mse": float(torch.mean((p - t) ** 2)),
    }


# --------------------------------------------------------------------------
# Noise ceiling — two independent measurements of the same line integrals
# --------------------------------------------------------------------------

def preprocess_frames(frames: np.ndarray, ctx) -> np.ndarray:
    """Flat-field + log a small stack of raw frames, identically to the
    backends' own preprocessing. Ring correction is deliberately skipped: the
    ring pattern is a STATIC detector column offset, identical on both sides
    of every noise-ceiling pair, so it cancels in the comparison.

    Air normalization is deliberately KEPT, for the mirror-image reason: the
    gain offset is per-frame, so it does NOT cancel — the two phases of a
    noise-ceiling pair are acquired at different times and drift apart by up
    to 0.023 on Scan_1510. Leaving it in would inflate the MSE floor with a
    constant the model is not being asked to reproduce, and would score the
    reconstruction against an air level its own sinogram no longer has.
    """
    from .preprocessing import preprocess_sinogram
    return preprocess_sinogram(
        np.ascontiguousarray(frames, dtype=np.float32),
        ctx.bright_field, ctx.dark_field,
        air_normalization=bool(getattr(ctx, 'air_normalization', True)),
    )


def _other_phase_frame(ctx, index: int, phase: str):
    """Load raw frame `index` of a DIFFERENT acquisition phase, downsampled
    like ctx.projections. Returns (frame, phase_tag) or (None, None)."""
    try:
        folder = Path(ctx.data_folder)
        tags = set()
        for p in folder.glob("acq*.vff"):
            m = re.search(r"-(\d+)-", p.name)
            if m:
                tags.add(m.group(1))
        others = sorted(t for t in tags if t != phase)
        if not others:
            return None, None
        other = others[0]
        paths = sorted(p for p in folder.glob("acq*.vff") if f"-{other}-" in p.name)
        if index >= len(paths):
            return None, None
        from .vff_io import read_vff
        _hdr, dat = read_vff(str(paths[index]), verbose=False)
        frame = np.asarray(dat, dtype=np.float32).squeeze()
        if frame.ndim != 2:
            return None, None
        if ctx.downsample > 1:
            from .preprocessing import downsample_projections
            frame = downsample_projections(frame[None], ctx.downsample)[0]
        return frame, other
    except Exception as e:
        print(f"  [noise-ceiling] other-phase frame unavailable "
              f"({type(e).__name__}: {e})")
        return None, None


def measure_noise_ceiling(ctx, eval_index: int, phase: str = "00",
                          verbose: bool = True):
    """Noise-limited SSIM/PSNR ceiling + MSE floor at the evaluation angle.

    Compares the measured evaluation projection against a second independent
    measurement of (nearly) the same line integrals:

      * OTHER PHASE (preferred): the same angle from another acquisition pass
        (e.g. acq-01). Differences are pure noise (+ any physiological
        motion), so SSIM/PSNR between the two IS the reachable ceiling, and
        with b = s+n, b' = s+n': E[(b-b')^2] = 2 sigma^2, so the best MSE any
        model can honestly reach is  0.5 * mean((b-b')^2).
      * NEIGHBOURING PROJECTION (fallback): one angular step of real signal
        change is included, so the ceiling is biased LOW (pessimistic) and
        the MSE floor biased HIGH.

    Returns a dict {ssim, psnr, mse, source, pair} (pair = the two
    preprocessed frames, kept for the ceiling panels of the diagnostic
    figures), or None if no second frame could be formed.
    """
    n = int(ctx.projections.shape[0])
    if n < 2:
        return None
    p0_raw = np.asarray(ctx.projections[eval_index], dtype=np.float32)

    pair_raw, other_tag = _other_phase_frame(ctx, eval_index, phase)
    if pair_raw is not None and pair_raw.shape == p0_raw.shape:
        source = f"other phase (acq-{other_tag}, same angle)"
        conservative = False
    else:
        j = eval_index + 1 if eval_index + 1 < n else eval_index - 1
        pair_raw = np.asarray(ctx.projections[j], dtype=np.float32)
        source = f"neighbouring projection (index {j})"
        conservative = True

    pp = preprocess_frames(np.stack([p0_raw, pair_raw]), ctx)
    # Same covered window as every other diagnostic in this module, so the
    # ceiling is measured on exactly the pixels the recon is scored on (and so
    # the stored pair aligns pixel-for-pixel with rendered predictions).
    # This matters more than it looks: outside the window the ceiling compares
    # two MEASUREMENTS, which share the flat field's residual air offset, while
    # the model — whose density cannot go negative — is structurally barred
    # from reproducing it. Scoring both there inflates the reported gap between
    # the reconstruction and its ceiling.
    try:
        b0, b1, a0, a1 = covered_detector_window(
            ctx.geometry, pp.shape[1], pp.shape[2])
        pp = pp[:, b0:b1, a0:a1]
    except (KeyError, TypeError):
        pass  # geometry incomplete (tests) — use the full frame
    p0, p1 = pp[0], pp[1]
    m = evaluate_projection(p1, p0)
    ceiling = {
        "ssim": m["ssim"],
        "psnr": m["psnr"],
        "mse": 0.5 * float(np.mean((p0 - p1) ** 2)),  # sigma^2-equivalent floor
        "source": source,
        "pair": (p0, p1),
    }
    if verbose:
        print(f"\n  [noise-ceiling] second measurement: {source}")
        print(f"  [noise-ceiling] ssim={ceiling['ssim']:.4f}  "
              f"psnr={ceiling['psnr']:.2f} dB  "
              f"mse floor (sigma^2-equivalent)={ceiling['mse']:.6e}")
        if conservative:
            print("  [noise-ceiling] (conservative: the neighbouring "
                  "projection also carries one angular step of real signal "
                  "change — ssim/psnr biased low, mse floor biased high)")
    return ceiling


# --------------------------------------------------------------------------
# Forward-project a finished volume at the evaluation angle
# --------------------------------------------------------------------------

def covered_detector_rows(geometry: dict) -> tuple[int, int]:
    """Detector-row interval [b_min, b_max) whose rays stay inside the
    reconstruction z-slab across the whole in-plane FOV chord.

    The reconstruction FOV is usually a THIN slab compared to the detector's
    axial coverage (e.g. 40 mm of z against ~65 mm at the isocenter): rays
    through the outer rows exit the slab and integrate through matter the
    volume simply does not contain, so any forward projection under-predicts
    there BY CONSTRUCTION. Comparing those rows would score FOV truncation,
    not reconstruction quality — every projection diagnostic in this module
    is therefore restricted to this interval.

    A row's ray reaches its largest |z| at the far edge of the FOV cylinder
    (distance R_s + r from the source): z = b_off * (R_s + r) / SDD.
    """
    R_s = float(geometry["R_s"])
    SDD = R_s + float(geometry["R_d"])
    db = float(geometry["db"])
    cpb = float(geometry["central_pixel_b"])
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    dx, dz = float(geometry["dx"]), float(geometry["dz"])
    oz = float(geometry["vol_origin"][2])
    hz = Nz * dz / 2.0
    r = min(Nx, Ny) * dx / 2.0
    reach = (R_s + r) / SDD
    b_lo = cpb + (oz - hz) / reach / db
    b_hi = cpb + (oz + hz) / reach / db
    return int(math.ceil(min(b_lo, b_hi))), int(math.floor(max(b_lo, b_hi))) + 1


def covered_detector_columns(geometry: dict) -> tuple[int, int]:
    """Detector-column interval [a_min, a_max) whose rays cross the in-plane
    reconstruction domain.

    The counterpart of ``covered_detector_rows`` for the other detector axis.
    The domain is a cylinder of radius r about the rotation axis — the
    renderer integrates nothing outside it — while the detector sees the whole
    fan. A ray through a column beyond this interval never enters the cylinder,
    so its forward projection is EXACTLY zero by construction. Scoring those
    columns measures the FOV, not the reconstruction, and it does so harshly:
    SSIM's luminance term is a ratio of local means, so a rendered zero against
    a measured air level that flat-fielding left slightly below zero collapses
    to ~0 (or negative) even though the absolute error is tiny.

    A ray at fan angle gamma passes the axis at perpendicular distance
    R_s*sin(gamma), so it clips the cylinder iff |sin(gamma)| <= r/R_s, i.e.

        |a_off| <= SDD * r / sqrt(R_s^2 - r^2).

    Note this is WIDER than the isocentre magnification r*SDD/R_s that the
    support measurement uses to map the other way: the tangent ray touches the
    cylinder nearer the source than the isocentre plane, so scaling by the
    isocentre magnification would clip real columns.

    An off-centre domain (an ROI grid under ``--model-domain off``) drifts
    relative to the source as the gantry turns, so its offset is added to the
    radius and the interval becomes the union over angles — never narrower
    than the truth, at worst a few columns too generous.
    """
    R_s = float(geometry["R_s"])
    SDD = R_s + float(geometry["R_d"])
    da = float(geometry["da"])
    cpa = float(geometry["central_pixel_a"])
    Nx, Ny, _Nz = (int(v) for v in geometry["vol_shape"])
    dx = float(geometry["dx"])
    ox, oy = (float(v) for v in geometry["vol_origin"][:2])
    r = min(Nx, Ny) * dx / 2.0 + math.hypot(ox, oy)
    if r >= R_s:
        # Domain reaches the source: every ray is inside it. Return an
        # interval the caller's clamp collapses onto the full detector.
        return 0, 1 << 30
    a_off = SDD * r / math.sqrt(R_s * R_s - r * r)
    return (int(math.ceil(cpa - a_off / da)),
            int(math.floor(cpa + a_off / da)) + 1)


def covered_detector_window(geometry: dict, n_b: int, n_a: int,
                            min_size: int = 16) -> tuple[int, int, int, int]:
    """(b_min, b_max, a_min, a_max): the detector rectangle the forward model
    can actually predict, clamped to the frame.

    Every projection diagnostic in the repo scores this window and nothing
    else, so SSIM/PSNR/MSE and the noise ceiling are all measured on the same
    pixels. An axis whose covered band comes out degenerate (< `min_size`
    pixels — pathological geometry, or the tiny synthetic frames in the tests)
    falls back to the full extent rather than to an unusable sliver.
    """
    b0, b1 = covered_detector_rows(geometry)
    b0, b1 = max(0, b0), min(int(n_b), b1)
    if b1 - b0 < min_size:
        b0, b1 = 0, int(n_b)
    a0, a1 = covered_detector_columns(geometry)
    a0, a1 = max(0, a0), min(int(n_a), a1)
    if a1 - a0 < min_size:
        a0, a1 = 0, int(n_a)
    return b0, b1, a0, a1


def render_projection_from_volume(volume, ctx, angle_index: int,
                                  measured: np.ndarray, *,
                                  volume_is_hu: bool = True,
                                  mu_water: float | None = None,
                                  downsample: int = 2,
                                  samples_per_ray: int | None = None,
                                  chunk_size: int = 8192,
                                  geometry: dict | None = None,
                                  window: tuple | None = None,
                                  device=None):
    """Render one projection through a reconstructed volume.

    Uses the canonical ray tracer (learning_based_iterative scene/renderer)
    with a trilinear VoxelGrid wrapping `volume`, so FDK/ASTRA volumes get
    the exact same forward model the learning-based backends train with.

    `volume` is (Nx, Ny, Nz) spanning geometry['vol_shape'] (the backend
    contract); `measured` is the PREPROCESSED evaluation projection
    (N_b, N_a). Returns (pred, target) numpy arrays at `downsample` stride.

    `geometry` overrides `ctx.geometry`. A driver reconstructing its own
    volume never needs it; a POST-HOC caller scoring a foreign volume does,
    because the grid that volume lives on is a property of the file, not of
    the scan — pass ctx.geometry with vol_shape/vol_origin/dx/dz replaced by
    the volume's own and every other entry (the projection geometry) left
    alone. That forward-projects each volume at its NATIVE resolution, which
    is what a comparison wants: resampling onto a common lattice would soften
    the sharper volume before measuring how sharp it is.

    `window` pins the scored detector rectangle to (b_min, b_max, a_min,
    a_max) instead of deriving it from the domain. THIS MATTERS FOR ANY
    COMPARISON: the window `covered_detector_window` returns depends on the
    reconstruction domain, so two volumes with different extents would
    otherwise be scored over different parts of the detector and their
    SSIM/PSNR would not be comparable. Pass the intersection over every
    volume in the comparison.
    """
    from ..learning_based_iterative.scene import Scene, model_domain_from_geometry
    from ..learning_based_iterative.projection import render_projection
    from ..learning_based_iterative.voxel.model import VoxelGrid
    from .calibration import MU_WATER_80KV
    from .hu_calibration import fixed_anchors

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("  [diag] rendering final projection on CPU (no CUDA) — "
                  "this can take a few minutes.")

    vol = np.asarray(volume, dtype=np.float32)
    if volume_is_hu:
        # Legacy path: the drivers all pass volume_is_hu=False now, because
        # every backend returns mu. Converting HU back to mu needs a scale,
        # and with no BHC to switch on there is only the one legacy constant.
        mu_w = MU_WATER_80KV if mu_water is None else float(mu_water)
        vol = fixed_anchors(mu_w).invert(vol)
    vol = np.clip(vol, 0.0, None)

    Nx, Ny, Nz = vol.shape
    model = VoxelGrid((Nz, Ny, Nx), init_density=0.0)
    model.load_volume(torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 1, 0))))
    model = model.to(device)

    geometry = dict(ctx.geometry if geometry is None else geometry)
    domain = model_domain_from_geometry(geometry, shape="cylinder")
    sino = torch.from_numpy(np.ascontiguousarray(measured, dtype=np.float32))[None]
    angles_t = torch.as_tensor([float(np.asarray(ctx.angles)[angle_index])],
                               dtype=torch.float32)
    scene = Scene(sinogram=sino, angles=angles_t, geometry=geometry,
                  scan_name="", model_domain=domain)

    if samples_per_ray is None:
        dx = float(geometry["dx"])
        chord = (2.0 * float(domain.radius_xy) if domain.radius_xy is not None
                 else float((domain.aabb_max - domain.aabb_min).max()))
        samples_per_ray = max(64, int(math.ceil(chord / (0.55 * dx))))

    n_b, n_a = scene.detector_shape
    if window is None:
        b_min, b_max, a_min, a_max = covered_detector_window(geometry, n_b, n_a)
    else:
        b_min, b_max, a_min, a_max = (int(v) for v in window)
    if (b_max - b_min, a_max - a_min) != (n_b, n_a):
        print(f"  [diag] evaluating detector rows [{b_min}, {b_max}) of {n_b} "
              f"and columns [{a_min}, {a_max}) of {n_a} — rays outside exit "
              f"the reconstruction domain and cannot be predicted from the "
              f"volume.")
    # The scene holds this one angle only, so the angle index is 0.
    pred, target = render_projection(
        model, scene, 0, samples_per_ray, device,
        downsample=downsample, b_range=(b_min, b_max), a_range=(a_min, a_max),
        chunk_size=chunk_size)
    return pred.cpu().numpy(), target.cpu().numpy()


# --------------------------------------------------------------------------
# Diagnostic figures (muNeRF's, without the FDK-baseline panels)
# --------------------------------------------------------------------------

def _match_to(pair, shape):
    """Stride-subsample a noise-ceiling pair to a prediction's grid."""
    out = []
    for q in pair:
        q = np.asarray(q, dtype=np.float32)
        fb = max(1, int(round(q.shape[0] / shape[0])))
        fa = max(1, int(round(q.shape[1] / shape[1])))
        q = q[::fb, ::fa][:shape[0], :shape[1]]
        out.append(q)
    return out


def ssim_heatmap_figure(pred, target, noise_pair=None, title: str = "") -> Figure:
    """Local-SSIM map + projections. 2x2: [SSIM map, ceiling map | residual]
    over [predicted projection, measured projection]."""
    p, t = _to_tensor(pred), _to_tensor(target)
    dr = float(t.max() - t.min()) or 1.0
    win = max(3, min(11, int(min(p.shape)) - (1 - int(min(p.shape)) % 2)))
    model_map = ssim_2d(p, t, data_range=dr, window_size=win,
                        return_map=True).numpy()

    ceil_map = None
    if noise_pair is not None:
        q0, q1 = _match_to(noise_pair, tuple(p.shape))
        if q0.shape == tuple(p.shape):
            qdr = float(q0.max() - q0.min()) or 1.0
            ceil_map = ssim_2d(_to_tensor(q1), _to_tensor(q0),
                               data_range=qdr, window_size=win,
                               return_map=True).numpy()

    fig = Figure(figsize=(11, 9), dpi=110)
    axes = fig.subplots(2, 2)

    im0 = axes[0, 0].imshow(model_map, aspect="auto", cmap="RdYlGn",
                            vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(f"SSIM: recon (mean={model_map.mean():.4f})",
                         fontsize=10)
    if ceil_map is not None:
        axes[0, 1].imshow(ceil_map, aspect="auto", cmap="RdYlGn",
                          vmin=0.0, vmax=1.0)
        axes[0, 1].set_title(
            f"SSIM: noise ceiling (mean={ceil_map.mean():.4f})", fontsize=10)
    else:
        resid = (p - t).numpy()
        lim = float(np.percentile(np.abs(resid), 99)) or 1.0
        axes[0, 1].imshow(resid, aspect="auto", cmap="RdBu_r",
                          vmin=-lim, vmax=lim)
        axes[0, 1].set_title("residual (pred − measured)", fontsize=10)

    vmin, vmax = float(t.min()), float(t.max())
    axes[1, 0].imshow(p.numpy(), aspect="auto", cmap="gray",
                      vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("projection: recon (forward-projected)", fontsize=10)
    axes[1, 1].imshow(t.numpy(), aspect="auto", cmap="gray",
                      vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("projection: measured", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    fig.colorbar(im0, cax=cax, label="local SSIM")
    fig.suptitle(f"Local SSIM & projections{title}", fontsize=12)
    return fig


def power_spectrum_figure(pred, target, det_px_mm: float,
                          noise_pair=None, title: str = "") -> Figure:
    """Row-wise projection power spectra: measured vs rendered vs (optional)
    noise floor, with 'missing signal' and 'reducible error' bands."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    def _row_ps(img):
        return (np.abs(np.fft.rfft(img, axis=1)) ** 2).mean(axis=0)

    ps_measured = _row_ps(target)
    ps_model = _row_ps(pred)
    ps_residual = _row_ps(pred - target)

    ps_noise = None
    if noise_pair is not None:
        q0, q1 = _match_to(noise_pair, pred.shape)
        if q0.shape == pred.shape:
            ps_noise = _row_ps((q1 - q0) / np.sqrt(2.0))

    freqs = np.fft.rfftfreq(pred.shape[1]) / det_px_mm  # lp/mm

    fig = Figure(figsize=(9, 5.5), dpi=110)
    ax = fig.add_subplot(111)
    ax.semilogy(freqs, ps_measured, label="measured (signal+noise)",
                color="tab:blue", lw=1.5, alpha=0.85)
    ax.semilogy(freqs, ps_model, label="recon forward-projected",
                color="tab:orange", lw=1.5, alpha=0.85)
    if ps_noise is not None:
        ax.semilogy(freqs, ps_noise, "--", label="noise floor",
                    color="tab:green", lw=1.5, alpha=0.85)
        ax.fill_between(freqs, ps_noise, ps_residual,
                        where=ps_residual > ps_noise,
                        alpha=0.15, color="purple", label="reducible error")
    ax.fill_between(freqs, ps_model, ps_measured,
                    where=ps_measured > ps_model,
                    alpha=0.15, color="red", label="missing signal")
    ax.set_xlabel("spatial frequency (lp/mm)")
    ax.set_ylabel("power")
    ax.set_title(f"Projection power spectra{title}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = ax.secondary_xaxis(
        "top", functions=(lambda x: x * det_px_mm, lambda x: x / det_px_mm))
    ax2.set_xlabel("cycles/pixel", fontsize=8)
    return fig


#: SSIM's Gaussian window; a scored plane must be at least this on both axes.
SSIM_WINDOW = 11


def noise_ceiling_all_angles(sino_a, sino_b, *, stride: int = 1,
                             window=None, verbose: bool = True) -> dict:
    """The noise ceiling averaged over EVERY view, not just the eval angle.

    ``measure_noise_ceiling`` scores one projection, which is enough to put a
    line on a plot but is a single sample of a quantity that varies with view:
    the ceiling depends on how much attenuation a given angle sees, so a lateral
    view and an AP view do not share it. Averaging over all of them gives the
    number a run should actually be compared against, and the spread says how
    meaningful the comparison is.

    Both arguments are PREPROCESSED sinograms (A, N_b, N_a) of the same scan
    measured twice — in practice two acquisition phases. With b = s+n and
    b' = s+n' independent, E[(b-b')^2] = 2*sigma^2, so the best mean-squared
    error any model can honestly reach is ``0.5 * mean((b-b')^2)``; that is the
    ``mse_floor`` returned here, in the sinogram's own units.

    ``stride`` subsamples the detector the way a diagnostic render does, so the
    ceiling is measured on the same grid the model is scored on — SSIM is not
    invariant to resolution, and a ceiling computed at full resolution would
    not be comparable to a downsampled diagnostic. ``window`` is an optional
    ``(b0, b1, a0, a1)`` crop, for scoring only the detector region whose rays
    stay inside the reconstruction domain.

    Returns ``{ssim, ssim_std, psnr, psnr_std, mse_floor, n_angles}``.
    """
    import numpy as np
    import torch

    a = torch.as_tensor(np.asarray(sino_a))
    b = torch.as_tensor(np.asarray(sino_b))
    if a.shape != b.shape:
        raise ValueError(f"the two sinograms differ in shape: "
                         f"{tuple(a.shape)} vs {tuple(b.shape)}")
    if window is not None:
        b0, b1, a0, a1 = (int(v) for v in window)
        a, b = a[:, b0:b1, a0:a1], b[:, b0:b1, a0:a1]
    if stride > 1:
        a, b = a[:, ::stride, ::stride], b[:, ::stride, ::stride]

    # SSIM convolves an 11x11 window, so a plane smaller than that fails deep
    # inside conv2d with an unhelpful message. Say which knob did it instead —
    # an over-aggressive diagnostic stride is the usual cause.
    if min(a.shape[1], a.shape[2]) < SSIM_WINDOW:
        raise ValueError(
            f"the scored plane is {a.shape[1]}x{a.shape[2]}, smaller than "
            f"SSIM's {SSIM_WINDOW}x{SSIM_WINDOW} window — stride={stride} "
            f"(and any window=) leaves too little detector to score.")

    ssims, psnrs = [], []
    for i in range(a.shape[0]):
        pa, pb = a[i].float(), b[i].float()
        dr = float(pa.max() - pa.min())
        if dr <= 0:
            continue                       # a blank view carries no ceiling
        ssims.append(float(ssim_2d(pa, pb, data_range=dr)))
        psnrs.append(float(psnr_2d(pa, pb, data_range=dr)))
    if not ssims:
        return {"ssim": None, "ssim_std": None, "psnr": None, "psnr_std": None,
                "mse_floor": None, "n_angles": 0}

    mse_floor = 0.5 * float(((a.float() - b.float()) ** 2).mean())
    out = {"ssim": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
           "psnr": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
           "mse_floor": mse_floor, "n_angles": len(ssims)}
    if verbose:
        print(f"[noise-ceiling] {out['n_angles']} angles: "
              f"ssim={out['ssim']:.4f}+-{out['ssim_std']:.4f}  "
              f"psnr={out['psnr']:.2f}+-{out['psnr_std']:.2f} dB  "
              f"mse floor={mse_floor:.6e}")
    return out
