"""Priors: terms that constrain the volume itself, not its projections.

A projection loss alone does not determine a reconstruction. Many volumes give
nearly the same line integrals, so the data term picks an equivalence class and
something else has to pick a member of it. That is what these do.

They divide by the domain they act on:

* VALUE-SPACE — ``wasserstein_census`` matches the distribution of attenuation
  values against a target histogram, without reference to where the values sit.
* SPATIAL — ``tv_3d_anisotropic`` penalises total variation on a voxel patch;
  ``modica_mortola_3d`` and ``concave_gradient_3d`` implement a double-well
  phase field, which prefers two materials with a thin interface between them
  over a continuous ramp.
* SPECTRAL — ``spectral_match_loss`` compares radially averaged power spectra,
  so it can ask for high-frequency content at a specific spatial frequency.
* WEIGHTING — ``build_fdk_edge_weight`` turns a reference volume's gradient
  magnitude into per-sample weights, concentrating a term near edges.
* MAGNITUDE — ``sparsity_l1`` and ``volume_norm_l2``, the two plain
  regularisers on the values themselves.

The bone-anchor helpers (``estimate_bone_anchors_hu``, ``bone_anchors_to_mu``,
``bone_fraction``) exist because a prior that talks about "bone" needs a
threshold, and a threshold pinned in absolute attenuation does not transfer
between two reconstructions on different scales. They express the band
dimensionlessly, relative to a volume's own tissue peak, and convert it onto
whichever scale is in use at the point of use.
"""

from __future__ import annotations

import math

import torch

from ..scene import Scene

def build_census_target(
    reference_vol: torch.Tensor,
    support_mask: torch.Tensor,
    num_quantiles: int = 256,
    denoise: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Target value-space census (quantile function) for the Wasserstein prior.

    The census is the distribution of μ values over the support, summarised as
    ``num_quantiles`` evenly-spaced quantiles (an empirical inverse-CDF). The
    Wasserstein census loss pulls the model's μ distribution toward this one,
    which — unlike any gradient-domain prior — points TOWARD the sharp
    solution: the blurry recon has valley fill and a truncated bone tail, the
    target does not.

    Target = DENOISED FDK volume (median filter). Rationale (measured):
      * The census is robust to FDK's dominant flaw, noise: zero-mean noise
        broadens each mode symmetrically but does not move mode centres or
        mode mass. Raw vs denoised FDK high quantiles are nearly identical
        (99%: 923 vs 899 HU), i.e. the bone tail is real, coherent signal.
      * Matching the full quantile function (not a one-sided tail COUNT) is
        what makes it noise-robust — a >T threshold count is not.
      * FDK's cupping (mono beam hardening) DOES distort the soft-tissue mode;
        keep num_quantiles modest / the weight gentle so the prior imports the
        bone-tail structure, not FDK's soft-tissue fine shape.

    Args:
        reference_vol:  (Nz, Ny, Nx) μ volume (mm⁻¹), e.g. resampled FDK.
        support_mask:   bool tensor, same shape — voxels to include (the
                        reconstructed region; MUST match the spatial support
                        the model is sampled over, or the air fraction differs
                        and W1 chases that instead of the tail).
        num_quantiles:  size of the quantile grid (inverse-CDF resolution).
        denoise:        median-filter (size 3) the reference before censusing.

    Returns:
        (num_quantiles,) tensor of target quantiles in μ units, on ``device``.
    """
    import numpy as np
    from scipy.ndimage import median_filter

    vol = reference_vol.detach().cpu().numpy()
    if denoise:
        vol = median_filter(vol, size=3)
    vals = vol[support_mask.detach().cpu().numpy()]
    qs = np.linspace(0.0, 1.0, num_quantiles)
    tq = np.quantile(vals.astype(np.float64), qs).astype(np.float32)
    return torch.from_numpy(tq).to(device)


def build_census_quantile_weights(
    quantile_grid: torch.Tensor,
    q_knee: float = 0.85,
    base_weight: float = 0.05,
    power: float = 1.0,
) -> torch.Tensor:
    """Per-quantile weights for a TAIL-FOCUSED census W1.

    A GLOBAL (uniform-weight) W1 spends its gradient on the cheapest transport
    first: draining the air/tissue valley (short μ distance) saturates long
    before the expensive mid→bone-tail transport gets budget. Measured
    (scratchpad/census_ab_finetune.py): weight 0.008 global drained the valley
    2 pp but grew the bone tail only 0.19 pp. These weights redirect the budget
    to the upper quantiles so the pressure lands on the bone tail.

    Shape: w(q) = base_weight for q < q_knee, ramping base_weight→1 over
    [q_knee, 1] as ((q−q_knee)/(1−q_knee))^power.
      * base_weight = 1.0 → uniform (recovers the global census).
      * base_weight = 0.0 → hard cutoff: match ONLY quantiles above q_knee.
      * default (q_knee 0.85, base 0.05) → mostly bone-tail, a little residual
        valley/tissue pressure so those regions don't drift.

    Returns (Q,) weights on the same device/dtype as ``quantile_grid``.
    """
    ramp = ((quantile_grid - q_knee) / max(1.0 - q_knee, 1e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        ramp = ramp ** power
    return base_weight + (1.0 - base_weight) * ramp


def wasserstein_census(
    mu_pred: torch.Tensor,
    target_quantiles: torch.Tensor,
    quantile_grid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """1-D Wasserstein-1 distance between the model's μ census and the target.

    W1 in 1-D is the L1 distance between inverse-CDFs (quantile functions):
        W1 = mean_q | Q_pred(q) − Q_target(q) |.
    Its gradient transports probability mass ALONG the μ axis — it physically
    pushes valley/mid voxels toward the bone tail to match the target. (A
    bin-wise divergence like KL compares CDF heights and gives no cross-bin
    pull, so it cannot move mass into an empty tail.)

    Because W1 is rearrangement-invariant it says how MANY voxels should be
    bone-valued, not WHICH — localisation comes from the data term + current
    estimate. Keep the weight modest so it doesn't amplify hot noise voxels
    into a false tail.

    Args:
        mu_pred:          (M,) sampled model μ values (poly: a1+a2).
        target_quantiles: (Q,) from build_census_target.
        quantile_grid:    (Q,) the q positions of target_quantiles in [0,1]
                          (torch.linspace(0,1,Q)); precompute once.
        weights:          optional (Q,) per-quantile weights (e.g. from
                          build_census_quantile_weights for a tail focus). If
                          None, all quantiles weighted equally (global W1).
    """
    pred_q = torch.quantile(mu_pred.flatten(), quantile_grid)
    err = (pred_q - target_quantiles).abs()
    if weights is not None:
        return (weights * err).sum() / weights.sum().clamp(min=1e-12)
    return err.mean()


# ---------------------------------------------------------------------------
# Volume power-spectrum ("green line") prior.
#
# The census failure mode is HF-noise injection: the value-space prior sets how
# MANY voxels are bone-valued but not WHICH, so it can satisfy the histogram by
# spraying broadband high-frequency noise (vol_hf_ratio ≫ 1 = speckle). This
# prior attacks that directly. From the volume power spectrum we can estimate
# the NOISELESS signal spectrum ("the green curve"): FDK's radial power spectrum
# minus its ramp-noise floor (∝ f², fit from the band above `noise_knee` where
# the true signal has collapsed). We then pull the model's radial power spectrum
# toward that target.
#
# The match is SHAPE-only: each spectrum is normalized by its power in a
# low-frequency anchor band (where signal dominates and model ≈ FDK), so this
# term governs only the DISTRIBUTION of energy across frequencies — how much HF
# relative to the bulk — and stays orthogonal to MSE/census which own the
# absolute HU level. It is TWO-SIDED: it penalizes HF EXCESS (speckle) and HF
# DEFICIT (blur) symmetrically. Like census it is spatially blind (a power
# spectrum discards phase); localization is supplied by the MS-SSIM/MSE data
# terms, which fix WHERE the energy lands. See build_spectral_target /
# spectral_match_loss and train.py's `training.spectral` block.
# ---------------------------------------------------------------------------

def build_radial_bins(h: int, w: int, device: torch.device | str = "cpu"):
    """Per-pixel radial-frequency bin index for an (h, w) 2-D FFT.

    Mirrors the diagnostic power-spectrum binning in train.py: shifted FFT,
    integer radius from the centre, frequency axis in cycles/pixel = r/max(h,w).

    Returns (r_index (h*w,) long, r_max int, freqs_cpp (r_max,) cycles/pixel).
    """
    cy, cx = h // 2, w // 2
    yy = torch.arange(h, device=device).float()
    xx = torch.arange(w, device=device).float()
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2).round().long()
    r_max = int(min(cx, cy))
    r = r.clamp(max=r_max - 1)
    freqs_cpp = torch.arange(r_max, device=device).float() / max(h, w)
    return r.flatten(), r_max, freqs_cpp


def _radial_power_spectrum(img2d: torch.Tensor, r_index: torch.Tensor,
                           r_max: int) -> torch.Tensor:
    """Differentiable radially-averaged power spectrum of a real 2-D image.

    `r_index`/`r_max` come from build_radial_bins for img2d's shape. Runs in
    fp32 (FFT is fragile under bf16 autocast — the same lesson as ssim_loss);
    the caller passes a .float() patch with autocast disabled.
    """
    f2 = torch.fft.fftshift(torch.fft.fft2(img2d.float()))
    ps = (f2.real ** 2 + f2.imag ** 2).flatten()
    sums = torch.zeros(r_max, device=img2d.device, dtype=ps.dtype)
    sums = sums.scatter_add(0, r_index, ps)
    counts = torch.zeros(r_max, device=img2d.device, dtype=ps.dtype)
    counts = counts.scatter_add(0, r_index, torch.ones_like(ps))
    return sums / counts.clamp(min=1.0)


def build_spectral_target(
    reference_vol: torch.Tensor,
    patch_n: int,
    voxel_mm: float,
    noise_knee_lpmm: float = 4.0,
    target_patches: int = 200,
    min_body_mu: float = 0.011,
    device: torch.device | str = "cpu",
):
    """Estimate the noiseless signal power spectrum ("green curve") from FDK.

    Averages the radial power spectrum over many random in-body ``patch_n`` ×
    ``patch_n`` axial patches of the reference (FDK) volume, then subtracts a
    fitted ``C·f²`` ramp-noise floor (C from the band above ``noise_knee_lpmm``,
    where the anatomical signal has collapsed and FDK is ≈ pure noise). Patches
    (not full slices) so the target frequency grid matches exactly the patches
    the model is scored on — no interpolation, apples to apples.

    Args:
        reference_vol:   (Nz, Ny, Nx) μ volume (mm⁻¹), e.g. resampled FDK.
        patch_n:         patch edge in voxels (== the model patch size).
        voxel_mm:        in-plane voxel size → frequency axis in lp/mm.
        noise_knee_lpmm: freq above which FDK ≈ ramp noise, used to fit C.
        target_patches:  number of in-body FDK patches to average.
        min_body_mu:     skip patches whose mean μ is below this (mostly air).

    Returns (green (r_max,) target power, freqs_lpmm (r_max,), N used).
    """
    import numpy as np

    vol = reference_vol.detach().float()
    Nz, Ny, Nx = vol.shape
    N = int(min(patch_n, Ny, Nx))
    r_index, r_max, freqs_cpp = build_radial_bins(N, N, device)
    acc = torch.zeros(r_max, device=device)
    cnt = 0
    tries = 0
    max_tries = target_patches * 20
    while cnt < target_patches and tries < max_tries:
        tries += 1
        z = int(torch.randint(0, Nz, (1,)).item())
        y0 = int(torch.randint(0, Ny - N + 1, (1,)).item())
        x0 = int(torch.randint(0, Nx - N + 1, (1,)).item())
        blk = vol[z, y0:y0 + N, x0:x0 + N].to(device)
        if float(blk.mean()) < min_body_mu:      # mostly air → skip
            continue
        acc = acc + _radial_power_spectrum(blk, r_index, r_max)
        cnt += 1
    ps = (acc / max(cnt, 1)).cpu().numpy()
    freqs_lpmm = (freqs_cpp / voxel_mm).cpu().numpy()
    hf = freqs_lpmm > noise_knee_lpmm
    if hf.sum() > 5:
        C = float((ps[hf] / (freqs_lpmm[hf] ** 2 + 1e-20)).mean())
        green = np.maximum(ps - C * freqs_lpmm ** 2, 0.0)
    else:
        green = ps
    return (
        torch.from_numpy(green.astype(np.float32)).to(device),
        torch.from_numpy(freqs_lpmm.astype(np.float32)).to(device),
        int(N),
    )


def spectral_split_db(ps_model, ps_ref, freqs_lpmm, noise_knee_lpmm: float = 4.0):
    """Split a reconstruction's spectral error into BLUR and SPECKLE, in dB.

    `diag/vol_hf_ratio` cannot rank reconstructions: it is a single number that
    rises both when a run resolves genuine detail and when it sprays broadband
    noise, and it is silent about the mid-band content that is actually missing.
    `fdk/ssim` is no better — it rewards copying FDK, so it penalises a run for
    out-resolving the reference. This splits the error into the two independent
    quantities, each of which must go to zero on its own:

      deficit_db  mean per-bin dB by which the model sits BELOW the estimated
                  noiseless signal spectrum over [0.5, f_cross] lp/mm — the
                  mid-band anatomy (ribs, organ boundaries) that blur removes.
      excess_db   mean per-bin dB by which the model sits ABOVE the reference's
                  own spectrum over [f_cross, Nyquist] — a band where the
                  reference is ≈ pure ramp noise, so anything extra is invented.

    `f_cross` is MEASURED as the frequency where the estimated true signal
    (reference minus its fitted C·f² ramp-noise floor) falls under that floor,
    i.e. where the reference stops carrying anatomy. It is not hardcoded, so it
    tracks the scan's own resolution limit.

    Both terms are clipped at zero so they cannot cancel: blurring lowers
    excess but raises deficit, and injecting noise does the reverse. Their sum
    `cost_db` is therefore the rankable single number — a run only improves it
    by moving power from the noise band into the signal band.

    Args:
        ps_model:        (R,) radially-averaged power spectrum of the model.
        ps_ref:          (R,) same for the reference (FDK) volume.
        freqs_lpmm:      (R,) matching frequency axis in lp/mm.
        noise_knee_lpmm: band above which the reference is treated as pure
                         noise when fitting the C·f² floor.

    Returns a dict with deficit_db, excess_db, cost_db and f_cross_lpmm; the
    dB entries are None when a band is empty (too few bins to judge).
    """
    import numpy as np

    f = np.asarray(freqs_lpmm, dtype=np.float64)
    pm = np.asarray(ps_model, dtype=np.float64)
    pr = np.asarray(ps_ref, dtype=np.float64)
    n = min(len(f), len(pm), len(pr))
    f, pm, pr = f[:n], pm[:n], pr[:n]
    eps = 1e-30

    hf = f > noise_knee_lpmm
    if hf.sum() > 5:
        C = float((pr[hf] / (f[hf] ** 2 + 1e-20)).mean())
        noise = C * f ** 2
    else:                                     # too narrow to fit a floor
        noise = np.zeros_like(f)
    signal = np.maximum(pr - noise, 0.0)

    # f_cross: first bin above 1 lp/mm at which the signal drops under the
    # noise floor, having been above it somewhere below.
    alive = (signal > noise) & (f > 1.0)
    idx = np.nonzero(alive)[0]
    if idx.size:
        dead = np.nonzero((~alive) & (f > f[idx[0]]))[0]
        f_cross = float(f[dead[0]]) if dead.size else float(f[-1])
    else:
        f_cross = float(noise_knee_lpmm)

    sig_band = (f >= 0.5) & (f < f_cross)
    noi_band = f >= f_cross

    deficit = excess = None
    if sig_band.sum() > 0:
        d = 10.0 * np.log10(np.maximum(signal[sig_band], eps)
                            / np.maximum(pm[sig_band], eps))
        deficit = float(np.maximum(d, 0.0).mean())
    if noi_band.sum() > 0:
        e = 10.0 * np.log10(np.maximum(pm[noi_band], eps)
                            / np.maximum(pr[noi_band], eps))
        excess = float(np.maximum(e, 0.0).mean())

    return {
        "deficit_db": deficit,
        "excess_db": excess,
        "cost_db": (None if deficit is None or excess is None
                    else deficit + excess),
        "f_cross_lpmm": f_cross,
    }


def spectral_match_loss(
    mu_patch2d: torch.Tensor,
    target_green: torch.Tensor,
    freqs_lpmm: torch.Tensor,
    r_index: torch.Tensor,
    r_max: int,
    anchor_lo: float = 0.5,
    anchor_hi: float = 1.5,
    hf_knee: float = 1.5,
    hf_base_weight: float = 0.25,
    log_floor_frac: float = 1e-3,
) -> torch.Tensor:
    """Shape-only, HF-weighted L1 distance (log-power) to the green target.

    Both the patch spectrum and the target are normalized by their mean power
    in the [anchor_lo, anchor_hi] lp/mm band (the ANCHOR band, signal-dominated
    and DC-free), so only the SHAPE (HF-relative-to-bulk) is constrained — the
    absolute level, and the DC bin in particular, are left to MSE/census. The
    target is floored at `log_floor_frac` of the anchor level, so where the true
    signal has collapsed the target reads "HF ≈ 0.1% of bulk": the loss then
    pulls the patch's HF toward that floor (despeckle) or up to it (deblur) —
    two-sided. Bins at/below `anchor_hi` get zero weight (already matched by the
    data term); above it the weight ramps `hf_base_weight`→1 past `hf_knee`.

    `mu_patch2d` is a (N, N) μ patch (call with autocast disabled, fp32).
    Returns a scalar in [0, ∞); 0 = spectral shape matches the green curve.
    """
    ps = _radial_power_spectrum(mu_patch2d, r_index, r_max)
    amask = (freqs_lpmm >= anchor_lo) & (freqs_lpmm <= anchor_hi)
    if amask.sum() < 1:                               # degenerate: mid band
        amask = (freqs_lpmm > 0) & (freqs_lpmm <= freqs_lpmm.max() * 0.5)
    # Normalize each spectrum by its own anchor-band mean (shape-only, DC-free).
    ps_a = ps[amask].mean().clamp(min=1e-20)
    gr_a = target_green[amask].mean().clamp(min=1e-20)
    ps_n = ps / ps_a
    # Target shape, floored RELATIVE TO THE ANCHOR (not the DC max): where the
    # signal has collapsed the target sits at `log_floor_frac` of the bulk.
    gr_n = (target_green / gr_a).clamp(min=log_floor_frac)
    lp = torch.log10(ps_n.clamp(min=log_floor_frac * 1e-3))
    lg = torch.log10(gr_n)
    diff = (lp - lg).abs()
    fmax = freqs_lpmm.max().clamp(min=hf_knee + 1e-6)
    ramp = ((freqs_lpmm - hf_knee) / (fmax - hf_knee)).clamp(0.0, 1.0)
    w = torch.where(
        freqs_lpmm > anchor_hi,
        hf_base_weight + (1.0 - hf_base_weight) * ramp,
        torch.zeros_like(freqs_lpmm),
    )
    return (w * diff).sum() / w.sum().clamp(min=1e-12)


def _gradient_magnitude(vol, sigma: float):
    """Gradient magnitude of *vol* after Gaussian smoothing at *sigma*."""
    from scipy.ndimage import gaussian_filter
    import numpy as np

    smoothed = gaussian_filter(vol, sigma=sigma)
    gz = (smoothed[2:, :, :] - smoothed[:-2, :, :]) / 2.0
    gy = (smoothed[:, 2:, :] - smoothed[:, :-2, :]) / 2.0
    gx = (smoothed[:, :, 2:] - smoothed[:, :, :-2]) / 2.0
    gz = np.pad(gz, ((1, 1), (0, 0), (0, 0)), mode="edge")
    gy = np.pad(gy, ((0, 0), (1, 1), (0, 0)), mode="edge")
    gx = np.pad(gx, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return np.sqrt(gx**2 + gy**2 + gz**2)


def build_fdk_edge_weight(
    fdk_resampled: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Pre-compute multi-scale edge weight map from the resampled FDK volume.

    Uses multi-scale persistence to separate real edges from noise: gradient
    magnitude is computed at three scales (σ/4, σ/2, σ) and combined via
    geometric mean.  Real edges persist across all scales (slow 1/σ decay),
    while noise gradients vanish at larger scales (fast 1/σ^{5/2} decay),
    giving orders-of-magnitude better edge/noise separation than a single
    scale.

    Returns a (Nz, Ny, Nx) tensor in [0, 1] where high values mark edges
    (bone boundaries, tissue interfaces) and near-zero values mark flat
    regions and noise.
    """
    import numpy as np

    vol = fdk_resampled.numpy()

    sigmas = [sigma / 4.0, sigma / 2.0, sigma]
    edge_maps = []
    for s in sigmas:
        gm = _gradient_magnitude(vol, s)
        gmax = gm.max()
        if gmax > 0:
            gm = gm / gmax  # normalise each scale to [0, 1]
        edge_maps.append(gm)

    # Geometric mean: product^(1/N).  Preserves edges (high at all scales),
    # crushes noise (low at any scale).
    product = edge_maps[0] * edge_maps[1] * edge_maps[2]
    geo_mean = np.power(product, 1.0 / len(sigmas))

    # Re-normalise to [0, 1].
    gmax = geo_mean.max()
    if gmax > 0:
        geo_mean = geo_mean / gmax

    print(f"  FDK edge weight (multi-scale): σ=[{sigmas[0]:.2f}, {sigmas[1]:.1f}, {sigmas[2]:.1f}], "
          f"range [{geo_mean.min():.4f}, {geo_mean.max():.4f}], "
          f"mean={geo_mean.mean():.4f}")
    return torch.from_numpy(geo_mean.astype(np.float32))


def sparsity_l1(
    model: torch.nn.Module,
    n_points: int,
    device: torch.device,
) -> torch.Tensor:
    """L1 sparsity penalty on the model's output at random domain points.

    Samples random coordinates in [-1, 1]^3 (the full model domain), queries
    the model, and returns mean(|μ|). This pushes μ toward zero everywhere.
    The projection MSE counteracts this for voxels where real attenuation
    exists (tissue, bone, bed). For air voxels — where the MSE gradient is
    noise-dominated and near zero — the L1 penalty wins and drives μ to zero.

    This breaks the degeneracy where the model spreads small positive μ
    into air regions to absorb scatter/flat-field offsets from the sinogram.
    """
    xyz = torch.rand(n_points, 3, device=device) * 2.0 - 1.0
    mu = model(xyz)
    return mu.abs().mean()


def volume_norm_l2(
    model: torch.nn.Module,
    n_points: int,
    device: torch.device,
) -> torch.Tensor:
    """L2 volume-norm penalty: mean(μ²) at random domain points.

    Replicates the minimum-norm (pseudoinverse) selection that FDK's ramp
    filter provides implicitly. Penalises null-space energy — volume
    components invisible to all projections — without referencing FDK's
    output or imposing hand-crafted structural priors (TV, smoothness).
    """
    xyz = torch.rand(n_points, 3, device=device) * 2.0 - 1.0
    mu = model(xyz)
    return (mu ** 2).mean()


def tv_3d_anisotropic(mu_patch: torch.Tensor) -> torch.Tensor:
    """Anisotropic 3D total variation on a (Pz, Py, Px) μ-patch.

    Sum of per-axis mean absolute first-difference. Anisotropic = each axis
    summed independently as L1 (rather than per-voxel L2 norm of the gradient
    vector). This is the formulation used by IntraTomo and NeRP — simpler
    gradient, same qualitative behaviour as isotropic TV.

    Returns a scalar tensor (mean over patch voxels per axis, summed across
    axes), so the TV magnitude is *patch-shape-independent* — a single
    `tv_weight` hyperparameter in the config carries the same meaning whether
    the patch is 4×32×32 or 8×64×64.
    """
    if mu_patch.dim() != 3:
        raise ValueError(f"tv_3d_anisotropic expects (Pz, Py, Px), got {tuple(mu_patch.shape)}")
    Pz, Py, Px = mu_patch.shape
    # Each axis only contributes if it has ≥ 2 voxels (otherwise the
    # difference tensor is empty → .mean() returns NaN).
    zero = mu_patch.new_zeros(())
    dx = (mu_patch[:, :, 1:] - mu_patch[:, :, :-1]).abs().mean() if Px > 1 else zero
    dy = (mu_patch[:, 1:, :] - mu_patch[:, :-1, :]).abs().mean() if Py > 1 else zero
    dz = (mu_patch[1:, :, :] - mu_patch[:-1, :, :]).abs().mean() if Pz > 1 else zero
    return dx + dy + dz


def sample_tv_patch_xyz(
    scene: Scene,
    patch_shape: tuple[int, int, int] = (4, 32, 32),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Random axis-aligned 3D patch inside the export ROI, in normalized coords.

    The patch center is sampled uniformly from the export ROI minus the
    patch half-extent (so the full patch lies in bounds). Voxel spacing
    matches the export grid's `dx`/`dz`, so the gradients computed by
    `tv_3d_anisotropic` reflect the same spatial scale the model is asked
    to render at inference.

    Args:
        scene: provides geometry (vol_shape, vol_origin, dx, dz) and
            model_domain.normalize for the [-1, 1]^3 mapping.
        patch_shape: (Pz, Py, Px) — number of voxels per axis. Defaults to a
            small 4-slice patch for compute-cheap regularization.
        generator: torch.Generator for reproducibility (optional).
        device: device for the returned tensor.

    Returns:
        xyz_norm: (Pz*Py*Px, 3) flat tensor of normalized coords, ready for
            `model(xyz_norm)`. The patch shape is returned alongside so the
            caller can reshape `model(xyz_norm)` back to (Pz, Py, Px).
        patch_shape: same tuple, echoed for reshape convenience.

    Note: the patch is sampled INSIDE the export ROI (where reconstruction
    quality is what we actually care about), not the full model_domain
    cylinder — TV applied uniformly across the full domain would waste budget
    on the empty-air margins between the export ROI and the integration
    domain boundary.
    """
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)

    geom = scene.geometry
    Nx, Ny, Nz = geom["vol_shape"]
    dx = float(geom["dx"])
    dz = float(geom["dz"])
    ox, oy, oz = (float(c) for c in geom["vol_origin"])
    Pz, Py, Px = patch_shape

    # Half-extents in mm: patch half-size and ROI half-size.
    half_pX = (Px - 1) * dx / 2.0
    half_pY = (Py - 1) * dx / 2.0
    half_pZ = (Pz - 1) * dz / 2.0
    half_X = Nx * dx / 2.0
    half_Y = Ny * dx / 2.0
    half_Z = Nz * dz / 2.0

    # Margin = ROI half - patch half. If patch is bigger than ROI on some
    # axis, clamp margin to 0 (patch will straddle the boundary; OK).
    mx = max(half_X - half_pX, 0.0)
    my = max(half_Y - half_pY, 0.0)
    mz = max(half_Z - half_pZ, 0.0)

    u = torch.rand(3, generator=generator, dtype=torch.float32) * 2.0 - 1.0  # (-1, 1)
    cx = ox + float(u[0]) * mx
    cy = oy + float(u[1]) * my
    cz = oz + float(u[2]) * mz

    xs = (torch.arange(Px, dtype=torch.float32, device=device) - (Px - 1) / 2.0) * dx + cx
    ys = (torch.arange(Py, dtype=torch.float32, device=device) - (Py - 1) / 2.0) * dx + cy
    zs = (torch.arange(Pz, dtype=torch.float32, device=device) - (Pz - 1) / 2.0) * dz + cz

    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    xyz = torch.stack([xx, yy, zz], dim=-1)  # (Pz, Py, Px, 3) in mm
    xyz_norm = scene.model_domain.normalize(xyz)
    return xyz_norm.reshape(-1, 3), patch_shape


# ---------------------------------------------------------------------------
# Sharp-interface bone prior (β double well + Modica-Mortola).
#
# HOW THIS DIFFERS FROM TV, as a matter of the functionals themselves. TV is L1
# on the first difference, and for a MONOTONE edge the absolute differences sum
# to the total jump however they are distributed — so TV is invariant to how
# wide that edge is (a unit step gives TV = 1.0000 at blur sigma 0, 0.5, 1, 2
# and 4 voxels alike). What varies TV is noise. Its derivative is also
# constant, so it pulls equally hard on a tall edge and a short one, which
# reduces contrast along with the noise.
#
# This prior is built to have the opposite two properties: it responds to edge
# WIDTH, and it leaves a tall edge alone. Work in bone-fraction space and use a
# NON-convex functional:
#
#     β(μ) = clamp((μ − μ_tissue) / (μ_bone − μ_tissue), 0, 1)
#     L    = mean[ W(β)/ε + ε·|∇β|² ],      W(β) = β²(1−β)²
#
# This is Modica-Mortola: as ε → 0 it Γ-converges to (bone surface area), and
# the optimal interface profile has width ~ε. BOTH terms are required — W
# alone binarises to a zero-width (aliased) edge, |∇β|² alone is just a
# smoothness prior i.e. blur.
#
# It produces two forces the projection loss cannot:
#   1. CONTRAST. dW/dβ = 2β(1−β)(1−2β) is negative for β ∈ (0.5, 1), so every
#      UNDER-CONTRASTED interior bone voxel is pushed UP, with no gradient
#      involved. MEASURED force: β=0.6 -> -0.096, β=0.8 -> -0.192 (max),
#      β=0.95 -> -0.086, β=1 -> 0. It self-extinguishes at the anchor instead
#      of overshooting. At the observed bone/tail_ratio 0.656 every bone voxel
#      pays W = 0.051. No gradient-domain prior sees absolute level at all.
#   2. INTERFACE WIDTH. ε is not a free knob — it IS the target edge width in
#      voxels. MEASURED preferred blur σ*: ε=0.5 -> 0.5, ε=0.75 -> 1.0,
#      ε=1.0 -> 1.5, ε=2.0 -> 3.0, i.e. σ* ≈ 1.4·ε.
#
# THE CLAMP IS THE GATE. Soft tissue sits at β=0 and deep bone at β=1, where
# W = 0 AND ∇β = 0 — exactly zero force. Only the transition band between the
# two anchors contributes, so no bone mask and no gating hyperparameter is
# needed. The corollary is that this prior does NOTHING about soft-tissue
# speckle; `concave_gradient_3d` is the companion term for that.
# ---------------------------------------------------------------------------

def estimate_bone_anchors_hu(
    reference_vol,
    nominal_mu_water: float = 0.0219,
    shoulder_mult: float = 2.5,
    bone_quantile: float = 0.995,
    denoise: bool = True,
    verbose: bool = False,
) -> tuple[float, float]:
    """Measure the bone band as DIMENSIONLESS HU from a reference (FDK) volume.

    Returns HU, NOT μ — and that is the contract, not a detail. Two
    reconstructions of the same object generally sit on DIFFERENT attenuation
    scales (a reference volume's tissue peak might be μ=0.0253 where a model's
    is μ=0.0173, a 1.46x difference), so a band measured as μ on one and
    applied as μ to the other lands in the wrong place. Measured here relative
    to the reference's own tissue peak and air, it transfers: pair it with
    `bone_anchors_to_mu` and the CONSUMER's own anchors at the point of use.

    HU is dimensionless, so it transfers between the two scales. Pair this
    with `bone_anchors_to_mu` at the point of use, feeding it the MODEL's own
    live two-point anchors. The reference then contributes only two
    scale-free numbers.

    NOTE the tissue shoulder is derived from the REFERENCE's soft-tissue
    width, which is noise-dominated (FDK). That biases the shoulder HIGH,
    which is the safe direction: it keeps more of the soft-tissue
    distribution out of the well's downward basin.

    μ_bone MUST be an EMPIRICAL quantile of the reference, never the physical
    pure-cortical-bone value. Mouse rib cortex is ~100-200 μm against a 75 μm
    voxel, so essentially every bone voxel in this scan is PARTIAL VOLUME and
    β = 1 is physically unreachable. The config's NIST-derived
    `mix_mu_bone` = 0.0814 mm⁻¹ is ≈2700 HU; anchoring there would put most
    bone voxels below β = 0.5, where the well pushes DOWN — it would destroy
    the bone rather than restore it. The reachable bone level is set by the
    resolution/partial-volume limit, not by material physics.

    Both anchors are noise-robust POPULATION statistics, which reduces the
    reference's role to TWO SCALARS. That matters for robustness: a scalar
    carries no geometry, so any spatial error in the reference — a scale
    mismatch, a residual misalignment — cannot reach the prior, unlike any term
    that consumes the reference volume itself. Noise inflates a raw high
    quantile by only ~2.6 % (923 vs 899 HU, raw vs denoised), and `denoise`
    removes most of that.

    μ_tissue is the soft-tissue SHOULDER (peak + `shoulder_mult`·σ), not the
    peak: β < 0.5 is the well's downward basin, so putting the lower anchor at
    the peak would drag the upper half of the soft-tissue distribution toward
    air. The shoulder keeps ambiguous soft tissue at β = 0, where the force is
    exactly zero.

    Args:
        reference_vol:     (Nz, Ny, Nx) μ volume (mm⁻¹), e.g. resampled FDK.
        nominal_mu_water:  IGNORED, kept for call compatibility. The submodule
                           estimator locates both anchors from the histogram's
                           shape alone and needs no seed value.
        shoulder_mult:     σ multiples above the tissue peak for μ_tissue.
        bone_quantile:     quantile of in-body voxels used for μ_bone.
        denoise:           median-filter (size 3) before measuring.

    Returns (hu_tissue, hu_bone) in HU on the REFERENCE's own two-point scale.
    Raises ValueError if the two collapse.
    """
    import numpy as np
    from scipy.ndimage import median_filter

    from ...ct_core.hu_calibration import find_attenuation_anchors

    vol = np.asarray(
        reference_vol.detach().cpu().numpy()
        if hasattr(reference_vol, "detach") else reference_vol,
        dtype=np.float64,
    )
    if denoise:
        vol = median_filter(vol, size=3)

    # Soft-tissue peak via the SAME scale-equivariant estimator the HU
    # self-calibration uses, so the anchor and the HU scale never disagree.
    # Only the anchor LOCATIONS in μ are used here — the band is converted to
    # HU and back with the same formula at both ends, so where the estimator
    # declares tissue to land (0 HU or the vendor's +120) cancels out.
    _anchors = find_attenuation_anchors(vol)
    mu_water = float(_anchors.value_tissue)
    mu_air = float(_anchors.value_air)

    # The submodule estimator REPORTS a bad fit rather than raising, because
    # most consumers would still rather have a flagged map than an exception.
    # This one would not: a double-well prior built on anchors that are really
    # two halves of ONE noise peak will actively push every voxel toward one of
    # two invented phases, and the resulting μ band is arbitrary.
    #
    # The signature is the anchor SEPARATION, not the peaks' shape: on a
    # near-uniform volume the estimator happily finds two "populations" with
    # convincing mass and prominence, sitting a fraction of the noise width
    # apart. Measuring the span against the estimator's own robust value range
    # keeps the test scale-free, exactly like the estimator itself — a real
    # specimen puts air and tissue across a large share of that range, while
    # here they were 0.6 % of it apart.
    _d = _anchors.diagnostics
    _centers = _d.get("centers")
    if _centers is not None and len(_centers) > 1:
        _range = float(_centers[-1] - _centers[0])
        _span = float(_anchors.value_tissue - _anchors.value_air)
        if _range > 0 and _span / _range < 0.05:
            raise ValueError(
                f"estimate_bone_anchors_hu: the reference has no distinct "
                f"material populations to anchor on — its air and tissue "
                f"anchors sit {100 * _span / _range:.1f} % of the volume's own "
                f"value range apart, i.e. they are two halves of a single "
                f"peak. A uniform or near-uniform volume cannot define a bone "
                f"band.")

    # Robust width of the soft-tissue mode: MAD over a window around the peak
    # that is wide enough to see the shoulder but excludes bone and lung.
    half = 0.5 * (mu_water - mu_air)          # ~500 HU in μ units
    win = vol[(vol > mu_water - half) & (vol < mu_water + half)]
    if win.size < 64:
        raise ValueError("estimate_bone_anchors: soft-tissue mode too small "
                         f"to measure ({win.size} voxels)")
    sigma = 1.4826 * float(np.median(np.abs(win - mu_water)))
    mu_tissue = float(mu_water + shoulder_mult * sigma)

    # Bone anchor: high quantile over IN-BODY voxels only (air would dominate
    # the population and drag the quantile down).
    body = vol[vol > mu_air + 0.25 * (mu_water - mu_air)]
    if body.size < 64:
        raise ValueError("estimate_bone_anchors: body support too small "
                         f"to measure ({body.size} voxels)")
    mu_bone = float(np.quantile(body, bone_quantile))

    if not (mu_bone > mu_tissue):
        raise ValueError(
            f"estimate_bone_anchors_hu: μ_bone ({mu_bone:.5f}) must exceed "
            f"μ_tissue ({mu_tissue:.5f}); the reference has no separable bone "
            "population (wrong units, or an all-soft-tissue volume?)")

    # Express the band in units of the reference's OWN air->tissue span. These
    # are NOT Hounsfield units and must not be compared with anything on the HU
    # scale: they are dimensionless band coordinates whose only job is to be
    # inverted by `bone_anchors_to_mu` against the MODEL's anchors, which is
    # what makes the band transferable between two different attenuation
    # scales. The ×1000 is cosmetic, so the numbers read like HU.
    span = mu_water - mu_air
    if span <= 0:
        raise ValueError(
            f"estimate_bone_anchors_hu: degenerate reference calibration "
            f"(mu_water {mu_water:.5f} <= mu_air {mu_air:.5f})")
    hu_tissue = 1000.0 * (mu_tissue - mu_water) / span
    hu_bone = 1000.0 * (mu_bone - mu_water) / span

    if verbose:
        print(f"  [bone anchors] reference tissue peak μ={mu_water:.5f} "
              f"(σ={sigma:.5f}) | shoulder {hu_tissue:+.0f} HU, "
              f"q{bone_quantile:.4g} bone {hu_bone:+.0f} HU "
              f"(DIMENSIONLESS — converted to the model's μ scale at use)")
    return hu_tissue, hu_bone


def bone_anchors_to_mu(hu_tissue: float, hu_bone: float,
                       mu_water: float, mu_air: float) -> tuple[float, float]:
    """Put the dimensionless bone band onto a specific μ scale.

    Inverts ``HU = 1000·(μ − μ_water)/(μ_water − μ_air)``. Feed it the anchors
    of the volume the band is about to be APPLIED to, never those of the volume
    it was measured on — the two scales differ, which is the whole reason the
    band travels dimensionlessly.

    Anchoring to the consumer's own soft-tissue peak carries no feedback risk:
    the prior acts on bone, so it cannot move the statistic that scales it.
    """
    span = float(mu_water) - float(mu_air)
    if span <= 0:
        raise ValueError(f"bone_anchors_to_mu needs mu_water > mu_air, got "
                         f"{mu_water} <= {mu_air}")
    mu_tissue = mu_water + (float(hu_tissue) / 1000.0) * span
    mu_bone = mu_water + (float(hu_bone) / 1000.0) * span
    if not (mu_bone > mu_tissue):
        raise ValueError(
            f"bone_anchors_to_mu: hu_bone ({hu_bone}) must exceed hu_tissue "
            f"({hu_tissue})")
    return mu_tissue, mu_bone


def bone_fraction(mu: torch.Tensor, mu_tissue: float,
                  mu_bone: float) -> torch.Tensor:
    """Partial-volume bone fraction β ∈ [0, 1] from attenuation.

    LINEAR, not a sigmoid: partial-volume mixing is linear in attenuation, so
    a voxel that is fraction f cortical bone has exactly
    μ = f·μ_bone + (1−f)·μ_tissue. The linear map therefore IS the physical
    bone fraction; a smoothstep would distort the mixture interpretation and
    flatten the gradient at precisely the two ends where force is needed.

    The clamp is deliberate and is what gates the prior — see the module note.
    """
    span = float(mu_bone) - float(mu_tissue)
    if span <= 0:
        raise ValueError(f"bone_fraction needs mu_bone > mu_tissue, "
                         f"got {mu_bone} <= {mu_tissue}")
    return ((mu - float(mu_tissue)) / span).clamp(0.0, 1.0)


def bone_band_fraction(beta: torch.Tensor) -> float:
    """Fraction of voxels strictly inside the bone band (0 < β < 1).

    THE health check for this prior, and worth logging on every run. Outside
    the band β is clamped, so W = 0 AND ∇β = 0: the term becomes *exactly*
    inert, contributing no loss and no gradient — while a loss curve alone
    shows only a regulariser that has settled at zero, which is
    indistinguishable from one that has converged. A value of 0 means the prior
    is doing nothing, not that it is satisfied.
    """
    with torch.no_grad():
        return float(((beta > 0.0) & (beta < 1.0)).float().mean())


def modica_mortola_3d(beta_patch: torch.Tensor, eps_voxels: float,
                      z_spacing_ratio: float = 1.0) -> torch.Tensor:
    """Sharp-interface energy mean[W(β)/ε + ε·|∇β|²] on a (Pz, Py, Px) patch.

    ``eps_voxels`` is in VOXELS and sets the preferred interface width
    (σ* ≈ 1.4·ε, measured). ``z_spacing_ratio`` = dz/dx converts the
    through-plane difference into the same in-plane voxel unit, so ε keeps one
    meaning on anisotropic grids.

    Means (not sums) on both terms, so the value is patch-shape independent
    and a single config weight carries the same meaning at any patch size —
    the same convention as `tv_3d_anisotropic`.
    """
    if beta_patch.dim() != 3:
        raise ValueError("modica_mortola_3d expects (Pz, Py, Px), got "
                         f"{tuple(beta_patch.shape)}")
    if eps_voxels <= 0:
        raise ValueError(f"eps_voxels must be > 0, got {eps_voxels}")
    b = beta_patch.float()
    Pz, Py, Px = b.shape
    zero = b.new_zeros(())

    well = (b ** 2 * (1.0 - b) ** 2).mean()

    # Forward differences; z rescaled to in-plane voxel units.
    gx = (b[:, :, 1:] - b[:, :, :-1]).pow(2).mean() if Px > 1 else zero
    gy = (b[:, 1:, :] - b[:, :-1, :]).pow(2).mean() if Py > 1 else zero
    gz = ((b[1:, :, :] - b[:-1, :, :]) / max(z_spacing_ratio, 1e-6)
          ).pow(2).mean() if Pz > 1 else zero

    return well / eps_voxels + eps_voxels * (gx + gy + gz)


def concave_gradient_3d(mu_patch: torch.Tensor, eps_grad: float,
                        z_spacing_ratio: float = 1.0) -> torch.Tensor:
    """Bounded concave edge penalty mean[φ(|∇μ|)], φ(g) = g²/(g² + ε²).

    The companion to `modica_mortola_3d`: β is inert outside the bone band, so
    this is what acts on soft tissue. Three structural properties, none of them
    tuned:

      * small g -> φ ≈ (g/ε)², quadratic: speckle and blur tails are pulled
        hard toward zero;
      * large g -> φ -> 1 with derivative -> 0: a genuine edge is free, and a
        TALLER edge costs no more than a short one, so unlike TV this does NOT
        shrink bone contrast;
      * concave throughout: one sharp jump is cheaper than the smeared ramp of
        the same total height, which is exactly what TV cannot express
        (measured: L^0.5 runs 1.0 -> 4.46 over blur σ 0 -> 3 where TV stays at
        1.0000).

    ``eps_grad`` is in μ units per voxel and sets the noise/edge changeover;
    anneal it down for graduated non-convexity.
    """
    if mu_patch.dim() != 3:
        raise ValueError("concave_gradient_3d expects (Pz, Py, Px), got "
                         f"{tuple(mu_patch.shape)}")
    if eps_grad <= 0:
        raise ValueError(f"eps_grad must be > 0, got {eps_grad}")
    m = mu_patch.float()
    Pz, Py, Px = m.shape
    zero = m.new_zeros(())
    e2 = float(eps_grad) ** 2

    def _phi(g):
        g2 = g.pow(2)
        return (g2 / (g2 + e2)).mean()

    px = _phi(m[:, :, 1:] - m[:, :, :-1]) if Px > 1 else zero
    py = _phi(m[:, 1:, :] - m[:, :-1, :]) if Py > 1 else zero
    pz = _phi((m[1:, :, :] - m[:-1, :, :]) / max(z_spacing_ratio, 1e-6)
              ) if Pz > 1 else zero
    return px + py + pz
