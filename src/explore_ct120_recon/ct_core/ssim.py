"""SSIM, once. Pure torch, no plotting, no policy.

This is the single implementation of Wang et al. (2004) in the package. Two
callers need slightly different envelopes around the same arithmetic and used
to carry a copy each:

  * ``ct_core.projection_diag.ssim_2d`` — a METRIC on one 2-D image, with an
    optional validity mask and an option to return the map instead of a scalar.
  * the learned backends' structural LOSS — batched, differentiable, and
    multi-scale, evaluated inside an autocast region.

Neither envelope is a subset of the other, which is exactly why the duplication
survived: each side looked like it had a reason to exist. What they share is the
part that can be silently wrong — the window, the stabiliser constants, and the
E[x^2] - E[x]^2 variance — so that part lives here and both import it.

NUMERICS, learned the hard way. The variance terms are a difference of two
similar quantities, so they cancel catastrophically in bf16 and the MS-SSIM
fractional powers amplify whatever is left into NaN. `ssim_components` therefore
forces fp32 and disables any surrounding autocast, and clamps the variances at
zero (rounding can push them slightly negative, which poisons the cs ratio).
The covariance is legitimately signed and is NOT clamped.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Wang et al. (2004) stabiliser constants.
K1 = 0.01
K2 = 0.03

# Wang et al. (2003) 5-scale MS-SSIM weights, coarse -> fine.
MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def gaussian_window_1d(window_size: int, sigma: float, dtype, device):
    """Normalised 1-D Gaussian, centred on the window."""
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_window_2d(window_size: int, sigma: float, dtype, device):
    """Separable 2-D Gaussian shaped (1, 1, w, w) for conv2d."""
    g = gaussian_window_1d(window_size, sigma, dtype, device)
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def to_bchw(x: torch.Tensor) -> torch.Tensor:
    """(H,W) or (B,H,W) -> (B,1,H,W). Already-4-D input passes through."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def ssim_components(pred: torch.Tensor, target: torch.Tensor, *,
                    data_range: float, window_size: int = 11,
                    sigma: float = 1.5):
    """Per-pixel SSIM and contrast-structure maps on (B,1,H,W) inputs.

    Returns ``(ssim_map, cs_map)``, both (B,1,H',W') with H' = H - window_size
    + 1 (a 'valid' convolution, matching skimage's ``gaussian_weights=True``).
    The cs map is returned separately because MS-SSIM needs it at every scale
    but the luminance term only at the finest.
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = gaussian_window_2d(window_size, sigma, pred.dtype, pred.device)

    mu_p = F.conv2d(pred, win)
    mu_t = F.conv2d(target, win)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    # Clamped: the subtraction can go slightly negative from rounding, which
    # would poison the cs ratio. The covariance below is legitimately signed.
    sig_p2 = (F.conv2d(pred * pred, win) - mu_p2).clamp(min=0.0)
    sig_t2 = (F.conv2d(target * target, win) - mu_t2).clamp(min=0.0)
    sig_pt = F.conv2d(pred * target, win) - mu_pt

    lum = (2.0 * mu_pt + C1) / (mu_p2 + mu_t2 + C1)
    cs = (2.0 * sig_pt + C2) / (sig_p2 + sig_t2 + C2)
    return lum * cs, cs


def ssim(pred: torch.Tensor, target: torch.Tensor, *, data_range: float,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM per image. Input (H,W), (B,H,W) or (B,1,H,W); output (B,)."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = to_bchw(pred).float(), to_bchw(target).float()
        s, _ = ssim_components(p, t, data_range=data_range,
                               window_size=window_size, sigma=sigma)
        return s.mean(dim=(1, 2, 3))


def msssim(pred: torch.Tensor, target: torch.Tensor, *, data_range: float,
           window_size: int = 11, sigma: float = 1.5,
           weights=None) -> torch.Tensor:
    """Mean MS-SSIM per image. Output (B,).

    The number of scales is clamped so the coarsest level still spans the
    window — small patches simply get fewer scales, with the weights
    renormalised — rather than failing or silently convolving past the edge.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = to_bchw(pred).float(), to_bchw(target).float()
        w = list(weights) if weights else list(MSSSIM_WEIGHTS)
        min_hw = min(p.shape[-2], p.shape[-1])
        max_lev = max(1, int(math.floor(math.log2(min_hw / window_size))) + 1)
        if len(w) > max_lev:
            w = w[:max_lev]
            total = sum(w)
            w = [x / total for x in w]

        out = None
        for i in range(len(w)):
            s_i, cs_i = ssim_components(p, t, data_range=data_range,
                                        window_size=window_size, sigma=sigma)
            # x**(weight) with weight < 1 blows up as x -> 0, so floor the base.
            if i < len(w) - 1:
                factor = cs_i.mean(dim=(1, 2, 3)).clamp(min=1e-3) ** w[i]
                p = F.avg_pool2d(p, kernel_size=2)
                t = F.avg_pool2d(t, kernel_size=2)
            else:
                # The finest remaining level contributes the full SSIM, i.e.
                # it is the only scale that carries the luminance term.
                factor = s_i.mean(dim=(1, 2, 3)).clamp(min=1e-3) ** w[i]
            out = factor if out is None else out * factor
        return out
