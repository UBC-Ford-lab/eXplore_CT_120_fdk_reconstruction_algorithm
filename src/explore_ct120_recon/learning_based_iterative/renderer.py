"""Differentiable X-ray (line-integral) forward operator.

For each ray we compute its segment inside the volume AABB, sample N points
along that segment, query a model `volume_fn` for densities (in the [-1, 1]^3
frame), and accumulate the line integral via a Riemann sum:

    p = integral_{t_near}^{t_far}  rho(ray(t)) dt
      ~= sum_i  rho_i  *  (t_far - t_near) / N

Densities are linear-attenuation values in mm^-1, matching the line integrals
produced by `ct_core.preprocessing.preprocess_sinogram` (which returns
-log(T) = integral mu ds, in the same units).

`render_rays_hierarchical` is the NeRF-style two-network variant: a coarse
network produces a per-bin density estimate, those values define a PDF
along each ray (weight_i = mu_i * delta_i — the bin's contribution to the
line integral), and a fine network is evaluated on the union of the coarse
samples and N_f importance-sampled points. Both networks are supervised by
their own line-integral MSE.

OPTIONAL KERNEL FUSION (`set_render_compile`). The forward is bandwidth-bound,
not compute-bound: for a 12288-ray batch at 1091 samples/ray each `(N, S, 3)`
intermediate is 161 MB, and eager mode materializes `t_unit`, `t_samples`,
`xyz_mm` and `xyz_norm` in turn. Everything between the ray-domain clip and
the model query is elementwise, so `torch.compile` fuses it into a single
kernel that writes only what the model actually consumes (`xyz_norm` and the
cylinder mask) — roughly halving the traffic, with the same saving again in
the backward pass. Note this is NOT AMP: `grid_sample`, the elementwise chain
and the reduction all stay float32 (autocast would downcast none of them
anyway, as it only targets matmul, of which this path has none).

ON by default since 2026-08-21. Fusion reorders floating-point operations, so
results move in the last bits and an eager baseline is not bit-comparable with
a compiled run — which is why the mode travels with the run's config rather
than why it stays off. Turn it off with `--compile off` (the driver) or
`set_render_compile('off')` when a run has to be bit-comparable with an eager
one.

The fusion needs Triton, which needs compute capability >= 7.0 (Volta). On
anything older — a P100 is Pascal, 6.0 — `fusion_supported` reports why and
the caller downgrades to eager, so asking for fusion on a card that cannot do
it costs nothing and says so.
"""

from __future__ import annotations

import warnings
from typing import Callable

import torch

from .scene import Scene, normalize_to_unit_cube

# --------------------------------------------------------------------------
# kernel fusion control (process-global; the renderer is called from many
# places and threading a flag through all of them would be worse)
# --------------------------------------------------------------------------

COMPILE_MODES = ("off", "on", "max-autotune")

_COMPILE_MODE = "off"
_COMPILED: dict = {}
_COMPILE_FAILED = False


def set_render_compile(mode: str = "off") -> str:
    """Enable/disable `torch.compile` fusion of the renderer's hot kernels.

    ``'off'``          eager (default) — bit-for-bit the pre-2026-08-13 path.
    ``'on'``           Inductor default mode.
    ``'max-autotune'`` benchmarks kernel variants at compile time; minutes of
                       extra compile for a few percent, worth it only for long
                       runs.

    Compilation happens on first call. If it raises for any reason (no C++
    toolchain, no Triton, unsupported GPU) the renderer warns once and falls
    back to eager for the rest of the process rather than failing the run.
    """
    global _COMPILE_MODE, _COMPILE_FAILED
    mode = str(mode).strip().lower()
    if mode in ("none", "false", "0"):
        mode = "off"
    if mode in ("true", "1", "default"):
        mode = "on"
    if mode not in COMPILE_MODES:
        raise ValueError(f"compile mode must be one of {COMPILE_MODES}, got {mode!r}")
    if mode != _COMPILE_MODE:
        _COMPILE_FAILED = False
    _COMPILE_MODE = mode
    return _COMPILE_MODE


#: Volta. Triton, torch.compile's Inductor backend, does not support Pascal.
#: Same gate as `training.TENSOR_CORE_CAPABILITY`, kept here so the renderer
#: does not import the training module for one integer.
FUSION_MIN_CAPABILITY = 7


def fusion_supported(device=None) -> tuple[bool, str]:
    """Whether `torch.compile` fusion can actually run here, and why not.

    Checked BEFORE compiling rather than caught after, so a run on a card that
    cannot fuse prints one honest line and proceeds eager, instead of paying a
    compile attempt and raising a RuntimeWarning through `_fused`.
    """
    if not hasattr(torch, "compile"):
        return False, "this torch has no torch.compile"
    if not torch.cuda.is_available():
        return False, "no CUDA device (Inductor's CPU backend is not the win here)"
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception as exc:                       # pragma: no cover - env dep
        return False, f"could not read the compute capability ({exc})"
    if major < FUSION_MIN_CAPABILITY:
        name = torch.cuda.get_device_name(device)
        return False, (f"{name} is compute capability {major}.{minor} < "
                       f"{FUSION_MIN_CAPABILITY}.0 — Triton/Inductor does not "
                       f"support Pascal")
    return True, ""


def render_compile_mode() -> str:
    """The fusion mode actually in force ('off' after a failed compile)."""
    return "off" if _COMPILE_FAILED else _COMPILE_MODE


def _fused(fn, *args, **kwargs):
    """Call `fn`, compiled when fusion is on, with a one-shot eager fallback.

    OOM is re-raised untouched — that is a real capacity signal the caller's
    batch sizing needs to see, not a compilation problem.
    """
    global _COMPILE_FAILED
    if _COMPILE_MODE == "off" or _COMPILE_FAILED:
        return fn(*args, **kwargs)
    key = (fn.__name__, _COMPILE_MODE)
    compiled = _COMPILED.get(key)
    if compiled is None:
        opts = {} if _COMPILE_MODE == "on" else {"mode": _COMPILE_MODE}
        compiled = torch.compile(fn, **opts)
        _COMPILED[key] = compiled
    try:
        return compiled(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:                       # pragma: no cover - env dep
        _COMPILE_FAILED = True
        warnings.warn(f"torch.compile failed ({type(exc).__name__}: {exc}); "
                      f"falling back to eager rendering for this process.",
                      RuntimeWarning, stacklevel=2)
        return fn(*args, **kwargs)


def scale_grad(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Value-preserving gradient scaler.

    Returns a tensor whose forward VALUE equals ``x`` but whose gradient w.r.t.
    ``x`` is scaled by ``s`` (treated as a constant). Used to inject the SIRT
    column preconditioner C into the forward projector without perturbing the
    physical line integral: forward stays ``sum mu ds``, backward becomes
    ``C * dL/dmu`` per sample.

        y = x * s + (x - x * s).detach()
        forward:  x*s + (x - x*s)      = x
        backward: d y / d x            = s
    """
    return x * s + (x - x * s).detach()


def ray_aabb_intersect(
    origins: torch.Tensor,
    directions: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
):
    """Slab method. Returns (t_near, t_far, valid), each shape (N,) / (N,) / (N,) bool.

    `t_near` is clamped to >= 0 so we never integrate behind the source.
    `valid` is True when the ray actually pierces the AABB ahead of its origin.
    """
    inv_dir = 1.0 / directions
    t1 = (aabb_min - origins) * inv_dir
    t2 = (aabb_max - origins) * inv_dir
    t_lo = torch.minimum(t1, t2).amax(dim=-1)
    t_hi = torch.maximum(t1, t2).amin(dim=-1)

    valid = (t_hi > t_lo) & (t_hi > 0)
    t_near = t_lo.clamp(min=0.0)
    t_far = t_hi
    return t_near, t_far, valid


def ray_cylinder_intersect(
    origins: torch.Tensor,
    directions: torch.Tensor,
    center_xy: tuple[float, float],
    radius_xy: float,
    z_min: torch.Tensor | float,
    z_max: torch.Tensor | float,
):
    """Ray vs a z-aligned finite cylinder. Same contract as `ray_aabb_intersect`.

    The cylinder is the intersection of an infinite tube about
    (center_xy, z-axis) with the slab z in [z_min, z_max], so the entry/exit
    interval is the tube's quadratic root pair clipped by the slab.
    """
    ox = origins[..., 0] - center_xy[0]
    oy = origins[..., 1] - center_xy[1]
    dx, dy, dz = directions[..., 0], directions[..., 1], directions[..., 2]

    # --- infinite tube: |o_xy + t d_xy|^2 = R^2 ---
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - float(radius_xy) ** 2
    disc = b * b - 4.0 * a * c

    # A ray parallel to the axis (a == 0) never crosses the tube wall: it is
    # either inside for all t or outside for all t. Both branches are computed
    # unconditionally and selected with `where` so no NaN reaches the output
    # (0/0 in the quadratic would poison even the discarded lane).
    a_safe = a.clamp(min=1e-12)
    sq = torch.sqrt(disc.clamp(min=0.0))
    t_tube_lo = (-b - sq) / (2.0 * a_safe)
    t_tube_hi = (-b + sq) / (2.0 * a_safe)

    inf = torch.full_like(t_tube_lo, float("inf"))
    parallel = a <= 1e-12
    inside_tube = c < 0.0
    t_tube_lo = torch.where(parallel, torch.where(inside_tube, -inf, inf), t_tube_lo)
    t_tube_hi = torch.where(parallel, torch.where(inside_tube, inf, -inf), t_tube_hi)

    # --- z slab. dz == 0 would give 0*inf = NaN via the reciprocal form, so
    # clamp the magnitude instead: a huge finite slope reproduces the correct
    # "inside forever / outside forever" limit without the indeterminate form.
    dz_safe = torch.where(dz.abs() < 1e-12,
                          torch.full_like(dz, 1e-12), dz)
    tz1 = (z_min - origins[..., 2]) / dz_safe
    tz2 = (z_max - origins[..., 2]) / dz_safe
    t_lo = torch.maximum(t_tube_lo, torch.minimum(tz1, tz2))
    t_hi = torch.minimum(t_tube_hi, torch.maximum(tz1, tz2))

    valid = (t_hi > t_lo) & (t_hi > 0)
    return t_lo.clamp(min=0.0), t_hi, valid


def ray_domain_intersect(origins: torch.Tensor, directions: torch.Tensor, domain):
    """Entry/exit t for the segment of each ray inside the MODEL DOMAIN.

    Dispatches on `domain.ray_clip`:

      * ``'domain'`` — the domain body itself (the inscribed cylinder for
        shape='cylinder'; the box otherwise). All `num_samples` quadrature
        points then land on real domain.
      * ``'aabb'`` — the bounding box always, which is what the renderer did
        before 2026-08-09. Samples in the AABB corners are still evaluated by
        the network and then zeroed by `cylinder_mask`, costing 19.5% of all
        samples on Scan_1510's geometry.

    Both integrate the same function: the corners are outside the model domain
    and hold no material by definition. Only the quadrature differs.
    """
    aabb_min = domain.aabb_min.to(origins)
    aabb_max = domain.aabb_max.to(origins)
    if (getattr(domain, "ray_clip", "domain") == "domain"
            and domain.shape == "cylinder" and domain.radius_xy is not None):
        return ray_cylinder_intersect(
            origins, directions, domain.center_xy, domain.radius_xy,
            aabb_min[2], aabb_max[2],
        )
    return ray_aabb_intersect(origins, directions, aabb_min, aabb_max)


def _quadrature(
    origins: torch.Tensor,
    directions: torch.Tensor,
    t_near: torch.Tensor,
    t_far: torch.Tensor,
    u: torch.Tensor | None,
    num_samples: int,
    center: torch.Tensor,
    half: torch.Tensor,
    cyl_center_x: float,
    cyl_center_y: float,
    cyl_radius: float | None,
    need_mm: bool,
):
    """Lay quadrature points along each ray and normalize them for the model.

    THE fusible block: one elementwise chain from the ray parameters to the
    model's input coordinates, with no data-dependent control flow, no RNG
    (the stratification jitter `u` is drawn by the caller so a torch.Generator
    never enters the graph) and no library calls. Everything it touches is
    `(N, num_samples[, 3])`.

    `need_mm` returns the mm-space points as well, which the SIRT column
    preconditioner needs; under fusion that forces the big intermediate to be
    materialized, so it is False on the hot path. It is a Python bool, so the
    two variants specialize into separate graphs rather than branching.

    Returns `(xyz_norm, cyl_mask, delta_t, xyz_mm | None)`.
    """
    span = (t_far - t_near).unsqueeze(-1)
    centers = (torch.arange(num_samples, device=origins.device,
                            dtype=origins.dtype) + 0.5) / num_samples
    if u is None:
        t_unit = centers.unsqueeze(0)
    else:
        t_unit = centers.unsqueeze(0) + (u - 0.5) / num_samples
    t_samples = t_near.unsqueeze(-1) + t_unit * span
    xyz_mm = origins.unsqueeze(1) + t_samples.unsqueeze(-1) * directions.unsqueeze(1)

    xyz_norm = (xyz_mm - center) / half

    if cyl_radius is None:
        cyl_mask = torch.ones(xyz_mm.shape[:-1], dtype=xyz_mm.dtype,
                              device=xyz_mm.device)
    else:
        dx = xyz_mm[..., 0] - cyl_center_x
        dy = xyz_mm[..., 1] - cyl_center_y
        r2 = dx ** 2 + dy ** 2
        cyl_mask = (r2 < cyl_radius ** 2).to(xyz_mm.dtype)

    delta_t = span / num_samples
    return xyz_norm, cyl_mask, delta_t, (xyz_mm if need_mm else None)


def _integrate(density: torch.Tensor, cyl_mask: torch.Tensor,
               delta_t: torch.Tensor) -> torch.Tensor:
    """Mask the samples and Riemann-sum them into a line integral.

    Kept in float32 deliberately: the sum runs over `num_samples` (~1091)
    terms and our projection residuals live at 1e-2, so a reduced-precision
    accumulator would put arithmetic noise at the size of the signal.
    """
    return ((density * cyl_mask) * delta_t).sum(dim=-1)


def _domain_affine(domain, ref: torch.Tensor):
    """`(center, half)` of the mm -> [-1,1] map, matching ModelDomain.normalize."""
    amin = domain.aabb_min.to(ref)
    amax = domain.aabb_max.to(ref)
    return (amin + amax) * 0.5, (amax - amin) * 0.5


def _cylinder_params(domain):
    """`(cx, cy, radius | None)`; radius None means "no mask" (box domain)."""
    if getattr(domain, "shape", None) == "box" or domain.radius_xy is None:
        return 0.0, 0.0, None
    return (float(domain.center_xy[0]), float(domain.center_xy[1]),
            float(domain.radius_xy))


def render_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    volume_fn: Callable[[torch.Tensor], torch.Tensor],
    scene: Scene,
    num_samples: int,
    stratified: bool = True,
    generator: torch.Generator | None = None,
    spectrum=None,
    grad_scale_fn=None,
) -> torch.Tensor:
    """Predict line integrals for a batch of rays.

    Args:
        origins:     (N, 3) ray origins in mm.
        directions:  (N, 3) unit-length ray directions.
        volume_fn:   model wrapped as Callable[(M, 3) -> (M,)], or, in
                     polychromatic mode, (M, 3) -> (M, 2) for (a1, a2).
        scene:       provides the AABB and the mm -> [-1, 1] affine.
        num_samples: samples per ray.
        stratified:  jitter each sample within its bin (default True).
        generator:   torch.Generator for the stratified jitter, optional.
        spectrum:    None → monochromatic p = ∫mu ds (volume_fn → (M,)).
                     A Spectrum → polychromatic: volume_fn → (M, 2), accumulate
                     A1=∫a1 ds, A2=∫a2 ds, and return the beam-hardened line
                     integral p = -ln(Σ_k w_k exp(-f_PE_k A1 - f_KN_k A2)).
        grad_scale_fn: optional Callable[(N, num_samples, 3) mm -> (N, num_samples)]
                     that returns a per-sample gradient scale applied to the
                     density via a value-preserving op (see sirt.scale_grad).
                     Forward output is UNCHANGED; only the backward pass is
                     scaled. Used for the SIRT column preconditioner C and the
                     handoff warmup's interior freeze. In polychromatic mode
                     the scale is applied to both material channels (the
                     constrained split is affine in mu, so scaling the split's
                     gradient equals scaling mu's gradient). Default None.

    Returns:
        (N,) predicted line integrals.
    """
    t_near, t_far, valid = ray_domain_intersect(origins, directions,
                                                scene.model_domain)

    # Axis-aligned rays that miss the volume produce ±inf t values; replace
    # them with a zero-length segment so downstream sampling stays finite.
    zero = torch.zeros_like(t_near)
    t_near = torch.where(valid, t_near, zero)
    t_far = torch.where(valid, t_far, zero)

    n = origins.shape[0]
    device = origins.device

    # RNG stays outside the fused kernel: a torch.Generator cannot be traced,
    # and drawing here keeps the random stream identical whether or not the
    # renderer is compiled, so eager and fused runs see the same rays.
    u = (torch.rand(n, num_samples, device=device, generator=generator,
                    dtype=origins.dtype) if stratified else None)

    # xyz_norm, the cylinder mask and the step width in one pass. `cyl_mask`
    # zeroes samples outside the cylinder: under ray_clip='domain' every
    # sample is already inside by construction and this is a no-op; it stays
    # as the guard for ray_clip='aabb' (where it does the real work) and for
    # the tangent-ray corner case where a root lands exactly on the wall.
    center, half = _domain_affine(scene.model_domain, origins)
    cx, cy, radius = _cylinder_params(scene.model_domain)
    xyz_norm, cyl_mask, delta_t, xyz_mm = _fused(
        _quadrature, origins, directions, t_near, t_far, u, num_samples,
        center, half, cx, cy, radius, grad_scale_fn is not None)

    if spectrum is not None:
        # Polychromatic: accumulate two line integrals (A1, A2), then apply
        # the beam-hardened detector collapse. Same ray sampling as mono.
        if getattr(spectrum, "constrained", False):
            # Single-DOF: the model outputs one channel mu; split it into
            # (a1, a2) on the NIST mixture line so beam hardening is retained
            # with only one per-voxel unknown. See Spectrum.mixture_split.
            mu = volume_fn(xyz_norm.reshape(-1, 3)).reshape(n, num_samples)
            mu = mu * cyl_mask
            a = spectrum.mixture_split(mu)                  # (n, num_samples, 2)
        else:
            a = volume_fn(xyz_norm.reshape(-1, 3)).reshape(n, num_samples, 2)
            a = a * cyl_mask.unsqueeze(-1)
        if grad_scale_fn is not None:
            s = grad_scale_fn(xyz_mm).detach()
            a = scale_grad(a, s.to(a.dtype).unsqueeze(-1))
        A = (a * delta_t.unsqueeze(-1)).sum(dim=1)          # (n, 2)
        return spectrum.collapse(A[..., 0], A[..., 1])

    density = volume_fn(xyz_norm.reshape(-1, 3)).reshape(n, num_samples)
    if grad_scale_fn is not None:
        # SIRT column preconditioner C: scale each sample's gradient by its
        # coverage weight without changing the forward line integral.
        s = grad_scale_fn(xyz_mm).detach()          # (n, num_samples)
        density = scale_grad(density, s.to(density.dtype))
    return _fused(_integrate, density, cyl_mask, delta_t)


def _sample_pdf(
    bin_edges: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Inverse-CDF sampling of a piecewise-constant PDF over t.

    Args:
        bin_edges:  (..., K+1) bin edges in t. Bin i spans [bin_edges[..., i],
                    bin_edges[..., i+1]]. Must be non-decreasing along the last
                    axis (we don't sort here).
        weights:    (..., K) non-negative per-bin weights (un-normalized PDF).
        n_samples:  number of samples to draw per row.
        generator:  optional torch.Generator for the uniform draw.
        eps:        added to weights to keep the PDF well-defined for rays that
                    miss every density region (all-zero weights → uniform).

    Returns:
        (..., n_samples) new t values, each within the support
        [bin_edges[..., 0], bin_edges[..., -1]].

    Mirrors NeRF's `sample_pdf` from the official PyTorch implementation,
    rewritten to operate on explicit bin edges (we already have them) rather
    than reconstructing them from sample midpoints.
    """
    weights = weights + eps
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (..., K+1)

    shape = list(weights.shape[:-1]) + [n_samples]
    u = torch.rand(*shape, device=weights.device, generator=generator,
                   dtype=weights.dtype)

    # Find the bin in which each u lands. searchsorted operates on the last
    # dim; cdf is shape (..., K+1), u is shape (..., n_samples).
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    edge_below = torch.gather(bin_edges, -1, below)
    edge_above = torch.gather(bin_edges, -1, above)

    # Linear interpolation within the picked bin.
    denom = (cdf_above - cdf_below).clamp(min=1e-5)
    alpha = (u - cdf_below) / denom
    return edge_below + alpha * (edge_above - edge_below)


def render_rays_hierarchical(
    origins: torch.Tensor,
    directions: torch.Tensor,
    coarse_fn: Callable[[torch.Tensor], torch.Tensor],
    fine_fn: Callable[[torch.Tensor], torch.Tensor],
    scene: Scene,
    num_coarse: int,
    num_fine: int,
    stratified: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeRF-style hierarchical rendering, adapted to Beer-Lambert.

    Procedure (mirrors NeRF Section 5.2):
      1. Stratified-sample N_c points uniformly along each ray's AABB segment.
      2. Evaluate the COARSE network at those points → mu_i. Compute the
         coarse line integral p_coarse = sum_i mu_i * delta (uniform delta).
      3. Build a PDF along each ray with weights w_i = mu_i * delta — the
         contribution of each bin to the line integral. Inverse-CDF sample
         N_f additional points from this PDF.
      4. Union the N_c coarse and N_f fine samples, sort along t, and
         evaluate the FINE network at all N_c + N_f points.
      5. Integrate using a midpoint-bin Riemann rule with bin edges anchored
         at t_near / t_far, so the support exactly matches the coarse pass.

    Why each piece:
      * weights = mu * delta (not mu alone). The integrand is mu(t); the
        contribution of bin i to the integral is mu_i * delta_i. This is the
        natural importance (∝ |integrand × measure|) and matches what NAF /
        IntraTomo use for CT-INR hierarchical sampling.
      * mu_c.detach() before forming the PDF. The PDF is a fixed sampling
        distribution from the optimizer's POV — we don't want gradients
        flowing through searchsorted into the coarse net (NeRF doesn't either;
        the coarse net is trained only via its own MSE loss).
      * Midpoint-bin edges anchored at t_near, t_far. NeRF's "delta_i =
        t_{i+1} - t_i with last delta = 1e10" trick is correct for alpha
        compositing (the 1e10 cancels via exp(-σ*1e10) ≈ 0) but would add
        a huge spurious term for our linear integral. The midpoint rule
        gives a proper Riemann sum where the bin widths sum to t_far - t_near,
        matching the coarse pass's integration support exactly.
      * eps in _sample_pdf keeps rays that miss the volume (all weights = 0)
        from producing NaNs; their CDF becomes uniform and they sample
        meaningless points, but their final mu_f * delta integrates to 0
        anyway because such rays are zeroed by the AABB-miss handling above.

    Returns:
        (p_coarse, p_fine), each shape (N,). Both are line integrals in the
        same units as render_rays. Train with `mse(p_coarse, target) +
        mse(p_fine, target)`.
    """
    t_near, t_far, valid = ray_domain_intersect(origins, directions,
                                                scene.model_domain)

    zero = torch.zeros_like(t_near)
    t_near = torch.where(valid, t_near, zero)
    t_far = torch.where(valid, t_far, zero)

    n = origins.shape[0]
    device = origins.device
    span = (t_far - t_near).unsqueeze(-1)  # (n, 1)

    # ---- coarse pass: stratified samples on uniform [0, 1] strata ----
    centers = (torch.arange(num_coarse, device=device, dtype=origins.dtype) + 0.5) / num_coarse
    if stratified:
        u = torch.rand(n, num_coarse, device=device, generator=generator, dtype=origins.dtype)
        t_unit_c = centers.unsqueeze(0) + (u - 0.5) / num_coarse
    else:
        t_unit_c = centers.unsqueeze(0).expand(n, num_coarse)
    t_c = t_near.unsqueeze(-1) + t_unit_c * span  # (n, N_c)

    xyz_c = origins.unsqueeze(1) + t_c.unsqueeze(-1) * directions.unsqueeze(1)
    xyz_c_norm = normalize_to_unit_cube(xyz_c, scene)
    mu_c = coarse_fn(xyz_c_norm.reshape(-1, 3)).reshape(n, num_coarse)
    mu_c = mu_c * scene.model_domain.cylinder_mask(xyz_c)

    delta_c = span / num_coarse  # (n, 1) uniform-bin width
    p_coarse = (mu_c * delta_c).sum(dim=-1)

    # ---- fine pass: importance-sample from coarse PDF, union, sort, integrate ----
    edges_unit = torch.linspace(0.0, 1.0, num_coarse + 1, device=device, dtype=origins.dtype)
    t_edges = t_near.unsqueeze(-1) + edges_unit.unsqueeze(0) * span  # (n, N_c+1)

    pdf_weights = (mu_c.detach() * delta_c).clamp(min=0.0)
    t_f = _sample_pdf(t_edges, pdf_weights, num_fine, generator=generator)  # (n, N_f)

    t_combined, _ = torch.sort(torch.cat([t_c, t_f], dim=-1), dim=-1)
    K = num_coarse + num_fine

    xyz_f = origins.unsqueeze(1) + t_combined.unsqueeze(-1) * directions.unsqueeze(1)
    xyz_f_norm = normalize_to_unit_cube(xyz_f, scene)
    mu_f = fine_fn(xyz_f_norm.reshape(-1, 3)).reshape(n, K)
    mu_f = mu_f * scene.model_domain.cylinder_mask(xyz_f)

    # Midpoint Riemann rule with outer edges at t_near, t_far. Sum of widths
    # = t_far - t_near (exactly the coarse-pass integration support).
    edges_f = torch.cat([
        t_near.unsqueeze(-1),
        0.5 * (t_combined[..., :-1] + t_combined[..., 1:]),
        t_far.unsqueeze(-1),
    ], dim=-1)  # (n, K+1)
    delta_f = edges_f[..., 1:] - edges_f[..., :-1]  # (n, K)
    p_fine = (mu_f * delta_f).sum(dim=-1)

    return p_coarse, p_fine
