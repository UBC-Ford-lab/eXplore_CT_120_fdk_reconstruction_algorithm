"""Render a whole projection, chunked — the one implementation.

`render_rays` takes a flat batch of rays. Turning that into "the projection at
angle k" needs three more decisions, and every caller that wanted a projection
was making them itself:

  * WHICH detector pixels — a stride over the panel, and optionally a
    restriction to the rows/columns whose rays stay inside the reconstruction
    domain;
  * HOW MANY AT ONCE — a chunk loop, because a full panel at a few hundred
    samples per ray is tens of millions of model queries;
  * DETERMINISM — `stratified=False`, so the same model and angle give the same
    projection every time. Jitter would put sampling noise into an SSIM curve
    that is supposed to be tracking the model.

There were two copies of that: muNeRF's `inr_pipeline.metrics.render_projection`
(model in, Scene in) and `ct_core.projection_diag.render_projection_from_volume`
(finished volume in, ScanContext in). They agreed on the arithmetic and differed
only in the envelope, which is the shape of duplication that stays correct right
up until one side is fixed and the other is not. The envelopes stay where they
were; the middle moved here.

Nothing in this module knows what the model is. `volume_fn` is any
Callable[(M,3) mm-normalised -> (M,)], so a trilinear voxel grid, a hash-grid
INR, an ROI indicator function and a closure over a vendor volume are all the
same thing to it — which is what makes one projection renderer enough.
"""

from __future__ import annotations

from typing import Callable

import torch

from .ray_sampler import rays_from_indices
from .renderer import render_rays
from .scene import Scene


def detector_indices(n: int, factor: int, start: int = 0,
                     stop: int | None = None) -> torch.Tensor:
    """Indices on a regular stride: `arange(start, stop, factor)`.

    A fixed stride rather than a random subset, so the same call at two
    different iterations compares the same pixels and an SSIM difference means
    the model changed. factor=1 keeps every pixel.
    """
    if factor < 1:
        raise ValueError(f"downsample factor must be >= 1, got {factor}")
    stop = n if stop is None else int(stop)
    return torch.arange(int(start), stop, int(factor))


def perturb_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    R_s: float,
    SDD: float,
    da: float = 0.0,
    db: float = 0.0,
    delta_cpa=None,
    delta_cpb=None,
    eta=None,
    phi=None,
    psi=None,
    origin_scale=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a cone-beam geometry perturbation to a batch of rays.

    This is the forward model of a *calibration*: the source distance, the
    central pixel and the three detector rotation angles are treated as
    parameters rather than as constants read from the scan header, so a caller
    that is fitting them can render through the geometry as currently estimated.

    Every argument may be a plain float or a tensor. Passing `nn.Parameter`s
    keeps the whole thing differentiable, which is what lets the corrections be
    LEARNED; passing floats gives the same arithmetic for a one-off render.

    Parameters
    ----------
    R_s, SDD : nominal source-to-isocentre and source-to-detector distances (mm).
    da, db : detector pitch along the column (a) and row (b) axes (mm), used to
        turn a central-pixel offset in PIXELS into a displacement in mm.
    delta_cpa, delta_cpb : central-pixel offset (pixels).
    eta, phi, psi : detector rotations (rad) about the beam, the row axis and
        the column axis respectively, to first order.
    origin_scale : multiplies the source position, i.e. a fractional change in
        R_s. Applied LAST and deliberately so — the tilt terms need the true
        source direction, so scaling the origin first would rotate the frame the
        tilt is expressed in and silently change what eta/phi/psi mean.

    Returns (origins, directions) with directions re-normalised.
    """
    want_dir = (delta_cpa is not None) or (eta is not None)
    if want_dir:
        s = -origins / R_s                                   # unit-ish source dir
        u = torch.stack([-s[:, 1], s[:, 0], torch.zeros_like(s[:, 0])], dim=-1)
        v = torch.zeros_like(s)
        v[:, 2] = 1.0
        dot_d_s = (directions * s).sum(dim=-1, keepdim=True)
        d_unnorm = directions * (SDD / dot_d_s)              # ray to detector plane
        corr = torch.zeros_like(d_unnorm)
        if delta_cpa is not None:
            corr = corr + (delta_cpa * da) * u
            corr = corr + (delta_cpb * db) * v
        if eta is not None:
            a_off = (d_unnorm * u).sum(dim=-1, keepdim=True)
            b_off = (d_unnorm * v).sum(dim=-1, keepdim=True)
            corr = corr + (eta * (a_off * v - b_off * u)
                           + (psi * a_off + phi * b_off) * s)
        d_new = d_unnorm + corr
        directions = d_new / d_new.norm(dim=-1, keepdim=True)
    if origin_scale is not None:
        origins = origin_scale * origins
    return origins, directions


def geometry_perturbation(tilt: dict | None = None, origin_scale=None):
    """`perturb_rays` packaged as a `ray_transform`, or None if nothing moves.

    `tilt` carries the nominal geometry (`R_s`, `SDD`, `da`, `db`) plus whichever
    of `delta_cpa`/`delta_cpb` and `eta`/`phi`/`psi` are being fitted. Returning
    None when there is nothing to apply means the ordinary path pays nothing,
    not even a closure call per chunk.
    """
    if tilt is None and origin_scale is None:
        return None
    t = dict(tilt or {})

    def transform(o_c, d_c):
        return perturb_rays(
            o_c, d_c,
            R_s=t.get("R_s", 1.0), SDD=t.get("SDD", 1.0),
            da=t.get("da", 0.0), db=t.get("db", 0.0),
            delta_cpa=t.get("delta_cpa"), delta_cpb=t.get("delta_cpb"),
            eta=t.get("eta"), phi=t.get("phi"), psi=t.get("psi"),
            origin_scale=origin_scale,
        )

    return transform


def render_rays_chunked(
    origins: torch.Tensor,
    directions: torch.Tensor,
    volume_fn: Callable[[torch.Tensor], torch.Tensor],
    scene: Scene,
    *,
    num_samples: int,
    chunk_size: int = 8192,
    stratified: bool = False,
    spectrum=None,
    ray_transform: Callable | None = None,
) -> torch.Tensor:
    """Line integrals for N rays, evaluated `chunk_size` rays at a time.

    Peak memory is bounded by `chunk_size * num_samples` model queries
    regardless of how many rays are asked for, which is what lets a full
    detector panel be rendered on a small GPU.

    `ray_transform`, if given, is applied per chunk as
    ``(origins, directions) -> (origins, directions)``. It exists so a caller
    that is CALIBRATING the geometry — perturbing the source distance, the
    central pixel, or the detector tilt angles — can inject that perturbation
    without the chunk loop having to know what a tilt is. Applying it per chunk
    rather than up front keeps the memory bound.
    """
    n_rays = int(origins.shape[0])
    out = torch.empty(n_rays, dtype=torch.float32, device=origins.device)
    for i in range(0, n_rays, chunk_size):
        j = min(i + chunk_size, n_rays)
        o, d = origins[i:j], directions[i:j]
        if ray_transform is not None:
            o, d = ray_transform(o, d)
        out[i:j] = render_rays(o, d, volume_fn, scene, num_samples=num_samples,
                               stratified=stratified, spectrum=spectrum)
    return out


def render_projection(
    volume_fn: Callable[[torch.Tensor], torch.Tensor],
    scene: Scene,
    angle_index: int,
    num_samples: int,
    device: torch.device | str,
    *,
    downsample: int = 1,
    b_range: tuple[int, int] | None = None,
    a_range: tuple[int, int] | None = None,
    chunk_size: int = 8192,
    stratified: bool = False,
    spectrum=None,
    ray_transform: Callable | None = None,
    tilt: dict | None = None,
    origin_scale=None,
    no_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one projection and return `(pred, target)`, both (n_b', n_a').

    `target` is the measured projection sampled at exactly the same pixels, so
    the two are directly comparable with no further alignment.

    `b_range` / `a_range` restrict the panel to a half-open window before the
    stride is applied — used when only part of the detector has rays that stay
    inside the reconstruction domain, since a ray that leaves the domain cannot
    be predicted and would only dilute the comparison.

    `tilt` / `origin_scale` render through a PERTURBED geometry — see
    `geometry_perturbation`. A caller fitting the geometry must diagnose through
    the geometry it has fitted, not the one in the scan header, or its own
    diagnostic disagrees with its own training rays. `ray_transform` is the raw
    escape hatch for anything else; it is mutually exclusive with these two.
    """
    if tilt is not None or origin_scale is not None:
        if ray_transform is not None:
            raise ValueError(
                "pass either ray_transform or tilt/origin_scale, not both — "
                "they would silently compose in an order nobody chose")
        ray_transform = geometry_perturbation(tilt, origin_scale)

    n_b, n_a = scene.detector_shape
    # Indices are built on the SINOGRAM's device, which is normally the CPU:
    # `rays_from_indices` does `scene.sinogram.to(angle_idx.device)`, so GPU
    # indices would try to move the entire multi-GB sinogram onto the GPU.
    sino_device = scene.sinogram.device
    b0, b1 = (0, n_b) if b_range is None else (int(b_range[0]), int(b_range[1]))
    a0, a1 = (0, n_a) if a_range is None else (int(a_range[0]), int(a_range[1]))
    b_keep = detector_indices(n_b, downsample, b0, b1).to(sino_device)
    a_keep = detector_indices(n_a, downsample, a0, a1).to(sino_device)

    bb, aa = torch.meshgrid(b_keep, a_keep, indexing="ij")
    b_flat, a_flat = bb.reshape(-1), aa.reshape(-1)
    angle = torch.full((b_flat.numel(),), int(angle_index),
                       dtype=torch.long, device=sino_device)

    origins, directions, target = rays_from_indices(scene, angle, b_flat, a_flat)
    origins = origins.to(device)
    directions = directions.to(device)
    target = target.to(device)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        pred = render_rays_chunked(
            origins, directions, volume_fn, scene, num_samples=num_samples,
            chunk_size=chunk_size, stratified=stratified, spectrum=spectrum,
            ray_transform=ray_transform)

    shape = (b_keep.numel(), a_keep.numel())
    return pred.reshape(shape), target.reshape(shape)
