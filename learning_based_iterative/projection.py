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
    no_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one projection and return `(pred, target)`, both (n_b', n_a').

    `target` is the measured projection sampled at exactly the same pixels, so
    the two are directly comparable with no further alignment.

    `b_range` / `a_range` restrict the panel to a half-open window before the
    stride is applied — used when only part of the detector has rays that stay
    inside the reconstruction domain, since a ray that leaves the domain cannot
    be predicted and would only dilute the comparison.
    """
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
