"""SART emulation: the classical simultaneous update expressed as a weighted
data term plus a gradient preconditioner.

WHAT IT IS. Classical SIRT solves ``A x = b`` with

    x^{k+1} = x^k + C A^T R (b - A x^k)

i.e. preconditioned steepest descent on the ROW-weighted misfit
``f(x) = 1/2 ||A x - b||_R^2``, where

  * ``R = diag(1 / sum_j A_ij)`` — per-ray normalisation by path length,
  * ``C = diag(1 / sum_i A_ij)`` — per-voxel normalisation by coverage.

A learned backend descending a plain projection MSE with R = C = I is therefore
already **Landweber**. Adding R and C, plus the non-negativity projection and
the dense voxel representation the backend already has, is what turns it into
the classical method.

WHY "SART" AND NOT "SIRT". SIRT is a *simultaneous* method: every update uses
every ray. This backend draws a random batch per step, which is the defining
feature of the SART / ordered-subset family rather than of SIRT. The batching is
kept deliberately — it is what makes the run fit in memory and finish — so the
emulation is named for the method it actually implements.

R is an exact reweighting of the misfit, so it is a genuine loss and is
registered as the ``sart`` data term. **C is not a loss**: it scales the
gradient without touching the forward value, so it is applied through the
renderer's ``grad_scale_fn`` hook.

THE DOMAIN CAVEAT, which is what the ROI machinery here is for. Textbook SIRT
assumes the reconstruction grid IS the object support. That does not hold when
the model domain is much larger than the reconstructed field — a common setup,
since the domain has to enclose everything that attenuates (the specimen, the
bed) while the field of interest is smaller. Measured over the full domain:

  * ``L_i`` is dominated by air path, so every object-carrying ray pins to
    roughly the same small weight;
  * ``1 / L_i`` is unbounded as L -> 0, so rays that merely graze the domain rim
    take orders of magnitude more weight than the median while carrying almost
    no signal;
  * coverage is near-flat inside the field (C does nothing where it matters) and
    ramps up in the outer annulus, which is the least-determined region.

So ``ray_support_lengths`` measures L over the ROI with outside path counted at
a reduced rate, ``sart_weighted_mse`` clamps the weights around their median,
and ``CoveragePreconditioner`` holds C neutral outside the ROI.

``outside_weight`` is deliberately > 0. Whatever sits outside the object ROI —
the bed, the holder — really does attenuate, so those rays carry signal. Giving
them zero length would drop them from the loss entirely, leaving that material
unconstrained and corrupting the forward model of the ROI rays themselves.
"""

from __future__ import annotations

import torch

from .renderer import ray_domain_intersect, render_rays


# --------------------------------------------------------------------------
# Object support
# --------------------------------------------------------------------------

def roi_bounds(scene, *, scale: float = 1.0):
    """Object-support bounds from the scene's export ROI.

    Returns ``(roi_min, roi_max, radius_xy, center_xy)`` in mm, using the
    inscribed cylinder because a CT field of view is circular. ``scale``
    inflates (>1) or shrinks (<1) the support about its centre — useful when the
    object is known to occupy less than the export extent, and the only way to
    get a support smaller than the reconstruction grid.
    """
    rmin = scene.export_aabb_min.clone()
    rmax = scene.export_aabb_max.clone()
    center = (rmin + rmax) * 0.5
    half = (rmax - rmin) * 0.5 * float(scale)
    rmin, rmax = center - half, center + half
    radius = float(torch.minimum(half[0], half[1]))
    return rmin, rmax, radius, (float(center[0]), float(center[1]))


def roi_mask(xyz_mm, roi_min, roi_max, *, radius, center_xy=(0.0, 0.0)):
    """Float mask (...,) — 1 inside the object support, 0 outside.

    A z-slab intersected with a cylinder in xy. Distinct from
    ``ModelDomain.cylinder_mask``, which bounds the RECONSTRUCTION DOMAIN and
    has no z extent; this bounds the smaller region the object occupies inside
    it, which is the whole point of the ROI weighting.
    """
    zmin, zmax = roi_min[2].to(xyz_mm), roi_max[2].to(xyz_mm)
    inside_z = (xyz_mm[..., 2] >= zmin) & (xyz_mm[..., 2] <= zmax)
    dx = xyz_mm[..., 0] - center_xy[0]
    dy = xyz_mm[..., 1] - center_xy[1]
    inside_xy = (dx * dx + dy * dy) < float(radius) ** 2
    return (inside_z & inside_xy).to(xyz_mm.dtype)


# --------------------------------------------------------------------------
# Row sums: L_i
# --------------------------------------------------------------------------

@torch.no_grad()
def ray_support_lengths(origins, dirs, scene, num_samples: int, *,
                        roi_min, roi_max, roi_radius, roi_center_xy=(0.0, 0.0),
                        outside_weight: float = 0.25) -> torch.Tensor:
    """Effective row length: path through the object ROI, plus path outside it
    counted at ``outside_weight``.

        L_i = sum_p dt * ( roi + outside_weight * (1 - roi) )

    Computed as a FORWARD PROJECTION of that per-point weight, so the quadrature
    is by construction the renderer's own. An earlier version replicated the
    renderer's sampling in order to reach the sample positions; a replica of a
    quadrature is a thing that can silently disagree with the operator it is
    meant to describe, and L_i only means anything if it is the row sum of the
    matrix actually being applied.
    """
    if outside_weight <= 0:
        raise ValueError(
            "outside_weight must be > 0 — material outside the ROI (bed, "
            "holder) attenuates, so those rays carry signal and must keep a "
            "meaningful row length or they drop out of the loss entirely.")

    domain = scene.model_domain
    amin, amax = domain.aabb_min, domain.aabb_max

    def _weight(xyz_norm: torch.Tensor) -> torch.Tensor:
        # The renderer hands normalised coordinates; the ROI is in mm.
        lo = amin.to(xyz_norm)
        hi = amax.to(xyz_norm)
        centre, half = (lo + hi) * 0.5, (hi - lo) * 0.5
        xyz_mm = centre + xyz_norm * half
        r = roi_mask(xyz_mm, roi_min, roi_max, radius=roi_radius,
                     center_xy=roi_center_xy)
        return r + float(outside_weight) * (1.0 - r)

    return render_rays(origins, dirs, _weight, scene,
                       num_samples=num_samples, stratified=False)


@torch.no_grad()
def ray_chord_lengths(origins, dirs, scene, num_samples: int) -> torch.Tensor:
    """L_i over the FULL model domain — the strict row sum for this A.

    A constant-1 forward projection. Kept because it is the textbook quantity
    and the reference the ROI-restricted version is judged against; see the
    domain caveat in the module docstring for why it is not the default.
    """
    def _ones(xyz_norm):
        return torch.ones(xyz_norm.shape[0], device=xyz_norm.device,
                          dtype=xyz_norm.dtype)
    return render_rays(origins, dirs, _ones, scene,
                       num_samples=num_samples, stratified=False)


# --------------------------------------------------------------------------
# R: the row-weighted data term
# --------------------------------------------------------------------------

def sart_weighted_mse(pred, target, chord, floor_frac: float = 1e-3,
                      w_clamp_lo: float = 0.25, w_clamp_hi: float = 4.0):
    """Row-weighted misfit ``sum_i w_i (pred_i - target_i)^2`` with
    ``w_i = 1/L_i`` (R), returned as a weighted MEAN so its scale tracks
    ordinary MSE and a learning rate transfers between the two.

    The weights are CLAMPED to ``[w_clamp_lo, w_clamp_hi] x median(w)``, because
    ``1/L`` is unbounded as ``L -> 0``: rays grazing the support take orders of
    magnitude more weight than the median while carrying almost no signal, only
    detector noise. Pass ``w_clamp_hi=None`` for strict, unbounded weighting.

    Rays with ~zero support are excluded — their row sum is undefined.

    Returns ``(loss, weights)``; the weights are returned so a caller can log
    what the reweighting actually did rather than assume it.
    """
    chord = chord.reshape(-1)
    pos = chord[chord > 0]
    Lmed = pos.median() if pos.numel() > 0 else chord.new_tensor(1.0)
    floor = floor_frac * Lmed
    w = torch.where(chord > floor, 1.0 / (chord + floor), torch.zeros_like(chord))
    if w_clamp_hi is not None:
        wpos = w[w > 0]
        if wpos.numel() > 0:
            wmed = wpos.median()
            w = torch.where(
                w > 0,
                w.clamp(min=float(w_clamp_lo) * wmed,
                        max=float(w_clamp_hi) * wmed),
                w)
    resid = (pred.reshape(-1) - target.reshape(-1)) ** 2
    denom = w.sum().clamp_min(1e-12)
    return (w * resid).sum() / denom, w


# --------------------------------------------------------------------------
# C: the coverage preconditioner
# --------------------------------------------------------------------------

@torch.no_grad()
def build_coverage_grid(scene, *, grid_res: int = 96, n_rays: int = 1_000_000,
                        cov_samples: int = 128, chunk: int = 40_000,
                        aabb_min=None, aabb_max=None, device=None):
    """Back-project an all-ones sinogram onto a grid: ``cov[z,y,x] ~ sum_i A_ij``.

    ``aabb_min/max`` select the grid EXTENT (default: the full model domain).
    Restricting the extent to the object ROI resolves C inside the object
    (~0.3 mm cells instead of ~0.9 mm) instead of spending resolution on air.
    The RAY SET is always all measured rays, so the values stay the true A^T 1.

    Returns ``(cov (G,G,G) indexed [z,y,x], aabb_min, aabb_max)``.
    """
    device = device if device is not None else scene.sinogram.device
    from .ray_sampler import sample_random_rays

    dom_min = scene.aabb_min.to(device)
    dom_max = scene.aabb_max.to(device)
    g_min = dom_min if aabb_min is None else aabb_min.to(device)
    g_max = dom_max if aabb_max is None else aabb_max.to(device)
    G = int(grid_res)
    grid = torch.zeros(G * G * G, device=device, dtype=torch.float32)
    centers = (torch.arange(cov_samples, device=device, dtype=torch.float32)
               + 0.5) / cov_samples

    done = 0
    while done < n_rays:
        m = min(chunk, n_rays - done)
        done += m
        origins, dirs, _ = sample_random_rays(scene, m, device=device)
        origins = origins.to(device).float()
        dirs = dirs.to(device).float()
        t_near, t_far, valid = ray_domain_intersect(origins, dirs,
                                                    scene.model_domain)
        zero = torch.zeros_like(t_near)
        t_near = torch.where(valid, t_near, zero)
        t_far = torch.where(valid, t_far, zero)
        span = (t_far - t_near).unsqueeze(-1)
        t_s = t_near.unsqueeze(-1) + centers.unsqueeze(0) * span
        xyz = origins.unsqueeze(1) + t_s.unsqueeze(-1) * dirs.unsqueeze(1)
        cyl = scene.model_domain.cylinder_mask(xyz)
        wpt = (span / cov_samples) * cyl
        # normalize into the GRID extent; drop samples outside it
        rel = (xyz - g_min) / (g_max - g_min).clamp_min(1e-12)   # [0,1] inside
        inside = ((rel >= 0).all(dim=-1) & (rel <= 1).all(dim=-1))
        idx = (rel.clamp(0, 1) * (G - 1)).round().long()
        flat = (idx[..., 2] * G + idx[..., 1]) * G + idx[..., 0]
        grid.scatter_add_(0, flat.reshape(-1),
                          (wpt * inside.to(wpt.dtype)).reshape(-1).float())

    return grid.reshape(G, G, G), g_min, g_max



class CoveragePreconditioner:
    """``C = 1 / coverage``, as a per-sample gradient weight.

    Holds a fixed, geometry-only coverage grid over ``aabb_min/max``. Calling it
    at sample positions returns a detached per-sample weight for the renderer's
    ``grad_scale_fn``, which scales the backward pass while leaving the forward
    value untouched — which is what makes C a preconditioner rather than a term
    in the objective.

    Normalised so the well-covered interior is ~1 and clamped to
    ``[c_min, c_max]``. Points OUTSIDE the grid extent return ``outside_value``
    (default 1.0 = neutral). With the grid built over the object ROI, that is
    what stops the outer annulus — partially sampled, truncation-prone — from
    being handed the largest steps in the volume.
    """

    def __init__(self, scene, cov_grid, aabb_min=None, aabb_max=None, *,
                 c_min: float = 0.2, c_max: float = 5.0, eps_frac: float = 0.02,
                 outside_value: float = 1.0, device=None):
        device = device if device is not None else cov_grid.device
        self.device = device
        self.outside_value = float(outside_value)
        G = cov_grid.shape[0]
        domain = scene.model_domain
        self.aabb_min = ((domain.aabb_min if aabb_min is None else aabb_min)
                         .to(device).float())
        self.aabb_max = ((domain.aabb_max if aabb_max is None else aabb_max)
                         .to(device).float())

        covered = cov_grid > 0
        pos = cov_grid[covered]
        ref = pos.median() if pos.numel() > 0 else cov_grid.new_tensor(1.0)
        cov_norm = cov_grid / ref.clamp_min(1e-12)               # interior ~1
        C = (1.0 / (cov_norm + eps_frac)).clamp(c_min, c_max)
        Cpos = C[covered]
        cref = Cpos.median() if Cpos.numel() > 0 else C.new_tensor(1.0)
        C = (C / cref.clamp_min(1e-12)).clamp(c_min, c_max)

        self.C = C.reshape(1, 1, G, G, G).to(device).float()     # (1,1,Dz,Hy,Wx)
        self.stats = dict(
            grid_res=int(G), cov_ref=float(ref), c_min=float(c_min),
            c_max=float(c_max), eps_frac=float(eps_frac),
            outside_value=self.outside_value,
            C_min=float(C.min()), C_max=float(C.max()), C_mean=float(C.mean()),
            covered_frac=float(covered.float().mean()),
        )

    def __call__(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        """xyz_mm (..., 3) -> per-point weight (...), detached."""
        shp = xyz_mm.shape[:-1]
        x = xyz_mm.to(self.device).float()
        rel = (x - self.aabb_min) / (self.aabb_max - self.aabb_min).clamp_min(1e-12)
        inside = ((rel >= 0).all(dim=-1) & (rel <= 1).all(dim=-1))
        # grid_sample wants [-1,1]; last dim (x,y,z) maps to (W,H,D), C is [z,y,x]
        gs = (rel.clamp(0, 1) * 2.0 - 1.0).reshape(1, -1, 1, 1, 3)
        samp = torch.nn.functional.grid_sample(
            self.C, gs, mode="bilinear", align_corners=True,
            padding_mode="border").reshape(-1)
        out = torch.where(inside.reshape(-1), samp,
                          torch.full_like(samp, self.outside_value))
        return out.reshape(shp).detach()
