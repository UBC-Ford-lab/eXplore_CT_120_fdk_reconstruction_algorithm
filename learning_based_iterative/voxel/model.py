"""Dense voxel-grid representation — SIRT's parameterization, trained by SGD.

Moved here (verbatim) from muNeRF's ``inr_pipeline/model.py`` 2026-08-12. The
voxel grid is the canonical learning-based-iterative representation: one free
parameter per voxel, optimized through the differentiable renderer — the same
argmin problem classical iterative recon solves, with the algorithm swapped
for autograd + Adam. muNeRF's ``inr_pipeline.model`` re-exports these names.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def voxel_grid_shape(model_domain_cfg: dict, voxel_mm) -> tuple[int, int, int]:
    """Grid dims (Dz, Hy, Wx) tiling the model-domain AABB at ``voxel_mm``.

    Derived from ``cfg['model_domain']`` ONLY — never from the runtime scene —
    so `build_model(cfg)` stays a pure function of the config and `infer.py`
    can reconstruct the identical shape for `load_state_dict` without loading
    a sinogram.

    Three sources, in order:

    * ``resolved_bounds`` — what a MEASURED domain (``auto``) wrote back once it
      was resolved. This is the normal path: the domain is measured from the
      projections when the scene loads, stamped into the config, and read here
      and by inference, so all three agree by construction.
    * ``half_extent`` or ``extent_xy`` + ``half_extent_z`` — a pinned domain.
    * neither, with ``auto`` still set — an error, because the measurement has
      not happened yet and a grid shape guessed from the export ROI would not
      match the one training used.

    ``voxel_mm`` is a scalar or [vx, vy, vz] in mm.
    """
    cfg = model_domain_cfg or {}
    rb = cfg.get("resolved_bounds")
    if rb is not None:
        hx = (float(rb["x_max"]) - float(rb["x_min"])) / 2.0
        hy = (float(rb["y_max"]) - float(rb["y_min"])) / 2.0
        hz = (float(rb["z_max"]) - float(rb["z_min"])) / 2.0
        return _tile(hx, hy, hz, voxel_mm)
    if bool(cfg.get("auto", True)) and cfg.get("half_extent") is None \
            and cfg.get("extent_xy") is None:
        raise ValueError(
            "model.type: voxel needs the domain to be known before the grid is "
            "built. With model_domain.auto the domain is MEASURED when the "
            "scene loads and written back as `resolved_bounds`; this call "
            "happened first, or the config predates the measurement. Load the "
            "scene before building the model, or pin extent_xy/half_extent_z.")
    he = cfg.get("half_extent")
    if he is not None:
        hx, hy, hz = (float(he[0]), float(he[1]), float(he[2]))
    else:
        exy, hez = cfg.get("extent_xy"), cfg.get("half_extent_z")
        if exy is None or hez is None:
            raise ValueError(
                "model.type: voxel needs model_domain.half_extent or "
                "extent_xy + half_extent_z")
        hx = hy = float(exy) / 2.0
        hz = float(hez)

    return _tile(hx, hy, hz, voxel_mm)


def _tile(hx: float, hy: float, hz: float, voxel_mm) -> tuple[int, int, int]:
    """(Dz, Hy, Wx) covering half-extents (hx, hy, hz) at pitch ``voxel_mm``."""
    v = [float(voxel_mm)] * 3 if not isinstance(voxel_mm, (list, tuple)) \
        else [float(x) for x in voxel_mm]
    if min(v) <= 0:
        raise ValueError(f"model.voxel.size_mm must be > 0, got {voxel_mm!r}")
    # ceil so the grid always COVERS the AABB (a short grid would silently
    # clip the domain the renderer integrates over).
    nx = max(1, math.ceil(2.0 * hx / v[0]))
    ny = max(1, math.ceil(2.0 * hy / v[1]))
    nz = max(1, math.ceil(2.0 * hz / v[2]))
    return (nz, ny, nx)


class VoxelGrid(nn.Module):
    """Dense trainable volume — one free parameter per voxel.

    Drop-in for `CoordMLP`: same `(M, 3) in [-1,1]^3 -> (M,)` (or `(M, C)`)
    contract, so `render_rays`, the metrics and the export path need no
    changes. What differs is the FUNCTION CLASS: `CoordMLP` maps coordinates
    through a hash grid whose finest level packs ~241 spatially scattered
    voxels into one table entry, then a 4x128 MLP. Those voxels cannot take
    independent values. Here they can — this is the same representation TIGRE
    SIRT optimises, so it isolates "is the representation the constraint?"
    from every loss/optimiser question.

    Storage is `(1, C, Dz, Hy, Wx)` so `F.grid_sample`'s `(x, y, z)` grid
    order maps straight onto `(W, H, D)` — the same convention (and the same
    call) as `sirt.CoveragePreconditioner.__call__`.

    ALIGN_CORNERS=False, deliberately. With N voxels TILING the AABB, voxel i
    is centred at `2(i+0.5)/N - 1` in normalised coords, which is exactly what
    `align_corners=False` inverts. `align_corners=True` would instead place
    samples AT the corners — a half-voxel shift plus an N/(N-1) scale error.
    That misregistration is invisible in a rendered image but would show up as
    a systematic sub-voxel offset against FDK. Pinned by
    `tests/test_voxel_grid.py::test_voxel_centres_map_to_their_own_value`.

    PADDING_MODE='zeros': outside the domain is air. The renderer already
    clips rays to the domain, so this only governs samples landing within a
    half-voxel of the boundary.

    Non-negativity is enforced as a PROJECTION (`clamp_nonneg()` after each
    optimizer step), not a softplus head. This matches SIRT's
    `res.clip(min=0)`, and avoids the failure mode a per-voxel softplus has
    but a shared MLP does not: air voxels driven to large negative
    pre-activations get vanishing gradients and go permanently dead.
    """

    def __init__(self, shape_zyx, out_channels: int = 1,
                 init_density: float = 0.02, padding_mode: str = "zeros"):
        super().__init__()
        D, H, W = (int(s) for s in shape_zyx)
        if min(D, H, W) < 2:
            raise ValueError(f"VoxelGrid needs >=2 voxels per axis, got {(D, H, W)}")
        self.out_channels = int(out_channels)
        self.padding_mode = str(padding_mode)
        if init_density < 0:
            raise ValueError(f"init_density must be >= 0, got {init_density}")
        per_ch = float(init_density) / self.out_channels
        self.mu = nn.Parameter(
            torch.full((1, self.out_channels, D, H, W), per_ch, dtype=torch.float32))

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.mu.shape[2:])

    def forward(self, xyz: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        # `extra` (polychromatic conditioning) is accepted for signature
        # compatibility with CoordMLP and ignored: a voxel grid has no shared
        # trunk to condition, every voxel is already free.
        grid = xyz.reshape(1, -1, 1, 1, 3).to(self.mu.dtype)
        out = F.grid_sample(self.mu, grid, mode="bilinear",
                            align_corners=False, padding_mode=self.padding_mode)
        out = out.reshape(self.out_channels, -1).transpose(0, 1)   # (M, C)
        if self.out_channels == 1:
            return out.squeeze(-1)
        return out

    @torch.no_grad()
    def clamp_nonneg(self) -> None:
        """SIRT's non-negativity projection. Call AFTER `optimizer.step()`."""
        self.mu.clamp_(min=0.0)

    @torch.no_grad()
    def load_volume(self, vol_zyx: torch.Tensor) -> None:
        """Initialise from a volume in (Z, Y, X) order, resampled to the grid.

        `vol_zyx` is assumed to span the SAME AABB as this grid. Used to warm
        start from FDK, which for a voxel grid replaces the entire
        `fdk_pretrain` stage: there is no function to fit, so the 100k-iteration
        coordinate->density regression (and the hash-collision scramble at the
        pretrain->main handoff it causes) collapses to this one assignment.
        """
        v = vol_zyx.to(self.mu.device, self.mu.dtype)
        if v.ndim == 3:
            v = v.unsqueeze(0).unsqueeze(0)            # (1,1,Z,Y,X)
        elif v.ndim == 4:
            v = v.unsqueeze(0)                          # (1,C,Z,Y,X)
        if v.shape[1] == 1 and self.out_channels > 1:
            v = v.expand(-1, self.out_channels, -1, -1, -1) / self.out_channels
        if tuple(v.shape[2:]) != self.shape_zyx:
            v = F.interpolate(v, size=self.shape_zyx, mode="trilinear",
                              align_corners=False)
        self.mu.copy_(v)
