"""Render a trained representation onto the export grid — whole volume or one plane.

A learned reconstruction is a FUNCTION, not an array: the volume only exists
once the model is evaluated at voxel centres. That evaluation is the same
arithmetic whether it is a finished export, a mid-training diagnostic or a
figure, and it is independent of what the representation is — a hash-grid INR,
a dense voxel grid and a plain MLP all satisfy the same contract
(``(M, 3) in [-1,1]^3 -> (M,)`` or ``(M, C)`` for a multi-channel head).

So it lives here, once. ``LearnedReconstructor.export_volume`` calls
``render_volume``; anything that needs a single plane out of a model — rather
than out of a finished array — calls ``render_slice``.

WHY A SLICE RENDERER EXISTS AT ALL. ``ct_core.wandb_logging.midplane_views``
also produces axial/coronal/sagittal images, but it INDEXES a finished array.
That is the right tool once a volume is in hand and the wrong one during
training: a per-evaluation diagnostic that wants one plane out of a 420 M-voxel
model would otherwise have to render all 420 M voxels to throw away every one
but a plane of them. ``render_slice`` evaluates only the plane.

COORDINATES. Voxel centres come from the geometry's export grid
(``vol_shape`` / ``vol_origin`` / ``dx`` / ``dz``), in mm; the model is queried
in the MODEL DOMAIN's normalised coordinates via ``domain.normalize``. The two
are not the same body — the domain is usually the larger one (it has to cover
whatever the rays cross, bed included) while the export grid is the ROI being
shipped — which is exactly why the domain has to be passed in rather than
inferred from the geometry.

AXIS ORDER is the package's ``(Nx, Ny, Nz)`` throughout, matching
``crop_to_export_roi``, ``postprocess_and_save`` and ``midplane_views``.
Callers wanting the ``(Nz, Ny, Nx)`` of a stacked image transpose explicitly at
the call site, so the conversion is visible where it happens.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .training import unwrap_model

SLICE_DIRECTIONS = ("axial", "coronal", "sagittal")

# ~4 M queries per model call: large enough that the launch overhead is
# amortised, small enough that the coordinate buffer stays well inside a
# single GPU allocation.
EXPORT_CHUNK = 1 << 22


def query_mu(model, xyz_norm: torch.Tensor) -> torch.Tensor:
    """Model output as mu at the reference energy.

    A polychromatic head returns ``(M, 2) = (a1, a2)``; the bases are
    normalised to 1 at ``e_ref``, so ``mu(e_ref) = a1 + a2``. A monochromatic
    model returns ``(M,)`` and passes through.
    """
    out = model(xyz_norm)
    return out.sum(dim=-1) if out.ndim == 2 else out


def grid_coords(geometry: dict):
    """Voxel-centre coordinates (x, y, z) of the export grid, in mm."""
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    ox, oy, oz = (float(v) for v in geometry["vol_origin"])
    dx, dz = float(geometry["dx"]), float(geometry["dz"])
    x = (torch.arange(Nx, dtype=torch.float32) - (Nx - 1) / 2.0) * dx + ox
    y = (torch.arange(Ny, dtype=torch.float32) - (Ny - 1) / 2.0) * dx + oy
    z = (torch.arange(Nz, dtype=torch.float32) - (Nz - 1) / 2.0) * dz + oz
    return x, y, z


def _resolve(scene_or_geometry, domain):
    """Accept either a ``Scene`` or an explicit ``(geometry, domain)`` pair.

    The training side already holds a Scene and the driver side holds the two
    pieces separately; taking both spares every caller an unpacking line.
    """
    if domain is None:
        return scene_or_geometry.geometry, scene_or_geometry.model_domain
    return scene_or_geometry, domain


class _EvalMode:
    """Evaluate with the module in eval mode, restoring what it was.

    Rendering is a read: a diagnostic that silently left the model in eval
    mode would change the next training step.
    """

    def __init__(self, model):
        self._m = unwrap_model(model)

    def __enter__(self):
        self._was_training = self._m.training
        self._m.eval()
        return self._m

    def __exit__(self, *exc):
        if self._was_training:
            self._m.train()
        return False


@torch.no_grad()
def render_volume(model, scene_or_geometry, domain=None, *,
                  device="cpu", chunk: int = EXPORT_CHUNK,
                  mask: bool = True, progress: bool = False) -> np.ndarray:
    """Evaluate `model` at every export voxel centre. Returns (Nx, Ny, Nz) mu.

    Correct for any representation, including one whose own grid has a
    different pitch or a larger domain than the export ROI — the usual case
    once the domain hook is in use. A backend whose parameters ARE the export
    grid should return them directly instead; resampling a grid onto itself
    only costs sharpness.

    `mask` zeroes voxels outside the model domain's cylinder. The model is
    undefined there — ``normalize`` maps them outside [-1, 1] and nothing ever
    trained them — so the default is on. It is a no-op for a box domain, and
    for any export ROI that sits wholly inside the domain.
    """
    geometry, domain = _resolve(scene_or_geometry, domain)
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    x, y, z = grid_coords(geometry)
    device = torch.device(device)
    model = model.to(device)

    out = torch.empty((Nx, Ny, Nz), dtype=torch.float32)
    # Slab along z so a chunk is a whole number of xy planes: contiguous in
    # the output and cheap to index.
    per_plane = Nx * Ny
    n_planes = max(1, min(Nz, chunk // max(1, per_plane)))
    gx, gy = torch.meshgrid(x, y, indexing="ij")

    slabs = range(0, Nz, n_planes)
    if progress:
        slabs = tqdm(slabs, desc="rendering", total=(Nz + n_planes - 1) // n_planes)

    with _EvalMode(model):
        for z0 in slabs:
            z1 = min(z0 + n_planes, Nz)
            pts = torch.stack([
                gx[..., None].expand(-1, -1, z1 - z0),
                gy[..., None].expand(-1, -1, z1 - z0),
                z[z0:z1].view(1, 1, -1).expand(Nx, Ny, -1),
            ], dim=-1).reshape(-1, 3).to(device)
            vals = query_mu(model, domain.normalize(pts)).float()
            if mask:
                vals = vals * domain.cylinder_mask(pts)
            out[:, :, z0:z1] = vals.reshape(Nx, Ny, z1 - z0).cpu()

    return np.ascontiguousarray(out.numpy(), dtype=np.float32)


@torch.no_grad()
def render_slice(model, scene_or_geometry, domain=None,
                 direction: str = "axial", slice_idx: int | None = None,
                 chunk: int = 65536, device="cpu",
                 mask: bool = True) -> np.ndarray:
    """One plane through the export grid, evaluated directly from the model.

    `slice_idx` indexes along the plane's normal axis and defaults to the
    middle of that axis. Returned shapes, chosen so the first axis is the one
    drawn downwards in a radiological view:

      * axial    (normal z) -> (Ny, Nx)
      * coronal  (normal y) -> (Nz, Nx)
      * sagittal (normal x) -> (Nz, Ny)
    """
    if direction not in SLICE_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {SLICE_DIRECTIONS}, got {direction!r}")

    geometry, domain = _resolve(scene_or_geometry, domain)
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    x, y, z = grid_coords(geometry)

    n_axis, axis_name = {"axial": (Nz, "axial"),
                         "coronal": (Ny, "coronal"),
                         "sagittal": (Nx, "sagittal")}[direction]
    if slice_idx is None:
        slice_idx = n_axis // 2
    if not 0 <= slice_idx < n_axis:
        raise IndexError(f"{axis_name} slice_idx must be in [0, {n_axis}), "
                         f"got {slice_idx}")

    if direction == "axial":
        rows, cols = Ny, Nx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        const = torch.full((rows * cols,), z[slice_idx].item())
        pts = torch.stack([xx.flatten(), yy.flatten(), const], dim=-1)
    elif direction == "coronal":
        rows, cols = Nz, Nx
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        const = torch.full((rows * cols,), y[slice_idx].item())
        pts = torch.stack([xx.flatten(), const, zz.flatten()], dim=-1)
    else:  # sagittal
        rows, cols = Nz, Ny
        zz, yy = torch.meshgrid(z, y, indexing="ij")
        const = torch.full((rows * cols,), x[slice_idx].item())
        pts = torch.stack([const, yy.flatten(), zz.flatten()], dim=-1)

    device = torch.device(device)
    model = model.to(device)
    pts = pts.to(device)
    xyz_norm = domain.normalize(pts)

    out = torch.empty(pts.shape[0], dtype=torch.float32, device=device)
    with _EvalMode(model):
        for i in range(0, pts.shape[0], chunk):
            out[i:i + chunk] = query_mu(model, xyz_norm[i:i + chunk]).float()
    if mask:
        out = out * domain.cylinder_mask(pts)

    return out.reshape(rows, cols).cpu().numpy()
