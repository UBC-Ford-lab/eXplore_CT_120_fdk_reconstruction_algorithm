"""Learning-based iterative reconstruction with a dense voxel grid.

Reconstruction as optimization: the volume is a `VoxelGrid` (one free
parameter per voxel — SIRT's representation) fitted to the measured line
integrals by Adam through the differentiable renderer. It follows the
ct_core backend contract, so it is a drop-in peer of the FDK / ASTRA / TIGRE
backends:

  consume  raw-count projections (N_angles, N_b, N_a) + angles (radians, FDK
           convention) + ct_core ``build_geometry`` dict
  return   float32 (Nx, Ny, Nz) volume of linear attenuation mu (mm^-1)
           via ``reconstructed_volume`` -- NOT HU; calibration is a single
           downstream step shared by every backend (ct_core.hu_calibration)

THE LOOP ITSELF LIVES IN ``..trainer.LearnedReconstructor``. This class is
what makes that loop a voxel-grid backend, and it is exactly three answers:

  * the model is a ``VoxelGrid`` over the export grid;
  * the integration domain IS the export FOV (this backend is compared
    against FDK/SIRT on the same grid, and has no bed to absorb);
  * the parameters ARE the volume, so export is a transpose, not a render.

Read it as the reference implementation of the three hooks — a second
representation is the same three answers with different content.

Recipe notes carried over from the validated config:
  * plain MSE data term (SIRT's objective; no structural terms)
  * non-negativity as a PROJECTION after each step (SIRT's clip, not softplus)
  * init near air, not water — gradient pressure raises mu only where the
    projections demand it
  * short LR warmup (500) + cosine decay; the optimum sits EARLY (~16k steps
    on Scan_1510), so long schedules mostly fit noise
  * quadrature finer than the voxel (~0.55 voxel steps) to avoid aliasing
    against a critically-sampled grid
  * one held-out projection is excluded from training and scored for
    early stopping — the same SIRT-style holdout the TIGRE backend uses

Like the classical backends, the reconstruction FOV must cover the object
(truncated projections are unmodeled); the model domain here IS the export
FOV cylinder.
"""

from __future__ import annotations

import numpy as np

from ..trainer import LearnedReconstructor
from .model import VoxelGrid


class VoxelReconstructor(LearnedReconstructor):
    """Voxel-grid learning-based iterative reconstruction (backend contract)."""

    def __init__(self, projections, angles, geometry, *,
                 init_density: float = 0.001,
                 lr: float = 1e-4,
                 **kwargs):
        # `init_density` is the one constructor argument that is specific to
        # this representation: a network's output scale is set by its head, but
        # here the parameter IS mu and its starting value is a modelling
        # choice. Near AIR, not water — outside the object that is already the
        # right answer, so gradient pressure only has to raise mu where the
        # projections demand it rather than walk every voxel down to zero.
        self.init_density = float(init_density)
        super().__init__(projections, angles, geometry, lr=lr, **kwargs)

    # ------------------------------------------------- representation hooks --
    # build_domain() is inherited: the default (integration domain == export
    # FOV) is the right one here and saying so by omission is the point.

    def build_model(self, domain, device):
        """A `VoxelGrid` tiling the export grid exactly.

        Shape is (Dz, Hy, Wx) — the storage order `F.grid_sample` wants — from
        the geometry's own (Nx, Ny, Nz), so the parameter grid and the exported
        volume are the same voxels and no resampling ever happens between them.
        """
        Nx, Ny, Nz = (int(v) for v in self.geometry["vol_shape"])
        print(f"  Voxel grid: {Nx} x {Ny} x {Nz} (x,y,z), init "
              f"{self.init_density:g} mm^-1")
        return VoxelGrid((Nz, Ny, Nx), init_density=self.init_density)

    def export_volume(self, model, domain, device) -> np.ndarray:
        """The parameter grid IS the volume — no render, no resampling.

        ``mu[0, 0]`` is (Dz, Hy, Wx) with indices increasing along +z/+y/+x;
        transpose to the FDK convention (Nx, Ny, Nz). Overriding the base
        class's evaluate-on-the-export-grid default matters here: that default
        would put a trilinear pass between the parameters and the output, which
        for a grid that already IS the export grid can only lose sharpness.
        """
        if self._export_fn is not None:
            return self._export_fn(model, self.geometry, domain, device)
        from ..training import unwrap_model
        mu = unwrap_model(model).mu.detach()[0, 0].cpu().numpy().transpose(2, 1, 0)
        return np.ascontiguousarray(mu, dtype=np.float32)
