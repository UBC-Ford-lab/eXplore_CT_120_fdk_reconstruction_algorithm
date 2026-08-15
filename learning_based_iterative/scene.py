"""Scene containers for learning-based (differentiable) reconstruction.

A `Scene` bundles four things:
  * preprocessed line integrals  (sinogram, after flat-field/log/ring)
  * cone-beam projection angles  (radians)
  * cone-beam geometry           (ct_core's plain dict; defines the EXPORT ROI)
  * `model_domain`               (the integration domain — usually larger than
                                  the export ROI to absorb out-of-FOV matter)

`scene.aabb_min/max` route to `model_domain` because that is what the renderer
integrates over. The geometry-derived `scene.export_aabb_min/max` is the box
that export crops the saved volume down to.

Coordinates are isocenter-centered millimetres throughout. The mm → [-1,1]
affine that the model sees uses the model domain, not the export ROI.

Moved here (verbatim) from muNeRF's ``inr_pipeline/dataset.py`` 2026-08-12 so
every learning-based algorithm — inside this submodule and in muNeRF — shares
one definition. muNeRF's ``inr_pipeline.dataset`` re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelDomain:
    """Integration domain for the renderer.

    The AABB bounds the domain. For shape='cylinder' the domain is the
    inscribed cylinder (radius_xy = extent_xy / 2), which lets the model cover
    bed/external matter while still telling the optimizer "outside the cylinder
    is air" — the AABB corners never reach the loss.

    ``ray_clip`` selects which body the renderer lays its quadrature points
    across (see ``renderer.ray_domain_intersect``):

      * ``'domain'`` (default) — the cylinder itself. Every sample lands on
        real domain, and the step is ``cylinder_chord / num_samples``.
      * ``'aabb'`` (legacy) — the bounding box, with out-of-cylinder samples
        evaluated and then multiplied by zero. MEASURED on Scan_1510's
        geometry: 19.5% of all samples are discarded this way (1 - pi/4 for a
        disc inscribed in a square, less the rays the z-slab clips first), and
        up to 71% on rays grazing the domain boundary. Both settings integrate
        the same function — the corners hold no material by definition — so
        this only trades compute and quadrature resolution, not correctness.
        Kept to reproduce runs from before 2026-08-09.

    Note the saving is concentrated in the PERIPHERY: a ray through the axis
    sees cylinder_chord ~= box_chord and gains nothing. On Scan_1510 the mouse
    sits at |b| < 13 mm, where the step improves by only 1.01x; the win there
    is throughput, not resolution.

    The AABB may be centered off-isocenter via the ``origin`` config key,
    which shifts the domain to better cover an off-center ROI (reducing
    wasted hash-grid capacity on empty space).
    """

    shape: str                       # 'cylinder' | 'box'
    aabb_min: torch.Tensor           # (3,) mm, float32
    aabb_max: torch.Tensor           # (3,) mm, float32
    radius_xy: float | None = None   # only meaningful for shape='cylinder'
    center_xy: tuple[float, float] = (0.0, 0.0)  # cylinder axis center in mm
    ray_clip: str = "domain"         # 'domain' | 'aabb' (legacy)

    def cylinder_mask(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        """Float (..., ) mask. 1 inside the cylinder (or always 1 for box)."""
        if self.shape == "box" or self.radius_xy is None:
            return torch.ones(
                xyz_mm.shape[:-1], dtype=xyz_mm.dtype, device=xyz_mm.device
            )
        dx = xyz_mm[..., 0] - self.center_xy[0]
        dy = xyz_mm[..., 1] - self.center_xy[1]
        r2 = dx ** 2 + dy ** 2
        return (r2 < self.radius_xy ** 2).to(xyz_mm.dtype)

    def normalize(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        amin = self.aabb_min.to(xyz_mm)
        amax = self.aabb_max.to(xyz_mm)
        center = (amin + amax) * 0.5
        half = (amax - amin) * 0.5
        return (xyz_mm - center) / half


@dataclass
class Scene:
    sinogram: torch.Tensor       # (N_angles, N_b, N_a) line integrals, float32
    angles: torch.Tensor         # (N_angles,) radians, float32
    geometry: dict[str, Any]     # ct_core build_geometry output (export ROI)
    scan_name: str
    model_domain: ModelDomain    # REQUIRED — no silent fallback to export ROI.

    @property
    def n_angles(self) -> int:
        return int(self.sinogram.shape[0])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return int(self.sinogram.shape[1]), int(self.sinogram.shape[2])

    @property
    def export_aabb_min(self) -> torch.Tensor:
        ox, oy, oz = self.geometry["vol_origin"]
        Nx, Ny, Nz = self.geometry["vol_shape"]
        dx, dz = self.geometry["dx"], self.geometry["dz"]
        return torch.tensor(
            [ox - Nx * dx / 2.0, oy - Ny * dx / 2.0, oz - Nz * dz / 2.0],
            dtype=torch.float32,
        )

    @property
    def export_aabb_max(self) -> torch.Tensor:
        ox, oy, oz = self.geometry["vol_origin"]
        Nx, Ny, Nz = self.geometry["vol_shape"]
        dx, dz = self.geometry["dx"], self.geometry["dz"]
        return torch.tensor(
            [ox + Nx * dx / 2.0, oy + Ny * dx / 2.0, oz + Nz * dz / 2.0],
            dtype=torch.float32,
        )

    # Renderer-facing AABB == the integration domain.
    @property
    def aabb_min(self) -> torch.Tensor:
        return self.model_domain.aabb_min

    @property
    def aabb_max(self) -> torch.Tensor:
        return self.model_domain.aabb_max


def normalize_to_unit_cube(xyz_mm: torch.Tensor, scene: Scene) -> torch.Tensor:
    """Map mm-space points (..., 3) into [-1, 1]^3 spanning the model domain.

    Per-axis affine — extents along x/y and z are not equal in general. Points
    outside the model domain return values outside [-1, 1]; the renderer
    handles masking via the AABB clip and (for cylinder domains) the cylinder
    mask.
    """
    return scene.model_domain.normalize(xyz_mm)


def model_domain_from_geometry(geometry: dict, *, shape: str = "cylinder",
                               ray_clip: str = "domain") -> ModelDomain:
    """A ModelDomain covering exactly the export ROI of a ct_core geometry.

    The convenience constructor for submodule backends, where the scan's
    reconstruction FOV (``fov_xy``/``fov_z`` via ``build_geometry``) IS the
    integration domain. muNeRF configs instead build enlarged domains that
    absorb out-of-FOV matter (scan bed) — that logic stays in muNeRF's
    ``dataset._build_model_domain``.
    """
    ox, oy, oz = (float(v) for v in geometry["vol_origin"])
    Nx, Ny, Nz = (int(v) for v in geometry["vol_shape"])
    dx, dz = float(geometry["dx"]), float(geometry["dz"])
    hx, hy, hz = Nx * dx / 2.0, Ny * dx / 2.0, Nz * dz / 2.0
    aabb_min = torch.tensor([ox - hx, oy - hy, oz - hz], dtype=torch.float32)
    aabb_max = torch.tensor([ox + hx, oy + hy, oz + hz], dtype=torch.float32)
    radius = min(hx, hy) if shape == "cylinder" else None
    return ModelDomain(shape=shape, aabb_min=aabb_min, aabb_max=aabb_max,
                       radius_xy=radius, center_xy=(ox, oy), ray_clip=ray_clip)
