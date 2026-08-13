"""Learning-based iterative reconstruction.

Reconstruction as optimization: a differentiable representation of the volume
(voxel grid, INR, ...) is fitted to the measured line integrals by gradient
descent through a differentiable forward projector. Classical iterative recon
solves the same argmin — this family swaps the hand-derived update rule for
autograd, which is what makes representations exchangeable.

This package holds the CANONICAL copies of the machinery every such algorithm
shares — muNeRF's ``inr_pipeline`` imports from here rather than duplicating:

* ``scene``          — Scene / ModelDomain containers, mm → [-1,1]^3 affine
* ``ray_sampler``    — cone-beam ray generation (the geometry convention)
* ``renderer``       — differentiable line-integral forward operator
* ``detector_warp``  — per-pixel detector distortion applied to ray geometry

Algorithms live in their own subfolder:

* ``voxel``          — dense voxel grid (SIRT's representation) + trainer.
                       Future siblings: nerf/, hashgrid/, gaussian_splatting/.
"""

from .scene import (
    ModelDomain,
    Scene,
    model_domain_from_geometry,
    normalize_to_unit_cube,
)
from .ray_sampler import (
    rays_for_projection,
    rays_from_indices,
    sample_projection_patch,
    sample_random_rays,
    sample_random_rows,
)
from .renderer import (
    COMPILE_MODES,
    ray_aabb_intersect,
    ray_cylinder_intersect,
    ray_domain_intersect,
    render_compile_mode,
    render_rays,
    render_rays_hierarchical,
    scale_grad,
    set_render_compile,
)
from .detector_warp import (
    DetectorWarp,
    detector_serial_from_scan,
    resolve_detector_warp,
)
from .voxel.model import VoxelGrid, voxel_grid_shape

__all__ = [
    "ModelDomain", "Scene", "model_domain_from_geometry",
    "normalize_to_unit_cube",
    "rays_for_projection", "rays_from_indices", "sample_projection_patch",
    "sample_random_rays", "sample_random_rows",
    "ray_aabb_intersect", "ray_cylinder_intersect", "ray_domain_intersect",
    "render_rays", "render_rays_hierarchical", "scale_grad",
    "COMPILE_MODES", "render_compile_mode", "set_render_compile",
    "DetectorWarp", "detector_serial_from_scan", "resolve_detector_warp",
    "VoxelGrid", "voxel_grid_shape",
]
