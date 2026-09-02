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
* ``training``       — precision, per-group LRs, model compile, grad clipping
* ``trainer``        — the optimization LOOP, independent of the representation

Algorithms live in their own subfolder, and each is only the three answers
``LearnedReconstructor`` cannot guess — what the model is, what the
integration domain is, and how a volume comes out:

* ``voxel``          — dense voxel grid (SIRT's representation).
                       Future siblings: nerf/, hashgrid/, gaussian_splatting/.

Each also ships a ``LearnedAlgorithm`` descriptor (``registry``) carrying the
three things the DRIVER cannot guess either — the flags it adds to the CLI,
the constructor arguments it takes, and how much of the machine it needs — and
registers it below. That is what ``--algorithm`` selects over, so adding a
representation touches this package and nothing else: not the driver, not
``ct_core``.
"""

from .scene import (
    DOMAIN_SPECS,
    ModelDomain,
    Scene,
    model_domain_from_bounds,
    model_domain_from_geometry,
    model_domain_from_spec,
    normalize_to_unit_cube,
    resolve_export_grid,
    resolve_export_roi,
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
    fusion_supported,
)
from .detector_warp import (
    DetectorWarp,
    detector_serial_from_scan,
    resolve_detector_warp,
)
from .training import (OPTIMIZERS, AdamBF16, autocast_ctx, build_optimizer,
                       build_param_groups, clip_grad_norm,
                       maybe_compile_model, project_nonneg,
                       resolve_amp_dtype, unwrap_model)
from .trainer import LearnedReconstructor
# Short names inside `registry` (get/names/describe read well qualified);
# unambiguous ones out here, where they sit beside Scene and render_rays.
from .registry import LearnedAlgorithm, algorithms
from .registry import describe as describe_algorithms
from .registry import get as get_algorithm
from .registry import names as algorithm_names
from .registry import register as register_algorithm
from .voxel.model import VoxelGrid, voxel_grid_shape
from .voxel.reconstructor import VoxelReconstructor
from .voxel import ALGORITHM as VOXEL_ALGORITHM

# ---------------------------------------------------------- registration --
# Importing this package is what makes an algorithm selectable, and every
# entry point already imports it. Registration is a call rather than a
# decorator so the set is visible in one place; a third-party representation
# calls `register_algorithm` from its own module instead.
register_algorithm(VOXEL_ALGORITHM)

__all__ = [
    "DOMAIN_SPECS", "ModelDomain", "Scene", "model_domain_from_bounds",
    "model_domain_from_geometry", "model_domain_from_spec",
    "normalize_to_unit_cube",
    "rays_for_projection", "rays_from_indices", "sample_projection_patch",
    "sample_random_rays", "sample_random_rows",
    "ray_aabb_intersect", "ray_cylinder_intersect", "ray_domain_intersect",
    "render_rays", "render_rays_hierarchical", "scale_grad",
    "COMPILE_MODES", "render_compile_mode", "set_render_compile",
    "fusion_supported",
    "DetectorWarp", "detector_serial_from_scan", "resolve_detector_warp",
    "OPTIMIZERS", "AdamBF16",
    "autocast_ctx", "build_optimizer", "build_param_groups",
    "clip_grad_norm", "maybe_compile_model", "project_nonneg",
    "resolve_amp_dtype", "unwrap_model",
    "LearnedReconstructor",
    "LearnedAlgorithm", "register_algorithm", "get_algorithm",
    "algorithm_names", "algorithms", "describe_algorithms",
    "VoxelGrid", "voxel_grid_shape", "VoxelReconstructor", "VOXEL_ALGORITHM",
    "resolve_export_grid", "resolve_export_roi",
]
