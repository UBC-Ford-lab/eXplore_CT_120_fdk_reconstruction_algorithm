"""Dense voxel-grid learning-based reconstruction (SIRT's representation)."""

from .model import VoxelGrid, voxel_grid_shape
from .reconstructor import VoxelReconstructor

__all__ = ["VoxelGrid", "voxel_grid_shape", "VoxelReconstructor"]
