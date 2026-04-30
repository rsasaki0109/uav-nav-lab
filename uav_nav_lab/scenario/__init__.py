from .base import SCENARIO_REGISTRY, Scenario
from . import grid_world, voxel_world  # noqa: F401  (registers backends)

__all__ = ["SCENARIO_REGISTRY", "Scenario"]
