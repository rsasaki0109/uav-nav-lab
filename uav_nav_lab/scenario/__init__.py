from .base import SCENARIO_REGISTRY, Scenario
from . import grid_world  # noqa: F401  (registers backend)

__all__ = ["SCENARIO_REGISTRY", "Scenario"]
