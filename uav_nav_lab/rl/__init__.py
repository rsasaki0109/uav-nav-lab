"""RL comparison baseline — gym.Env wrappers around the framework's scenarios.

Provides `VoxelNavEnv` (voxel_world) and `GridNavEnv` (grid_world) for
training stable-baselines3 agents against the same obstacle fields used
by the plan-based (MPC/MPPI/A*) planners.
"""

from .env import VoxelNavEnv, GridNavEnv

__all__ = ["VoxelNavEnv", "GridNavEnv"]
