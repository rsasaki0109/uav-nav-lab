"""Dynamic-obstacle predictor interface.

A predictor takes the *currently observed* dynamic obstacles (with positions
and velocities) and returns the planner's *belief* about where they will be
at a list of future time offsets. This is what the planner uses to score
trajectory candidates against moving threats.

By splitting this from the MPC body we get a clean ablation lever: swap a
constant-velocity predictor for a noisy / Kalman / learned one and re-run.

Interface
---------
    predictor.predict(dynamic_obstacles, horizon_dts) -> np.ndarray
        shape: [n_obstacles, len(horizon_dts), ndim]

`horizon_dts` is the array of time offsets *from the planning instant*
(e.g. [dt, 2*dt, ..., H*dt] for a sampling MPC with `horizon` rollout steps).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from ..registry import Registry


class Predictor(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Predictor": ...

    def reset(self, *, seed: int | None = None) -> None:
        """Reset any internal RNG / filter state. Default: no-op."""

    @abstractmethod
    def predict(
        self,
        dynamic_obstacles: list[dict],
        horizon_dts: np.ndarray,
    ) -> np.ndarray:
        """Return [n_obstacles, len(horizon_dts), ndim] predicted positions."""


PREDICTOR_REGISTRY: Registry[Predictor] = Registry("predictor")


def build_predictor(cfg: Mapping[str, Any] | None) -> Predictor:
    """Build a predictor from a config dict, defaulting to constant_velocity.

    A missing or empty config gives the historical MPC behavior, so existing
    YAMLs continue to work without an explicit `predictor:` block.
    """
    if not cfg:
        cfg = {"type": "constant_velocity"}
    name = str(cfg.get("type", "constant_velocity"))
    return PREDICTOR_REGISTRY.get(name).from_config(cfg)
