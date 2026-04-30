"""Episode log recorder.

One JSON file per episode keeps things easy to inspect, diff, and re-process
later. For larger studies we can swap in a binary backend without touching the
runner — only the recorder API surface matters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeRecorder:
    def __init__(self, episode_index: int, seed: int) -> None:
        self.meta = {"episode": episode_index, "seed": seed}
        self.steps: list[dict[str, Any]] = []
        self.replans: list[dict[str, Any]] = []
        self.outcome: str = "unknown"
        self.summary: dict[str, Any] = {}

    def log_step(
        self,
        t: float,
        true_pos: np.ndarray,
        true_vel: np.ndarray,
        observed_pos: np.ndarray,
        cmd: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        self.steps.append(
            {
                "t": float(t),
                "true_pos": [float(v) for v in true_pos],
                "true_vel": [float(v) for v in true_vel],
                "observed_pos": [float(v) for v in observed_pos],
                "cmd": [float(v) for v in cmd],
                "collision": bool(info.get("collision", False)),
                "goal_reached": bool(info.get("goal_reached", False)),
            }
        )

    def log_replan(self, t: float, plan_length: int, planner_dt_ms: float) -> None:
        self.replans.append(
            {"t": float(t), "plan_length": int(plan_length), "planner_dt_ms": float(planner_dt_ms)}
        )

    def set_outcome(self, outcome: str, **extra: Any) -> None:
        self.outcome = outcome
        self.summary.update(extra)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "outcome": self.outcome,
            "summary": self.summary,
            "replans": self.replans,
            "steps": self.steps,
        }

    def save(self, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
