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
        sim_extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "t": float(t),
            "true_pos": [float(v) for v in true_pos],
            "true_vel": [float(v) for v in true_vel],
            "observed_pos": [float(v) for v in observed_pos],
            "cmd": [float(v) for v in cmd],
            "collision": bool(info.get("collision", False)),
            "goal_reached": bool(info.get("goal_reached", False)),
        }
        # Surface a small summary of sim-side sensor side-channels so they
        # show up in the episode JSON. Currently: lidar point counts per
        # configured sensor name. Full point clouds stay in memory only —
        # writing them per step would balloon JSON sizes (e.g. a 16-beam
        # lidar at 10 Hz → ~10⁴ floats × 1500 steps).
        if sim_extra:
            lidar_pts = sim_extra.get("lidar_points")
            if isinstance(lidar_pts, dict) and lidar_pts:
                row["lidar_points"] = {
                    str(name): int(np.asarray(pc).shape[0])
                    for name, pc in lidar_pts.items()
                }
        self.steps.append(row)

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
