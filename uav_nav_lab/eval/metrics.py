"""Episode-level and run-level metrics.

Per-episode metrics:
  - outcome (success / collision / timeout)
  - final_t
  - path_length
  - average speed
  - replanning count
  - ATE: RMS of (observed - true) position. With a delayed/noisy sensor this
    is what the planner had to work with vs. ground truth — i.e. perception error.

Run-level metrics aggregate across episodes (mean, std, success rate, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize_episode(ep: dict[str, Any]) -> dict[str, Any]:
    steps = ep.get("steps", [])
    replans = ep.get("replans", [])
    if not steps:
        return {
            "outcome": ep.get("outcome", "unknown"),
            "final_t": 0.0,
            "path_length": 0.0,
            "avg_speed": 0.0,
            "replans": len(replans),
            "ate_rms": 0.0,
        }
    true_pos = np.asarray([s["true_pos"] for s in steps], dtype=float)
    obs_pos = np.asarray([s["observed_pos"] for s in steps], dtype=float)
    deltas = np.diff(true_pos, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    path_length = float(seg_lengths.sum())
    final_t = float(steps[-1]["t"])
    avg_speed = path_length / final_t if final_t > 1e-9 else 0.0
    ate_rms = float(np.sqrt(np.mean(np.sum((obs_pos - true_pos) ** 2, axis=1))))
    return {
        "outcome": ep.get("outcome", "unknown"),
        "final_t": final_t,
        "path_length": path_length,
        "avg_speed": avg_speed,
        "replans": len(replans),
        "ate_rms": ate_rms,
    }


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    episode_files = sorted(run_dir.glob("episode_*.json"))
    if not episode_files:
        raise FileNotFoundError(f"no episode_*.json under {run_dir}")
    per_ep = []
    for ef in episode_files:
        with ef.open("r", encoding="utf-8") as f:
            ep = json.load(f)
        per_ep.append(summarize_episode(ep))

    n = len(per_ep)
    successes = sum(1 for e in per_ep if e["outcome"] == "success")
    collisions = sum(1 for e in per_ep if e["outcome"] == "collision")
    timeouts = sum(1 for e in per_ep if e["outcome"] == "timeout")

    def _agg(key: str) -> dict[str, float]:
        vals = np.asarray([e[key] for e in per_ep], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max())}

    summary = {
        "run_dir": str(run_dir),
        "n_episodes": n,
        "success_rate": successes / n,
        "collision_rate": collisions / n,
        "timeout_rate": timeouts / n,
        "final_t": _agg("final_t"),
        "path_length": _agg("path_length"),
        "avg_speed": _agg("avg_speed"),
        "replans": _agg("replans"),
        "ate_rms": _agg("ate_rms"),
        "episodes": per_ep,
    }
    out_path = run_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def format_summary_text(summary: dict[str, Any]) -> str:
    n = summary["n_episodes"]
    lines = [
        f"run: {summary['run_dir']}",
        f"  episodes:       {n}",
        f"  success rate:   {summary['success_rate']:.2%}",
        f"  collision rate: {summary['collision_rate']:.2%}",
        f"  timeout rate:   {summary['timeout_rate']:.2%}",
        f"  avg speed:      {summary['avg_speed']['mean']:.2f} m/s "
        f"(±{summary['avg_speed']['std']:.2f})",
        f"  path length:    {summary['path_length']['mean']:.2f} m "
        f"(±{summary['path_length']['std']:.2f})",
        f"  replans/ep:     {summary['replans']['mean']:.2f}",
        f"  ATE (rms):      {summary['ate_rms']['mean']:.3f} m",
    ]
    return "\n".join(lines)
