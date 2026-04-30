"""Episode-level and run-level metrics.

Per-episode metrics:
  - outcome (success / collision / timeout)
  - final_t
  - path_length
  - average speed
  - replanning count
  - ATE: RMS of (observed - true) position. With a delayed/noisy sensor this
    is what the planner had to work with vs. ground truth — i.e. perception error.

Run-level metrics aggregate across episodes:
  - Binary outcome rates use the Wilson score interval (95%) — robust at
    small N and near boundaries (rates of 0% / 100%), unlike the normal
    approximation.
  - Continuous metrics get mean ± 1.96 · SEM (= std / sqrt(N)). Good enough
    for the N=3-50 range typical of replanning sweeps; falls back to a 0
    half-width when N=1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_Z95 = 1.959963984540054  # two-sided 95% normal quantile


def _wilson(successes: int, n: int, z: float = _Z95) -> tuple[float, float, float]:
    """95% Wilson score interval. Returns (point_estimate, lower, upper)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _continuous_ci(values: np.ndarray, z: float = _Z95) -> dict[str, float]:
    n = int(values.size)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "min": 0.0, "max": 0.0, "n": 0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    half = z * sem
    return {"mean": mean, "std": std, "sem": sem,
            "ci_lo": mean - half, "ci_hi": mean + half,
            "min": float(values.min()), "max": float(values.max()), "n": n}


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
    # Multi-drone runs write per-drone trajectory logs *and* a separate
    # `episode_<j>_joint.json` summary that captures per-drone outcomes for
    # joint-success aggregation. The trajectory glob below intentionally
    # excludes those joint files so we do not double-count them.
    episode_files = sorted(
        f for f in run_dir.glob("episode_*.json") if not f.name.endswith("_joint.json")
    )
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
        return _continuous_ci(vals)

    s_p, s_lo, s_hi = _wilson(successes, n)
    c_p, c_lo, c_hi = _wilson(collisions, n)
    t_p, t_lo, t_hi = _wilson(timeouts, n)
    summary = {
        "run_dir": str(run_dir),
        "n_episodes": n,
        "success_rate": s_p,
        "success_ci95": [s_lo, s_hi],
        "collision_rate": c_p,
        "collision_ci95": [c_lo, c_hi],
        "timeout_rate": t_p,
        "timeout_ci95": [t_lo, t_hi],
        "final_t": _agg("final_t"),
        "path_length": _agg("path_length"),
        "avg_speed": _agg("avg_speed"),
        "replans": _agg("replans"),
        "ate_rms": _agg("ate_rms"),
        "episodes": per_ep,
    }
    # Joint (multi-drone) aggregation: read the per-episode joint summaries
    # if present and append a `joint_*` block to the run summary. Per-drone
    # rates above are still useful, but the joint rate is the metric an
    # operator would actually report ("did *the mission* succeed?").
    joint_files = sorted(run_dir.glob("episode_*_joint.json"))
    if joint_files:
        joint_eps = []
        for jf in joint_files:
            with jf.open("r", encoding="utf-8") as f:
                joint_eps.append(json.load(f))
        n_j = len(joint_eps)
        j_succ = sum(1 for e in joint_eps if e["outcome"] == "success")
        j_coll = sum(1 for e in joint_eps if e["outcome"] == "collision")
        j_to = sum(1 for e in joint_eps if e["outcome"] == "timeout")
        sp, slo, shi = _wilson(j_succ, n_j)
        cp, clo, chi = _wilson(j_coll, n_j)
        tp, tlo, thi = _wilson(j_to, n_j)
        summary["joint_n_episodes"] = n_j
        summary["joint_n_drones"] = int(joint_eps[0]["meta"].get("n_drones", 0))
        summary["joint_success_rate"] = sp
        summary["joint_success_ci95"] = [slo, shi]
        summary["joint_collision_rate"] = cp
        summary["joint_collision_ci95"] = [clo, chi]
        summary["joint_timeout_rate"] = tp
        summary["joint_timeout_ci95"] = [tlo, thi]
        summary["joint_episodes"] = joint_eps

    out_path = run_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _fmt_rate(p: float, ci: list[float]) -> str:
    return f"{p:.1%}  [{ci[0]:.1%}, {ci[1]:.1%}]"


def _fmt_cont(stats: dict, fmt: str = ".2f", unit: str = "") -> str:
    half = (stats["ci_hi"] - stats["ci_lo"]) / 2.0
    return f"{stats['mean']:{fmt}}{unit} ± {half:{fmt}}{unit} (n={stats['n']})"


def format_summary_text(summary: dict[str, Any]) -> str:
    n = summary["n_episodes"]
    drone_label = "drone-episodes" if "joint_n_episodes" in summary else "episodes"
    lines = [
        f"run: {summary['run_dir']}",
        f"  {drone_label}:  {n}",
        f"  success rate:   {_fmt_rate(summary['success_rate'], summary['success_ci95'])}",
        f"  collision rate: {_fmt_rate(summary['collision_rate'], summary['collision_ci95'])}",
        f"  timeout rate:   {_fmt_rate(summary['timeout_rate'], summary['timeout_ci95'])}",
        f"  avg speed:      {_fmt_cont(summary['avg_speed'], unit=' m/s')}",
        f"  path length:    {_fmt_cont(summary['path_length'], unit=' m')}",
        f"  replans/ep:     {_fmt_cont(summary['replans'], fmt='.1f')}",
        f"  ATE (rms):      {_fmt_cont(summary['ate_rms'], fmt='.3f', unit=' m')}",
    ]
    if "joint_n_episodes" in summary:
        nj = summary["joint_n_episodes"]
        nd = summary.get("joint_n_drones", 0)
        lines += [
            "",
            f"  joint (n={nd} drones, {nj} episodes):",
            f"    success rate:   {_fmt_rate(summary['joint_success_rate'], summary['joint_success_ci95'])}",
            f"    collision rate: {_fmt_rate(summary['joint_collision_rate'], summary['joint_collision_ci95'])}",
            f"    timeout rate:   {_fmt_rate(summary['joint_timeout_rate'], summary['joint_timeout_ci95'])}",
        ]
    return "\n".join(lines)
