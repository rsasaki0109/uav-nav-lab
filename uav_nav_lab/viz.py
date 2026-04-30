"""Trajectory / sweep visualization.

Reads a run directory (the JSON files written by the recorder) and renders one
PNG per episode plus an optional `summary.png` showing the obstacle map.
The scenario is reconstructed from the saved `config.yaml` so we can draw
the same obstacle layout the runner used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig
from .scenario import SCENARIO_REGISTRY


def _need_mpl() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")  # default to headless; --show overrides via plt.show()
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for `uav-nav viz`. Install with: "
            "pip install -e '.[viz]'"
        ) from e


def _load_run(run_dir: Path) -> tuple[ExperimentConfig, list[dict]]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found — viz needs the saved config")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = ExperimentConfig.from_dict(yaml.safe_load(f))
    episodes = []
    for ef in sorted(run_dir.glob("episode_*.json")):
        with ef.open("r", encoding="utf-8") as f:
            episodes.append(json.load(f))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.json in {run_dir}")
    return cfg, episodes


def _render_episode_2d(plt, ax, cfg, ep, scenario) -> None:
    occ = scenario.occupancy
    res = scenario.resolution
    ax.imshow(
        occ.T,
        origin="lower",
        extent=(0, occ.shape[0] * res, 0, occ.shape[1] * res),
        cmap="Greys",
        alpha=0.4,
        interpolation="nearest",
    )
    steps = ep["steps"]
    if steps:
        true_xy = [(s["true_pos"][0], s["true_pos"][1]) for s in steps]
        obs_xy = [(s["observed_pos"][0], s["observed_pos"][1]) for s in steps]
        tx, ty = zip(*true_xy)
        ox, oy = zip(*obs_xy)
        ax.plot(tx, ty, "-", color="tab:blue", lw=1.5, label="true")
        ax.plot(ox, oy, "--", color="tab:orange", lw=0.8, alpha=0.6, label="observed")
    for r in ep["replans"]:
        t = r["t"]
        for s in steps:
            if abs(s["t"] - t) < 1e-6:
                ax.plot(s["true_pos"][0], s["true_pos"][1], "x", color="tab:red", ms=6)
                break
    ax.plot(*scenario.start, "go", ms=10, label="start")
    ax.plot(*scenario.goal, "r*", ms=14, label="goal")
    ax.set_xlim(0, occ.shape[0] * res)
    ax.set_ylim(0, occ.shape[1] * res)
    ax.set_aspect("equal")
    ax.set_title(_title_for(ep))
    ax.legend(loc="lower right", fontsize=8)


def _render_episode_3d(plt, ax, cfg, ep, scenario) -> None:
    import numpy as np

    occ = scenario.occupancy
    res = scenario.resolution
    # sub-sample obstacle voxels so dense maps remain readable
    ix, iy, iz = np.where(occ)
    ax.scatter((ix + 0.5) * res, (iy + 0.5) * res, (iz + 0.5) * res,
               c="gray", alpha=0.3, s=18, marker="s")
    steps = ep["steps"]
    if steps:
        tx = [s["true_pos"][0] for s in steps]
        ty = [s["true_pos"][1] for s in steps]
        tz = [s["true_pos"][2] for s in steps]
        ax.plot(tx, ty, tz, "-", color="tab:blue", lw=1.5, label="true")
    for r in ep["replans"]:
        t = r["t"]
        for s in steps:
            if abs(s["t"] - t) < 1e-6:
                ax.scatter([s["true_pos"][0]], [s["true_pos"][1]], [s["true_pos"][2]],
                           c="red", s=22, marker="x")
                break
    ax.scatter(*scenario.start, c="green", s=80, label="start")
    ax.scatter(*scenario.goal, c="red", marker="*", s=160, label="goal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(_title_for(ep))
    ax.legend(loc="upper left", fontsize=8)


def _title_for(ep: dict) -> str:
    outcome = ep.get("outcome", "?")
    final_t = ep.get("summary", {}).get("final_t", 0.0)
    return (
        f"ep {ep['meta']['episode']:03d}  outcome={outcome}  "
        f"replans={len(ep['replans'])}  t={final_t:.1f}s"
    )


def viz_run(run_dir: Path, *, show: bool = False) -> list[Path]:
    plt = _need_mpl()
    run_dir = Path(run_dir)
    cfg, episodes = _load_run(run_dir)
    scenario_cls = SCENARIO_REGISTRY.get(cfg.scenario.get("type", "grid_world"))
    scenario = scenario_cls.from_config(cfg.scenario)
    is_3d = scenario.ndim == 3
    if scenario.ndim not in (2, 3):
        raise NotImplementedError(
            f"viz_run supports 2D / 3D scenarios (got ndim={scenario.ndim})."
        )
    # reseed each episode the way the runner did so obstacle layouts match
    saved: list[Path] = []
    for ep in episodes:
        seed = ep["meta"]["seed"]
        scenario.reseed(seed)
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            _render_episode_3d(plt, ax, cfg, ep, scenario)
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            _render_episode_2d(plt, ax, cfg, ep, scenario)
        out = run_dir / f"episode_{ep['meta']['episode']:03d}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        if show:  # pragma: no cover
            plt.show()
        plt.close(fig)
        saved.append(out)
    return saved
