"""Episode replay → animated GIF.

Re-walks a recorded episode at simulator time and renders one GIF per
episode (sub-sampled to a target FPS). Dynamic obstacles, lidar memory
state, and the drone trajectory build up frame by frame so the animation
matches what the planner actually saw at each replan.

The animation re-reads the saved config so the scenario seed and dynamic
obstacle setup match the original run exactly. Both 2D (matplotlib
axes) and 3D (Axes3D with rotating camera) renderings are supported;
the dispatcher picks based on `scenario.ndim`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig
from .scenario import SCENARIO_REGISTRY


def _need_mpl_anim() -> tuple[Any, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        return plt, animation
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for `uav-nav anim`. "
            "Install with: pip install -e '.[viz]'"
        ) from e


def _animate_episode_2d(plt, animation, cfg: ExperimentConfig, ep: dict, scenario, fps: int) -> Any:
    import numpy as np

    res = scenario.resolution
    nx, ny = scenario.occupancy.shape
    steps = ep["steps"]
    if not steps:
        return None

    dt = float(cfg.simulator.get("dt", 0.05))
    sim_fps = 1.0 / dt
    stride = max(1, int(round(sim_fps / fps)))
    frame_indices = list(range(0, len(steps), stride))
    if frame_indices[-1] != len(steps) - 1:
        frame_indices.append(len(steps) - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, nx * res)
    ax.set_ylim(0, ny * res)
    ax.set_aspect("equal")

    ax.imshow(
        scenario._static_occ.T,  # always show static layer behind everything
        origin="lower",
        extent=(0, nx * res, 0, ny * res),
        cmap="Greys",
        alpha=0.5,
        interpolation="nearest",
    )
    ax.plot(*scenario.start, "go", ms=10, label="start", zorder=4)
    ax.plot(*scenario.goal, "r*", ms=14, label="goal", zorder=4)

    (traj_line,) = ax.plot([], [], "-", color="tab:blue", lw=1.5, label="true", zorder=3)
    (drone_pt,) = ax.plot([], [], "o", color="tab:blue", ms=8, zorder=5)
    dyn_scatter = ax.scatter([], [], s=120, c="tab:red", marker="o",
                             edgecolors="black", zorder=4, label="dynamic")

    sensor_cfg = cfg.sensor
    sensor_range = float(sensor_cfg.get("range", sensor_cfg.get("range_m", 0.0)))
    sensor_circle = None
    if sensor_range > 0 and sensor_cfg.get("type") == "lidar":
        # draw a circle representing visibility around the drone
        from matplotlib.patches import Circle
        sensor_circle = Circle((0, 0), sensor_range, fill=False, color="tab:cyan",
                               lw=1.0, alpha=0.6, zorder=2)
        ax.add_patch(sensor_circle)

    title = ax.set_title("")
    ax.legend(loc="lower right", fontsize=8)

    # We re-step the scenario to recover dynamic obstacle positions per frame.
    scenario.reseed(ep["meta"]["seed"])
    sim_time_at_step: dict[int, float] = {i: i * dt for i in range(len(steps))}

    def update(idx_in_frames: int):
        i = frame_indices[idx_in_frames]
        # walk scenario forward to step i
        # (re-using the same scenario object across frames; reset on first frame)
        if idx_in_frames == 0:
            scenario.reseed(ep["meta"]["seed"])
            scenario._steps_advanced = 0  # bookkeeping
        target = i
        cur = getattr(scenario, "_steps_advanced", 0)
        for _ in range(cur, target):
            scenario.advance(dt)
        scenario._steps_advanced = target

        true_pos = [(s["true_pos"][0], s["true_pos"][1]) for s in steps[: i + 1]]
        if true_pos:
            tx, ty = zip(*true_pos)
            traj_line.set_data(tx, ty)
            drone_pt.set_data([tx[-1]], [ty[-1]])

        dyn = scenario.dynamic_obstacles
        if dyn:
            xs = [d["position"][0] for d in dyn]
            ys = [d["position"][1] for d in dyn]
            dyn_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            dyn_scatter.set_offsets(np.zeros((0, 2)))

        if sensor_circle is not None and steps:
            sensor_circle.center = (steps[i]["true_pos"][0], steps[i]["true_pos"][1])

        outcome = ep.get("outcome", "?")
        title.set_text(
            f"ep {ep['meta']['episode']:03d}  outcome={outcome}  "
            f"t={sim_time_at_step[i]:.2f}s"
        )
        return traj_line, drone_pt, dyn_scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )
    return fig, anim


def _animate_episode_3d(plt, animation, cfg: ExperimentConfig, ep: dict, scenario, fps: int) -> Any:
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection

    res = scenario.resolution
    nx, ny, nz = scenario.occupancy.shape
    steps = ep["steps"]
    if not steps:
        return None

    dt = float(cfg.simulator.get("dt", 0.05))
    sim_fps = 1.0 / dt
    stride = max(1, int(round(sim_fps / fps)))
    frame_indices = list(range(0, len(steps), stride))
    if frame_indices[-1] != len(steps) - 1:
        frame_indices.append(len(steps) - 1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, nx * res)
    ax.set_ylim(0, ny * res)
    ax.set_zlim(0, nz * res)

    # Static obstacles: scatter once (cheap vs re-drawing each frame).
    static_occ = getattr(scenario, "_static_occ", scenario.occupancy)
    ix, iy, iz = np.where(static_occ)
    if ix.size > 0:
        ax.scatter(
            (ix + 0.5) * res,
            (iy + 0.5) * res,
            (iz + 0.5) * res,
            c="gray", alpha=0.25, s=18, marker="s",
        )
    ax.scatter(*scenario.start, c="green", s=80, label="start")
    ax.scatter(*scenario.goal, c="red", marker="*", s=160, label="goal")

    (traj_line,) = ax.plot([], [], [], "-", color="tab:blue", lw=1.5, label="true")
    drone_pt = ax.scatter([], [], [], c="tab:blue", s=60, depthshade=True)
    dyn_scatter = ax.scatter([], [], [], s=120, c="tab:red", marker="o", edgecolors="black")
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)

    # Re-derive dynamic obstacle trajectories from the replay config so we
    # show their actual positions at each frame (rather than the stale
    # snapshots saved in step records, which only have what the *sensor*
    # reported).
    dyn_specs = list(cfg.scenario.get("dynamic_obstacles", []) or [])

    def dyn_at_step_index(j: int) -> tuple[list[float], list[float], list[float]]:
        if not dyn_specs:
            return [], [], []
        # Re-compute deterministic motion to step j (with reflection).
        bounds = (nx * res, ny * res, nz * res)
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for spec in dyn_specs:
            pos = list(map(float, spec["start"]))
            vel = list(map(float, spec.get("velocity", [0, 0, 0])))
            reflect = bool(spec.get("reflect", True))
            for _ in range(j):
                for k in range(3):
                    pos[k] += vel[k] * dt
                    if reflect:
                        if pos[k] < 0:
                            pos[k] = -pos[k]
                            vel[k] = -vel[k]
                        elif pos[k] > bounds[k]:
                            pos[k] = 2 * bounds[k] - pos[k]
                            vel[k] = -vel[k]
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
        return xs, ys, zs

    def update(frame_i: int) -> tuple[Any, ...]:
        j = frame_indices[frame_i]
        # Trajectory up to j.
        tx = [steps[k]["true_pos"][0] for k in range(j + 1)]
        ty = [steps[k]["true_pos"][1] for k in range(j + 1)]
        tz = [steps[k]["true_pos"][2] for k in range(j + 1)]
        traj_line.set_data(tx, ty)
        traj_line.set_3d_properties(tz)
        drone_pt._offsets3d = ([tx[-1]], [ty[-1]], [tz[-1]])
        dx, dy, dz = dyn_at_step_index(j)
        dyn_scatter._offsets3d = (dx, dy, dz)
        # Slow rotating view, +120° over the episode for a sense of depth.
        ax.view_init(elev=22.0, azim=-60.0 + (frame_i / max(1, len(frame_indices) - 1)) * 120.0)
        title.set_text(
            f"ep {ep['meta']['episode']:03d}  outcome={ep.get('outcome','?')}  "
            f"t={steps[j]['t']:.1f}s"
        )
        return traj_line, drone_pt, dyn_scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )
    return fig, anim


def viz_anim(run_dir: Path, fps: int = 20) -> list[Path]:
    plt, animation = _need_mpl_anim()
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found — anim needs the saved config")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = ExperimentConfig.from_dict(yaml.safe_load(f))
    scenario_cls = SCENARIO_REGISTRY.get(cfg.scenario.get("type", "grid_world"))
    scenario = scenario_cls.from_config(cfg.scenario)
    if scenario.ndim not in (2, 3):
        raise NotImplementedError(f"anim supports 2D / 3D scenarios (got ndim={scenario.ndim}).")

    saved: list[Path] = []
    for ef in sorted(run_dir.glob("episode_*.json")):
        if "_drone_" in ef.stem or ef.stem.endswith("_joint"):
            continue  # multi-drone artifacts handled by 2D path elsewhere
        with ef.open("r", encoding="utf-8") as f:
            ep = json.load(f)
        if scenario.ndim == 3:
            result = _animate_episode_3d(plt, animation, cfg, ep, scenario, fps=fps)
        else:
            result = _animate_episode_2d(plt, animation, cfg, ep, scenario, fps=fps)
        if result is None:
            continue
        fig, anim = result
        out = run_dir / f"episode_{ep['meta']['episode']:03d}.gif"
        anim.save(out, writer="pillow", fps=fps)
        plt.close(fig)
        saved.append(out)
    return saved
