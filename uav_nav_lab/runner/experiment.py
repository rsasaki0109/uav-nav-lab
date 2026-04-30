"""Experiment orchestrator — runs N episodes for a single config and writes logs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import ExperimentConfig
from ..planner import PLANNER_REGISTRY, Plan, Planner
from ..recorder import EpisodeRecorder
from ..scenario import SCENARIO_REGISTRY
from ..sensor import SENSOR_REGISTRY
from ..sim import SIM_REGISTRY


def _follow_plan(plan: Plan, observation: np.ndarray, max_speed: float) -> np.ndarray:
    """Convert a Plan into a velocity setpoint.

    If the planner already chose a velocity (e.g. MPC), apply it directly.
    Otherwise run pure-pursuit on the waypoint sequence:
      1. project observation onto the plan via the closest waypoint
      2. walk forward to the first waypoint at least `lookahead` away
    Lookahead scales with speed so high-speed runs do not chase points that
    have already been overtaken due to sensor lag / noise.
    """
    if plan.target_velocity is not None:
        v = np.asarray(plan.target_velocity, dtype=float).ravel()
        n = float(np.linalg.norm(v))
        if n > max_speed:
            v = v * (max_speed / n)
        return v
    if plan.is_empty:
        ndim = int(np.asarray(observation).shape[0])
        return np.zeros(ndim)
    wps = plan.waypoints
    ndim = int(wps.shape[1])
    obs = np.asarray(observation, dtype=float)[:ndim]
    dists = np.linalg.norm(wps - obs[None, :], axis=1)
    closest = int(np.argmin(dists))
    lookahead = max(1.5, 0.3 * max_speed)
    target_idx = len(wps) - 1
    for i in range(closest, len(wps)):
        if np.linalg.norm(wps[i] - obs) >= lookahead:
            target_idx = i
            break
    vec = wps[target_idx] - obs
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.zeros(ndim)
    return (vec / n) * max_speed


def _build(cfg: ExperimentConfig) -> tuple[Any, Planner, Any, Any]:
    scenario_cls = SCENARIO_REGISTRY.get(cfg.scenario.get("type", "grid_world"))
    scenario = scenario_cls.from_config(cfg.scenario)

    sim_cls = SIM_REGISTRY.get(cfg.simulator.get("type", "dummy_2d"))
    sim = sim_cls.from_config(cfg.simulator, scenario)

    planner_cls = PLANNER_REGISTRY.get(cfg.planner.get("type", "straight"))
    planner = planner_cls.from_config(cfg.planner)

    sensor_cfg = dict(cfg.sensor)
    sensor_cfg.setdefault("dt", cfg.simulator.get("dt", 0.05))
    sensor_cls = SENSOR_REGISTRY.get(sensor_cfg.get("type", "perfect"))
    sensor = sensor_cls.from_config(sensor_cfg)

    return sim, planner, sensor, scenario


def _run_episode(
    sim: Any,
    planner: Planner,
    sensor: Any,
    *,
    seed: int,
    replan_period: float,
    max_steps: int,
    episode_index: int,
) -> EpisodeRecorder:
    rec = EpisodeRecorder(episode_index=episode_index, seed=seed)
    state = sim.reset(seed=seed)
    sensor.reset(seed=seed)
    planner.reset()

    plan: Plan | None = None
    last_replan_t = -float("inf")
    dt = sim.dt

    for step in range(max_steps):
        t = state.t
        observation = sensor.observe(t, state.position)

        if plan is None or (t - last_replan_t) >= replan_period:
            t0 = time.perf_counter()
            perceived_map = sensor.observe_map(t, state.position, sim.obstacle_map)
            perceived_dyn = sensor.observe_dynamics(
                t, state.position, sim.scenario.dynamic_obstacles
            )
            plan = planner.plan(
                observation,
                sim.goal,
                perceived_map,
                dynamic_obstacles=perceived_dyn,
            )
            planner_dt_ms = (time.perf_counter() - t0) * 1000.0
            last_replan_t = t
            rec.log_replan(t=t, plan_length=int(plan.waypoints.shape[0]), planner_dt_ms=planner_dt_ms)

        cmd = _follow_plan(plan, observation, planner.max_speed)
        next_state, info = sim.step(cmd)

        rec.log_step(
            t=t,
            true_pos=state.position,
            true_vel=state.velocity,
            observed_pos=observation,
            cmd=cmd,
            info={"collision": info.collision, "goal_reached": info.goal_reached},
        )

        state = next_state
        if info.collision:
            rec.set_outcome("collision", final_t=float(state.t))
            return rec
        if info.goal_reached:
            rec.set_outcome("success", final_t=float(state.t))
            return rec
        if info.truncated:
            rec.set_outcome("timeout", final_t=float(state.t))
            return rec

    rec.set_outcome("timeout", final_t=float(state.t))
    return rec


def run_experiment(cfg: ExperimentConfig, output_dir: Path) -> Path:
    """Run all episodes for a single config; return the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    sim, planner, sensor, _ = _build(cfg)
    replan_period = float(cfg.planner.get("replan_period", 0.5))
    max_steps = int(cfg.simulator.get("max_steps", 2000))

    print(f"[run] {cfg.name}: {cfg.num_episodes} episode(s) → {output_dir}")
    for ep in range(cfg.num_episodes):
        seed = cfg.seed + ep
        rec = _run_episode(
            sim,
            planner,
            sensor,
            seed=seed,
            replan_period=replan_period,
            max_steps=max_steps,
            episode_index=ep,
        )
        rec.save(output_dir / f"episode_{ep:03d}.json")
        print(f"  ep {ep:03d} seed={seed} outcome={rec.outcome} t={rec.summary.get('final_t', 0.0):.2f}s")

    return output_dir
