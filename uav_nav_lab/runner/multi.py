"""Multi-drone episode runner.

One scenario, N drones. Each drone gets its own simulator / sensor / planner
instance; all instances share the same scenario object so the static map
and any scenario-owned dynamic obstacles stay consistent. Per global step:

  1. each drone observes the *other* drones' true poses + velocities as
     dynamic obstacles, filtered through its own sensor (so a range-limited
     sensor cleanly stops seeing peers beyond `range_m`)
  2. each drone replans (if its replan timer has elapsed) and acts
  3. the runner advances the scenario clock once (so dynamic obstacles do
     not advance N times per global tick)
  4. inter-drone collisions are detected pairwise on the freshly-stepped
     true positions
  5. once a drone finishes (success / collision), its position is frozen
     and it remains a peer obstacle for the rest until the episode ends

The episode ends when *all* drones have finished or `max_steps` elapses.
Per-drone trajectories are logged to `episode_<j>_drone_<i>.json` so the
existing single-drone eval pipeline aggregates over them automatically.
"""

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
from .experiment import _follow_plan


def _build_multi(cfg: ExperimentConfig) -> tuple[Any, list[Any], list[Planner], list[Any]]:
    scenario_cls = SCENARIO_REGISTRY.get(cfg.scenario.get("type", "multi_drone_grid"))
    scenario = scenario_cls.from_config(cfg.scenario)
    n = scenario.n_drones

    sim_cls = SIM_REGISTRY.get(cfg.simulator.get("type", "dummy_2d"))
    planner_cls = PLANNER_REGISTRY.get(cfg.planner.get("type", "straight"))

    sensor_cfg = dict(cfg.sensor)
    sensor_cfg.setdefault("dt", cfg.simulator.get("dt", 0.05))
    sensor_cls = SENSOR_REGISTRY.get(sensor_cfg.get("type", "perfect"))

    sims: list[Any] = []
    planners: list[Planner] = []
    sensors: list[Any] = []
    for i in range(n):
        # Only sim 0 advances the shared scenario; the rest are passive.
        sim = sim_cls.from_config(cfg.simulator, scenario)
        if i > 0 and hasattr(sim, "_advance_scenario"):
            sim._advance_scenario = False
        sim.set_goal(scenario.drones[i].goal)
        sims.append(sim)
        planners.append(planner_cls.from_config(cfg.planner))
        sensors.append(sensor_cls.from_config(sensor_cfg))
    return scenario, sims, planners, sensors


def _peers_view(
    states: list[Any], radii: list[float], finished: list[bool], me: int
) -> list[dict]:
    """Build a `dynamic_obstacles`-shaped list describing the *other* drones.

    A finished peer (success or collision) is reported with zero velocity —
    the simplest reasonable model of "stuck in place". This is what a
    downstream tracker would feed back to the planner anyway.
    """
    peers = []
    for j, s in enumerate(states):
        if j == me:
            continue
        v = s.velocity if not finished[j] else np.zeros_like(s.velocity)
        peers.append(
            {
                "position": [float(x) for x in s.position],
                "velocity": [float(x) for x in v],
                "radius": float(radii[j]),
            }
        )
    return peers


def _check_peer_collision(
    states: list[Any], radii: list[float], drone_radius: float
) -> list[bool]:
    """Returns a boolean per drone: True if it is currently overlapping a peer."""
    n = len(states)
    hit = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            r = drone_radius + radii[j]  # treat both with their own radii
            d2 = float(np.sum((states[i].position - states[j].position) ** 2))
            if d2 <= r * r:
                hit[i] = True
                hit[j] = True
    return hit


def run_episode_multi(
    scenario: Any,
    sims: list[Any],
    planners: list[Planner],
    sensors: list[Any],
    *,
    seed: int,
    replan_period: float,
    max_steps: int,
    episode_index: int,
) -> list[EpisodeRecorder]:
    n = len(sims)
    radii = [d.radius for d in scenario.drones]
    drone_radius = float(getattr(sims[0].p, "drone_radius", 0.4))

    # Reset all drones; only sim 0 reseeds the scenario (so the static layout
    # is reproducible and consistent across all drones in this episode).
    states = []
    for i, sim in enumerate(sims):
        s = sim.reset(
            seed=seed if i == 0 else None,
            initial_position=scenario.drones[i].start,
        )
        states.append(s)
        sensors[i].reset(seed=seed + 1000 * i)
        planners[i].reset()

    recorders = [EpisodeRecorder(episode_index=episode_index, seed=seed) for _ in range(n)]
    for i, rec in enumerate(recorders):
        rec.meta["drone_id"] = i
        rec.meta["drone_name"] = scenario.drones[i].name

    plans: list[Plan | None] = [None] * n
    last_replan_t = [-float("inf")] * n
    finished = [False] * n
    final_states = list(states)

    t = 0.0
    for step in range(max_steps):
        # 1. observations + replanning (per drone, in parallel order)
        observations = []
        for i in range(n):
            obs_i = sensors[i].observe(t, states[i].position)
            observations.append(obs_i)
            if finished[i]:
                continue
            if plans[i] is None or (t - last_replan_t[i]) >= replan_period:
                t0 = time.perf_counter()
                perceived_map = sensors[i].observe_map(
                    t, states[i].position, sims[i].obstacle_map,
                    sim_extra=states[i].extra or None,
                )
                scenario_dyn = scenario.dynamic_obstacles
                peer_dyn = _peers_view(states, radii, finished, me=i)
                # Filter through the sensor — a range-limited sensor will
                # drop peers / scene-dyn obstacles beyond its range.
                perceived_dyn = sensors[i].observe_dynamics(
                    t, states[i].position, scenario_dyn + peer_dyn
                )
                plans[i] = planners[i].plan(
                    obs_i,
                    sims[i].goal,
                    perceived_map,
                    dynamic_obstacles=perceived_dyn,
                )
                planner_dt_ms = (time.perf_counter() - t0) * 1000.0
                last_replan_t[i] = t
                recorders[i].log_replan(
                    t=t, plan_length=int(plans[i].waypoints.shape[0]),
                    planner_dt_ms=planner_dt_ms,
                )

        # 2. step each drone's sim (only sim 0 advances the scenario clock)
        new_states: list[Any] = list(states)
        infos = [None] * n
        for i in range(n):
            if finished[i]:
                # frozen — no integration, peers see vel=0 via _peers_view
                continue
            cmd = _follow_plan(
                plans[i], observations[i], planners[i].max_speed,
                t_since_replan=float(t - last_replan_t[i]),
            )
            ns, info = sims[i].step(cmd)
            new_states[i] = ns
            infos[i] = info
            recorders[i].log_step(
                t=t,
                true_pos=states[i].position,
                true_vel=states[i].velocity,
                observed_pos=observations[i],
                cmd=cmd,
                info={"collision": info.collision, "goal_reached": info.goal_reached},
                sim_extra=dict(ns.extra) if ns.extra else None,
            )

        # 3. peer-vs-peer collision check on the freshly stepped positions
        peer_hit = _check_peer_collision(new_states, radii, drone_radius)

        # 4. resolve outcomes
        for i in range(n):
            if finished[i]:
                continue
            info = infos[i]
            if info.collision or peer_hit[i]:
                recorders[i].set_outcome("collision", final_t=float(new_states[i].t))
                finished[i] = True
                final_states[i] = new_states[i]
                continue
            if info.goal_reached:
                recorders[i].set_outcome("success", final_t=float(new_states[i].t))
                finished[i] = True
                final_states[i] = new_states[i]
                continue

        states = new_states
        t = states[0].t  # all unfrozen sims share the same dt
        if all(finished):
            break

    # any still-running drones at this point are timed out
    for i in range(n):
        if not finished[i]:
            recorders[i].set_outcome("timeout", final_t=float(states[i].t))
            final_states[i] = states[i]

    return recorders


def run_experiment_multi(cfg: ExperimentConfig, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    scenario, sims, planners, sensors = _build_multi(cfg)
    n = scenario.n_drones
    replan_period = float(cfg.planner.get("replan_period", 0.5))
    max_steps = int(cfg.simulator.get("max_steps", 2000))

    print(f"[run] {cfg.name}: {cfg.num_episodes} episode(s), {n} drone(s) → {output_dir}")
    for ep in range(cfg.num_episodes):
        seed = cfg.seed + ep
        recs = run_episode_multi(
            scenario, sims, planners, sensors,
            seed=seed,
            replan_period=replan_period,
            max_steps=max_steps,
            episode_index=ep,
        )
        outcomes = [r.outcome for r in recs]
        for i, rec in enumerate(recs):
            rec.save(output_dir / f"episode_{ep:03d}_drone_{i:02d}.json")
        joint_outcome = (
            "success" if all(o == "success" for o in outcomes)
            else "collision" if any(o == "collision" for o in outcomes)
            else "timeout"
        )
        joint = {
            "meta": {"episode": ep, "seed": seed, "n_drones": n},
            "outcome": joint_outcome,
            "per_drone_outcomes": outcomes,
            "drone_names": [d.name for d in scenario.drones],
            "final_t": max(float(r.summary.get("final_t", 0.0)) for r in recs),
        }
        with (output_dir / f"episode_{ep:03d}_joint.json").open("w", encoding="utf-8") as f:
            import json as _json
            _json.dump(joint, f, indent=2)
        print(f"  ep {ep:03d} seed={seed} per-drone={outcomes} joint={joint_outcome}")

    return output_dir
