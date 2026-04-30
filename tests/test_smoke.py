"""End-to-end smoke test: run → eval → sweep without crashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np  # noqa: F401
import pytest

from uav_nav_lab.cli import build_parser, main
from uav_nav_lab.config import ExperimentConfig
from uav_nav_lab.eval import evaluate_run
from uav_nav_lab.planner import PLANNER_REGISTRY
from uav_nav_lab.runner import expand_sweep, run_experiment

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _basic_cfg(overrides: dict | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_basic.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 600
    if overrides:
        cfg.raw.update(overrides)
    return cfg


def test_registries_populated() -> None:
    assert "astar" in PLANNER_REGISTRY.names()
    assert "straight" in PLANNER_REGISTRY.names()
    assert "mpc" in PLANNER_REGISTRY.names()


def test_run_then_eval(tmp_path: Path) -> None:
    cfg = _basic_cfg()
    run_dir = run_experiment(cfg, tmp_path / "run")
    eps = sorted(run_dir.glob("episode_*.json"))
    assert len(eps) == 2
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert (run_dir / "summary.json").exists()


def test_straight_baseline_runs(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_straight.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "straight")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_sweep_expansion() -> None:
    base = ExperimentConfig.from_yaml(EXAMPLES / "exp_sweep.yaml")
    cfgs = expand_sweep(base, [("planner.max_speed", "5:15:5"), ("planner.type", "astar,straight")])
    assert len(cfgs) == 2 * 2
    speeds = {c.planner["max_speed"] for c in cfgs}
    types = {c.planner["type"] for c in cfgs}
    assert speeds == {5, 10}
    assert types == {"astar", "straight"}


def test_cli_list_runs() -> None:
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.func(args) == 0


def test_3d_viz(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from uav_nav_lab.viz import viz_run

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_3d.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 200
    run_dir = run_experiment(cfg, tmp_path / "viz_3d")
    saved = viz_run(run_dir)
    assert len(saved) == 1
    assert saved[0].exists() and saved[0].stat().st_size > 0


def test_anim_gif(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from uav_nav_lab.anim import viz_anim

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_dynamic.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 80   # very short — keep the test fast
    run_dir = run_experiment(cfg, tmp_path / "anim_run")
    saved = viz_anim(run_dir, fps=10)
    assert len(saved) == 1
    assert saved[0].suffix == ".gif"
    assert saved[0].stat().st_size > 1000   # something more than an empty file


def test_bridge_stubs_registered() -> None:
    """AirSim and ROS2 backends register at import time but should fail with
    a clear message if their heavy deps are not installed."""
    from uav_nav_lab.sim import SIM_REGISTRY

    assert "airsim" in SIM_REGISTRY.names()
    assert "ros2" in SIM_REGISTRY.names()


def test_3d_runs(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_3d.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "3d")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1
    # confirm the run was actually 3D (3 components per logged position)
    import json

    ep0 = json.loads((run_dir / "episode_000.json").read_text())
    assert len(ep0["steps"][0]["true_pos"]) == 3


def test_3d_mpc_runs(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_3d_mpc.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    cfg.scenario["obstacles"]["count"] = 30  # keep it loose; MPC only needs to plan, not succeed
    cfg.planner["n_samples"] = 16
    run_dir = run_experiment(cfg, tmp_path / "3d_mpc")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_lidar_dynamics_filtered_by_range() -> None:
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    lidar_cls = SENSOR_REGISTRY.get("lidar")
    s = lidar_cls.from_config({"range": 5.0, "delay": 0.0, "memory": False})
    s.reset(seed=0)
    dyn = [
        {"position": [3.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.5},  # in range
        {"position": [50.0, 0.0], "velocity": [0.0, 0.0], "radius": 0.5},  # out of range
    ]
    seen = s.observe_dynamics(0.0, np.array([0.0, 0.0]), dyn)
    assert len(seen) == 1
    assert seen[0]["position"] == [3.0, 0.0]


def test_mpc_prediction_changes_action() -> None:
    """A rollout that reaches the goal but passes through a *predicted*
    obstacle position should be rejected when use_prediction=True. Without
    prediction the planner picks the goal-direction action even though it
    walks through the obstacle's future location."""
    from uav_nav_lab.planner.mpc import SamplingMPCPlanner

    occ = np.zeros((30, 30), dtype=bool)
    goal = np.array([20.0, 5.0])
    pos = np.array([2.0, 5.0])
    # Cross-path threat: at h=20 (1.0s @ dt_plan=0.05) it sits on the
    # straight-line drone path between (10, 5) and (12, 5).
    dyn = [{"position": [11.0, 12.0], "velocity": [0.0, -7.0], "radius": 1.5}]

    args = dict(max_speed=10.0, horizon=40, dt_plan=0.05, n_samples=32, inflate=0,
                safety_margin=0.5)
    pw = SamplingMPCPlanner(use_prediction=True, **args)
    pw.reset()
    aw = pw.plan(pos, goal, occ, dynamic_obstacles=dyn).target_velocity

    pwo = SamplingMPCPlanner(use_prediction=False, **args)
    pwo.reset()
    awo = pwo.plan(pos, goal, occ, dynamic_obstacles=dyn).target_velocity

    # The "without prediction" planner has no reason to deviate from the
    # straight goal direction (1, 0) * max_speed.
    assert np.allclose(awo, [10.0, 0.0], atol=1e-3)
    # The "with prediction" planner must steer off-axis.
    assert not np.allclose(aw, awo, atol=0.1)


def test_dynamic_obstacle_motion(tmp_path: Path) -> None:
    """A dynamic obstacle should appear in occupancy at the right cells over
    time, and the lidar memory should clear when it moves out of range."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cfg = {
        "size": [20, 20],
        "start": [2.0, 2.0],
        "goal": [18.0, 18.0],
        "obstacles": {"type": "none"},
        "dynamic_obstacles": [
            {"start": [10.0, 10.0], "velocity": [5.0, 0.0], "reflect": False, "radius": 0.6}
        ],
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(cfg)
    occ_t0 = scn.occupancy.copy()
    assert occ_t0[10, 10]
    scn.advance(1.0)  # move 5 cells in x
    assert scn.occupancy[15, 10]
    assert not scn.occupancy[10, 10]  # the dynamic obstacle is no longer here

    # lidar memory: cell (10,10) was seen as obstacle, then we observe again
    # with obstacle gone — should be cleared from memory.
    lidar_cls = SENSOR_REGISTRY.get("lidar")
    lidar = lidar_cls.from_config({"range": 5.0, "delay": 0.0, "memory": True})
    lidar.reset(seed=0)
    seen0 = lidar.observe_map(0.0, np.array([10.0, 10.0]), occ_t0)
    assert seen0[10, 10]
    seen1 = lidar.observe_map(1.0, np.array([10.0, 10.0]), scn.occupancy)
    assert not seen1[10, 10]


def test_lidar_sensor_partial_map() -> None:
    """Lidar should only mark obstacles within `range` of the drone, and
    accumulate them across observations when memory=True."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    occ = np.zeros((20, 20), dtype=bool)
    occ[5, 5] = True   # near start
    occ[15, 15] = True  # far from start

    sensor_cls = SENSOR_REGISTRY.get("lidar")
    s = sensor_cls.from_config({"range": 4.0, "delay": 0.0, "resolution": 1.0, "memory": True})
    s.reset(seed=0)

    seen0 = s.observe_map(0.0, np.array([5.0, 5.0]), occ)
    assert seen0[5, 5]            # close obstacle is visible
    assert not seen0[15, 15]      # distant one is not

    # drone moves close to the far obstacle; both should now be in memory
    seen1 = s.observe_map(0.1, np.array([15.0, 15.0]), occ)
    assert seen1[5, 5] and seen1[15, 15]


def test_dynamic_run(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_dynamic.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "dyn_run")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_lidar_run(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_lidar.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "lidar_run")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_sample_unit_directions_3d() -> None:
    from uav_nav_lab.planner._grid import sample_unit_directions

    base = np.array([1.0, 0.0, 0.0])
    dirs = sample_unit_directions(3, 16, base)
    assert dirs.shape == (16, 3)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # first sample is the goal direction
    assert np.allclose(dirs[0], base)


def test_mpc_runs(tmp_path: Path) -> None:
    cfg = _basic_cfg()
    cfg.planner = {"type": "mpc", "max_speed": 5.0, "replan_period": 0.5, "horizon": 30}
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "mpc")
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 1


def test_parallel_sweep(tmp_path: Path) -> None:
    from uav_nav_lab.runner import run_sweep

    base = ExperimentConfig.from_yaml(EXAMPLES / "exp_sweep.yaml")
    base.num_episodes = 1
    base.simulator["max_steps"] = 200
    out = run_sweep(
        base,
        [("planner.max_speed", "5,10")],
        tmp_path / "psweep",
        parallel=2,
    )
    assert (out / "run_000" / "config.yaml").exists()
    assert (out / "run_001" / "config.yaml").exists()
    assert (out / "sweep_manifest.json").exists()


def test_get_dotted() -> None:
    from uav_nav_lab.config import get_dotted

    d = {"a": {"b": {"c": 7}, "x": "v"}}
    assert get_dotted(d, "a.b.c") == 7
    assert get_dotted(d, "a.x") == "v"
    assert get_dotted(d, "a.b.missing", default=42) == 42
    assert get_dotted(d, "nope.nope", default=None) is None


def test_sweep_viz(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from uav_nav_lab.runner import run_sweep
    from uav_nav_lab.sweep_viz import sweep_viz

    base = ExperimentConfig.from_yaml(EXAMPLES / "exp_sweep.yaml")
    base.num_episodes = 1
    base.simulator["max_steps"] = 200
    out = run_sweep(
        base,
        [("planner.max_speed", "5,10"), ("planner.type", "astar,straight")],
        tmp_path / "sviz",
    )
    img = sweep_viz(out)
    assert img.exists() and img.stat().st_size > 0


def test_viz(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from uav_nav_lab.viz import viz_run

    cfg = _basic_cfg()
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 200
    run_dir = run_experiment(cfg, tmp_path / "viz_run")
    saved = viz_run(run_dir)
    assert len(saved) == 1
    assert saved[0].exists() and saved[0].stat().st_size > 0


def test_cli_run_eval_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    # short, deterministic runs
    base_args = ["run", str(EXAMPLES / "exp_basic.yaml"), "-o", str(out_a)]
    rc = main(base_args)
    assert rc == 0
    rc = main(["run", str(EXAMPLES / "exp_straight.yaml"), "-o", str(out_b)])
    assert rc == 0
    rc = main(["eval", str(out_a)])
    assert rc == 0
    rc = main(["compare", str(out_a), str(out_b)])
    assert rc == 0
    # confirm log files exist
    assert any(out_a.glob("episode_*.json"))
    assert (out_a / "summary.json").exists()
