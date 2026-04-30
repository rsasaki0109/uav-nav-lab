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
