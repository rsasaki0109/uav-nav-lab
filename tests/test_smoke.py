"""End-to-end smoke test: run → eval → sweep without crashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
