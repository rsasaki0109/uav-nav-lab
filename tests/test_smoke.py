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
