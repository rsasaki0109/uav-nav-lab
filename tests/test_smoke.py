"""End-to-end smoke test: run → eval → sweep without crashes."""

from __future__ import annotations

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


def test_voxel_world_dynamic_obstacles_advance_and_collide() -> None:
    """voxel_world should support dynamic obstacles symmetrically with grid_world."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    voxel_cls = SCENARIO_REGISTRY.get("voxel_world")
    sc = voxel_cls.from_config(
        {
            "size": [20, 20, 10],
            "start": [2.0, 2.0, 5.0],
            "goal": [17.0, 17.0, 5.0],
            "resolution": 1.0,
            "obstacles": {"type": "none"},
            "dynamic_obstacles": [
                {"start": [10.0, 10.0, 5.0], "velocity": [1.0, 0.0, 0.0], "radius": 0.5},
            ],
        }
    )
    # property reflects current state
    assert len(sc.dynamic_obstacles) == 1
    assert sc.dynamic_obstacles[0]["position"][0] == pytest.approx(10.0)
    # advance moves the obstacle linearly
    sc.advance(1.0)
    assert sc.dynamic_obstacles[0]["position"][0] == pytest.approx(11.0)
    # collision check uses sphere-sphere distance against true position
    assert sc.is_collision(np.array([11.0, 10.0, 5.0]), radius=0.4)
    assert not sc.is_collision(np.array([15.0, 15.0, 5.0]), radius=0.4)


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


def test_dummy_sim_wind_blows_drone() -> None:
    """A drone with zero command + non-zero wind should drift along the wind."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn_cfg = {
        "size": [50, 50], "start": [10.0, 10.0], "goal": [40.0, 40.0],
        "obstacles": {"type": "none"},
    }
    scn = SCENARIO_REGISTRY.get("grid_world").from_config(scn_cfg)
    sim_cfg = {
        "dt": 0.1, "max_steps": 100, "max_accel": 100.0,
        "disturbance": {"wind": [3.0, 0.0]},
    }
    sim = SIM_REGISTRY.get("dummy_2d").from_config(sim_cfg, scn)
    sim.reset(seed=0)
    initial_x = sim.state.position[0]
    for _ in range(10):
        sim.step(np.array([0.0, 0.0]))  # zero velocity command
    drift = sim.state.position[0] - initial_x
    # 3 m/s wind for 1.0s should produce ~3m of drift
    assert 2.5 < drift < 3.5


def test_sweep_vector_param_parsing() -> None:
    from uav_nav_lab.runner.sweep import _parse_spec

    vals = _parse_spec("[0,0],[3,0],[6,0]")
    assert vals == [[0, 0], [3, 0], [6, 0]]
    # mixed list + scalar
    vals = _parse_spec("[0,3]")
    assert vals == [[0, 3]]


def test_wilson_ci_bounds() -> None:
    """Wilson interval should: (1) bracket the point estimate, (2) widen at
    small N, (3) stay inside [0,1] even at boundary outcomes (0/N or N/N)."""
    from uav_nav_lab.eval.metrics import _wilson

    p, lo, hi = _wilson(3, 5)
    assert lo <= p <= hi
    assert lo > 0 and hi < 1
    assert (hi - lo) > 0.4  # N=5 is wide

    _, lo, hi = _wilson(50, 100)
    assert (hi - lo) < 0.2  # N=100 is much tighter

    _, lo, hi = _wilson(0, 5)  # boundary
    assert lo == 0.0 and 0 < hi < 1
    _, lo, hi = _wilson(5, 5)  # other boundary
    assert hi == 1.0 and 0 < lo < 1


def test_summary_includes_ci(tmp_path: Path) -> None:
    cfg = _basic_cfg()
    run_dir = run_experiment(cfg, tmp_path / "ci_run")
    summary = evaluate_run(run_dir)
    assert "success_ci95" in summary
    lo, hi = summary["success_ci95"]
    assert 0.0 <= lo <= summary["success_rate"] <= hi <= 1.0
    assert "ci_lo" in summary["avg_speed"]
    assert "sem" in summary["avg_speed"]


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


def test_predictor_registry_has_defaults() -> None:
    from uav_nav_lab.predictor import PREDICTOR_REGISTRY

    names = PREDICTOR_REGISTRY.names()
    assert "constant_velocity" in names
    assert "noisy_velocity" in names


def test_constant_velocity_predictor_extrapolates_linearly() -> None:
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    p = build_predictor({"type": "constant_velocity"})
    obs = [{"position": [0.0, 0.0], "velocity": [1.0, 2.0], "radius": 0.5}]
    traj = p.predict(obs, np.array([1.0, 2.0, 3.0]))
    # one obstacle, three time points, 2D
    assert traj.shape == (1, 3, 2)
    assert np.allclose(traj[0, 0], [1.0, 2.0])
    assert np.allclose(traj[0, 2], [3.0, 6.0])


def test_kalman_predictor_basic_roundtrip() -> None:
    """Kalman predictor must (a) register, (b) produce the right shape,
    (c) eventually agree with the truth on a clean constant-velocity
    target after a few measurement updates."""
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    p = build_predictor({"type": "kalman_velocity", "dt": 0.1,
                         "process_noise_std": 0.1, "measurement_noise_std": 0.05})
    # Simulate a target moving at v=(2, 0) starting from (0, 0) for ~10 dt
    truth_v = np.array([2.0, 0.0])
    pos = np.zeros(2)
    horizon_dts = np.array([0.1, 0.5, 1.0])
    for _ in range(10):
        obs = [{"position": list(pos), "velocity": list(truth_v), "radius": 0.5}]
        traj = p.predict(obs, horizon_dts)
        assert traj.shape == (1, 3, 2)
        pos = pos + truth_v * 0.1
    # After 10 updates the KF velocity estimate should be close to truth
    track_v = p._tracks[0]["x"][2:]
    assert np.allclose(track_v, truth_v, atol=0.2)


def test_kalman_track_associates_across_calls() -> None:
    """A drifting target observed across multiple calls should remain a
    single track (not be re-spawned every call as a brand-new one)."""
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    p = build_predictor({"type": "kalman_velocity", "dt": 0.2,
                         "association_threshold": 5.0})
    horizon_dts = np.array([0.2])
    pos = np.array([10.0, 10.0])
    for _ in range(5):
        obs = [{"position": list(pos), "velocity": [1.0, 0.0], "radius": 0.5}]
        p.predict(obs, horizon_dts)
        pos[0] += 0.2  # drift x by dt·v = 0.2 per call
    assert len(p._tracks) == 1, "track was duplicated on each call"


def test_kalman_delayed_sensor_recovers_current_pose() -> None:
    """Kalman-delayed sensor should converge to the true current pose
    on a constant-velocity target after the buffer fills + a few KF updates."""
    import numpy as np

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("kalman_delayed")
    sensor = cls.from_config({
        "delay": 0.2, "dt": 0.05,
        "process_noise_std": 0.5, "measurement_noise_std": 0.05,
    })
    sensor.reset(seed=42)
    pos = np.zeros(2)
    truth_v = np.array([2.0, 0.0])
    last_obs = None
    last_truth = None
    for k in range(40):
        last_truth = pos.copy()
        last_obs = sensor.observe(k * 0.05, pos)
        pos = pos + truth_v * 0.05
    # by step 40 the KF should be tracking close to the current truth
    assert np.allclose(last_obs, last_truth, atol=0.3)


def test_delayed_sensor_velocity_window_smooths_noisy_position() -> None:
    """A larger velocity_window should reduce the variance of the
    extrapolated estimate when position observations are noisy."""
    import numpy as np

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("delayed")
    rng = np.random.default_rng(0)
    truth_v = np.array([2.0, 0.0])

    def run(window: int) -> float:
        sensor = cls.from_config({
            "delay": 0.2, "dt": 0.05, "extrapolate": True,
            "position_noise_std": 0.0, "velocity_window": window,
        })
        sensor.reset(seed=42)
        pos = np.zeros(2)
        outs = []
        for k in range(40):
            # noisy true position (sim of imperfect localization input)
            noisy_pos = pos + rng.normal(0.0, 0.05, size=2)
            outs.append(sensor.observe(k * 0.05, noisy_pos))
            pos = pos + truth_v * 0.05
        # measure variance of the estimate's deviation from truth
        outs = np.asarray(outs[10:])  # let buffer fill
        true_traj = np.stack([truth_v * (k * 0.05) for k in range(10, 40)])
        return float(np.std(outs - true_traj))

    err_w1 = run(window=1)
    err_w5 = run(window=5)
    # window=5 should produce a noticeably smaller error stdev than window=1
    assert err_w5 < err_w1, f"window=5 ({err_w5:.3f}) did not improve over window=1 ({err_w1:.3f})"


def test_delayed_sensor_extrapolate_recovers_current_pose() -> None:
    """With `extrapolate=True`, a stale measurement should be projected
    forward by `delay`, recovering close to the true current pose for a
    constant-velocity target."""
    import numpy as np

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("delayed")
    sensor = cls.from_config({"delay": 0.1, "dt": 0.05, "extrapolate": True})
    sensor.reset()
    # constant-velocity true motion: pos = [t, 0]; v=(1,0)
    pos = np.zeros(2)
    last_obs = None
    for k in range(10):
        last_obs = sensor.observe(k * 0.05, pos)
        pos = pos + np.array([1.0, 0.0]) * 0.05
    # final true position is [10*0.05, 0] = [0.5, 0]
    # extrapolated obs should be close to current truth (within numerical noise)
    assert np.allclose(last_obs, [0.5, 0.0], atol=0.1)


def test_kalman_delay_compensation_extrapolates_forward() -> None:
    """With delay_compensation set, the output should sit ahead of the
    raw rollout by delay_compensation × velocity."""
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    base = build_predictor({"type": "kalman_velocity", "dt": 0.1,
                            "delay_compensation": 0.0})
    leaded = build_predictor({"type": "kalman_velocity", "dt": 0.1,
                              "delay_compensation": 0.5})
    obs = [{"position": [0.0, 0.0], "velocity": [3.0, 0.0], "radius": 0.5}]
    dts = np.array([0.1])
    a = base.predict(obs, dts)
    b = leaded.predict(obs, dts)
    # b should be 0.5 * 3.0 = 1.5 m further along x than a (with first-call
    # bootstrap velocity from observation)
    assert np.isclose(b[0, 0, 0] - a[0, 0, 0], 1.5, atol=0.05)


def test_noisy_predictor_seed_is_deterministic() -> None:
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    cfg = {"type": "noisy_velocity", "velocity_noise_std": 1.0}
    obs = [{"position": [0.0, 0.0], "velocity": [1.0, 0.0], "radius": 0.5}]
    dts = np.array([1.0, 2.0])
    p1 = build_predictor(cfg)
    p1.reset(seed=123)
    a = p1.predict(obs, dts)
    p2 = build_predictor(cfg)
    p2.reset(seed=123)
    b = p2.predict(obs, dts)
    assert np.allclose(a, b)
    # but a fresh seed should produce a different draw
    p3 = build_predictor(cfg)
    p3.reset(seed=456)
    c = p3.predict(obs, dts)
    assert not np.allclose(a, c)


def test_mpc_uses_configured_predictor(tmp_path: Path) -> None:
    """MPC must accept a `planner.predictor` block and pass it through."""
    from uav_nav_lab.predictor import constant_velocity as cv
    from uav_nav_lab.predictor import noisy as ny

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_predictor_noise.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 200
    cfg.planner["predictor"] = {"type": "noisy_velocity", "velocity_noise_std": 0.5}
    planner_cls = PLANNER_REGISTRY.get(cfg.planner["type"])
    p = planner_cls.from_config(cfg.planner)
    assert isinstance(p._predictor, ny.NoisyVelocityPredictor)

    cfg.planner["predictor"] = {"type": "constant_velocity"}
    p = planner_cls.from_config(cfg.planner)
    assert isinstance(p._predictor, cv.ConstantVelocityPredictor)


def test_multi_drone_runs_and_logs(tmp_path: Path) -> None:
    """Two drones, head-on; per-drone episode logs land alongside each other."""
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "multi")
    drone_logs = sorted(run_dir.glob("episode_*_drone_*.json"))
    # 2 episodes × 2 drones
    assert len(drone_logs) == 4
    # parent eval treats each drone-episode as its own row
    summary = evaluate_run(run_dir)
    assert summary["n_episodes"] == 4

    # at least one drone log should have a sane outcome string
    import json as _json
    log = _json.loads(drone_logs[0].read_text())
    assert log["outcome"] in {"success", "collision", "timeout"}
    assert "drone_id" in log["meta"]


def test_multi_drone_scenario_validates_drones() -> None:
    from uav_nav_lab.scenario import SCENARIO_REGISTRY

    cls = SCENARIO_REGISTRY.get("multi_drone_grid")
    with pytest.raises(ValueError):
        # missing `drones` block must be rejected
        cls.from_config({"size": [10, 10], "obstacles": {"type": "none"}})


def test_multi_drone_joint_metrics_in_summary(tmp_path: Path) -> None:
    """Joint episode summaries are picked up and aggregated separately from
    the per-drone-episode rows."""
    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 600
    run_dir = run_experiment(cfg, tmp_path / "multi_joint")
    # joint files exist alongside per-drone trajectories
    assert sorted(p.name for p in run_dir.glob("episode_*_joint.json")) == [
        "episode_000_joint.json",
        "episode_001_joint.json",
    ]
    summary = evaluate_run(run_dir)
    # per-drone-episode rows: 2 episodes × 2 drones
    assert summary["n_episodes"] == 4
    # joint rows: 2 episodes
    assert summary["joint_n_episodes"] == 2
    assert summary["joint_n_drones"] == 2
    assert "joint_success_rate" in summary
    assert "joint_collision_ci95" in summary


def test_summary_includes_planner_dt_compute_metrics(tmp_path: Path) -> None:
    """The recorder logs `planner_dt_ms` per replan; eval must aggregate
    that into mean / p95 / max compute cost so compute-budget studies
    do not need a second pass over the raw episode logs."""
    cfg = _basic_cfg()
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 200
    run_dir = run_experiment(cfg, tmp_path / "compute")
    summary = evaluate_run(run_dir)
    for key in ("planner_dt_ms_mean", "planner_dt_ms_p95", "planner_dt_ms_max"):
        assert key in summary, f"summary missing {key}"
        assert summary[key]["mean"] >= 0.0
        # consistency across statistics: max ≥ p95 ≥ mean
    assert summary["planner_dt_ms_max"]["mean"] >= summary["planner_dt_ms_p95"]["mean"]
    assert summary["planner_dt_ms_p95"]["mean"] >= summary["planner_dt_ms_mean"]["mean"]


def test_multi_drone_viz_groups_drones_per_episode(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from uav_nav_lab.viz import viz_run

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 2
    cfg.simulator["max_steps"] = 400
    run_dir = run_experiment(cfg, tmp_path / "multi_viz")
    saved = viz_run(run_dir)
    # one PNG per episode (not per drone)
    assert len(saved) == 2
    for p in saved:
        assert p.exists() and p.stat().st_size > 0
