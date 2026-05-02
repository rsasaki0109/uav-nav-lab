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


def test_airsim_bridge_step_round_trips_enu_via_mock_client() -> None:
    """Verify the AirSim bridge's ENU/NED conversions and step plumbing
    against an injected mock client — no AirSim install required."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge, _enu_to_ned, _ned_to_enu

    # Mathematical sanity on the conversion helpers.
    assert np.allclose(_enu_to_ned(np.array([1.0, 2.0, 3.0])), np.array([2.0, 1.0, -3.0]))
    assert np.allclose(_ned_to_enu(np.array([2.0, 1.0, -3.0])), np.array([1.0, 2.0, 3.0]))

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeKin:
        # NED kinematics: drone is at NED (4, 3, -1) → ENU (3, 4, 1).
        class _V:
            x_val = 4.0
            y_val = 3.0
            z_val = -1.0
        position = _V()
        linear_velocity = _V()

    class FakeState:
        kinematics_estimated = FakeKin()

    class FakeCollision:
        has_collided = False

    class FakeClient:
        def __init__(self) -> None:
            self.commands = []  # capture moveByVelocityAsync args

        def confirmConnection(self) -> None:
            pass

        def enableApiControl(self, _on: bool, _vehicle: str) -> None:
            pass

        def armDisarm(self, _on: bool, _vehicle: str) -> None:
            pass

        def reset(self) -> None:
            pass

        def simSetVehiclePose(self, *_args, **_kwargs) -> None:  # noqa: D401
            pass

        def simPause(self, _on: bool) -> None:
            pass

        def simContinueForTime(self, _dt: float) -> None:
            pass

        def moveByVelocityAsync(self, vx, vy, vz, dt, vehicle_name=None):
            self.commands.append((vx, vy, vz, dt, vehicle_name))

            class _Future:
                def join(self) -> None:
                    pass

            return _Future()

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return FakeState()

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return FakeCollision()

    fake = FakeClient()
    bridge = AirSimBridge(dt=0.05, scenario=sc, client=fake)
    state = bridge.reset()
    assert state.position.shape[0] == 2

    # ENU velocity (1, 2) → NED (2, 1, 0). 2D scenario pads vz=0.
    out_state, info = bridge.step(np.array([1.0, 2.0]))
    last = fake.commands[-1]
    assert last[0] == 2.0  # NED x = ENU y
    assert last[1] == 1.0  # NED y = ENU x
    assert last[2] == 0.0  # 2D scenario → vz = 0
    # Returned state in ENU (3, 4) [from FakeKin (4, 3, -1) NED].
    assert np.allclose(out_state.position, np.array([3.0, 4.0]))
    assert info.collision is False


def test_airsim_lidar_to_pointcloud_occupancy_to_planner_pipeline_via_mocks() -> None:
    """Guard rail for the AirSim-LiDAR perception pipeline (PRs #4 / #5 / #6).

    Drives the inner experiment loop with a mock AirSim client + the real
    AirSimBridge + the real pointcloud_occupancy sensor + a spy planner.
    Verifies the per-step LiDAR returns flow:
      AirSim mock → bridge.step → state.extra → recorder JSON summary
      AirSim mock → bridge.step → state.extra → sensor.observe_map →
        perceived_map the spy planner sees
    Each layer is unit-tested separately; this test catches regressions
    where the layers stop composing."""
    from types import SimpleNamespace

    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _run_episode
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0],
         "obstacles": {"type": "none"}, "resolution": 1.0}
    )

    class FakeKin:
        # Drone parks at NED (1, 1, 0) → ENU (1, 1, 0) every step. The
        # static position keeps the lidar hits landing on the same cells
        # each iteration so the assertions stay simple.
        class _V:
            x_val = 1.0
            y_val = 1.0
            z_val = 0.0
        position = _V()
        linear_velocity = _V()

    class FakeClient:
        def __init__(self) -> None:
            self.lidar_calls: list[str] = []

        def confirmConnection(self): pass
        def enableApiControl(self, _o, _v): pass
        def armDisarm(self, _o, _v): pass
        def reset(self): pass
        def simSetVehiclePose(self, *_a, **_k): pass
        def simPause(self, _o): pass
        def simContinueForTime(self, _dt): pass

        def moveByVelocityAsync(self, *_a, **_k):
            class _F:
                def join(self): pass
            return _F()

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(kinematics_estimated=FakeKin())

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(has_collided=False)

        def getLidarData(self, name, vehicle_name=None):  # noqa: ARG002
            self.lidar_calls.append(name)
            # NED (1, 0, 0), (0, 1, 0) → ENU (0, 1, 0), (1, 0, 0).
            # Drone at world (1, 1) → world cells [1, 2] and [2, 1].
            return SimpleNamespace(point_cloud=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    fake = FakeClient()
    bridge = AirSimBridge(dt=0.05, scenario=sc, client=fake, lidars=["L1"])

    pc_sensor_cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = pc_sensor_cls.from_config({"resolution": 1.0, "memory": True})

    captured = {"perceived_map": None, "calls": 0}

    class SpyPlanner:
        max_speed = 1.0

        def reset(self) -> None: pass

        def plan(self, observation, goal, perceived_map, dynamic_obstacles=None):  # noqa: ARG002
            captured["perceived_map"] = np.asarray(perceived_map).copy()
            captured["calls"] += 1
            wpts = np.array([observation, goal[: observation.shape[0]]])
            return Plan(waypoints=wpts, meta={"status": "ok"})

    rec = _run_episode(
        sim=bridge, planner=SpyPlanner(), sensor=sensor,
        seed=0, replan_period=0.05, max_steps=3, episode_index=0,
    )

    # Lidar polled at every bridge.step (3 steps).
    assert fake.lidar_calls == ["L1", "L1", "L1"]
    # Recorder surfaced the per-step lidar count summary on every row.
    assert len(rec.steps) == 3
    assert all(s["lidar_points"] == {"L1": 2} for s in rec.steps)
    # Spy planner saw an occupancy grid with the lidar-derived hits.
    pm = captured["perceived_map"]
    assert pm is not None
    assert pm.shape == (10, 10)
    assert pm[1, 2] and pm[2, 1]
    assert pm.sum() == 2
    # Replan ran each step (replan_period == dt), so the planner saw the
    # accumulated map every time, not a stale snapshot.
    assert captured["calls"] == 3


def test_airsim_bridge_polls_cameras_and_stashes_png_bytes_via_mock_client() -> None:
    """When `cameras: [{name, image_type}]` is configured, AirSimBridge.step()
    should call client.simGetImages() with a list of airsim.ImageRequest
    objects and stash the response bytes at state.extra["camera_images"][name].

    The bridge lazy-imports airsim only when cameras are configured; the
    test therefore injects a minimal fake airsim module into sys.modules
    so the import succeeds and ImageRequest construction works."""
    import sys
    from types import ModuleType, SimpleNamespace

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    # Minimal stand-in for `airsim.ImageRequest` / `airsim.ImageType` so
    # `_build_image_requests` does not have to know about CI vs prod.
    class _ImgType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        DepthPlanar = 1
        Segmentation = 5
        SurfaceNormals = 6
        Infrared = 7

    class _ImgReq:
        def __init__(self, camera_name, image_type, pixels_as_float, compress):  # noqa: ARG002
            self.camera_name = camera_name
            self.image_type = image_type
            self.compress = compress

    # `reset()` also touches airsim.Pose / Vector3r / to_quaternion for the
    # teleport step, so the fake module needs those too — otherwise the
    # narrow `except ImportError` around teleport leaks an AttributeError.
    class _Vec3:
        def __init__(self, x, y, z):
            self.x_val, self.y_val, self.z_val = x, y, z

    class _Pose:
        def __init__(self, position, orientation):
            self.position, self.orientation = position, orientation

    fake_airsim = ModuleType("airsim")
    fake_airsim.ImageType = _ImgType
    fake_airsim.ImageRequest = _ImgReq
    fake_airsim.Vector3r = _Vec3
    fake_airsim.Pose = _Pose
    fake_airsim.to_quaternion = lambda *_a, **_k: object()
    saved = sys.modules.get("airsim")
    sys.modules["airsim"] = fake_airsim
    try:
        class FakeKin:
            class _V:
                x_val = 0.0
                y_val = 0.0
                z_val = 0.0
            position = _V()
            linear_velocity = _V()

        captured_requests: list[list[Any]] = []  # noqa: F821

        class FakeClient:
            def confirmConnection(self): pass
            def enableApiControl(self, _o, _v): pass
            def armDisarm(self, _o, _v): pass
            def reset(self): pass
            def simSetVehiclePose(self, *_a, **_k): pass
            def simPause(self, _o): pass
            def simContinueForTime(self, _dt): pass
            def moveByVelocityAsync(self, *_a, **_k):
                class _F:
                    def join(self): pass
                return _F()
            def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
                return SimpleNamespace(kinematics_estimated=FakeKin())
            def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
                return SimpleNamespace(has_collided=False)
            def simGetImages(self, requests, vehicle_name=None):  # noqa: ARG002
                captured_requests.append(list(requests))
                # Two cameras configured → two responses with distinct PNG bytes.
                return [
                    SimpleNamespace(image_data_uint8=b"PNG_BYTES_FRONT"),
                    SimpleNamespace(image_data_uint8=b"PNG_BYTES_DEPTH"),
                ]

        fake = FakeClient()
        bridge = AirSimBridge(
            dt=0.05, scenario=sc, client=fake,
            cameras=[
                {"name": "front_center", "image_type": "scene"},
                {"name": "front_depth", "image_type": "depth_vis"},
            ],
        )
        bridge.reset()
        out_state, _ = bridge.step(np.array([0.0, 0.0]))

        # simGetImages received two ImageRequests with the right names + types.
        assert len(captured_requests) == 1
        reqs = captured_requests[0]
        assert reqs[0].camera_name == "front_center"
        assert reqs[0].image_type == _ImgType.Scene
        assert reqs[0].compress is True
        assert reqs[1].camera_name == "front_depth"
        assert reqs[1].image_type == _ImgType.DepthVis

        # Both PNG payloads landed in state.extra under their camera names.
        cams = out_state.extra["camera_images"]
        assert cams["front_center"] == b"PNG_BYTES_FRONT"
        assert cams["front_depth"] == b"PNG_BYTES_DEPTH"
    finally:
        if saved is None:
            del sys.modules["airsim"]
        else:
            sys.modules["airsim"] = saved


def test_runner_saves_camera_frames_to_disk_when_configured(tmp_path: Path) -> None:
    """When `output.save_camera_frames: true`, the runner should write each
    step's camera_images bytes to `<run_dir>/frames_NNN/step_NNNN_<name>.png`."""
    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _run_episode

    class FakeSim:
        # Point-mass-like stub that reports collision/goal_reached on a fixed
        # step so the episode terminates predictably.
        dt = 0.05
        def __init__(self) -> None:
            self.scenario = SimpleNamespace(  # noqa: F821
                dynamic_obstacles=[], ndim=2,
            )
            self.obstacle_map = np.zeros((10, 10), dtype=bool)
            self.goal = np.array([9.0, 9.0])
            self._t = 0.0
            self._step = 0
        def reset(self, *, seed=None):  # noqa: ARG002
            self._t = 0.0
            self._step = 0
            from uav_nav_lab.sim.base import SimState
            return SimState(t=0.0, position=np.array([1.0, 1.0]),
                            velocity=np.zeros(2),
                            extra={"camera_images": {"cam0": b"PNG_INIT"}})
        def step(self, _cmd):
            from uav_nav_lab.sim.base import SimState, SimStepInfo
            self._t += self.dt
            self._step += 1
            ns = SimState(
                t=self._t, position=np.array([1.0, 1.0]), velocity=np.zeros(2),
                extra={"camera_images": {"cam0": f"PNG_{self._step:04d}".encode()}},
            )
            done = self._step >= 3
            return ns, SimStepInfo(collision=False, goal_reached=done, truncated=False)

    class SpyPlanner:
        max_speed = 1.0
        def reset(self): pass
        def plan(self, observation, goal, perceived_map, dynamic_obstacles=None):  # noqa: ARG002
            wpts = np.array([observation, goal[: observation.shape[0]]])
            return Plan(waypoints=wpts, meta={"status": "ok"})

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("perfect").from_config({})
    from types import SimpleNamespace  # noqa: F811
    sim = FakeSim()
    fdir = tmp_path / "frames_000"
    rec = _run_episode(
        sim=sim, planner=SpyPlanner(), sensor=sensor,
        seed=0, replan_period=0.05, max_steps=10, episode_index=0,
        frame_dir=fdir,
    )
    assert rec.outcome == "success"
    # 3 steps logged, 3 frame files written (step indices 0000-0002).
    assert sorted(p.name for p in fdir.iterdir()) == [
        "step_0000_cam0.png", "step_0001_cam0.png", "step_0002_cam0.png",
    ]
    # Bytes round-trip — recorder writes verbatim, no re-encoding.
    assert (fdir / "step_0000_cam0.png").read_bytes() == b"PNG_0001"
    assert (fdir / "step_0002_cam0.png").read_bytes() == b"PNG_0003"


def test_video_stitch_run_produces_one_mp4_per_episode_camera(tmp_path: Path) -> None:
    """Smoke test the `uav-nav video` end of the pipeline: write a few
    valid PNGs into a frames_NNN/ directory and verify `stitch_run`
    emits the expected mp4 path. Skip if ffmpeg or PIL is missing —
    both are realistic prereqs for the actual user workflow."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed; skipping video stitching test")
    pil_image = pytest.importorskip("PIL.Image")

    from uav_nav_lab.video import stitch_run

    # h264 needs even dimensions; 16×16 is small enough to keep the test
    # fast but real enough to encode cleanly.
    run_dir = tmp_path / "run"
    fdir = run_dir / "frames_000"
    fdir.mkdir(parents=True)
    for i in range(3):
        img = pil_image.new("RGB", (16, 16), color=(i * 80, 100, 200))
        img.save(fdir / f"step_{i:04d}_cam0.png")

    saved = stitch_run(run_dir, fps=10)
    assert len(saved) == 1
    assert saved[0] == run_dir / "episode_000_cam0.mp4"
    assert saved[0].stat().st_size > 0


def test_airsim_camera_to_video_end_to_end_via_mocks(tmp_path: Path) -> None:
    """End-to-end: a real `AirSimBridge` (with an injected fake airsim
    client returning *valid* PIL-encoded PNG bytes) drives `_run_episode`
    with `frame_dir=tmp_path/frames_000`; the recorder writes those bytes
    verbatim to disk; `stitch_run` then ffmpegs them into an MP4.

    This is the only test that exercises *all three* layers
    (bridge → runner frame writer → ffmpeg encoder) against the same
    bytes, catching any contract drift between them."""
    import io
    import shutil
    import sys
    from types import ModuleType, SimpleNamespace

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed; skipping camera→video e2e test")
    pil_image = pytest.importorskip("PIL.Image")

    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _run_episode
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge
    from uav_nav_lab.video import stitch_run

    # Build N distinct VALID PNG buffers up front so the fake client just
    # serves them back per step. h264 needs even dimensions — 16×16 is the
    # same size used by the pure-stitch test above.
    def _make_png(color: tuple[int, int, int]) -> bytes:
        buf = io.BytesIO()
        pil_image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
        return buf.getvalue()

    # 4 steps × 1 camera, varying color so the encoded video is non-trivial.
    canned_pngs = [_make_png((i * 60, 100, 200)) for i in range(4)]

    # --- inject minimal fake airsim module (mirrors the AirSim camera test) ---
    # Bridge's `_build_image_requests` looks up every entry of its
    # `image_type` map at module-import time on `airsim.ImageType`, so the
    # fake must carry all of them even though we only request `scene`.
    class _ImgType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        DepthPlanar = 1
        Segmentation = 5
        SurfaceNormals = 6
        Infrared = 7
    class _ImgReq:
        def __init__(self, camera_name, image_type, pixels_as_float, compress):  # noqa: ARG002
            self.camera_name = camera_name
            self.image_type = image_type
            self.compress = compress
    class _Vec3:
        def __init__(self, x, y, z):
            self.x_val, self.y_val, self.z_val = x, y, z
    class _Pose:
        def __init__(self, position, orientation):
            self.position, self.orientation = position, orientation
    fake_airsim = ModuleType("airsim")
    fake_airsim.ImageType = _ImgType
    fake_airsim.ImageRequest = _ImgReq
    fake_airsim.Vector3r = _Vec3
    fake_airsim.Pose = _Pose
    fake_airsim.to_quaternion = lambda *_a, **_k: object()
    saved_mod = sys.modules.get("airsim")
    sys.modules["airsim"] = fake_airsim

    try:
        # Drive the FakeClient through 4 simGetImages() calls then collide.
        # 4 frames is enough to verify the index→bytes mapping survives
        # bridge→recorder→writer without becoming flaky on fast machines.
        class FakeKin:
            class _V:
                x_val = 0.0
                y_val = 0.0
                z_val = 0.0
            position = _V()
            linear_velocity = _V()

        class FakeClient:
            def __init__(self) -> None:
                self._call = 0
            def confirmConnection(self): pass
            def enableApiControl(self, _o, _v): pass
            def armDisarm(self, _o, _v): pass
            def reset(self): pass
            def simSetVehiclePose(self, *_a, **_k): pass
            def simPause(self, _o): pass
            def simContinueForTime(self, _dt): pass
            def moveByVelocityAsync(self, *_a, **_k):
                class _F:
                    def join(self): pass
                return _F()
            def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
                return SimpleNamespace(kinematics_estimated=FakeKin())
            def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
                # End the episode on step 4 so we get exactly 4 frames written.
                hit = self._call >= len(canned_pngs)
                return SimpleNamespace(has_collided=hit)
            def simGetImages(self, requests, vehicle_name=None):  # noqa: ARG002
                idx = min(self._call, len(canned_pngs) - 1)
                self._call += 1
                return [SimpleNamespace(image_data_uint8=canned_pngs[idx])]

        grid_cls = SCENARIO_REGISTRY.get("grid_world")
        sc = grid_cls.from_config(
            {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0],
             "obstacles": {"type": "none"}}
        )

        bridge = AirSimBridge(
            dt=0.05, scenario=sc, client=FakeClient(),
            cameras=[{"name": "front_center", "image_type": "scene"}],
        )

        class StubPlanner:
            max_speed = 1.0
            def reset(self): pass
            def plan(self, observation, goal, perceived_map, dynamic_obstacles=None):  # noqa: ARG002
                wpts = np.array([observation, goal[: observation.shape[0]]])
                return Plan(waypoints=wpts, meta={"status": "ok"})

        sensor = SENSOR_REGISTRY.get("perfect").from_config({})
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        fdir = run_dir / "frames_000"

        rec = _run_episode(
            sim=bridge, planner=StubPlanner(), sensor=sensor,
            seed=0, replan_period=0.05, max_steps=20, episode_index=0,
            frame_dir=fdir,
        )
        # Episode terminated on the simulated collision after 4 successful
        # camera polls; recorder+writer should have produced 4 PNGs.
        assert rec.outcome == "collision"
        png_files = sorted(fdir.glob("step_*_front_center.png"))
        assert len(png_files) == 4
        # Each on-disk PNG decodes back to the canned image of the same step.
        for i, p in enumerate(png_files):
            decoded = pil_image.open(p)
            assert decoded.size == (16, 16)
            assert decoded.getpixel((0, 0)) == (i * 60, 100, 200)

        # Now stitch them into an MP4 — this is the part that catches any
        # bytes-validity bug between the bridge layer and ffmpeg.
        saved = stitch_run(run_dir, fps=10)
        assert len(saved) == 1
        assert saved[0] == run_dir / "episode_000_front_center.mp4"
        assert saved[0].stat().st_size > 0
    finally:
        if saved_mod is None:
            del sys.modules["airsim"]
        else:
            sys.modules["airsim"] = saved_mod


def test_pointcloud_occupancy_marks_world_cells_from_local_points() -> None:
    """Drone at world (5, 5); lidar reports local points (1, 0, 0) and
    (-1, 1, 0) → world (6, 5) and (4, 6) → cells [6][5] and [4][6]
    flipped on the (10, 10) occupancy grid. Other cells stay free."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True})
    sensor.reset()

    occ_shape = (10, 10)
    base = np.zeros(occ_shape, dtype=bool)
    pos = np.array([5.0, 5.0])
    cloud = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
    out = sensor.observe_map(
        t=0.0, true_position=pos, true_obstacle_map=base,
        sim_extra={"lidar_points": {"FrontLidar": cloud}},
    )
    assert out.shape == occ_shape
    assert out[6, 5]
    assert out[4, 6]
    # exactly 2 cells set
    assert out.sum() == 2


def test_pointcloud_occupancy_memory_accumulates_then_clears_when_off() -> None:
    """memory=True sweeps OR together; memory=False keeps only the latest."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])

    s_mem = cls.from_config({"resolution": 1.0, "memory": True})
    s_mem.reset()
    s_mem.observe_map(0.0, pos, base, sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 0.0]])}})
    out = s_mem.observe_map(0.05, pos, base, sim_extra={"lidar_points": {"L": np.array([[-1.0, 0.0, 0.0]])}})
    assert out.sum() == 2  # both points retained

    s_nom = cls.from_config({"resolution": 1.0, "memory": False})
    s_nom.reset()
    s_nom.observe_map(0.0, pos, base, sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 0.0]])}})
    out = s_nom.observe_map(0.05, pos, base, sim_extra={"lidar_points": {"L": np.array([[-1.0, 0.0, 0.0]])}})
    assert out.sum() == 1  # only the latest sweep


def test_pointcloud_occupancy_handles_empty_or_missing_extras() -> None:
    """No sim_extra / no lidar_points / unknown lidar name → empty grid
    (or accumulated memory). Should not crash and should not pretend to
    see ground-truth occupancy."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True})
    sensor.reset()
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])

    assert sensor.observe_map(0.0, pos, base, sim_extra=None).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={}).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={"lidar_points": {}}).sum() == 0
    # malformed point cloud (not (N, 3)) → silently skipped
    assert sensor.observe_map(
        0.0, pos, base, sim_extra={"lidar_points": {"L": np.array([1.0, 2.0])}}
    ).sum() == 0


def test_pointcloud_occupancy_3d_grid_uses_z_component() -> None:
    """In 3D scenarios the sensor should index into 3D occupancy with
    the world-frame z component, not silently project to 2D."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True})
    sensor.reset()
    base = np.zeros((10, 10, 8), dtype=bool)
    pos = np.array([5.0, 5.0, 4.0])
    # local (1, 0, 1) → world (6, 5, 5) → cell [6][5][5]
    out = sensor.observe_map(
        0.0, pos, base,
        sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 1.0]])}},
    )
    assert out.shape == (10, 10, 8)
    assert out[6, 5, 5]
    assert out.sum() == 1


def test_pointcloud_occupancy_inflate_dilates_each_hit() -> None:
    """`inflate: 1` should add a 1-cell ring around every hit cell
    (cross-shaped via separable shifts; 4 neighbors in 2D)."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("pointcloud_occupancy")
    sensor = cls.from_config({"resolution": 1.0, "memory": True, "inflate": 1})
    sensor.reset()
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])
    # Single point → cell [6, 5] + 4-cell ring around it.
    out = sensor.observe_map(
        0.0, pos, base,
        sim_extra={"lidar_points": {"L": np.array([[1.0, 0.0, 0.0]])}},
    )
    assert out[6, 5]
    assert out[5, 5] and out[7, 5] and out[6, 4] and out[6, 6]
    assert out.sum() == 5  # center + 4 neighbors, no diagonals


def test_depth_image_occupancy_projects_pixel_to_correct_world_cell() -> None:
    """Drone at world (5, 5), 2D occupancy. A single non-sky pixel at
    column 40 row 24 with depth 3 m, intrinsics fx=fy=32, cx=32, cy=24:
        camera frame: x = (40-32) * 3 / 32 = 0.75, y = 0, z = 3
    With identity rotation that's body (0.75, 0, 3); projecting to 2D
    occupancy (xy) with resolution 1.0 lands the hit at cell (5, 5)
    (5 + 0.75 → floor → 5). All other cells stay free."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": True, "stride": 1, "max_depth": 10.0}
    )
    sensor.reset()
    depth = np.full((48, 64), 100.0, dtype=np.float32)  # everything is sky
    depth[24, 40] = 3.0
    base = np.zeros((20, 20), dtype=bool)
    out = sensor.observe_map(
        t=0.0, true_position=np.array([5.0, 5.0]), true_obstacle_map=base,
        sim_extra={"depth_images": {"front": {
            "depth": depth,
            "intrinsics": {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 24.0},
        }}},
    )
    assert out.sum() == 1
    assert out[5, 5]


def test_depth_image_occupancy_drops_out_of_range_pixels() -> None:
    """`max_depth: M` should drop pixels reporting d > M (sky / no-return).
    With max_depth=5 m and a depth image of all 8 m, the resulting
    occupancy must stay empty — no false positives from saturated pixels."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": True, "stride": 1, "max_depth": 5.0}
    )
    sensor.reset()
    depth = np.full((10, 10), 8.0, dtype=np.float32)  # all beyond max_depth
    base = np.zeros((20, 20), dtype=bool)
    out = sensor.observe_map(
        0.0, np.array([10.0, 10.0]), base,
        sim_extra={"depth_images": {"front": {
            "depth": depth,
            "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 5.0, "cy": 5.0},
        }}},
    )
    assert out.sum() == 0


def test_depth_image_occupancy_handles_missing_or_malformed_payload() -> None:
    """No sim_extra / no depth_images / missing intrinsics / wrong-shape
    depth array should all return the (memory-accumulated, possibly
    empty) grid without crashing — same forgiving behaviour as
    pointcloud_occupancy."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    sensor = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": True}
    )
    sensor.reset()
    base = np.zeros((10, 10), dtype=bool)
    pos = np.array([5.0, 5.0])

    assert sensor.observe_map(0.0, pos, base, sim_extra=None).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={}).sum() == 0
    assert sensor.observe_map(0.0, pos, base, sim_extra={"depth_images": {}}).sum() == 0
    # Missing intrinsics → silently skip.
    assert sensor.observe_map(
        0.0, pos, base, sim_extra={"depth_images": {"f": {"depth": np.ones((4, 4), np.float32)}}}
    ).sum() == 0
    # 1D depth array → silently skip.
    assert sensor.observe_map(
        0.0, pos, base, sim_extra={"depth_images": {"f": {
            "depth": np.ones(16, np.float32),
            "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 2.0},
        }}},
    ).sum() == 0


def test_depth_image_occupancy_stride_subsamples_for_compute() -> None:
    """`stride: 2` halves the pixel grid in each axis. A 4x4 depth image
    with all pixels valid should mark *fewer* cells when stride=2 vs
    stride=1 (cost-vs-coverage tradeoff)."""
    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("depth_image_occupancy")
    base = np.zeros((30, 30), dtype=bool)
    pos = np.array([15.0, 15.0])
    # 16 unique depths so each pixel projects to a distinct (X, Y).
    depth = np.linspace(1.0, 5.0, 16, dtype=np.float32).reshape(4, 4)
    payload = {"f": {
        "depth": depth,
        "intrinsics": {"fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 2.0},
    }}
    s1 = cls.from_config({"resolution": 0.5, "memory": True, "stride": 1, "max_depth": 10.0})
    s1.reset()
    s2 = cls.from_config({"resolution": 0.5, "memory": True, "stride": 2, "max_depth": 10.0})
    s2.reset()
    n1 = int(s1.observe_map(0.0, pos, base, sim_extra={"depth_images": payload}).sum())
    n2 = int(s2.observe_map(0.0, pos, base, sim_extra={"depth_images": payload}).sum())
    assert n1 > 0
    assert n2 > 0
    assert n2 < n1  # subsampling marks fewer cells


def test_airsim_bridge_polls_depth_cameras_and_stashes_float_depth_via_mock_client() -> None:
    """When `depths: [{name, fov_deg, width, height}]` is configured,
    AirSimBridge.step() should call client.simGetImages() with
    pixels_as_float=True and stash {depth, intrinsics} at
    state.extra["depth_images"][name]. Intrinsics derive from the
    configured fov + image size."""
    import sys
    from types import ModuleType, SimpleNamespace

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class _ImgType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        DepthPlanar = 1
        Segmentation = 5
        SurfaceNormals = 6
        Infrared = 7
    class _ImgReq:
        def __init__(self, camera_name, image_type, pixels_as_float, compress):  # noqa: ARG002
            self.camera_name = camera_name
            self.image_type = image_type
            self.pixels_as_float = pixels_as_float
            self.compress = compress
    class _Vec3:
        def __init__(self, x, y, z): self.x_val, self.y_val, self.z_val = x, y, z
    class _Pose:
        def __init__(self, position, orientation): self.position, self.orientation = position, orientation
    fake_airsim = ModuleType("airsim")
    fake_airsim.ImageType = _ImgType
    fake_airsim.ImageRequest = _ImgReq
    fake_airsim.Vector3r = _Vec3
    fake_airsim.Pose = _Pose
    fake_airsim.to_quaternion = lambda *_a, **_k: object()
    saved = sys.modules.get("airsim")
    sys.modules["airsim"] = fake_airsim
    try:
        captured: list[list[Any]] = []  # noqa: F821

        class FakeKin:
            class _V:
                x_val = 0.0
                y_val = 0.0
                z_val = 0.0
            position = _V()
            linear_velocity = _V()

        class FakeClient:
            def confirmConnection(self): pass
            def enableApiControl(self, _o, _v): pass
            def armDisarm(self, _o, _v): pass
            def reset(self): pass
            def simSetVehiclePose(self, *_a, **_k): pass
            def simPause(self, _o): pass
            def simContinueForTime(self, _dt): pass
            def moveByVelocityAsync(self, *_a, **_k):
                class _F:
                    def join(self): pass
                return _F()
            def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
                return SimpleNamespace(kinematics_estimated=FakeKin())
            def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
                return SimpleNamespace(has_collided=False)
            def simGetImages(self, requests, vehicle_name=None):  # noqa: ARG002
                captured.append(list(requests))
                # Return one float-pixel response per request: 8x6 = 48 floats.
                return [SimpleNamespace(image_data_float=[2.5] * (8 * 6))]

        bridge = AirSimBridge(
            dt=0.05, scenario=sc, client=FakeClient(),
            depths=[{
                "name": "fwd", "image_type": "depth_planar",
                "fov_deg": 90.0, "width": 8, "height": 6,
            }],
        )
        bridge.reset()
        out_state, _ = bridge.step(np.array([0.0, 0.0]))

        # The right ImageRequest was issued: depth_planar + float + uncompressed.
        assert len(captured) == 1
        req = captured[0][0]
        assert req.camera_name == "fwd"
        assert req.image_type == _ImgType.DepthPlanar
        assert req.pixels_as_float is True
        assert req.compress is False

        # Payload landed correctly under state.extra["depth_images"][name].
        depth_bag = out_state.extra["depth_images"]
        payload = depth_bag["fwd"]
        assert payload["depth"].shape == (6, 8)
        assert float(payload["depth"][0, 0]) == pytest.approx(2.5)
        # 90deg fov on 8 px wide → fx = 4 / tan(45) = 4. cx = 4, cy = 3.
        assert payload["intrinsics"]["fx"] == pytest.approx(4.0)
        assert payload["intrinsics"]["fy"] == pytest.approx(4.0)
        assert payload["intrinsics"]["cx"] == pytest.approx(4.0)
        assert payload["intrinsics"]["cy"] == pytest.approx(3.0)
    finally:
        if saved is None:
            del sys.modules["airsim"]
        else:
            sys.modules["airsim"] = saved


def test_recorder_summarizes_lidar_points_into_step_row() -> None:
    """When a sim backend populates state.extra['lidar_points'] with
    name-keyed (N, 3) arrays, EpisodeRecorder.log_step should surface
    {name: count} into the step row so episode JSONs show that lidar
    was actually being polled. Full clouds stay in memory only."""
    from uav_nav_lab.recorder import EpisodeRecorder

    rec = EpisodeRecorder(episode_index=0, seed=0)
    pos = np.array([1.0, 2.0])
    cloud_a = np.zeros((42, 3))
    cloud_b = np.zeros((7, 3))

    # Step 1: lidar populated → counts persisted.
    rec.log_step(
        t=0.0, true_pos=pos, true_vel=pos, observed_pos=pos, cmd=pos,
        info={"collision": False, "goal_reached": False},
        sim_extra={"lidar_points": {"FrontLidar": cloud_a, "RearLidar": cloud_b}},
    )
    # Step 2: empty extra dict → no lidar key in row.
    rec.log_step(
        t=0.05, true_pos=pos, true_vel=pos, observed_pos=pos, cmd=pos,
        info={"collision": False, "goal_reached": False},
        sim_extra={},
    )
    # Step 3: extra carries something else but no lidar → no lidar key.
    rec.log_step(
        t=0.10, true_pos=pos, true_vel=pos, observed_pos=pos, cmd=pos,
        info={"collision": False, "goal_reached": False},
        sim_extra={"depth_image": "ignored"},
    )

    assert rec.steps[0]["lidar_points"] == {"FrontLidar": 42, "RearLidar": 7}
    assert "lidar_points" not in rec.steps[1]
    assert "lidar_points" not in rec.steps[2]


def test_airsim_bridge_polls_lidar_and_converts_to_enu_via_mock_client() -> None:
    """When `lidars: [name]` is configured, AirSimBridge.step() should call
    client.getLidarData(name) and stash an (N, 3) ENU point cloud at
    state.extra['lidar_points'][name] — converted from AirSim's NED
    (x, y, z) → (y, x, -z) the same way poses are."""
    from types import SimpleNamespace

    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.airsim_bridge import AirSimBridge, _ned_pointcloud_to_enu

    # Helper sanity: NED [(1,2,-3), (4,5,-6)] flat → ENU [(2,1,3), (5,4,6)].
    pts = _ned_pointcloud_to_enu([1.0, 2.0, -3.0, 4.0, 5.0, -6.0])
    assert pts.shape == (2, 3)
    assert np.allclose(pts, np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]))
    # Empty / malformed readouts return shape (0, 3) instead of crashing.
    assert _ned_pointcloud_to_enu([]).shape == (0, 3)
    assert _ned_pointcloud_to_enu([1.0, 2.0]).shape == (0, 3)

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeKin:
        class _V:
            x_val = 0.0
            y_val = 0.0
            z_val = 0.0
        position = _V()
        linear_velocity = _V()

    class FakeClient:
        def __init__(self) -> None:
            self.lidar_calls: list[tuple[str, str | None]] = []

        def confirmConnection(self) -> None: pass
        def enableApiControl(self, _on, _vehicle): pass
        def armDisarm(self, _on, _vehicle): pass
        def reset(self): pass
        def simSetVehiclePose(self, *_a, **_kw): pass
        def simPause(self, _on): pass
        def simContinueForTime(self, _dt): pass

        def moveByVelocityAsync(self, *_a, **_kw):
            class _F:
                def join(self): pass
            return _F()

        def getMultirotorState(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(kinematics_estimated=FakeKin())

        def simGetCollisionInfo(self, vehicle_name=None):  # noqa: ARG002
            return SimpleNamespace(has_collided=False)

        def getLidarData(self, name, vehicle_name=None):
            self.lidar_calls.append((name, vehicle_name))
            # Two NED points that map to predictable ENU rows.
            return SimpleNamespace(point_cloud=[1.0, 2.0, -3.0, 4.0, 5.0, -6.0])

    fake = FakeClient()
    bridge = AirSimBridge(dt=0.05, scenario=sc, client=fake, lidars=["FrontLidar"])
    bridge.reset()
    out_state, _ = bridge.step(np.array([0.0, 0.0]))

    # Lidar polled with the configured name + vehicle.
    assert fake.lidar_calls == [("FrontLidar", "Drone1")]
    # Points landed in state.extra under the lidar name.
    cloud = out_state.extra["lidar_points"]["FrontLidar"]
    assert cloud.shape == (2, 3)
    assert np.allclose(cloud, np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]))


def test_ros2_bridge_step_round_trips_enu_via_mock_adapter() -> None:
    """Verify the ROS 2 bridge's publish-spin-read plumbing against an
    injected mock adapter — no rclpy install required."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeAdapter:
        def __init__(self) -> None:
            self.commands: list[tuple[float, float, float]] = []
            self.teleports: list[np.ndarray] = []
            # Canned ENU pose / velocity returned from /odom on every tick.
            self._pose = np.array([3.0, 4.0, 1.0])
            self._vel = np.array([0.5, 0.6, 0.0])
            self._collision = False

        def publish_velocity(self, vx: float, vy: float, vz: float) -> None:
            self.commands.append((vx, vy, vz))

        def latest_pose_velocity(self):
            return (self._pose.copy(), self._vel.copy())

        def latest_collision(self) -> bool:
            return self._collision

        def tick(self, _timeout_s: float) -> None:
            pass

        def teleport(self, pos_enu: np.ndarray) -> None:
            self.teleports.append(np.asarray(pos_enu).copy())

    fake = FakeAdapter()
    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=fake)
    state = bridge.reset()
    assert state.position.shape[0] == 2
    # reset() teleports the (3D-padded) start pose.
    assert len(fake.teleports) == 1
    assert np.allclose(fake.teleports[0][:2], np.array([1.0, 1.0]))
    # Initial state taken from the canned odom (ENU pass-through, no flip).
    assert np.allclose(state.position, np.array([3.0, 4.0]))

    # ENU velocity (1, 2) → adapter sees (1, 2, 0). No frame flip vs AirSim's NED.
    out_state, info = bridge.step(np.array([1.0, 2.0]))
    last = fake.commands[-1]
    assert last[0] == 1.0
    assert last[1] == 2.0
    assert last[2] == 0.0  # 2D scenario → vz = 0
    assert np.allclose(out_state.position, np.array([3.0, 4.0]))
    assert info.collision is False


def test_ros2_bridge_surfaces_lidar_camera_via_mock_adapter() -> None:
    """When `lidars`/`cameras` topics are configured, Ros2Bridge.step() should
    populate `state.extra['lidar_points'][topic]` and `['camera_images'][topic]`
    from the adapter — same keys as AirSimBridge so the pointcloud_occupancy
    sensor and `uav-nav video` CLI consume both backends transparently."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeAdapter:
        def __init__(self) -> None:
            self._pose = np.array([2.0, 3.0, 1.0])
            self._vel = np.array([0.0, 0.0, 0.0])
            self._clouds = {
                "/front_lidar": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5]], dtype=np.float32),
                "/rear_lidar": np.array([[-1.0, -1.0, 0.0]], dtype=np.float32),
            }
            self._images = {
                "/front_camera/image_raw": b"PNG_FRONT",
                "/down_camera/image_raw": b"PNG_DOWN",
            }

        def publish_velocity(self, vx, vy, vz):  # noqa: ARG002
            pass

        def latest_pose_velocity(self):
            return (self._pose.copy(), self._vel.copy())

        def latest_collision(self) -> bool:
            return False

        def tick(self, _timeout_s: float) -> None:
            pass

        def teleport(self, _pos_enu: np.ndarray) -> None:
            pass

        def latest_lidar_clouds(self):
            return {k: v.copy() for k, v in self._clouds.items()}

        def latest_camera_images(self):
            return dict(self._images)

    fake = FakeAdapter()
    bridge = Ros2Bridge(
        dt=0.05,
        scenario=sc,
        lidars=["/front_lidar", "/rear_lidar"],
        cameras=["/front_camera/image_raw", "/down_camera/image_raw"],
        adapter=fake,
    )
    bridge.reset()
    out_state, _ = bridge.step(np.array([0.0, 0.0]))

    clouds = out_state.extra["lidar_points"]
    assert set(clouds.keys()) == {"/front_lidar", "/rear_lidar"}
    assert clouds["/front_lidar"].shape == (2, 3)
    assert clouds["/rear_lidar"].shape == (1, 3)
    # Pass-through: bridge does NOT flip frames for ROS 2 (REP-103 is ENU).
    assert np.allclose(clouds["/front_lidar"][0], np.array([1.0, 0.0, 0.0]))

    cams = out_state.extra["camera_images"]
    assert cams["/front_camera/image_raw"] == b"PNG_FRONT"
    assert cams["/down_camera/image_raw"] == b"PNG_DOWN"


def test_ros2_bridge_omits_extras_when_lidars_cameras_not_configured() -> None:
    """If neither lidars nor cameras are configured the bridge must not
    poll those adapter methods — keeps the no-sensor case lightweight and
    means adapters that don't implement them still work."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class MinimalAdapter:
        # Deliberately omits latest_lidar_clouds / latest_camera_images.
        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def tick(self, _t): pass
        def teleport(self, _p): pass

    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=MinimalAdapter())
    bridge.reset()
    out_state, _ = bridge.step(np.array([0.0, 0.0]))
    assert "lidar_points" not in out_state.extra
    assert "camera_images" not in out_state.extra


def test_ros2_bridge_sim_time_advances_state_t_via_clock_not_wall() -> None:
    """When `use_sim_time: true`, Ros2Bridge.step() should advance
    `state.t` based on the adapter's `wait_for_sim_time_advance` return
    value rather than the wall-clock dt. Confirms the bridge prefers
    the sim's `/clock` over its own wall-clock counter — the load-bearing
    behaviour for PX4-SITL fast-forward."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class FakeSimTimeAdapter:
        """Mock adapter whose sim clock is driven by the bridge's wait
        calls rather than by wall-clock — `wait_for_sim_time_advance`
        instantly jumps the clock to the requested target. Lets us check
        the bridge wired sim-time through end-to-end without sleeping."""
        def __init__(self) -> None:
            self._sim_t = 0.0
            self.wait_targets: list[float] = []
            self.wait_timeouts: list[float] = []
            # Intentionally no `tick` method on this adapter so the test
            # would fail loudly if the bridge fell back to wall-clock.

        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def teleport(self, _p): pass
        def latest_sim_time(self): return self._sim_t

        def wait_for_sim_time_advance(self, *, target_time, wall_timeout):
            self.wait_targets.append(float(target_time))
            self.wait_timeouts.append(float(wall_timeout))
            # Simulate the sim's clock jumping forward to the target.
            self._sim_t = float(target_time)
            return self._sim_t

        # reset() still needs to spin once for the first odom — provide
        # tick() but assert it's only called from reset(), never from step().
        def tick(self, _t): pass

    fake = FakeSimTimeAdapter()
    bridge = Ros2Bridge(
        dt=0.05, scenario=sc, adapter=fake,
        use_sim_time=True, sim_time_wall_timeout=2.0,
    )
    bridge.reset()
    out_state_a, _ = bridge.step(np.array([0.0, 0.0]))
    out_state_b, _ = bridge.step(np.array([0.0, 0.0]))
    # state.t tracks sim-time, not the wall-clock the test ran at.
    assert out_state_a.t == pytest.approx(0.05)
    assert out_state_b.t == pytest.approx(0.10)
    # Targets re-anchor on the previously-observed clock value, not on
    # accumulated dt — protects against drift over a long episode.
    assert fake.wait_targets == [pytest.approx(0.05), pytest.approx(0.10)]
    # The configured wall-clock safety timeout reaches the adapter.
    assert all(t == pytest.approx(2.0) for t in fake.wait_timeouts)


def test_ros2_bridge_sim_time_falls_back_to_wall_clock_for_legacy_adapters() -> None:
    """A user-supplied mock adapter that doesn't implement
    `wait_for_sim_time_advance` should keep working — bridge should
    silently fall back to wall-clock `tick()` rather than crashing.
    Lets users opt into use_sim_time without re-writing existing
    test fakes."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class WallClockOnlyAdapter:
        # Note absence of wait_for_sim_time_advance / latest_sim_time.
        def __init__(self) -> None:
            self.tick_calls = 0
        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def teleport(self, _p): pass
        def tick(self, _t): self.tick_calls += 1

    fake = WallClockOnlyAdapter()
    bridge = Ros2Bridge(dt=0.1, scenario=sc, adapter=fake, use_sim_time=True)
    bridge.reset()
    out, _ = bridge.step(np.array([0.0, 0.0]))
    # tick() was called twice (once in reset, once in step's wall fallback).
    assert fake.tick_calls == 2
    # Wall-clock fallback advances state.t by the configured dt.
    assert out.t == pytest.approx(0.1)


def test_ros2_bridge_sim_time_disabled_uses_wall_clock_dt() -> None:
    """Default mode (`use_sim_time=False`) must keep the original
    behaviour — `state.t` advances by `dt` regardless of any sim-time
    methods on the adapter. Regression guard against accidentally
    routing default runs through the sim-time path."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    grid_cls = SCENARIO_REGISTRY.get("grid_world")
    sc = grid_cls.from_config(
        {"size": [10, 10], "start": [1.0, 1.0], "goal": [9.0, 9.0], "obstacles": {"type": "none"}}
    )

    class AmbiguousAdapter:
        # Implements *both* tick and the sim-time methods; the bridge
        # should still pick wall-clock when use_sim_time is off.
        def __init__(self) -> None:
            self.wait_calls = 0
            self.tick_calls = 0
        def publish_velocity(self, *_args): pass
        def latest_pose_velocity(self):
            return (np.zeros(3), np.zeros(3))
        def latest_collision(self): return False
        def teleport(self, _p): pass
        def tick(self, _t): self.tick_calls += 1
        def latest_sim_time(self): return 999.0
        def wait_for_sim_time_advance(self, **_kw):
            self.wait_calls += 1
            return 999.0

    fake = AmbiguousAdapter()
    bridge = Ros2Bridge(dt=0.05, scenario=sc, adapter=fake)  # use_sim_time defaults False
    bridge.reset()
    out, _ = bridge.step(np.array([0.0, 0.0]))
    assert fake.wait_calls == 0  # sim-time path NEVER taken
    assert fake.tick_calls == 2  # one in reset, one in step
    assert out.t == pytest.approx(0.05)


def test_rrt_star_returns_shorter_path_than_rrt_on_open_world() -> None:
    """RRT* rewiring should produce a path no longer than plain RRT on
    average. We compare on a wide-open world where rewiring has clear
    headroom (zigzag RRT path → near-straight RRT* path)."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((30, 30), dtype=bool)
    start = np.array([2.0, 2.0])
    goal = np.array([28.0, 28.0])

    rrt = PLANNER_REGISTRY.get("rrt").from_config(
        {"step_size": 1.5, "goal_tolerance": 1.0, "max_samples": 800, "seed": 1}
    )
    rrt_star = PLANNER_REGISTRY.get("rrt_star").from_config(
        {
            "step_size": 1.5,
            "rewire_radius": 4.0,
            "goal_tolerance": 1.0,
            "max_samples": 800,
            "seed": 1,
        }
    )
    p_rrt = rrt.plan(start, goal, occ)
    p_star = rrt_star.plan(start, goal, occ)
    assert p_rrt.meta["status"] == "ok"
    assert p_star.meta["status"] == "ok"

    def path_len(wps):
        return float(np.sum(np.linalg.norm(np.diff(wps, axis=0), axis=1)))

    # On open ground RRT* should not be longer than RRT (it's allowed to
    # be equal in the rare case both find the same path).
    assert path_len(p_star.waypoints) <= path_len(p_rrt.waypoints) + 1e-6
    # And it should report a path_cost in its metadata.
    assert "path_cost" in p_star.meta


def test_rrt_planner_finds_path_around_a_wall() -> None:
    """RRT should find *some* path around a wall and reach goal_tolerance."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    rrt_cls = PLANNER_REGISTRY.get("rrt")
    rrt = rrt_cls.from_config(
        {
            "max_speed": 10.0,
            "step_size": 2.0,
            "goal_tolerance": 1.5,
            "max_samples": 1000,
            "goal_bias": 0.2,
            "seed": 42,
        }
    )
    occ = np.zeros((20, 20), dtype=bool)
    occ[10, 5:15] = True  # wall down the middle with two openings
    plan = rrt.plan(np.array([2.0, 10.0]), np.array([18.0, 10.0]), occ)
    assert plan.meta["status"] == "ok"
    assert plan.waypoints.shape[0] >= 2
    # last waypoint must land within goal_tolerance of the goal
    last = plan.waypoints[-1]
    assert float(np.linalg.norm(last - np.array([18.0, 10.0]))) <= 1.5


def test_chomp_open_world_returns_essentially_straight_line() -> None:
    """With no obstacles the smoothness term dominates and CHOMP should
    converge to the straight-line trajectory (zero second-difference).
    Endpoints stay pinned at start / goal exactly."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 20, "n_iters": 50}
    )
    occ = np.zeros((20, 20), dtype=bool)
    plan = chomp.plan(np.array([1.0, 1.0]), np.array([18.0, 18.0]), occ)
    assert plan.meta["status"] == "ok"
    # Endpoints clamped to start / goal exactly.
    assert np.allclose(plan.waypoints[0], [1.0, 1.0])
    assert np.allclose(plan.waypoints[-1], [18.0, 18.0])
    # Straight line ⇒ second differences ≈ 0.
    assert float(np.linalg.norm(np.diff(plan.waypoints, n=2, axis=0))) < 1e-6


def test_chomp_routes_around_a_horizontal_bar() -> None:
    """A straight-line init from (2, 8) to (18, 12) crosses a horizontal
    bar at y=10 (x ∈ [5, 14]). CHOMP's local optimisation should detour
    *under* the bar (lower y) and report `status=ok`. Asymmetric start /
    goal y-coordinates break the symmetry that traps the box test."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 30, "n_iters": 100}
    )
    occ = np.zeros((20, 20), dtype=bool)
    occ[5:15, 10] = True
    plan = chomp.plan(np.array([2.0, 8.0]), np.array([18.0, 12.0]), occ)
    assert plan.meta["status"] == "ok"
    # No waypoint inside the raw obstacle.
    cells = np.clip(np.round(plan.waypoints).astype(int), 0, 19)
    assert int(occ[tuple(cells.T)].sum()) == 0
    # The detour reaches y ≤ 9 — actually goes below the bar.
    assert float(plan.waypoints[:, 1].min()) <= 9.0


def test_chomp_reports_local_minimum_when_init_cannot_escape() -> None:
    """Symmetric box: start (2, 15), goal (28, 15), box at x∈[10,19],
    y∈[10,19]. The straight-line init is symmetric across y=15 so the
    obstacle gradient cancels in the y-axis — CHOMP cannot decide to go
    up or down. The planner should detect this and return
    `status=local_minimum`, not silently produce a colliding plan."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 30, "n_iters": 100}
    )
    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True
    plan = chomp.plan(np.array([2.0, 15.0]), np.array([28.0, 15.0]), occ)
    assert plan.meta["status"] == "local_minimum"


def test_chomp_smoothness_hessian_inverse_keeps_step_stable_at_n50() -> None:
    """Plain GD on K diverges around n≈20 because λ_max(K) ≳ 16. The
    M⁻¹-preconditioned step should stay bounded for n=50 + 200 iters with
    a high obstacle weight. Regression guard against re-introducing the
    `K @ x` raw-gradient bug."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {"n_waypoints": 50, "n_iters": 200, "w_obs": 10.0}
    )
    occ = np.zeros((30, 30), dtype=bool)
    occ[14, 14] = True  # single-cell obstacle near the path
    plan = chomp.plan(np.array([2.0, 14.0]), np.array([28.0, 14.0]), occ)
    # No waypoint blew outside a generous box around the world.
    assert float(np.abs(plan.waypoints).max()) < 100.0
    # And no NaN / Inf.
    assert np.all(np.isfinite(plan.waypoints))


def test_chomp_init_rrt_escapes_box_that_traps_straight_init() -> None:
    """The symmetric box scenario locks straight-line init in a saddle
    (see test_chomp_reports_local_minimum_when_init_cannot_escape).
    `init: rrt` uses an RRT path as the warm start, which detours around
    the box, so CHOMP smooths a *collision-free* trajectory and reports
    `status=ok` — the whole point of the feature."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {
            "n_waypoints": 30,
            "n_iters": 100,
            "init": "rrt",
            "rrt_max_samples": 1000,
            "rrt_seed": 42,
            "rrt_goal_bias": 0.2,
        }
    )
    chomp.reset()
    plan = chomp.plan(np.array([2.0, 15.0]), np.array([28.0, 15.0]), occ)
    assert plan.meta["status"] == "ok"
    assert plan.meta["init"] == "rrt"
    cells = np.clip(np.round(plan.waypoints).astype(int), 0, 29)
    assert int(occ[tuple(cells.T)].sum()) == 0


def test_chomp_init_rrt_falls_back_to_straight_when_rrt_fails() -> None:
    """When the inner RRT exhausts max_samples without finding a path,
    CHOMP must keep working — fall back to straight-line init and report
    `init=rrt_fallback_straight` so the failure is observable in the
    plan log without crashing the run."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True

    chomp = PLANNER_REGISTRY.get("chomp").from_config(
        {
            "n_waypoints": 30,
            "n_iters": 50,
            "init": "rrt",
            "rrt_max_samples": 5,  # nowhere near enough to find a path
            "rrt_seed": 0,
        }
    )
    plan = chomp.plan(np.array([2.0, 15.0]), np.array([28.0, 15.0]), occ)
    assert plan.meta["init"] == "rrt_fallback_straight"
    assert plan.meta["status"] == "local_minimum"


def test_chomp_resample_polyline_uniform_arc_length() -> None:
    """The internal arc-length resampler should turn a 3-vertex L-shape
    (segments 10 + 6 = 16 m) into n equally-spaced points along the
    polyline. Index 10 of 17 sits exactly at the joint (10/16 of arc
    length)."""
    from uav_nav_lab.planner.chomp import _resample_polyline

    wps = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0]])
    out = _resample_polyline(wps, 17)
    assert out.shape == (17, 2)
    assert np.allclose(out[0], wps[0])
    assert np.allclose(out[-1], wps[-1])
    assert np.allclose(out[10], np.array([10.0, 0.0]))
    # Single-vertex degenerate case returns the start repeated.
    out2 = _resample_polyline(np.array([[3.0, 4.0]]), 5)
    assert out2.shape == (5, 2)
    assert np.allclose(out2, np.tile([3.0, 4.0], (5, 1)))


def test_chomp_init_invalid_value_raises() -> None:
    """Typo in the init field should fail loud at construction, not
    silently default — guards `init: 'RRT'` / `init: 'random'` from
    looking like they did something."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    with pytest.raises(ValueError, match="init must be"):
        PLANNER_REGISTRY.get("chomp").from_config({"init": "random"})


def test_chomp_registry_and_from_config_round_trip() -> None:
    """Registration + from_config wiring smoke test."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    cls = PLANNER_REGISTRY.get("chomp")
    chomp = cls.from_config(
        {
            "max_speed": 7.0,
            "n_waypoints": 25,
            "n_iters": 10,
            "learning_rate": 0.1,
            "w_obs": 3.0,
            "epsilon": 1.5,
            "resolution": 0.5,
            "inflate": 1,
        }
    )
    assert chomp.max_speed == 7.0
    assert chomp.n_waypoints == 25
    assert chomp.n_iters == 10
    assert chomp.epsilon == 1.5


def test_mpc_chomp_smooths_mpc_rollout_corners() -> None:
    """The MPC rollout is a piecewise-straight constant-velocity prediction;
    CHOMP smoothing should reduce the trajectory's second-difference norm
    versus the raw MPC waypoints (acceleration profile), without breaking
    the start position. We use an obstacle-free world so the only force on
    the optimiser is the smoothness term — the smoothed path stays on the
    same direct route as the raw MPC rollout."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    obs = np.array([2.0, 2.0])
    goal = np.array([18.0, 18.0])
    occ = np.zeros((20, 20), dtype=bool)

    mpc = PLANNER_REGISTRY.get("mpc").from_config(
        {"horizon": 30, "dt_plan": 0.05, "n_samples": 16, "max_speed": 5.0}
    )
    raw = mpc.plan(obs, goal, occ)

    hybrid = PLANNER_REGISTRY.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 30,
            "w_smooth": 1.0,
            "w_obs": 0.0,
            "mpc": {
                "horizon": 30,
                "dt_plan": 0.05,
                "n_samples": 16,
                "max_speed": 5.0,
            },
        }
    )
    smoothed = hybrid.plan(obs, goal, occ)

    # target_velocity must be cleared so the runner pure-pursues the
    # smoothed waypoints instead of the constant rollout velocity.
    assert smoothed.target_velocity is None
    assert smoothed.meta["smoothed"] is True
    # Same horizon-length output as the raw MPC rollout.
    assert smoothed.waypoints.shape == raw.waypoints.shape
    # Smoothness improved (lower second-difference norm).
    raw_acc = float(np.linalg.norm(np.diff(raw.waypoints, n=2, axis=0)))
    sm_acc = float(np.linalg.norm(np.diff(smoothed.waypoints, n=2, axis=0)))
    assert sm_acc <= raw_acc + 1e-6


def test_mpc_chomp_clears_target_velocity_for_pure_pursuit() -> None:
    """Even when the underlying MPC sets target_velocity (its normal mode),
    the wrapper must clear it so the runner falls back to pure-pursuit on
    the smoothed waypoints. Otherwise the smoothing is purely cosmetic."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((20, 20), dtype=bool)
    raw = PLANNER_REGISTRY.get("mpc").from_config(
        {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0}
    ).plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    # Sanity: raw MPC does set target_velocity.
    assert raw.target_velocity is not None

    hybrid = PLANNER_REGISTRY.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 5,
            "mpc": {"horizon": 20, "n_samples": 8, "max_speed": 5.0},
        }
    )
    plan = hybrid.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    assert plan.target_velocity is None


def test_mpc_chomp_short_rollout_passthrough() -> None:
    """When the MPC rollout has fewer than 3 waypoints there's nothing to
    smooth — wrapper should pass the plan through unchanged (preserving
    target_velocity) so the goal-reach behaviour is identical to plain MPC.
    Regression guard against trying to invert a 1×1 interior Hessian."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((20, 20), dtype=bool)
    # horizon=2 ⇒ MPC returns at most 2 waypoints (rollout[1:] of length 2),
    # which is below the wrapper's smoothing threshold.
    hybrid = PLANNER_REGISTRY.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "mpc": {
                "horizon": 2,
                "dt_plan": 0.05,
                "n_samples": 8,
                "max_speed": 5.0,
            },
        }
    )
    plan = hybrid.plan(np.array([1.0, 1.0]), np.array([18.0, 18.0]), occ)
    assert plan.waypoints.shape[0] == 2
    assert plan.meta["smoothed"] is False
    # Pass-through must preserve MPC's target_velocity (no pure-pursuit
    # fallback when there's nothing to smooth).
    assert plan.target_velocity is not None


def test_mpc_chomp_registry_and_from_config_round_trip() -> None:
    """Registration + from_config wiring smoke test."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    cls = PLANNER_REGISTRY.get("mpc_chomp")
    p = cls.from_config(
        {
            "max_speed": 7.0,
            "n_smooth_iters": 5,
            "w_obs": 3.0,
            "mpc": {"horizon": 10, "n_samples": 4, "max_speed": 7.0},
        }
    )
    assert p.max_speed == 7.0
    assert p.n_smooth_iters == 5
    assert p.w_obs == 3.0
    assert p.output == "waypoints"  # default


def test_mpc_chomp_velocity_profile_output_shape_and_dt() -> None:
    """`output: velocity_profile` must emit a (T, ndim) profile aligned to
    the MPC's dt_plan grid: one velocity per smoothed waypoint, each
    representing the displacement traveled in one dt_plan tick. T equals
    the smoothed waypoint count (forward differences from the obs+wps stack
    yield exactly len(waypoints) velocities)."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((20, 20), dtype=bool)
    p = PLANNER_REGISTRY.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 5,
            "output": "velocity_profile",
            "mpc": {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0},
        }
    )
    plan = p.plan(np.array([1.0, 1.0]), np.array([18.0, 18.0]), occ)
    assert plan.velocity_profile is not None
    assert plan.profile_dt == pytest.approx(0.05)
    assert plan.velocity_profile.shape == (plan.waypoints.shape[0], 2)
    # Must clear target_velocity so the runner picks the profile path.
    assert plan.target_velocity is None
    # Profile speeds within the planner's max_speed (no negative-step glitch).
    speeds = np.linalg.norm(plan.velocity_profile, axis=1)
    assert float(speeds.max()) <= 5.0 + 1e-6


def test_mpc_chomp_velocity_profile_invalid_output_raises() -> None:
    """Typo in `output` must fail loud at construction, mirroring the
    `init` validation in the inner CHOMP planner."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    with pytest.raises(ValueError, match="output must be"):
        PLANNER_REGISTRY.get("mpc_chomp").from_config(
            {"output": "spline", "mpc": {"horizon": 10, "n_samples": 4}}
        )


def test_follow_plan_velocity_profile_indexed_by_elapsed_time() -> None:
    """The runner's `_follow_plan` must pick the profile bin that matches
    `t_since_replan / profile_dt` and clip past the end. This is the
    machinery that lets a smoothing planner drive a varying velocity
    instead of one constant target_velocity."""
    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _follow_plan

    profile = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    plan = Plan(
        waypoints=np.zeros((3, 2)),
        velocity_profile=profile,
        profile_dt=0.1,
    )
    obs = np.array([0.0, 0.0])
    # t=0 → bin 0
    assert np.allclose(_follow_plan(plan, obs, max_speed=5.0, t_since_replan=0.0),
                       [1.0, 0.0])
    # t=0.15 → bin 1 (floor)
    assert np.allclose(_follow_plan(plan, obs, max_speed=5.0, t_since_replan=0.15),
                       [0.0, 1.0])
    # t past end → clip to last bin
    assert np.allclose(_follow_plan(plan, obs, max_speed=5.0, t_since_replan=10.0),
                       [-1.0, 0.0])
    # max_speed cap still applies (profile entry is unit norm; cap=0.5 halves it)
    assert np.allclose(_follow_plan(plan, obs, max_speed=0.5, t_since_replan=0.0),
                       [0.5, 0.0])


def test_mpc_chomp_w_action_jump_meta_and_round_trip() -> None:
    """w_action_jump round-trips through from_config, surfaces in meta on
    velocity_profile emit, and is accepted as a non-zero default. (The
    knob's *empirical effect* on smoothness is documented in the YAML
    header as a negative result — it modifies x[1] but the smoothness
    Hessian's neighbour coupling reverses the gain at sample 1.)"""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    p = PLANNER_REGISTRY.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "w_action_jump": 0.5,
            "output": "velocity_profile",
            "mpc": {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0},
        }
    )
    assert p.w_action_jump == 0.5
    occ = np.zeros((20, 20), dtype=bool)
    plan = p.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    assert plan.meta["w_action_jump"] == 0.5


def test_mpc_chomp_w_action_jump_only_active_after_first_replan() -> None:
    """Episode-start replan has prev_emitted=None so the jump cost is
    inactive — otherwise the first plan of every episode would be biased
    by stale state. Tests the dispatch guard and the reset() hook clearing
    the cache."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    occ = np.zeros((20, 20), dtype=bool)
    p = PLANNER_REGISTRY.get("mpc_chomp").from_config(
        {
            "max_speed": 5.0,
            "n_smooth_iters": 5,
            "w_action_jump": 100.0,  # huge — would dominate if active
            "output": "velocity_profile",
            "mpc": {"horizon": 20, "dt_plan": 0.05, "n_samples": 8, "max_speed": 5.0},
        }
    )
    plan1 = p.plan(np.array([2.0, 2.0]), np.array([18.0, 18.0]), occ)
    # First plan: prev was None, cache populated at emit time.
    assert p._prev_emitted_velocity is not None
    assert np.allclose(p._prev_emitted_velocity, plan1.velocity_profile[0])
    # reset() must clear the cache so a new episode starts unbiased.
    p.reset()
    assert p._prev_emitted_velocity is None


def test_follow_plan_velocity_profile_takes_priority_over_target_velocity() -> None:
    """If both velocity_profile and target_velocity are set on the same
    Plan, the profile must win — it's the more-specific signal. Regression
    guard for the dispatch order in `_follow_plan`."""
    from uav_nav_lab.planner.base import Plan
    from uav_nav_lab.runner.experiment import _follow_plan

    plan = Plan(
        waypoints=np.zeros((1, 2)),
        target_velocity=np.array([5.0, 0.0]),
        velocity_profile=np.array([[0.0, 3.0]]),
        profile_dt=0.1,
    )
    cmd = _follow_plan(plan, np.array([0.0, 0.0]), max_speed=10.0, t_since_replan=0.0)
    assert np.allclose(cmd, [0.0, 3.0])


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


def test_mppi_low_temperature_collapses_to_argmin() -> None:
    """As temperature → 0 the softmax weights become a one-hot at the
    argmin sample, so MPPI's chosen action should converge to MPC's. We
    verify with both planners on the same scenario at temperature=0.001 —
    the goal-direction sample dominates because the reach-goal bonus
    (-1e6) gives it a cost ~1e6 lower than any other."""
    from uav_nav_lab.planner.mpc import SamplingMPCPlanner
    from uav_nav_lab.planner.mppi import MPPIPlanner

    occ = np.zeros((30, 30), dtype=bool)
    obs = np.array([2.0, 2.0])
    goal = np.array([20.0, 20.0])

    args = dict(max_speed=5.0, horizon=20, n_samples=16, inflate=0)
    mpc = SamplingMPCPlanner(**args)
    mpc.reset()
    mpc_action = mpc.plan(obs, goal, occ).target_velocity

    mppi = MPPIPlanner(temperature=0.001, **args)
    mppi.reset()
    mppi_action = mppi.plan(obs, goal, occ).target_velocity

    # Both should pick the goal-direction action at ~max_speed.
    assert np.allclose(mppi_action, mpc_action, atol=0.1)


def test_mppi_high_temperature_attenuates_speed() -> None:
    """As temperature → ∞ the softmax weights flatten to uniform, so the
    chosen action converges to the *mean* of all sample directions. For
    an n=16 set of evenly-spread 2D directions the mean magnitude collapses
    toward zero — this is MPPI's signature failure mode (and why the
    framework's default temperature=10 sits in a deliberate middle range)."""
    from uav_nav_lab.planner.mppi import MPPIPlanner

    occ = np.zeros((30, 30), dtype=bool)
    obs = np.array([2.0, 2.0])
    goal = np.array([20.0, 20.0])

    low = MPPIPlanner(max_speed=5.0, horizon=20, n_samples=16, temperature=0.1, inflate=0)
    low.reset()
    speed_low = float(np.linalg.norm(low.plan(obs, goal, occ).target_velocity))

    high = MPPIPlanner(max_speed=5.0, horizon=20, n_samples=16, temperature=1e6, inflate=0)
    high.reset()
    speed_high = float(np.linalg.norm(high.plan(obs, goal, occ).target_velocity))

    # Low temp keeps the chosen sample's full speed; high temp dilutes it.
    assert speed_low > speed_high + 1.0


def test_mppi_invalid_temperature_raises() -> None:
    """Zero or negative temperature would make the softmax explode (divide
    by zero in the exp argument). Construction must reject it loudly."""
    from uav_nav_lab.planner import PLANNER_REGISTRY

    with pytest.raises(ValueError, match="temperature must be"):
        PLANNER_REGISTRY.get("mppi").from_config({"temperature": 0.0})
    with pytest.raises(ValueError, match="temperature must be"):
        PLANNER_REGISTRY.get("mppi").from_config({"temperature": -1.0})


def test_mppi_meta_carries_softmax_diagnostics() -> None:
    """weight_max ∈ [1/n, 1] and weight_entropy ∈ [0, log(n)] tell the
    user how concentrated MPPI's selection was. These appear in the plan
    metadata for downstream eval / sweep filtering."""
    from uav_nav_lab.planner import PLANNER_REGISTRY
    import math

    p = PLANNER_REGISTRY.get("mppi").from_config(
        {"horizon": 20, "n_samples": 16, "max_speed": 5.0, "temperature": 5.0}
    )
    occ = np.zeros((30, 30), dtype=bool)
    plan = p.plan(np.array([2.0, 2.0]), np.array([20.0, 20.0]), occ)
    assert plan.meta["planner"] == "mppi"
    assert 1.0 / 16 <= plan.meta["weight_max"] <= 1.0 + 1e-6
    assert 0.0 <= plan.meta["weight_entropy"] <= math.log(16) + 1e-6


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


def test_dummy_sim_synthetic_lidar_emits_in_range_obstacle_cells() -> None:
    """When `synthetic_perception.lidar_range > 0`, dummy sim should
    populate `state.extra["lidar_points"]["omni"]` with vehicle-local
    points for every occupied cell within the configured range — letting
    the same dummy world drive `pointcloud_occupancy` for ablations
    that don't require AirSim / ROS 2."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [20, 20], "start": [10.0, 10.0], "goal": [18.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_2d").from_config(
        {"dt": 0.05, "synthetic_perception": {"lidar_range": 8.0}}, scn,
    )
    # reset() reseeds the scenario which clears the obstacle grid; plant
    # AFTER reset so the test obstacles survive.
    sim.reset(seed=0)
    scn.occupancy[15, 10] = True   # 5.5 m from drone at (10, 10)
    scn.occupancy[18, 18] = True   # ~11.4 m, beyond 8 m range
    state, _ = sim.step(np.array([0.0, 0.0]))
    cloud = state.extra["lidar_points"]["omni"]
    assert cloud.shape[1] == 3
    assert cloud.shape[0] == 1                # only the in-range cell
    # Vehicle-local: world (15.5, 10.5) − drone (10.0, 10.0) ≈ (5.5, 0.5, 0).
    # Drone position drifts slightly during step() so allow ±1.0 tolerance.
    assert 4.5 < float(cloud[0, 0]) < 6.5
    assert abs(float(cloud[0, 1])) < 1.5


def test_dummy_sim_synthetic_depth_camera_projects_forward_obstacles() -> None:
    """`synthetic_perception.depth: {fov_deg, width, height, max_depth}`
    should populate `state.extra["depth_images"]["front"]` with a
    pinhole depth image plus intrinsics. Pixels with no obstacle stay
    at `max_depth + 1` so the depth_image_occupancy sensor's cap drops
    them; pixels with an obstacle in front carry the cell's depth."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [20, 20], "start": [5.0, 10.0], "goal": [18.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_2d").from_config(
        {"dt": 0.05, "synthetic_perception": {
            "depth": {"fov_deg": 90, "width": 16, "height": 12, "max_depth": 6.0}
        }}, scn,
    )
    sim.reset(seed=0)
    # Single obstacle directly ahead of drone: world (10, 10), depth ≈ 5.5 m.
    scn.occupancy[10, 10] = True
    state, _ = sim.step(np.array([0.0, 0.0]))
    payload = state.extra["depth_images"]["front"]
    depth = payload["depth"]
    assert depth.shape == (12, 16)
    # The obstacle's centre projects to image centre (drone faces +x,
    # obstacle at (10.5, 10.5) is bearing 0° + slight offset). Closest
    # depth in the image should be ~5.5 m (cell centre at x=10.5, drone x=5).
    assert 5.0 < float(depth.min()) < 6.0
    # Non-hit pixels stay at the sentinel (max_depth + 1).
    assert float(depth.max()) == pytest.approx(7.0)
    # Intrinsics: 90° FOV on 16-wide → fx = 8 / tan(45°) = 8.
    assert payload["intrinsics"]["fx"] == pytest.approx(8.0)
    # And R_cam_to_body so the depth_image_occupancy sensor reverses
    # the bridge's world→cam projection correctly.
    assert payload["R_cam_to_body"].shape == (3, 3)


def test_dummy_sim_synthetic_perception_round_trip_through_sensors() -> None:
    """Both pointcloud_occupancy and depth_image_occupancy fed by the
    same dummy sim should mark the obstacle cell that's directly in
    front of the drone — proves the camera-frame rotation in the
    synthetic depth payload is consistent with the sensor's reverse
    projection."""
    from uav_nav_lab.scenario import SCENARIO_REGISTRY
    from uav_nav_lab.sensor import SENSOR_REGISTRY
    from uav_nav_lab.sim import SIM_REGISTRY

    scn = SCENARIO_REGISTRY.get("grid_world").from_config(
        {"size": [20, 20], "start": [5.0, 10.0], "goal": [18.0, 10.0],
         "obstacles": {"type": "none"}}
    )
    sim = SIM_REGISTRY.get("dummy_2d").from_config(
        {"dt": 0.05, "synthetic_perception": {
            "lidar_range": 8.0,
            "depth": {"fov_deg": 90, "width": 32, "height": 24, "max_depth": 8.0},
        }}, scn,
    )
    sim.reset(seed=0)
    scn.occupancy[10, 10] = True
    state, _ = sim.step(np.array([0.0, 0.0]))

    pc = SENSOR_REGISTRY.get("pointcloud_occupancy").from_config(
        {"resolution": 1.0, "memory": False}
    )
    pc.reset()
    pc_occ = pc.observe_map(0.05, state.position, scn.occupancy, sim_extra=state.extra)

    dp = SENSOR_REGISTRY.get("depth_image_occupancy").from_config(
        {"resolution": 1.0, "memory": False, "stride": 1, "max_depth": 8.0}
    )
    dp.reset()
    dp_occ = dp.observe_map(0.05, state.position, scn.occupancy, sim_extra=state.extra)

    # Both sensors should mark cell (10, 10) — the obstacle itself.
    assert pc_occ[10, 10]
    assert dp_occ[10, 10]


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


def test_multi_drone_anim_groups_drones_per_episode(tmp_path: Path) -> None:
    """`uav-nav anim` on a multi-drone run dispatches to the multi-drone
    animator: one GIF per episode (not per drone), all N drone trajectories
    rendered together with a per-drone palette colour. Mirrors the
    `viz_run` test for parity."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from uav_nav_lab.anim import viz_anim

    cfg = ExperimentConfig.from_yaml(EXAMPLES / "exp_multi_drone.yaml")
    cfg.num_episodes = 1
    cfg.simulator["max_steps"] = 100   # very short — keep test fast
    run_dir = run_experiment(cfg, tmp_path / "multi_anim")
    saved = viz_anim(run_dir, fps=10)
    # one GIF per episode (not per drone)
    assert len(saved) == 1
    p = saved[0]
    assert p.suffix == ".gif"
    assert p.stat().st_size > 1000  # non-empty animation
