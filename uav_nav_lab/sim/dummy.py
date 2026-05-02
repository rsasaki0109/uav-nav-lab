"""N-D point-mass dummy simulator.

Simple Euler-integrated kinematics with a 2D / 3D occupancy grid for
collisions. The same class backs both `dummy_2d` and `dummy_3d`; dimension
is inferred from the scenario's `ndim` so YAML stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import SIM_REGISTRY, SimInterface, SimState, SimStepInfo


@dataclass
class _DummyParams:
    dt: float = 0.05
    max_steps: int = 2000
    max_accel: float = 50.0
    goal_radius: float = 1.0
    drone_radius: float = 0.4
    # Constant wind (m/s) added to drone velocity each step, plus optional
    # Gaussian gust on top. Ndim is detected from the scenario at construction.
    wind: tuple[float, ...] = ()
    gust_std: float = 0.0
    # Synthetic-perception knobs. When `synth_lidar_range > 0`, every step
    # surfaces a (N, 3) point cloud of in-range occupied cells at
    # state.extra["lidar_points"]["omni"]. When `synth_depth_*` are set,
    # a forward-facing pinhole depth image lands at
    # state.extra["depth_images"]["front"]. Together they let the
    # framework's pointcloud_occupancy / depth_image_occupancy sensors run
    # head-to-head against the *same* underlying obstacle set without
    # needing an external sim (AirSim / ROS 2). Synthesis is geometric,
    # not physical: no occlusion, no surface noise, no time lag.
    synth_lidar_range: float = 0.0
    synth_depth_fov_deg: float = 0.0
    synth_depth_width: int = 0
    synth_depth_height: int = 0
    synth_depth_max: float = 10.0


class DummySim(SimInterface):
    """N-D headless point-mass over an occupancy grid (2D or 3D).

    Command convention: N-D velocity setpoint. The integrator clamps the
    instantaneous accel to `max_accel` so step responses stay realistic.
    """

    def __init__(
        self,
        params: _DummyParams,
        scenario: Any,
        *,
        advance_scenario: bool = True,
    ) -> None:
        self.p = params
        self.dt = params.dt
        self.scenario = scenario
        self._advance_scenario = bool(advance_scenario)
        self._ndim = scenario.ndim
        self._state: SimState | None = None
        self._step_count = 0
        # Per-drone goal override (set by the multi-drone runner). When unset
        # we delegate to the scenario's single goal — preserving single-drone
        # behavior unchanged.
        self._goal_override: np.ndarray | None = None
        wind = np.asarray(params.wind, dtype=float) if params.wind else np.zeros(self._ndim)
        if wind.shape[0] < self._ndim:
            wind = np.concatenate([wind, np.zeros(self._ndim - wind.shape[0])])
        self._wind = wind[: self._ndim]
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], scenario: Any) -> "DummySim":
        dist = dict(cfg.get("disturbance", {}))
        synth = dict(cfg.get("synthetic_perception", {}))
        depth = dict(synth.get("depth", {}))
        params = _DummyParams(
            dt=float(cfg.get("dt", 0.05)),
            max_steps=int(cfg.get("max_steps", 2000)),
            max_accel=float(cfg.get("max_accel", 50.0)),
            goal_radius=float(cfg.get("goal_radius", 1.0)),
            drone_radius=float(cfg.get("drone_radius", 0.4)),
            wind=tuple(dist.get("wind", ())),
            gust_std=float(dist.get("gust_std", 0.0)),
            synth_lidar_range=float(synth.get("lidar_range", 0.0)),
            synth_depth_fov_deg=float(depth.get("fov_deg", 0.0)),
            synth_depth_width=int(depth.get("width", 0)),
            synth_depth_height=int(depth.get("height", 0)),
            synth_depth_max=float(depth.get("max_depth", 10.0)),
        )
        return cls(params, scenario)

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_position: np.ndarray | None = None,
    ) -> SimState:
        if seed is not None:
            self.scenario.reseed(seed)
            self._rng = np.random.default_rng(seed)
        if initial_position is not None:
            start = np.asarray(initial_position, dtype=float).reshape(self._ndim)
        else:
            start = np.asarray(self.scenario.start, dtype=float)
        self._state = SimState(t=0.0, position=start.copy(), velocity=np.zeros(self._ndim))
        self._step_count = 0
        return self._state.copy()

    def step(self, command: np.ndarray) -> tuple[SimState, SimStepInfo]:
        assert self._state is not None, "call reset() first"
        cmd = np.asarray(command, dtype=float).reshape(self._ndim)

        dv = cmd - self._state.velocity
        max_dv = self.p.max_accel * self.dt
        norm = float(np.linalg.norm(dv))
        if norm > max_dv:
            dv *= max_dv / norm
        self._state.velocity = self._state.velocity + dv
        # external disturbance: wind + gust. Affects position but does not
        # alter the controller's velocity tracking — the drone is "blown".
        disturbance = self._wind.copy()
        if self.p.gust_std > 0.0:
            disturbance = disturbance + self._rng.normal(
                0.0, self.p.gust_std, size=self._ndim
            )
        self._state.position = self._state.position + (
            self._state.velocity + disturbance
        ) * self.dt
        self._state.t += self.dt
        self._step_count += 1
        if self._advance_scenario:
            self.scenario.advance(self.dt)  # no-op for static-only scenarios

        # Synthetic perception — emit lidar / depth payloads when configured
        # so the same dummy sim can drive both pointcloud_occupancy and
        # depth_image_occupancy sensors for head-to-head ablations. No
        # raycasting / occlusion: we expose the geometry truthfully and
        # let the sensor models trade fidelity for cost.
        if self.p.synth_lidar_range > 0.0:
            self._state.extra["lidar_points"] = {
                "omni": self._synth_lidar_cloud()
            }
        if self.p.synth_depth_fov_deg > 0.0 and self.p.synth_depth_width > 0:
            self._state.extra["depth_images"] = {
                "front": self._synth_depth_image()
            }

        collision = self.scenario.is_collision(self._state.position, self.p.drone_radius)
        goal_pos = (
            self._goal_override if self._goal_override is not None else self.scenario.goal
        )
        goal_reached = bool(
            np.linalg.norm(self._state.position - goal_pos) <= self.p.goal_radius
        )
        truncated = self._step_count >= self.p.max_steps
        return self._state.copy(), SimStepInfo(
            collision=collision, goal_reached=goal_reached, truncated=truncated
        )

    def _synth_lidar_cloud(self) -> np.ndarray:
        """Vehicle-local (N, 3) ENU points for every occupied cell within
        `synth_lidar_range` of the drone. No occlusion — every visible
        obstacle cell is reported once."""
        occ = np.asarray(self.scenario.occupancy, dtype=bool)
        idx = np.argwhere(occ)
        if idx.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # Cell centres (cell_idx + 0.5) × resolution. The scenario's
        # `resolution` defaults to 1.0 if unset.
        res = float(getattr(self.scenario, "resolution", 1.0))
        world = (idx.astype(np.float32) + 0.5) * res
        if world.shape[1] == 2:
            world = np.column_stack([world, np.zeros(world.shape[0], dtype=np.float32)])
        pos3 = np.zeros(3, dtype=np.float32)
        pos3[: min(3, self._state.position.size)] = self._state.position[:3]
        local = world - pos3
        d2 = np.sum(local * local, axis=1)
        r2 = float(self.p.synth_lidar_range) ** 2
        return local[d2 <= r2]

    def _synth_depth_image(self) -> dict[str, Any]:
        """Forward-camera pinhole depth image. For each occupied cell within
        `synth_depth_max`, project to (u, v) and write the closest depth
        per pixel. Sky pixels stay at `synth_depth_max + 1` so the
        sensor's max-depth filter drops them. Camera frame: +z forward
        (drone faces +x in world ENU; we use a 90° world-yaw so camera
        +z aligns with world +x), +x right, +y down — standard ROS /
        OpenCV convention. No occlusion correction beyond per-pixel min."""
        h, w = int(self.p.synth_depth_height), int(self.p.synth_depth_width)
        max_d = float(self.p.synth_depth_max)
        depth = np.full((h, w), max_d + 1.0, dtype=np.float32)
        # Intrinsics (fx = fy from horizontal fov, square pixels).
        fov = float(self.p.synth_depth_fov_deg) * np.pi / 180.0
        fx = (w / 2.0) / float(np.tan(fov / 2.0))
        cx = w / 2.0
        cy = h / 2.0
        intr = {"fx": fx, "fy": fx, "cx": cx, "cy": cy}
        # Camera→body rotation. Below we project world → cam by mapping
        # (x_w, y_w, z_w) → (X = -y, Y = -z, Z = x) so optical +z aligns
        # with world +x (forward). The sensor reverse-projects assuming
        # camera-frame, then applies R_cam_to_body — supply the inverse
        # of our forward map so its world-frame reconstruction lands
        # back where the obstacle actually is.
        R_cam_to_body = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )

        occ = np.asarray(self.scenario.occupancy, dtype=bool)
        idx = np.argwhere(occ)
        if idx.size == 0:
            return {"depth": depth, "intrinsics": intr, "R_cam_to_body": R_cam_to_body}
        res = float(getattr(self.scenario, "resolution", 1.0))
        world = (idx.astype(np.float32) + 0.5) * res
        if world.shape[1] == 2:
            world = np.column_stack([world, np.zeros(world.shape[0], dtype=np.float32)])
        pos3 = np.zeros(3, dtype=np.float32)
        pos3[: min(3, self._state.position.size)] = self._state.position[:3]
        # Camera looks along world +x. Map world (x, y, z) - drone → camera
        # (X = -y, Y = -z, Z = x) so the +z optical axis matches world +x.
        rel = world - pos3
        cam_z = rel[:, 0]
        cam_x = -rel[:, 1]
        cam_y = -rel[:, 2]
        # In front of camera + within max depth.
        keep = (cam_z > 0.0) & (cam_z <= max_d)
        if not keep.any():
            return {"depth": depth, "intrinsics": intr, "R_cam_to_body": R_cam_to_body}
        cam_x, cam_y, cam_z = cam_x[keep], cam_y[keep], cam_z[keep]
        u = (cam_x * fx / cam_z + cx).astype(int)
        v = (cam_y * fx / cam_z + cy).astype(int)
        in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, cam_z = u[in_frame], v[in_frame], cam_z[in_frame]
        # Per-pixel min: scatter via flat index, take min via a sort.
        flat_idx = v * w + u
        order = np.argsort(cam_z)
        flat_idx = flat_idx[order]
        cam_z = cam_z[order]
        # First write of each pixel wins (sorted by depth ascending).
        seen = np.zeros(h * w, dtype=bool)
        for i in range(flat_idx.shape[0]):
            fi = int(flat_idx[i])
            if not seen[fi]:
                seen[fi] = True
                depth.flat[fi] = cam_z[i]
        return {"depth": depth, "intrinsics": intr, "R_cam_to_body": R_cam_to_body}

    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state.copy()

    @property
    def goal(self) -> np.ndarray:
        if self._goal_override is not None:
            return np.asarray(self._goal_override, dtype=float)
        return np.asarray(self.scenario.goal, dtype=float)

    def set_goal(self, goal: np.ndarray) -> None:
        """Override the goal used by the goal-reached check (multi-drone)."""
        self._goal_override = np.asarray(goal, dtype=float).reshape(self._ndim)

    @property
    def obstacle_map(self) -> np.ndarray:
        return self.scenario.occupancy


# Register the same class under both names so the YAML stays explicit about
# the expected dimension. Mismatch (e.g. dummy_3d + grid_world) is caught at
# build time by the scenario's ndim.
SIM_REGISTRY.register("dummy_2d")(DummySim)
SIM_REGISTRY.register("dummy_3d")(DummySim)
SIM_REGISTRY.register("dummy")(DummySim)
