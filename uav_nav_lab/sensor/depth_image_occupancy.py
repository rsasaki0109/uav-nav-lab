"""Depth-image → occupancy-grid sensor.

Consumes raw depth images the simulator backend exposes via
`state.extra["depth_images"]` and projects each valid pixel to a 3D
point in world coordinates, then rasterises into the framework's
occupancy grid the same way `pointcloud_occupancy` does.

Why a separate sensor (not a flag on `pointcloud_occupancy`):
  - Different upstream payload: a (H, W) float depth array plus
    `{fx, fy, cx, cy}` intrinsics, vs an (N, 3) point cloud.
  - Different cost profile: depth-image projection is O(H·W) per call
    even when most pixels are sky / max-range, so a stride-based
    subsample (`stride: N`) is the load-bearing tuning knob — separate
    from the point-cloud path's range filter.
  - Bridges can produce both side-by-side (e.g. a forward LiDAR plus a
    downward depth camera); keeping the consumers separate avoids
    overloading one config block with mutually exclusive sources.

Expected payload shape:
    state.extra["depth_images"] = {
        camera_name: {
            "depth":      np.ndarray (H, W) float32, in metres
                          (NaN / non-finite / ≤0 values are dropped)
            "intrinsics": {"fx": float, "fy": float, "cx": float, "cy": float},
            # Optional. The depth pixel z-axis is +x in vehicle frame
            # (camera looking forward); rotation from camera to ENU
            # body falls back to identity if the bridge does not supply
            # one. AirSimBridge populates this from simGetCameraInfo.
            "R_cam_to_body": np.ndarray (3, 3),
        }
    }

Coordinate convention:
  - Depth is along the camera's optical (forward) axis; pixel (u, v)
    with depth d projects to camera-frame (X, Y, Z) where
        X = (u - cx) * d / fx        # right
        Y = (v - cy) * d / fy        # down
        Z = d                        # forward
  - Camera frame is the standard ROS / OpenCV convention: +Z forward,
    +X right, +Y down.
  - `R_cam_to_body` rotates camera-frame points into vehicle ENU (the
    bridge owns this — defaults to identity if missing).
  - World point = `true_position + R_cam_to_body @ p_cam`.

`memory: true` (default) accumulates seen-occupied cells across the
episode; `memory: false` returns only the latest sweep.

`stride: N` subsamples the depth pixel grid by N along each axis —
the right knob to dial cost down for high-resolution depth cameras.
Default 4 (every 4th row, every 4th column).

`max_depth: M` drops pixels reporting d > M (sky / no-return on most
depth sensors). Default 10.0 metres."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import SENSOR_REGISTRY, SensorModel


@SENSOR_REGISTRY.register("depth_image_occupancy")
class DepthImageOccupancySensor(SensorModel):
    def __init__(
        self,
        resolution: float = 1.0,
        memory: bool = True,
        inflate: int = 0,
        stride: int = 4,
        max_depth: float = 10.0,
        cameras: list[str] | None = None,
    ) -> None:
        self.resolution = float(resolution)
        self.memory = bool(memory)
        self.inflate = max(0, int(inflate))
        # Subsample factor along both axes — 1 keeps every pixel.
        self.stride = max(1, int(stride))
        # Drop pixels reporting depth beyond this; AirSim returns
        # ~100 m for sky / no-return, real sensors saturate similarly.
        self.max_depth = float(max_depth)
        # Optional: restrict to a subset of camera names.
        self.cameras: list[str] | None = (
            None if cameras is None else [str(name) for name in cameras]
        )
        self._seen: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DepthImageOccupancySensor":
        cams_cfg = cfg.get("cameras")
        return cls(
            resolution=float(cfg.get("resolution", 1.0)),
            memory=bool(cfg.get("memory", True)),
            inflate=int(cfg.get("inflate", 0)),
            stride=int(cfg.get("stride", 4)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            cameras=None if cams_cfg is None else list(cams_cfg),
        )

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002
        # No randomness — seed accepted for interface parity.
        self._seen = None

    def observe(self, t: float, true_position: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.asarray(true_position, dtype=float).copy()

    def observe_map(
        self,
        t: float,  # noqa: ARG002
        true_position: np.ndarray,
        true_obstacle_map: np.ndarray,
        sim_extra: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        occ_shape = np.asarray(true_obstacle_map).shape
        ndim = len(occ_shape)
        if self._seen is None or self._seen.shape != occ_shape:
            self._seen = np.zeros(occ_shape, dtype=bool)
        if not self.memory:
            self._seen.fill(False)

        depth_bag = self._collect_depth_payloads(sim_extra)
        if not depth_bag:
            return self._seen

        pos = np.asarray(true_position, dtype=float)
        pos3 = np.zeros(3)
        pos3[: min(3, pos.size)] = pos[:3]

        for payload in depth_bag:
            world = self._project_to_world(payload, pos3)
            if world is None or world.size == 0:
                continue
            self._mark(world, ndim, occ_shape)

        if self.inflate > 0:
            self._dilate(occ_shape)
        return self._seen

    def _collect_depth_payloads(
        self, sim_extra: Mapping[str, Any] | None
    ) -> list[Mapping[str, Any]]:
        if not sim_extra:
            return []
        bag = sim_extra.get("depth_images")
        if not isinstance(bag, dict):
            return []
        names = self.cameras if self.cameras is not None else list(bag.keys())
        out: list[Mapping[str, Any]] = []
        for name in names:
            payload = bag.get(name)
            # Forgive both dict-shaped payloads and bare arrays — the latter
            # will be skipped by `_project_to_world` (no intrinsics).
            if isinstance(payload, Mapping):
                out.append(payload)
        return out

    def _project_to_world(
        self, payload: Mapping[str, Any], drone_pos3: np.ndarray
    ) -> np.ndarray | None:
        depth = payload.get("depth")
        intr = payload.get("intrinsics")
        if depth is None or not isinstance(intr, Mapping):
            return None
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2 or d.size == 0:
            return None
        fx = float(intr.get("fx", 0.0))
        fy = float(intr.get("fy", fx))
        cx = float(intr.get("cx", d.shape[1] / 2.0))
        cy = float(intr.get("cy", d.shape[0] / 2.0))
        if fx <= 0.0 or fy <= 0.0:
            return None

        # Subsample then mask out invalid / out-of-range depths in one shot.
        d = d[:: self.stride, :: self.stride]
        h, w = d.shape
        finite = np.isfinite(d) & (d > 0.0) & (d <= self.max_depth)
        if not finite.any():
            return None

        v_idx, u_idx = np.indices((h, w))
        # Rescale pixel coords to original image coords for intrinsics.
        u = u_idx[finite] * self.stride
        v = v_idx[finite] * self.stride
        z = d[finite]
        x = (u.astype(np.float32) - cx) * z / fx
        y = (v.astype(np.float32) - cy) * z / fy
        cam_pts = np.stack([x, y, z], axis=1)                    # (M, 3)

        R = payload.get("R_cam_to_body")
        if R is not None:
            R = np.asarray(R, dtype=np.float32)
            if R.shape != (3, 3):
                R = None
        body_pts = cam_pts if R is None else cam_pts @ R.T       # (M, 3)
        return body_pts + drone_pos3.astype(np.float32)

    def _mark(self, world: np.ndarray, ndim: int, occ_shape: tuple[int, ...]) -> None:
        cells = np.floor(world[:, :ndim] / self.resolution).astype(int)
        in_bounds = np.ones(cells.shape[0], dtype=bool)
        for i in range(ndim):
            in_bounds &= (cells[:, i] >= 0) & (cells[:, i] < occ_shape[i])
        cells = cells[in_bounds]
        if cells.size == 0:
            return
        idx = tuple(cells[:, i] for i in range(ndim))
        self._seen[idx] = True

    def _dilate(self, occ_shape: tuple[int, ...]) -> None:
        # Cheap separable dilation — same scheme as pointcloud_occupancy.
        out = self._seen.copy()
        for axis in range(len(occ_shape)):
            for shift in range(1, self.inflate + 1):
                out |= np.roll(self._seen, shift, axis=axis)
                out |= np.roll(self._seen, -shift, axis=axis)
        self._seen = out

    def observe_dynamics(
        self,
        t: float,  # noqa: ARG002
        true_position: np.ndarray,  # noqa: ARG002
        dynamic_obstacles: list[dict],  # noqa: ARG002
    ) -> list[dict]:
        # Depth images do not segment moving threats — that is downstream.
        return []
