"""Record an AirSim multi-drone demo flight as a GIF.

End-to-end:
  1. resets the running AirSim server (clears stale state on Drones 1-4),
  2. pitches Drone1's front-center camera ~17° down so the other
     drones stay in frame as they cross at the centre,
  3. runs `uav-nav run examples/exp_airsim_multi_demo.yaml` (the multi
     runner drives all 4 vehicles through their MPC plans + CV peer
     prediction; only Drone1's bridge captures camera frames),
  4. ffmpegs the per-step PNGs into docs/images/demo_airsim_multi.gif.

Run from the project root, with an AirSim server already up *and*
~/Documents/AirSim/settings.json declaring Drone1..Drone4:
  python3 scripts/record_airsim_multi_demo.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"
RUN_DIR = REPO_ROOT / "results" / "airsim_multi_demo"
GIF_OUT = REPO_ROOT / "docs" / "images" / "demo_airsim_multi.gif"


def _setup_camera() -> None:
    import airsim  # type: ignore[import-not-found]
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.reset()
    time.sleep(2.0)
    # Pitch Drone1's front-center camera ~17° down so the other
    # drones (also at altitude 30 m) stay in frame as they cross.
    cam_pose = airsim.Pose(
        airsim.Vector3r(0.50, 0.0, 0.0),
        airsim.to_quaternion(-0.30, 0.0, 0.0),
    )
    c.simSetCameraPose("front_center", cam_pose, vehicle_name="Drone1")
    time.sleep(0.3)


def _run_experiment() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    cmd = [sys.executable, "-c",
           "import sys; sys.argv=['uav-nav','run',str(r'%s')];"
           "from uav_nav_lab.cli import main; main()" % YAML]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _frames_to_gif(
    frames_dir: Path,
    out: Path,
    fps: int = 12,
    width: int = 320,
    target_seconds: float = 5.5,
) -> None:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"{frames_dir} not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = sum(1 for p in frames_dir.iterdir()
                   if p.suffix == ".png" and "front_center" in p.name)
    desired_frames = max(1, int(round(fps * target_seconds)))
    keep_every = max(1, n_frames // desired_frames)
    palette = frames_dir / "_palette.png"
    pattern = str(frames_dir / "step_%04d_front_center.png")
    vf = (
        f"select='not(mod(n,{keep_every}))',"
        f"setpts=N/{fps}/TB,"
        f"scale={width}:-1:flags=lanczos"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern,
         "-vf", f"{vf},palettegen=stats_mode=diff",
         str(palette)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern, "-i", str(palette),
         "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
         "-loop", "0",
         str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB)  "
          f"({n_frames} frames @ every {keep_every}, ~{n_frames // keep_every / fps:.1f}s)")


def main() -> int:
    print("[1/3] setup AirSim camera")
    _setup_camera()
    print("[2/3] run experiment (4 drones)")
    _run_experiment()
    # Multi runner writes per-drone frames to frames_000_drone_NN/ —
    # but we only configured cameras on Drone1, so frames_000 has the
    # frames we want.
    frames_dir = RUN_DIR / "frames_000_drone_00"
    if not frames_dir.is_dir():
        # fallback for single-drone-style frames dir
        frames_dir = RUN_DIR / "frames_000"
    print(f"[3/3] frames → GIF (from {frames_dir.name}/)")
    _frames_to_gif(frames_dir, GIF_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
