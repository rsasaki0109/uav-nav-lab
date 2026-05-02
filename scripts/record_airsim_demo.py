"""Record an AirSim demo flight as a GIF for the README hero.

End-to-end:
  1. resets the running AirSim server (clears stale collision flags),
  2. pitches the front-center camera ~17° down so the cube clusters
     stay visible as we fly over them,
  3. runs `uav-nav run examples/exp_airsim_demo.yaml` (the bridge
     drives the airsim instance through the planner-generated path
     and saves one PNG per step under results/airsim_demo/frames_000/),
  4. ffmpegs those PNGs into docs/images/demo_airsim.gif.

Why a script instead of a CLI feature: this is a one-off recording
operation tied to a specific scene composition (camera pitch + scenario
geometry). Folding it into the bridge would make the bridge carry
demo-specific concerns; a 60-line driver script keeps the framework
clean.

Run from the project root, with an AirSim server already up:
  python3 scripts/record_airsim_demo.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "examples" / "exp_airsim_demo.yaml"
RUN_DIR = REPO_ROOT / "results" / "airsim_demo"
GIF_OUT = REPO_ROOT / "docs" / "images" / "demo_airsim.gif"


def _setup_camera() -> None:
    import airsim  # type: ignore[import-not-found]
    c = airsim.MultirotorClient()
    c.confirmConnection()
    # Clear any stale collision flag from prior sessions.
    c.reset()
    time.sleep(2.0)
    # Pitch front-center camera ~17° (-0.30 rad) down so the cube
    # clusters stay in frame as we fly over them at 30 m altitude.
    cam_pose = airsim.Pose(
        airsim.Vector3r(0.50, 0.0, 0.0),
        airsim.to_quaternion(-0.30, 0.0, 0.0),
    )
    c.simSetCameraPose("front_center", cam_pose)
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
    fps: int = 15,
    width: int = 480,
) -> None:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"{frames_dir} not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Two-pass ffmpeg: build palette then quantise. Downscale to
    # `width` px and drop to `fps` so the GIF lands in the README's
    # ~1–2 MB target band rather than the 8 MB raw render produces.
    palette = frames_dir / "_palette.png"
    pattern = str(frames_dir / "step_%04d_front_center.png")
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern,
         "-vf", f"{scale},palettegen=stats_mode=diff",
         str(palette)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern, "-i", str(palette),
         "-lavfi", f"{scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
         str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB)")


def main() -> int:
    print("[1/3] setup AirSim camera")
    _setup_camera()
    print("[2/3] run experiment")
    _run_experiment()
    print("[3/3] frames → GIF")
    _frames_to_gif(RUN_DIR / "frames_000", GIF_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
