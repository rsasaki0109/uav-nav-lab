"""Record AirSim multi-drone demo from a top-down fixed camera.

Places a camera at (30, 30, 55) looking straight down so all 4 drones
are visible as they cross at the centre. Captures frames via AirSim
API directly (not through the framework's camera pipeline).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "results" / "airsim_topdown_frames"
GIF_OUT = REPO_ROOT / "docs" / "images" / "demo_airsim_multi.gif"


def run_experiment() -> None:
    run_dir = REPO_ROOT / "results" / "airsim_multi_demo"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        sys.executable, "-c",
        "import sys; sys.argv=['uav-nav','run',str(r'%s')];"
        "from uav_nav_lab.cli import main; main()"
        % str(REPO_ROOT / "examples" / "exp_airsim_multi_demo.yaml"),
    ]
    # Disable camera frames in the experiment — we capture externally
    env = {"UAV_NAV_NO_CAMERA": "1"}
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env={**__import__("os").environ, **env})


def capture_topdown(fps: int = 10, duration_s: float = 8.0) -> list[Path]:
    """Capture frames from a fixed top-down camera via AirSim API."""
    import airsim
    from PIL import Image  # type: ignore[import-not-found]

    client = airsim.MultirotorClient()
    client.confirmConnection()
    # Don't reset — the experiment runner handles that.

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old frames
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()

    # Place camera at centre, high up, looking straight down
    cam_name = "topdown"
    cam_pose = airsim.Pose(
        airsim.Vector3r(30.0, 30.0, -55.0),  # NED: (30N, 30E, 55Up)
        airsim.to_quaternion(np.pi / 2, 0.0, 0.0),  # pitch 90° down
    )
    # Use simSetCameraPose on a dummy vehicle (Drone1)
    try:
        client.simSetCameraPose(cam_name, cam_pose, vehicle_name="Drone1")
    except Exception:
        pass

    n_frames = int(fps * duration_s)
    interval = 1.0 / fps
    paths = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        try:
            responses = client.simGetImages(
                [airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)],
                vehicle_name="Drone1",
            )
            if responses:
                img_data = responses[0].image_data_uint8
                if img_data:
                    # AirSim returns BGRA
                    arr = np.frombuffer(img_data, dtype=np.uint8).reshape(
                        responses[0].height, responses[0].width, 4
                    )
                    img = Image.fromarray(arr[:, :, :3][:, :, ::-1])  # BGR→RGB
                    path = FRAMES_DIR / f"frame_{i:04d}.png"
                    img.save(str(path))
                    paths.append(path)
        except Exception as e:
            print(f"  frame {i} error: {e}")

        elapsed = time.perf_counter() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    return paths


def frames_to_gif(
    frames_dir: Path, out: Path, fps: int = 10, width: int = 640
) -> None:
    import subprocess
    n = sum(1 for p in frames_dir.iterdir() if p.suffix == ".png")
    if n == 0:
        print("No frames captured!")
        return
    palette = frames_dir / "_palette.png"
    pattern = str(frames_dir / "frame_%04d.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern, "-vf", f"{vf},palettegen=stats_mode=diff",
         str(palette)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", pattern, "-i", str(palette),
         "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
         "-loop", "0", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    palette.unlink(missing_ok=True)
    print(f"GIF: {out}  ({out.stat().st_size // 1024} KB)  ({n} frames)")


def main() -> int:
    print("[1/3] run experiment (4 drones crossing)")
    run_experiment()

    print("[2/3] capture top-down frames")
    paths = capture_topdown(fps=10, duration_s=8.0)
    print(f"  captured {len(paths)} frames")

    print("[3/3] frames → GIF")
    frames_to_gif(FRAMES_DIR, GIF_OUT, fps=10, width=640)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
