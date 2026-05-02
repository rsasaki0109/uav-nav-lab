"""Stitch per-step camera frames into per-episode MP4s.

Writing one PNG per step keeps the runner side trivial (no encoder
dependency, easy to inspect with `ls`/an image viewer); the encoder
step lives here so it only runs when the user explicitly asks for a
video. ffmpeg is the only outside tool this needs.

Run via `uav-nav video <run_dir>`:
  - Walks `<run_dir>/frames_NNN/` directories
  - Groups PNGs by camera name (the suffix in `step_NNNN_<name>.png`)
  - ffmpegs each (episode × camera) sequence into
    `<run_dir>/episode_NNN_<name>.mp4`
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# step_NNNN_<camera_name>.png — camera_name may contain underscores so
# the regex grabs everything after the step index up to the extension.
_FRAME_RE = re.compile(r"^step_(\d+)_(?P<cam>.+)\.png$")


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise SystemExit(
            "ffmpeg is not on PATH. Install ffmpeg (e.g. `apt install ffmpeg`) "
            "and retry — `uav-nav video` shells out to it."
        )
    return path


def _group_frames_by_camera(frames_dir: Path) -> dict[str, list[Path]]:
    """Return {camera_name: [PNG paths sorted by step]} for one frames_NNN/."""
    groups: dict[str, list[Path]] = {}
    for png in frames_dir.iterdir():
        m = _FRAME_RE.match(png.name)
        if not m:
            continue
        groups.setdefault(m.group("cam"), []).append(png)
    for cam in groups:
        groups[cam].sort(key=lambda p: int(_FRAME_RE.match(p.name).group(1)))  # type: ignore[union-attr]
    return groups


def stitch_run(run_dir: Path, fps: int = 20) -> list[Path]:
    """Walk `<run_dir>/frames_NNN/` and write `<run_dir>/episode_NNN_<cam>.mp4`
    per camera. Returns the list of mp4 paths actually produced."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} does not exist")
    ffmpeg = _ensure_ffmpeg()

    saved: list[Path] = []
    frame_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("frames_"))
    if not frame_dirs:
        raise FileNotFoundError(
            f"no frames_NNN/ subdirectories under {run_dir} — run with "
            "`output.save_camera_frames: true` and at least one camera "
            "configured to produce them."
        )

    for fdir in frame_dirs:
        # frames_NNN → NNN
        ep_idx = fdir.name.split("_", 1)[1]
        groups = _group_frames_by_camera(fdir)
        if not groups:
            continue
        for cam_name, pngs in groups.items():
            if not pngs:
                continue
            # ffmpeg's -i with %04d expects a contiguous numbered run starting
            # at the same digit width. Our writer uses 4-digit step indices
            # but episodes can terminate early, so we do a glob-pattern input
            # via concat-demuxer file-list to be safe with gaps.
            list_file = fdir / f".concat_{cam_name}.txt"
            with list_file.open("w", encoding="utf-8") as f:
                for png in pngs:
                    # ffmpeg concat-demuxer wants `file '<path>'` lines plus
                    # an explicit `duration` so frames advance at fps.
                    f.write(f"file '{png.resolve()}'\n")
                    f.write(f"duration {1.0 / max(1, fps):.6f}\n")
                # Repeat the last frame so concat produces the final image.
                f.write(f"file '{pngs[-1].resolve()}'\n")
            out = run_dir / f"episode_{ep_idx}_{cam_name}.mp4"
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-vsync", "vfr",
                "-pix_fmt", "yuv420p",
                "-vf", f"fps={fps}",
                str(out),
            ]
            subprocess.run(cmd, check=True)
            list_file.unlink(missing_ok=True)
            saved.append(out)
    return saved
