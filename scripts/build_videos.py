#!/usr/bin/env python3
"""Prepare local demo videos without changing their visual content."""

import argparse
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizer-dir", type=Path, default=root.parent / "Spatial-Interactor-Visualizer")
    args = parser.parse_args()
    output = root / "assets" / "videos"
    output.mkdir(parents=True, exist_ok=True)
    videos = [
        ("trajectory-integration", args.visualizer_dir / "assets/spatial_interactor_minimal_cot_demo.mp4", 16),
        ("path-shape", args.visualizer_dir / "assets/spatial_interactor_path_shape_demo.mp4", 16),
    ]
    for name, source, timestamp in videos:
        # Stream-copy preserves the original video; faststart enables progressive playback.
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart",
            str(output / f"{name}.mp4"),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(timestamp),
            "-i", str(source), "-frames:v", "1", "-c:v", "libwebp", "-quality", "92",
            str(output / f"{name}.webp"),
        ], check=True)
        print(f"Prepared {name}", flush=True)


if __name__ == "__main__":
    main()
