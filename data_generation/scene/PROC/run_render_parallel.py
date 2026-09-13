#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


SCENE_ROOT = Path("/path/to/workspace/SCENE/PROC")
DEFAULT_SCENE_BANK_ROOT = Path("/path/to/workspace/SCENEOUTPUT/PROC/output")
DEFAULT_RENDER_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/PROC/render_output")


def _scene_dirs(scene_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for path in sorted(scene_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        if (path / "state_bank.jsonl").is_file() and (path / "proposals.jsonl").is_file():
            dirs.append(path)
    return dirs


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-limit:]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量执行 ProcTHOR scene bank proposals，生成最终渲染样本。")
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_BANK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RENDER_OUTPUT_ROOT)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--classes", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--subcats", type=str, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--target-per-subcat", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--use-cloud-rendering", action="store_true")
    parser.add_argument("--hardware-rendering", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _selected_scenes(scene_root: Path, raw_scenes: str | None) -> List[Path]:
    all_dirs = _scene_dirs(scene_root)
    if raw_scenes is None:
        return all_dirs
    wanted = {part.strip() for part in raw_scenes.split(",") if part.strip()}
    return [path for path in all_dirs if path.name in wanted]


def _scene_is_complete(scene_output: Path) -> bool:
    summary_path = scene_output / "execution_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(summary, list) or not summary:
        return False
    return all(int(item.get("returncode", 1)) == 0 for item in summary if "returncode" in item)


def _run_one_scene(args: argparse.Namespace, scene_dir: Path) -> Dict[str, object]:
    scene_name = scene_dir.name
    scene_output = args.output_root / scene_name
    stdout_log = scene_output / "subprocess.stdout.log"
    stderr_log = scene_output / "subprocess.stderr.log"
    scene_output.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and _scene_is_complete(scene_output):
        return {
            "scene": scene_name,
            "status": "skipped_existing",
            "returncode": 0,
            "output_root": str(scene_output),
        }

    cmd = [
        str(SCENE_ROOT / "run_execute_all_tasks.sh"),
        "--state-bank",
        str(scene_dir / "state_bank.jsonl"),
        "--proposals",
        str(scene_dir / "proposals.jsonl"),
        "--output-root",
        str(scene_output),
        "--classes",
        args.classes,
        "--target-per-subcat",
        str(args.target_per_subcat),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--grid-size",
        str(args.grid_size),
        "--field-of-view",
        str(args.field_of_view),
        "--visibility-distance",
        str(args.visibility_distance),
    ]
    if args.subcats:
        cmd.extend(["--subcats", args.subcats])
    if args.max_proposals is not None:
        cmd.extend(["--max-proposals", str(args.max_proposals)])
    if args.use_cloud_rendering:
        cmd.append("--use-cloud-rendering")
    if args.reset_output:
        cmd.append("--reset-output")

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":99")
    env.setdefault("HOME", "/path/to/workspace/ai2thor_runtime_home")
    env["LIBGL_ALWAYS_SOFTWARE"] = "0" if args.hardware_rendering else "1"

    with stdout_log.open("w", encoding="utf-8") as fout, stderr_log.open("w", encoding="utf-8") as ferr:
        cp = subprocess.run(cmd, stdout=fout, stderr=ferr, text=True, env=env)

    return {
        "scene": scene_name,
        "status": "ok" if cp.returncode == 0 else "failed",
        "returncode": cp.returncode,
        "output_root": str(scene_output),
        "stdout_tail": _tail_text(stdout_log),
        "stderr_tail": _tail_text(stderr_log),
    }


def main() -> int:
    args = _parse_args()
    scenes = _selected_scenes(args.scene_root, args.scenes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    max_workers = max(1, min(int(args.num_workers), len(scenes) if scenes else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_one_scene, args, scene_dir): scene_dir.name
            for scene_dir in scenes
        }
        for future in as_completed(future_map):
            results.append(future.result())

    results.sort(key=lambda item: str(item["scene"]))
    summary = {
        "scene_root": str(args.scene_root),
        "output_root": str(args.output_root),
        "scene_count": len(results),
        "num_workers": max_workers,
        "classes": args.classes,
        "subcats": args.subcats,
        "max_proposals": args.max_proposals,
        "target_per_subcat": args.target_per_subcat,
        "use_cloud_rendering": bool(args.use_cloud_rendering),
        "hardware_rendering": bool(args.hardware_rendering),
        "successful_scenes": [item["scene"] for item in results if item["status"] in {"ok", "skipped_existing"}],
        "failed_scenes": [item["scene"] for item in results if item["status"] == "failed"],
        "results": results,
    }
    summary_path = args.output_root / "render_parallel_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0 if not summary["failed_scenes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
