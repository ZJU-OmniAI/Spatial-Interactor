#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


SCENE_ROOT = Path("/path/to/workspace/SCENE/AI2THOR")
DEFAULT_SCENE_BANK_ROOT = Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/output")
DEFAULT_RENDER_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/render_output")
DEFAULT_NUM_WORKERS = min(32, max(1, (os.cpu_count() or 32)))

CLASS_SCRIPTS = {
    1: "execute_class01_bank.py",
    2: "execute_class02_bank.py",
    3: "execute_class03_bank.py",
    4: "execute_class04_bank.py",
    5: "execute_class05_bank.py",
    6: "execute_class06_bank.py",
    7: "execute_class07_bank.py",
    8: "execute_class08_bank.py",
    9: "execute_class09_bank.py",
    10: "execute_class10_bank.py",
}

CLASS_ORDER = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    9: 8,
    8: 9,
    10: 10,
}


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _scene_dirs(scene_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for path in sorted(scene_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        if not path.name.startswith("FloorPlan"):
            continue
        if (path / "state_bank.jsonl").is_file() and (path / "proposals.jsonl").is_file():
            dirs.append(path)
    return dirs


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-limit:]


def _result_path(scene_output: Path, class_id: int) -> Path:
    return scene_output / f"class_{class_id:02d}_result.json"


def _class_output_root(scene_output: Path, class_id: int) -> Path:
    return scene_output / str(class_id)


def _class_log_paths(scene_output: Path, class_id: int) -> Tuple[Path, Path]:
    log_root = scene_output / "_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    return (
        log_root / f"class_{class_id:02d}.stdout.log",
        log_root / f"class_{class_id:02d}.stderr.log",
    )


def _load_json(path: Path) -> Dict[str, object] | List[object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _legacy_scene_class_success(scene_output: Path, class_id: int) -> bool:
    summary = _load_json(scene_output / "execution_summary.json")
    if not isinstance(summary, list):
        return False
    for item in summary:
        if int(item.get("class_id", -1)) != int(class_id):
            continue
        return int(item.get("returncode", 1)) == 0
    return False


def _class_is_complete(scene_output: Path, class_id: int) -> bool:
    result_data = _load_json(_result_path(scene_output, class_id))
    if isinstance(result_data, dict):
        if int(result_data.get("returncode", 1)) == 0:
            return True
    class_output = _class_output_root(scene_output, class_id)
    stats_ok = (class_output / "meta" / "stats.json").is_file()
    records_ok = (class_output / "meta" / "records.json").is_file()
    if stats_ok and records_ok:
        return True
    return _legacy_scene_class_success(scene_output, class_id)


def _write_result(scene_output: Path, class_id: int, result: Dict[str, object]) -> None:
    scene_output.mkdir(parents=True, exist_ok=True)
    _result_path(scene_output, class_id).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _refresh_scene_summary(scene_output: Path, classes: List[int]) -> None:
    entries: List[Dict[str, object]] = []
    for class_id in classes:
        data = _load_json(_result_path(scene_output, class_id))
        if isinstance(data, dict):
            entries.append(data)
    entries.sort(key=lambda item: int(item["class_id"]))
    if entries:
        (scene_output / "execution_summary.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量执行 AI2THOR scene bank proposals，生成最终渲染样本。")
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_BANK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RENDER_OUTPUT_ROOT)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--classes", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--subcats", type=str, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--target-per-subcat", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
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


def _task_sort_key(task: Tuple[Path, int]) -> Tuple[int, int]:
    scene_dir, class_id = task
    scene_num = int(scene_dir.name.replace("FloorPlan", ""))
    return (scene_num, CLASS_ORDER.get(class_id, 999), class_id)


def _run_one_task(args: argparse.Namespace, scene_dir: Path, class_id: int) -> Dict[str, object]:
    scene_name = scene_dir.name
    scene_output = args.output_root / scene_name
    scene_output.mkdir(parents=True, exist_ok=True)
    class_output = _class_output_root(scene_output, class_id)
    stdout_log, stderr_log = _class_log_paths(scene_output, class_id)

    if args.skip_existing and _class_is_complete(scene_output, class_id):
        result = {
            "scene": scene_name,
            "class_id": class_id,
            "status": "skipped_existing",
            "returncode": 0,
            "script": str(SCENE_ROOT / CLASS_SCRIPTS[class_id]),
            "output_root": str(class_output),
            "stdout_tail": _tail_text(stdout_log),
            "stderr_tail": _tail_text(stderr_log),
        }
        _write_result(scene_output, class_id, result)
        return result

    cmd = [
        sys.executable,
        str(SCENE_ROOT / CLASS_SCRIPTS[class_id]),
        "--state-bank",
        str(scene_dir / "state_bank.jsonl"),
        "--proposals",
        str(scene_dir / "proposals.jsonl"),
        "--output-root",
        str(class_output),
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
        "--target-per-subcat",
        str(args.target_per_subcat),
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

    result = {
        "scene": scene_name,
        "class_id": class_id,
        "status": "ok" if cp.returncode == 0 else "failed",
        "returncode": cp.returncode,
        "script": str(SCENE_ROOT / CLASS_SCRIPTS[class_id]),
        "output_root": str(class_output),
        "stdout_tail": _tail_text(stdout_log),
        "stderr_tail": _tail_text(stderr_log),
    }
    _write_result(scene_output, class_id, result)
    return result


def main() -> int:
    args = _parse_args()
    classes = [class_id for class_id in _parse_csv_ints(args.classes) if class_id in CLASS_SCRIPTS]
    scenes = _selected_scenes(args.scene_root, args.scenes)
    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks = [(scene_dir, class_id) for scene_dir in scenes for class_id in classes]
    tasks.sort(key=_task_sort_key)
    max_workers = max(1, min(int(args.num_workers), len(tasks) if tasks else 1))

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_one_task, args, scene_dir, class_id): (scene_dir.name, class_id)
            for scene_dir, class_id in tasks
        }
        for future in as_completed(future_map):
            scene_name, class_id = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "scene": scene_name,
                    "class_id": class_id,
                    "status": "failed",
                    "returncode": 1,
                    "script": str(SCENE_ROOT / CLASS_SCRIPTS[class_id]),
                    "output_root": str(_class_output_root(args.output_root / scene_name, class_id)),
                    "stdout_tail": "",
                    "stderr_tail": str(exc),
                }
                _write_result(args.output_root / scene_name, class_id, result)
            results.append(result)
            _refresh_scene_summary(args.output_root / scene_name, classes)
            print(
                f"[{result['status'].upper()}] scene={scene_name} class={class_id} "
                f"returncode={result['returncode']}",
                flush=True,
            )

    results.sort(key=lambda item: (str(item["scene"]), int(item["class_id"])))
    summary = {
        "scene_root": str(args.scene_root),
        "output_root": str(args.output_root),
        "scene_count": len(scenes),
        "task_count": len(results),
        "num_workers": max_workers,
        "requested_num_workers": int(args.num_workers),
        "classes": classes,
        "subcats": args.subcats,
        "max_proposals": args.max_proposals,
        "target_per_subcat": args.target_per_subcat,
        "use_cloud_rendering": bool(args.use_cloud_rendering),
        "hardware_rendering": bool(args.hardware_rendering),
        "successful_tasks": [
            {"scene": item["scene"], "class_id": item["class_id"]}
            for item in results
            if item["status"] in {"ok", "skipped_existing"}
        ],
        "failed_tasks": [
            {"scene": item["scene"], "class_id": item["class_id"]}
            for item in results
            if item["status"] == "failed"
        ],
        "results": results,
    }
    summary_path = args.output_root / "render_parallel_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0 if not summary["failed_tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
