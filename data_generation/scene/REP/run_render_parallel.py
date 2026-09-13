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


SCENE_ROOT = Path("/path/to/workspace/SCENE/REP")
DEFAULT_SCENE_BANK_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REP/output_prefilter_full_20260422")
DEFAULT_RENDER_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REP/render_output_prop_20260423")
DEFAULT_NUM_WORKERS = 3

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


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


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
    return path.read_text(encoding="utf-8", errors="ignore")[-limit:]


def _load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def _class_is_complete(scene_output: Path, class_id: int) -> bool:
    result = _load_json(_result_path(scene_output, class_id))
    if isinstance(result, dict) and int(result.get("returncode", 1)) == 0:
        return True
    class_output = _class_output_root(scene_output, class_id)
    stats_ok = (class_output / "meta" / "stats.json").is_file()
    records_ok = (class_output / "meta" / "records.json").is_file()
    return stats_ok and records_ok


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
    parser = argparse.ArgumentParser(description="批量执行 ReplicaCAD scene bank proposals，逐proposal生成最终渲染样本。")
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_BANK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RENDER_OUTPUT_ROOT)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--classes", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--subcats", type=str, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--target-per-subcat", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--gpu-ids", type=str, default="1,2,3")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _selected_scenes(scene_root: Path, raw_scenes: str | None) -> List[Path]:
    all_dirs = _scene_dirs(scene_root)
    if raw_scenes is None:
        return all_dirs
    wanted = {part.strip() for part in raw_scenes.split(",") if part.strip()}
    return [path for path in all_dirs if path.name in wanted]


def _parse_gpu_ids(raw: str | None) -> List[int]:
    if raw is None:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _run_one_task(args: argparse.Namespace, scene_dir: Path, class_id: int, gpu_id: int | None) -> Dict[str, object]:
    scene_name = scene_dir.name
    scene_output = args.output_root / scene_name
    class_output = _class_output_root(scene_output, class_id)
    stdout_log, stderr_log = _class_log_paths(scene_output, class_id)

    if args.skip_existing and _class_is_complete(scene_output, class_id):
        result = {
            "scene": scene_name,
            "class_id": class_id,
            "status": "skipped_existing",
            "returncode": 0,
            "gpu_id": gpu_id,
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
        "--max-episode-retry",
        "1",
    ]
    if args.subcats:
        cmd.extend(["--subcats", args.subcats])
    if args.max_proposals is not None:
        cmd.extend(["--max-proposals", str(args.max_proposals)])
    if args.target_per_subcat is not None:
        cmd.extend(["--target-per-subcat", str(args.target_per_subcat)])
    if args.reset_output:
        cmd.append("--reset-output")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["HABITAT_GPU_DEVICE_ID"] = "0"

    with stdout_log.open("w", encoding="utf-8") as fout, stderr_log.open("w", encoding="utf-8") as ferr:
        cp = subprocess.run(cmd, stdout=fout, stderr=ferr, text=True, env=env)

    result = {
        "scene": scene_name,
        "class_id": class_id,
        "status": "ok" if cp.returncode == 0 else "failed",
        "returncode": cp.returncode,
        "gpu_id": gpu_id,
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
    tasks = [(scene_dir, class_id) for scene_dir in scenes for class_id in classes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_workers = max(1, min(int(args.num_workers), len(tasks) if tasks else 1))

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for idx, (scene_dir, class_id) in enumerate(tasks):
            gpu_id = gpu_ids[idx % len(gpu_ids)] if gpu_ids else None
            future = executor.submit(_run_one_task, args, scene_dir, class_id, gpu_id)
            future_map[future] = (scene_dir.name, class_id)
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
                    "gpu_id": None,
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
                f"gpu={result.get('gpu_id')} returncode={result['returncode']}",
                flush=True,
            )

    summary = {
        "scene_root": str(args.scene_root),
        "output_root": str(args.output_root),
        "scene_count": len(scenes),
        "task_count": len(results),
        "num_workers": max_workers,
        "requested_num_workers": int(args.num_workers),
        "gpu_ids": gpu_ids,
        "classes": classes,
        "subcats": args.subcats,
        "max_proposals": args.max_proposals,
        "target_per_subcat": args.target_per_subcat,
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
        "results": sorted(results, key=lambda item: (str(item["scene"]), int(item["class_id"]))),
    }
    summary_path = args.output_root / "render_parallel_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0 if not summary["failed_tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
