#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_motion_run_index as base  # noqa: E402
import build_motion_segments_033_pilot as segments  # noqa: E402
import generate_arkit_ry_rz_large_sweep as arkit  # noqa: E402


DEFAULT_SCENE_LIST = Path("/path/to/workspace/DATA/arkit_missing_downloaded_not_segmented_training.json")
DEFAULT_OUTPUT = Path("/path/to/workspace/MOTION_SEGMENTS_033_FAST/_arkit_shards/shard_128_incremental/arkit")


def load_scene_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in payload:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("scene"):
            out.append(str(item["scene"]))
    return sorted(dict.fromkeys(out))


def existing_scenes(path: Path) -> set[str]:
    scenes: set[str] = set()
    if not path.exists():
        return scenes
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("scene"):
                scenes.add(str(row["scene"]))
    return scenes


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_segment_args() -> SimpleNamespace:
    return SimpleNamespace(
        trans_unit_m=0.05,
        rot_unit_deg=4.0,
        min_step_norm=0.12,
        step_ratio=1.15,
        min_steps=4,
        max_run_steps=90,
        max_impure_steps=18,
        min_purity=0.76,
        min_monotonicity=0.94,
        min_translation_m=0.35,
        min_rotation_deg=30.0,
        long_steps=24,
        long_translation_m=1.2,
        long_rotation_deg=90.0,
    )


def process_scene(scene_id: str) -> dict[str, Any]:
    scene_dir = arkit.DATA_ROOT / scene_id
    scene = arkit.load_scene_pose(scene_dir)
    if scene is None:
        return {"scene": scene_id, "status": "failed", "error": "load_scene_pose_none", "segments": []}
    width, height = base.arkit_size(scene)
    loaded = base.LoadedScene(
        "arkit",
        scene.scene,
        scene.frame_ids,
        scene,
        arkit.motion_between,
        base.axes_arkit,
        {
            "pose_rule": "size split; 1920x1440: dx right dz forward ry right-turn rx up-turn; 1440x1920: dy right dz forward rx right-turn -ry up-turn",
            "width": width,
            "height": height,
            "incremental_source": "downloaded_training_not_previously_segmented",
        },
    )
    rows, stats = segments.segment_scene(loaded, make_segment_args())
    return {"scene": scene_id, "status": "done", "segments": rows, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ARKit motion segments for downloaded Training scenes missing from old shards.")
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()

    segment_path = args.output_dir / "segments.jsonl"
    summary_path = args.output_dir / "summary.json"
    scene_ids = load_scene_ids(args.scene_list)
    done_scenes = existing_scenes(segment_path)
    todo = [scene_id for scene_id in scene_ids if scene_id not in done_scenes]
    if args.max_scenes:
        todo = todo[: args.max_scenes]

    segment_type_counts: Counter = Counter()
    action_counts: Counter = Counter()
    scene_stats = []
    failed = []
    appended_segments = 0

    print(json.dumps({"todo": len(todo), "existing_scenes_in_incremental_output": len(done_scenes), "workers": args.workers}, ensure_ascii=False), flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_scene, scene_id): scene_id for scene_id in todo}
        for idx, future in enumerate(as_completed(futures), start=1):
            scene_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failed.append({"scene": scene_id, "error": repr(exc)})
                print(f"[segments-incremental] {idx}/{len(todo)} failed {scene_id}: {exc!r}", flush=True)
                continue
            if result.get("status") != "done":
                failed.append({"scene": scene_id, "error": result.get("error")})
                print(f"[segments-incremental] {idx}/{len(todo)} failed {scene_id}: {result.get('error')}", flush=True)
                continue
            rows = result["segments"]
            append_jsonl(segment_path, rows)
            appended_segments += len(rows)
            stats = dict(result["stats"])
            stats["scene"] = scene_id
            scene_stats.append(stats)
            for row in rows:
                segment_type_counts[row.get("segment_type")] += 1
                if row.get("action_label"):
                    action_counts[row["action_label"]] += 1
            if idx % 10 == 0 or idx == len(todo):
                print(
                    f"[segments-incremental] {idx}/{len(todo)} scenes, appended_segments={appended_segments}, failed={len(failed)}",
                    flush=True,
                )

    summary = {
        "scene_list": str(args.scene_list),
        "output_dir": str(args.output_dir),
        "scenes_requested": len(scene_ids),
        "scenes_skipped_existing_in_incremental_output": len(done_scenes),
        "scenes_attempted_this_run": len(todo),
        "scenes_done_this_run": len(scene_stats),
        "failed": failed,
        "segments_appended_this_run": appended_segments,
        "segment_type_counts_this_run": dict(segment_type_counts),
        "action_counts_pure_segments_this_run": dict(action_counts),
        "scene_stats": scene_stats,
        "segments_jsonl": str(segment_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
