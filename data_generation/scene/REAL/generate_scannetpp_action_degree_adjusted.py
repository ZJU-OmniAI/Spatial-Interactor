#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BASE_SCRIPT = Path("/path/to/workspace/DATA/CODE/SCENE/REAL/generate_scannetpp_three_tasks.py")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/SCANNETPP_ADJUSTED")


def load_base_module():
    spec = importlib.util.spec_from_file_location("scannetpp_three_tasks", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_scene_poses(base, scene_paths: Iterable[Path]):
    for idx, scene_dir in enumerate(scene_paths, start=1):
        pose = base.load_scene_pose(scene_dir)
        if pose is None:
            print(f"[pose] skip {idx}: {scene_dir.name}", flush=True)
            continue
        print(f"[pose] loaded {idx}: {scene_dir.name} frames={len(pose.frame_ids)}", flush=True)
        yield pose


def save_task_rows(output_root: Path, task: str, rows: list[dict]) -> None:
    write_json(output_root / task / "qa_data.json", rows)
    write_jsonl(output_root / task / "qa_data.jsonl", rows)


def stream_action_rows(base, scene_paths: list[Path], target: int, output_root: Path) -> list[dict]:
    rows = []
    gaps = [50, 60, 80, 100, 120, 150, 180, 220]
    label_cap = math.ceil(target / len(base.ACTION_OPTIONS)) if target > 0 else 10**9
    label_counts = defaultdict(int)
    used_segments: set[tuple[str, int, int]] = set()

    for scene_idx, scene in enumerate(iter_scene_poses(base, scene_paths), start=1):
        if target > 0 and len(rows) >= target:
            break
        local = []
        for gap in gaps:
            if gap >= len(scene.frame_ids):
                continue
            step = max(15, gap // 3)
            for start in range(0, len(scene.frame_ids) - gap, step):
                end = start + gap
                motion = base.motion_between(scene, start, end)
                label, family, magnitude, score = base.classify_motion(motion)
                if not label or not family:
                    continue
                if label_counts[label] >= label_cap:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append((score, scene.scene, start, end, label, family, float(magnitude), motion))
        local.sort(key=lambda item: item[0], reverse=True)
        scene_counts = defaultdict(int)
        for score, scene_name, start, end, label, family, magnitude, motion in local:
            if target > 0 and len(rows) >= target:
                return rows
            if label_counts[label] >= label_cap:
                continue
            if scene_counts[label] >= 4:
                continue
            key = (scene_name, start, end)
            if key in used_segments:
                continue
            loaded = base.load_frames(scene_name, [motion["start_frame_index"], motion["end_frame_index"]])
            if loaded is None:
                continue
            frames, quality = loaded
            if quality["sharpness_min"] < 8 or quality["pair_gray_mad"][0] < 20:
                continue
            qa_id = f"scannetpp_action_inference_{len(rows) + 1:04d}"
            sample_dir = output_root / "action_inference" / qa_id
            paths = base.save_frames(sample_dir, [("frame_A", frames[0]), ("frame_B", frames[1])])
            rows.append(
                {
                    "qa_id": qa_id,
                    "task_type": "action_inference",
                    "dataset": "scannetpp",
                    "scene": scene_name,
                    "pose_source": "scannetpp_aligned_pose_relative_local_rule_v1_human_verified",
                    "question": "图A到图B之间，相机主要执行了什么动作？",
                    "answer": label,
                    "answer_with_value": base.action_value_text(label, family, magnitude),
                    "options": base.ACTION_OPTIONS,
                    "label_source": "predicted_by_confirmed_scannetpp_pose_rule",
                    "input": {
                        "frame_paths": [paths["frame_A"], paths["frame_B"]],
                        "frame_A": paths["frame_A"],
                        "frame_B": paths["frame_B"],
                        "contact_sheet": paths["contact_sheet"],
                    },
                    "gt": {
                        "motion": motion,
                        "approx_action": {
                            "text": base.action_value_text(label, family, magnitude),
                            "value": round(magnitude, 3),
                            "unit": "deg" if family == "rotation" else "m",
                        },
                        "quality": {"pose_score": score, **quality},
                    },
                }
            )
            used_segments.add(key)
            scene_counts[label] += 1
            label_counts[label] += 1
        if rows:
            save_task_rows(output_root, "action_inference", rows)
        print(
            "[action] scene=%s idx=%d total=%d labels=%s"
            % (scene.scene, scene_idx, len(rows), dict(label_counts)),
            flush=True,
        )
    return rows


def load_scenes_for_degree(base, scene_paths: list[Path]) -> list:
    scenes = []
    for scene in iter_scene_poses(base, scene_paths):
        scenes.append(scene)
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--target-action", type=int, default=2000)
    parser.add_argument("--target-degree", type=int, default=2000)
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()

    base = load_base_module()
    random.seed(20260513)

    args.output_root.mkdir(parents=True, exist_ok=True)

    scene_paths = base.scene_dirs(base.DATA_ROOT)
    if args.max_scenes > 0:
        scene_paths = scene_paths[: args.max_scenes]

    action_rows = stream_action_rows(base, scene_paths, args.target_action, args.output_root)
    print("[degree] loading poses for capped degree comparison", flush=True)
    scenes = load_scenes_for_degree(base, scene_paths)
    print(f"[degree] loaded scenes={len(scenes)}; collecting target={args.target_degree}", flush=True)
    degree_rows = base.collect_degree_candidates(scenes, args.target_degree, args.output_root)

    task_rows = {
        "action_inference": action_rows,
        "movement_degree_comparison": degree_rows,
    }
    for task, rows in task_rows.items():
        save_task_rows(args.output_root, task, rows)

    summary = {
        "dataset": "scannetpp",
        "output_root": str(args.output_root),
        "scenes_loaded": len(scenes),
        "target_action": args.target_action,
        "target_degree": args.target_degree,
        "counts": {task: len(rows) for task, rows in task_rows.items()},
        "action_label_counts": {
            label: sum(1 for row in action_rows if row.get("answer") == label)
            for label in base.ACTION_OPTIONS
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
