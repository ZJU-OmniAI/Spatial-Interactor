#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


TASKS = [
    "action_inference",
    "movement_sequence_sorting",
    "movement_degree_comparison",
    "motion_family_discrimination",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild REAL dataset while filtering only family QA.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/path/to/workspace/FINAL/REAL"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/path/to/workspace/FINAL/REAL_familyfix_v2"),
    )
    parser.add_argument("--family-frame-gap-max", type=int, default=160)
    parser.add_argument("--family-pose-gap-max", type=int, default=2)
    parser.add_argument("--translation-min", type=float, default=0.78)
    parser.add_argument("--translation-yaw-max", type=float, default=16.0)
    parser.add_argument("--translation-pitch-max", type=float, default=12.0)
    parser.add_argument("--rotation-yaw-min", type=float, default=40.0)
    parser.add_argument("--rotation-translation-max", type=float, default=0.50)
    parser.add_argument("--rotation-pitch-max", type=float, default=20.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def replace_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [replace_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, path_map) for key, item in value.items()}
    return value


def family_keep(row: dict[str, Any], args: argparse.Namespace) -> bool:
    m = row["gt"]["motion"]
    if m["frame_gap"] > args.family_frame_gap_max or m["pose_gap"] > args.family_pose_gap_max:
        return False
    if m["family"] == "translation":
        return (
            m["translation_m"] >= args.translation_min
            and abs(m["yaw_deg"]) <= args.translation_yaw_max
            and abs(m["pitch_deg"]) <= args.translation_pitch_max
        )
    if m["family"] == "rotation":
        return (
            abs(m["yaw_deg"]) >= args.rotation_yaw_min
            and m["translation_m"] <= args.rotation_translation_max
            and abs(m["pitch_deg"]) <= args.rotation_pitch_max
        )
    return False


def main() -> int:
    args = parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    copied_counts: dict[str, int] = {}
    family_removed_ids: list[str] = []
    kept_all: list[dict[str, Any]] = []

    for task in TASKS:
        src_task_dir = args.input_root / task
        dst_task_dir = args.output_root / task
        dst_task_dir.mkdir(parents=True, exist_ok=True)
        rows = read_json(src_task_dir / "qa_data.json")
        kept_rows: list[dict[str, Any]] = []

        for row in rows:
            if task == "motion_family_discrimination" and not family_keep(row, args):
                family_removed_ids.append(str(row["qa_id"]))
                continue

            qa_id = str(row["qa_id"])
            src_paths = [Path(p) for p in row["input"]["frame_paths"]]
            src_sample_dir = src_paths[0].parent
            dst_sample_dir = dst_task_dir / qa_id
            shutil.copytree(src_sample_dir, dst_sample_dir)
            path_map = {str(src): str(dst_sample_dir / src.name) for src in src_paths}

            new_row = deepcopy(row)
            new_row["input"] = replace_paths(new_row["input"], path_map)
            kept_rows.append(new_row)
            kept_all.append(new_row)

        copied_counts[task] = len(kept_rows)
        write_json(dst_task_dir / "qa_data.json", kept_rows)

    (args.output_root / "family_removed_qa_ids.txt").write_text(
        "\n".join(family_removed_ids) + ("\n" if family_removed_ids else ""),
        encoding="utf-8",
    )
    write_json(
        args.output_root / "summary.json",
        {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "policy": {
                "other_tasks": "unchanged from input root",
                "motion_family_discrimination": {
                    "frame_gap_max": args.family_frame_gap_max,
                    "pose_gap_max": args.family_pose_gap_max,
                    "translation_rule": {
                        "translation_m_min": args.translation_min,
                        "abs_yaw_deg_max": args.translation_yaw_max,
                        "abs_pitch_deg_max": args.translation_pitch_max,
                    },
                    "rotation_rule": {
                        "abs_yaw_deg_min": args.rotation_yaw_min,
                        "translation_m_max": args.rotation_translation_max,
                        "abs_pitch_deg_max": args.rotation_pitch_max,
                    },
                },
            },
            "copied_counts": copied_counts,
            "removed_family_count": len(family_removed_ids),
            "kept_total": len(kept_all),
        },
    )
    write_json(
        args.output_root / "qa_data_all.json",
        {
            "qa_count": len(kept_all),
            "data": kept_all,
        },
    )
    print(json.dumps(read_json(args.output_root / "summary.json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
