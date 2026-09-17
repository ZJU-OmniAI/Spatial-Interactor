#!/usr/bin/env python3
"""Assign generated LSI rows to the three curriculum levels."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TASK_LEVELS = {
    # Passive world-state transitions.
    "object_state_attribute_changes": "l1",
    "object_position_swapping": "l1",
    "dynamic_movement_occlusion": "l1",
    "pure_two_frame_motion_magnitude": "l1",
    "action_to_image_choice": "l1",
    "temporal_sequence_sorting": "l1",
    "long_horizon_manipulation_program": "l1",
    # Active self-state transitions.
    "action_inference": "l2",
    "multi_image_overlap_localization": "l2",
    "parallax_depth_inference": "l2",
    "movement_sequence_sorting": "l2",
    "movement_degree_comparison": "l2",
    "imagined_perspective_taking": "l2",
    "imagined_movement_consequence": "l2",
    "motion_family_discrimination": "l2",
    "distance_to_start_comparison": "l2",
    "return_to_start_detection": "l2",
    # Long-horizon interaction trajectories.
    "node_reverse_path_planning": "l3",
    "video_reverse_path": "l3",
    "roomtour3d_metric_pose_interval_summary": "l3",
    "roomtour3d_pose_interval_summary": "l3",
    "roomtour3d_coarse_video_motion": "l3",
    "roomtour3d_simple_path_shape": "l3",
    "roomtour3d_turning_trajectory_interval": "l3",
    "roomtour3d_revisited_location_pair": "l3",
    "roomtour3d_main_turn_interval": "l3",
    "trajectory_interval_summary": "l3",
    "trajectory_metric_05m_5deg": "l3",
    "simple_path_shape": "l3",
    "turning_trajectory_interval": "l3",
    "revisited_location_pair": "l3",
    "main_turn_interval": "l3",
    "pose_interval_summary": "l3",
    "coarse_video_motion": "l3",
}

CLASS_TASKS = {
    1: "action_inference",
    2: "multi_image_overlap_localization",
    3: "parallax_depth_inference",
    4: "movement_sequence_sorting",
    5: "movement_degree_comparison",
    6: "object_state_attribute_changes",
    7: "object_position_swapping",
    8: "dynamic_movement_occlusion",
    9: "imagined_perspective_taking",
    10: "imagined_movement_consequence",
}


def read_rows(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} is not an object")
                    yield path, line_number, row


def explicit_task(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(
        row.get("task_type")
        or metadata.get("task_type")
        or metadata.get("original_question_type")
        or ""
    )


def infer_task(row: dict[str, Any], mapping: dict[str, str]) -> str:
    task = explicit_task(row)
    if task:
        return task
    row_id = str(row.get("id") or row.get("qa_id") or "").lower()
    for candidate in sorted(mapping, key=len, reverse=True):
        if candidate.lower() in row_id:
            return candidate
    class_match = re.search(r"(?:^|__)class(\d{1,2})(?:__|$)", row_id)
    if class_match:
        return CLASS_TASKS.get(int(class_match.group(1)), "")
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Optional JSON object mapping additional task_type values to l1/l2/l3.",
    )
    parser.add_argument("--paper-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = dict(DEFAULT_TASK_LEVELS)
    if args.mapping:
        custom = json.loads(args.mapping.read_text(encoding="utf-8"))
        if not isinstance(custom, dict):
            raise ValueError("--mapping must contain a JSON object")
        mapping.update({str(key): str(value).lower() for key, value in custom.items()})
    invalid_levels = {value for value in mapping.values() if value not in {"l1", "l2", "l3"}}
    if invalid_levels:
        raise ValueError(f"Invalid curriculum levels in mapping: {invalid_levels}")

    grouped: dict[str, list[dict[str, Any]]] = {"l1": [], "l2": [], "l3": []}
    task_counts = Counter()
    unresolved = []
    for path, line_number, row in read_rows(args.input):
        metadata = row.get("metadata") or {}
        explicit_level = str(metadata.get("curriculum_level") or "").lower()
        task = infer_task(row, mapping)
        level = explicit_level or mapping.get(task, "")
        if level not in grouped:
            unresolved.append(
                {
                    "location": f"{path}:{line_number}",
                    "id": row.get("id") or row.get("qa_id"),
                    "task_type": task,
                }
            )
            continue
        output_row = dict(row)
        output_metadata = dict(metadata)
        output_metadata["curriculum_level"] = level
        output_metadata["curriculum_task_type"] = task
        output_row["metadata"] = output_metadata
        grouped[level].append(output_row)
        task_counts[(level, task)] += 1

    if unresolved:
        preview = json.dumps(unresolved[:20], indent=2)
        raise RuntimeError(
            f"Could not assign {len(unresolved)} rows. Add explicit metadata or --mapping.\n{preview}"
        )

    counts = {level: len(rows) for level, rows in grouped.items()}
    if args.paper_counts:
        expected = {
            "l1": 15_109,
            "l2": 69_487,
            "l1_l2": 84_596,
            "l3": 22_922,
            "total": 107_518,
        }
        actual = {
            "l1": counts["l1"],
            "l2": counts["l2"],
            "l1_l2": counts["l1"] + counts["l2"],
            "l3": counts["l3"],
            "total": sum(counts.values()),
        }
        if actual != expected:
            raise RuntimeError(f"LSI-108K count mismatch: expected {expected}, got {actual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for level, rows in grouped.items():
        path = args.output_dir / f"lsi_{level}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "total_rows": sum(counts.values()),
        "level_counts": counts,
        "l1_l2_rows": counts["l1"] + counts["l2"],
        "task_counts": {
            f"{level}:{task}": count for (level, task), count in sorted(task_counts.items())
        },
    }
    (args.output_dir / "curriculum_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
