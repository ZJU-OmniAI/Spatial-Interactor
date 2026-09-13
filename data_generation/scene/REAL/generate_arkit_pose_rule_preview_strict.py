#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_arkit_pose_rule_preview_rawtraj as base  # noqa: E402


OUTPUT_ROOT = Path("/path/to/workspace/DATA/ARKIT_ACTION_PREVIEW_STRICT")


def strict_classify(motion: dict) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta_hypothesis_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    translation = float(motion["translation_m"])
    gap = int(motion["frame_gap"])

    abs_rx, abs_ry, abs_rz = abs(rx), abs(ry), abs(rz)
    rot_primary = max(abs_rx, abs_ry)
    rot_secondary = min(abs_rx, abs_ry)

    # Pure rotations: clear 30-55 degree pitch/yaw, very little translation.
    # ARKit raw yaw follows right-hand convention: +ry looks visually like left turn.
    if (
        40 <= gap <= 260
        and translation <= 0.22
        and max(abs(dx), abs(dy), abs(dz)) <= 0.20
        and 30.0 <= rot_primary <= 55.0
        and rot_primary >= 3.5 * max(rot_secondary, abs_rz, 1e-6)
        and abs_rz <= 7.0
    ):
        if abs_rx >= abs_ry:
            label = "向上转动" if rx > 0 else "向下转动"
            magnitude = abs_rx
        else:
            label = "向左转动" if ry > 0 else "向右转动"
            magnitude = abs_ry
        score = magnitude * 3.0 - translation * 60.0 - rot_secondary * 4.0 - abs_rz * 4.0
        return label, "rotation", magnitude, score

    # Pure translations: dominant local x/z movement, tiny rotation and vertical drift.
    if (
        40 <= gap <= 320
        and 0.35 <= translation <= 1.4
        and abs(dy) <= 0.12
        and rot_primary <= 5.0
        and abs_rz <= 4.0
    ):
        if abs(dx) >= abs(dz):
            primary = abs(dx)
            secondary = abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary = abs(dz)
            secondary = abs(dx)
            label = "向前移动" if dz > 0 else "向后移动"
        if primary < 0.35 or primary < 4.0 * max(secondary, abs(dy), 1e-6):
            return None, None, 0.0, 0.0
        score = primary * 60.0 + primary / max(secondary, abs(dy), 1e-6) * 3.0 - rot_primary * 8.0 - abs_rz * 5.0
        return label, "translation", primary, score

    return None, None, 0.0, 0.0


def collect_strict_candidates(scenes: list[base.ScenePose]) -> dict[str, list[base.Candidate]]:
    per_label: dict[str, list[base.Candidate]] = defaultdict(list)
    gaps = [4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36]
    for scene in scenes:
        local: list[base.Candidate] = []
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(2, gap_slots // 2)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = base.motion_between(scene, start, end, local_rule="flip_z")
                label, family, magnitude, score = strict_classify(motion)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(base.Candidate(scene.scene, scene.scene_dir, start, end, label, family, magnitude, motion, score))
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.label] >= 2:
                continue
            per_label[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in base.ACTION_OPTIONS:
            per_label[label].sort(key=lambda item: item.score, reverse=True)
            del per_label[label][40:]
        print("[strict-scan] %s %s" % (scene.scene, {k: len(per_label[k]) for k in base.ACTION_OPTIONS}), flush=True)
        if all(per_label[label] for label in base.ACTION_OPTIONS):
            # Keep scanning a few more scenes via max-scenes for quality; do not break early.
            pass
    return per_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=base.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=120)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    base.ensure_dir(args.output_root / "action_inference")

    loaded: list[base.ScenePose] = []
    for scene_dir in base.scene_dirs(args.data_root)[: args.max_scenes]:
        scene = base.load_scene_pose(scene_dir, inverse=False)
        if scene is not None:
            loaded.append(scene)

    per_label = collect_strict_candidates(loaded)
    rows = base.build_rows(per_label, args.output_root, "raw_arkit_strict", "flip_z", "flip_yaw", False)
    base.write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    base.write_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", rows)
    base.make_overview(args.output_root)

    summary = {
        "dataset": "arkitscenes",
        "rule_name": "raw_arkit_strict",
        "output_root": str(args.output_root),
        "scenes_loaded": len(loaded),
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in base.ACTION_OPTIONS},
        "candidate_counts": {label: len(per_label[label]) for label in base.ACTION_OPTIONS},
        "pose_rule": {
            "traj_order": "timestamp tx ty tz rx ry rz",
            "local_rule": "flip_z",
            "yaw_visual_rule": "+ry is left turn, -ry is right turn",
            "strict_purity": "translation rot<=5deg and dominant axis>=4x; rotation translation<=0.22m and dominant angle>=3.5x",
        },
    }
    base.write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
