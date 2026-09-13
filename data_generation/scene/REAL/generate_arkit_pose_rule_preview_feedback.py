#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_arkit_pose_rule_preview_rawtraj as base  # noqa: E402


OUTPUT_ROOT = Path("/path/to/workspace/DATA/ARKIT_ACTION_PREVIEW")


def motion_between_feedback(scene: base.ScenePose, start: int, end: int) -> dict:
    motion = base.motion_between(scene, start, end, local_rule="flip_z")
    # User validation on ARKit rx-first preview: local x sign was reversed
    # (a pose-left sample looked visually right). Keep z flipped for camera forward.
    d = motion["local_delta_hypothesis_xyz"]
    d[0] *= -1.0
    motion["local_delta_hypothesis_xyz"] = d
    return motion


def classify_feedback(motion: dict) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta_hypothesis_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    translation = float(motion["translation_m"])
    gap = int(motion["frame_gap"])

    arx, ary, arz = abs(rx), abs(ry), abs(rz)

    # Pitch is the most reliable ARKit cue after user feedback:
    # rx < 0 visually up, rx > 0 visually down.
    if (
        40 <= gap <= 320
        and 24.0 <= arx <= 55.0
        and arx >= 2.2 * max(ary, arz, 1e-6)
        and translation <= 0.45
    ):
        label = "向上转动" if rx < 0 else "向下转动"
        score = arx * 3.0 - ary * 7.0 - arz * 5.0 - translation * 20.0
        return label, "rotation", arx, score

    # Yaw samples are kept only when pitch/roll are small; sign still needs user validation.
    if (
        40 <= gap <= 320
        and 25.0 <= ary <= 55.0
        and ary >= 2.5 * max(arx, arz, 1e-6)
        and translation <= 0.35
    ):
        label = "向右转动" if ry > 0 else "向左转动"
        score = ary * 3.0 - arx * 8.0 - arz * 5.0 - translation * 25.0
        return label, "rotation", ary, score

    # Translation: much stricter rotation limit to avoid up/down-turn samples being used as move.
    if (
        40 <= gap <= 340
        and 0.35 <= translation <= 1.6
        and abs(dy) <= 0.18
        and arx <= 3.5
        and ary <= 4.0
        and arz <= 4.0
    ):
        if abs(dx) >= abs(dz):
            primary, secondary = abs(dx), abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary, secondary = abs(dz), abs(dx)
            label = "向前移动" if dz > 0 else "向后移动"
        if primary < 0.35 or primary < 3.5 * max(secondary, abs(dy), 1e-6):
            return None, None, 0.0, 0.0
        score = primary * 70.0 + primary / max(secondary, abs(dy), 1e-6) * 2.0 - (arx + ary + arz) * 8.0
        return label, "translation", primary, score

    return None, None, 0.0, 0.0


def collect(scenes: list[base.ScenePose]) -> dict[str, list[base.Candidate]]:
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
                motion = motion_between_feedback(scene, start, end)
                label, family, magnitude, score = classify_feedback(motion)
                if not label:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(base.Candidate(scene.scene, scene.scene_dir, start, end, label, family, magnitude, motion, score))
        local.sort(key=lambda c: c.score, reverse=True)
        counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if counts[cand.label] >= 3:
                continue
            per_label[cand.label].append(cand)
            counts[cand.label] += 1
        for label in base.ACTION_OPTIONS:
            per_label[label].sort(key=lambda c: c.score, reverse=True)
            del per_label[label][50:]
        print("[feedback-scan]", scene.scene, {k: len(per_label[k]) for k in base.ACTION_OPTIONS}, flush=True)
        if all(per_label[k] for k in base.ACTION_OPTIONS):
            # Good enough for fast preview.
            break
    return per_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=base.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=80)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    base.ensure_dir(args.output_root / "action_inference")

    scenes: list[base.ScenePose] = []
    for scene_dir in base.scene_dirs(args.data_root)[: args.max_scenes]:
        scene = base.load_scene_pose(scene_dir, inverse=False, traj_order="rx_ry_rz_tx_ty_tz")
        if scene is not None:
            scenes.append(scene)

    per_label = collect(scenes)
    rows = base.build_rows(per_label, args.output_root, "arkit_feedback_v2", "flip_x_flip_z", "pitch_rx_neg_up", False)
    base.write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    base.write_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", rows)
    base.make_overview(args.output_root)
    summary = {
        "dataset": "arkitscenes",
        "rule_name": "arkit_feedback_v2",
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in base.ACTION_OPTIONS},
        "candidate_counts": {label: len(per_label[label]) for label in base.ACTION_OPTIONS},
        "feedback_used": [
            "traj order is timestamp rx ry rz tx ty tz",
            "local x sign flipped from previous preview",
            "pitch rx<0 is visual up and rx>0 is visual down",
            "translation requires very small rx/ry/rz to avoid pitch samples",
        ],
    }
    base.write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
