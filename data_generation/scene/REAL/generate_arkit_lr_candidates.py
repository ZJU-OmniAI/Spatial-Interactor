#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_arkit_pose_rule_preview_rawtraj as base  # noqa: E402


OUTPUT_ROOT = Path("/path/to/workspace/DATA/ARKIT_LR_CANDIDATES")
LABELS = ["左移候选", "右移候选", "左转候选", "右转候选"]


def classify_lr(motion: dict) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta_hypothesis_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    trans = float(motion["translation_m"])
    gap = int(motion["frame_gap"])

    arx, ary, arz = abs(rx), abs(ry), abs(rz)

    # Based on user feedback for ARKit rx-first:
    # previous x-dominant "left/right" looked like front/back, z-dominant looked vertical.
    # So search y-dominant translations as left/right candidates.
    if (
        40 <= gap <= 360
        and 0.35 <= abs(dy) <= 1.20
        and abs(dy) >= 2.8 * max(abs(dx), abs(dz), 1e-6)
        and max(arx, ary, arz) <= 6.0
    ):
        # User feedback: previous dy sign likely reversed.
        label = "右移候选" if dy > 0 else "左移候选"
        score = abs(dy) * 80.0 + abs(dy) / max(abs(dx), abs(dz), 1e-6) * 4.0 - (arx + ary + arz) * 3.0
        return label, "translation", abs(dy), score

    # User feedback: rx and ry candidates looked like up/down turns.
    # Search roll/z-axis dominant rotations as left/right turn candidates.
    if (
        40 <= gap <= 360
        and 18.0 <= arz <= 55.0
        and arz >= 2.0 * max(arx, ary, 1e-6)
        and trans <= 0.60
    ):
        # User feedback: previous left/right turn labels were reversed.
        label = "右转候选" if rz > 0 else "左转候选"
        score = arz * 4.0 - max(arx, ary) * 8.0 - trans * 20.0
        return label, "rotation", arz, score

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
                motion = base.motion_between(scene, start, end, local_rule="none")
                label, family, mag, score = classify_lr(motion)
                if not label:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(mag)
                local.append(base.Candidate(scene.scene, scene.scene_dir, start, end, label, family, mag, motion, score))

        local.sort(key=lambda c: c.score, reverse=True)
        counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if counts[cand.label] >= 2:
                continue
            per_label[cand.label].append(cand)
            counts[cand.label] += 1
        for label in LABELS:
            per_label[label].sort(key=lambda c: c.score, reverse=True)
            del per_label[label][20:]
        print("[lr-scan]", scene.scene, {k: len(per_label[k]) for k in LABELS}, flush=True)
        if all(per_label[k] for k in LABELS):
            break
    return per_label


def build_rows(per_label: dict[str, list[base.Candidate]], out: Path) -> list[dict]:
    rows: list[dict] = []
    for label in LABELS:
        # Keep two examples per label so user can judge direction and purity quickly.
        for cand in per_label[label][:2]:
            m = cand.motion
            frame_paths = [
                cand.scene_dir / "color" / f"{m['start_frame_index']:06d}.jpg",
                cand.scene_dir / "color" / f"{m['end_frame_index']:06d}.jpg",
            ]
            q = base.image_quality(frame_paths)
            if q is None or q["sharpness_min"] < 10 or q["pair_gray_mad"][0] < 5:
                continue
            qa_id = f"arkit_lr_{len(rows)+1:03d}"
            paths = base.save_frames(out / "action_inference" / qa_id, frame_paths)
            rows.append({
                "qa_id": qa_id,
                "task_type": "action_inference",
                "dataset": "arkitscenes",
                "scene": cand.scene,
                "question": "图A到图B之间，相机主要执行了什么动作？",
                "answer": label,
                "answer_with_value": base.action_value_text(label, cand.family, cand.magnitude),
                "options": LABELS,
                "label_source": "lr_axis_candidate_needs_user_validation",
                "input": {"frame_paths": [paths["frame_A"], paths["frame_B"]], "frame_A": paths["frame_A"], "frame_B": paths["frame_B"], "contact_sheet": paths["contact_sheet"]},
                "gt": {"motion": m, "approx_action": {"text": base.action_value_text(label, cand.family, cand.magnitude), "value": round(cand.magnitude, 3), "unit": "deg" if cand.family == "rotation" else "m"}, "quality": {"pose_score": cand.score, **q}},
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=base.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=100)
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
    rows = build_rows(per_label, args.output_root)
    base.write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    base.write_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", rows)
    base.make_overview(args.output_root)
    summary = {
        "dataset": "arkitscenes",
        "rule_name": "lr_candidates_y_translation_z_rotation",
        "count": len(rows),
        "label_counts": {label: sum(1 for r in rows if r["answer"] == label) for label in LABELS},
        "candidate_counts": {label: len(per_label[label]) for label in LABELS},
        "notes": "Only left/right movement/turn candidates. Not final labels until user validates.",
    }
    base.write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
