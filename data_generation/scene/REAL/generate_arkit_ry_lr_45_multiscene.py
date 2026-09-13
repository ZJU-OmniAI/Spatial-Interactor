#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import generate_arkit_ry_rz_large_sweep as base


DATA_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/QUICK_EIGHT_ACTIONS/ARKIT_RY_LR_45_MULTISCENE")
LABELS = ["向左转动", "向右转动"]
EXCLUDE_SCENES = {"41007589"}


def classify_ry_turn(motion: dict) -> tuple[str | None, float, float]:
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    dx, dy, dz = [float(x) for x in motion["local_delta_camera_xyz"]]
    trans = float(motion["translation_m"])
    gap = int(motion["frame_gap"])
    ary = abs(ry)
    if not (60 <= gap <= 300):
        return None, 0.0, 0.0
    if not (34.0 <= ary <= 58.0):
        return None, 0.0, 0.0
    if ary < 2.6 * max(abs(rx), abs(rz), 1e-6):
        return None, 0.0, 0.0
    if trans > 0.45:
        return None, 0.0, 0.0
    if max(abs(dx), abs(dy), abs(dz)) > 0.42:
        return None, 0.0, 0.0
    label = "向左转动" if ry < 0 else "向右转动"
    target_penalty = abs(ary - 45.0)
    score = 150.0 - target_penalty * 4.5 - max(abs(rx), abs(rz)) * 7.5 - trans * 28.0
    return label, ary, score


def collect(scenes: list[base.ScenePose], per_label: int) -> dict[str, list[base.Candidate]]:
    per_label_cands: dict[str, list[base.Candidate]] = defaultdict(list)
    gaps = [6, 8, 10, 12, 15, 18, 21, 24, 27, 30]
    for scene in scenes:
        if scene.scene in EXCLUDE_SCENES:
            continue
        local = []
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(2, gap_slots // 2)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = base.motion_between(scene, start, end)
                label, mag, score = classify_ry_turn(motion)
                if label is None:
                    continue
                local.append(base.Candidate(scene, start, end, label, mag, motion, score))
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.bucket] >= 1:
                continue
            per_label_cands[cand.bucket].append(cand)
            scene_counts[cand.bucket] += 1
        for label in LABELS:
            per_label_cands[label].sort(key=lambda item: item.score, reverse=True)
            del per_label_cands[label][per_label * 4 :]
        print("[arkit-ry-lr45]", scene.scene, {k: len(per_label_cands[k]) for k in LABELS}, flush=True)
        if all(len(per_label_cands[k]) >= per_label * 2 for k in LABELS):
            break
    return per_label_cands


def add_scene_candidates(scene: base.ScenePose, per_label_cands: dict[str, list[base.Candidate]], per_label: int) -> None:
    if scene.scene in EXCLUDE_SCENES:
        return
    gaps = [6, 8, 10, 12, 15, 18, 21, 24, 27, 30]
    local = []
    for gap_slots in gaps:
        if gap_slots >= len(scene.frame_ids):
            continue
        step = max(2, gap_slots // 2)
        for start in range(0, len(scene.frame_ids) - gap_slots, step):
            end = start + gap_slots
            motion = base.motion_between(scene, start, end)
            label, mag, score = classify_ry_turn(motion)
            if label is None:
                continue
            local.append(base.Candidate(scene, start, end, label, mag, motion, score))
    local.sort(key=lambda item: item.score, reverse=True)
    scene_counts: dict[str, int] = defaultdict(int)
    for cand in local:
        if scene_counts[cand.bucket] >= 1:
            continue
        per_label_cands[cand.bucket].append(cand)
        scene_counts[cand.bucket] += 1
    for label in LABELS:
        per_label_cands[label].sort(key=lambda item: item.score, reverse=True)
        del per_label_cands[label][per_label * 4 :]


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 56), (255, 255, 255), -1)
    cv2.putText(out, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 220), 2, cv2.LINE_AA)
    return out


def build_rows(cands: dict[str, list[base.Candidate]], out: Path, per_label: int) -> list[dict]:
    rows = []
    used_scenes: dict[str, set[str]] = {label: set() for label in LABELS}
    panels = []
    report = [
        "# ARKit ry left/right 45 degree multiscene",
        "",
        "Confirmed working assumption from user judgment: ry < 0 is visual left turn; ry > 0 is paired right turn.",
        "All samples target about 45 degrees and avoid scene 41007589.",
    ]
    for label in LABELS:
        kept = 0
        for cand in cands[label]:
            if cand.scene.scene in used_scenes[label]:
                continue
            m = cand.motion
            frame_paths = [
                cand.scene.scene_dir / "color" / f"{m['start_frame_index']:06d}.jpg",
                cand.scene.scene_dir / "color" / f"{m['end_frame_index']:06d}.jpg",
            ]
            q = base.image_quality(frame_paths)
            if q is None or q["sharpness_min"] < 10 or q["pair_gray_mad"] < 6:
                continue
            qa_id = f"arkit_ry_lr45_{len(rows) + 1:03d}"
            paths = base.save_pair(out / "action_inference" / qa_id, q["frames"])
            rx, ry, rz = [float(x) for x in m["relative_rotvec_deg_xyz"]]
            text = f"{len(rows)+1}. {label} ry={ry:.1f} rx={rx:.1f} rz={rz:.1f}"
            contact = cv2.imread(paths["contact_sheet"])
            if contact is not None:
                panels.append(put_label(cv2.resize(contact, (960, 360)), text))
            rows.append({
                "qa_id": qa_id,
                "task_type": "action_inference",
                "dataset": "arkitscenes",
                "scene": cand.scene.scene,
                "pose_source": "arkitscenes_official_traj_axis_angle_translation_inverted_to_c2w",
                "question": "图A到图B之间，相机主要执行了什么动作？",
                "answer": label,
                "answer_with_value": f"{label}约{cand.axis_value:.1f}度",
                "options": LABELS,
                "label_source": "ry_turn_rule_user_seeded_needs_final_visual_check",
                "input": {
                    "frame_paths": [paths["frame_A"], paths["frame_B"]],
                    "frame_A": paths["frame_A"],
                    "frame_B": paths["frame_B"],
                    "contact_sheet": paths["contact_sheet"],
                },
                "gt": {
                    "motion": m,
                    "approx_action": {"text": f"{label}约{cand.axis_value:.1f}度", "value": round(cand.axis_value, 3), "unit": "deg"},
                    "quality": {
                        "pose_score": cand.score,
                        "sharpness_min": q["sharpness_min"],
                        "pair_gray_mad": q["pair_gray_mad"],
                    },
                    "pose_rule_hypothesis": {
                        "left_right_turn_rule": "ry < 0 left turn, ry > 0 right turn",
                        "target": "about 45 degrees",
                    },
                },
            })
            report.extend([
                "",
                f"## {len(rows)}. {label} ry={ry:.3f}",
                f"- scene: {cand.scene.scene}",
                f"- frames: {m['start_frame_index']} -> {m['end_frame_index']}, gap={m['frame_gap']}",
                f"- translation_m: {m['translation_m']:.4f}",
                f"- local_delta_camera_xyz: {[round(x, 4) for x in m['local_delta_camera_xyz']]}",
                f"- relative_rotvec_deg_xyz: {[round(x, 3) for x in m['relative_rotvec_deg_xyz']]}",
                f"- contact_sheet: {paths['contact_sheet']}",
            ])
            used_scenes[label].add(cand.scene.scene)
            kept += 1
            if kept >= per_label:
                break
    if panels:
        lines = []
        for idx in range(0, len(panels), 2):
            pair = panels[idx : idx + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            lines.append(np.concatenate(pair, axis=1))
        cv2.imwrite(str(out / "action_inference" / "overview_contact_sheet.png"), np.concatenate(lines, axis=0))
    (out / "action_inference" / "ry_lr45_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=140)
    parser.add_argument("--per-label", type=int, default=6)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    base.ensure_dir(args.output_root / "action_inference")

    cands: dict[str, list[base.Candidate]] = defaultdict(list)
    for idx, scene_dir in enumerate(base.scene_dirs(args.data_root)[: args.max_scenes], 1):
        scene = base.load_scene_pose(scene_dir)
        print(f"[load] {idx}/{args.max_scenes} {scene_dir.name} usable={scene is not None}", flush=True)
        if scene is not None:
            add_scene_candidates(scene, cands, args.per_label)
            print("[arkit-ry-lr45]", scene.scene, {k: len(cands[k]) for k in LABELS}, flush=True)
            if all(len(cands[k]) >= args.per_label * 2 for k in LABELS):
                break

    rows = build_rows(cands, args.output_root, args.per_label)
    base.write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    summary = {
        "dataset": "arkitscenes",
        "rule_name": "ry_left_right_45_multiscene",
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in LABELS},
        "candidate_counts": {label: len(cands[label]) for label in LABELS},
        "excluded_scenes": sorted(EXCLUDE_SCENES),
        "overview": str(args.output_root / "action_inference" / "overview_contact_sheet.png"),
        "report": str(args.output_root / "action_inference" / "ry_lr45_report.md"),
        "qa_data": str(args.output_root / "action_inference" / "qa_data.json"),
    }
    base.write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
