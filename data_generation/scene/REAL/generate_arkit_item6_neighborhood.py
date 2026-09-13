#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import generate_arkit_ry_rz_large_sweep as base


DATA_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/QUICK_EIGHT_ACTIONS/ARKIT_ITEM6_RY_MINUS_45DEG")
SCENE_ID = "41007589"
CENTER_START_FRAME = 1260
CENTER_END_FRAME = 1620


def nearest_slot(frame_ids: np.ndarray, frame_id: int) -> int:
    return int(np.argmin(np.abs(frame_ids.astype(np.int64) - int(frame_id))))


def score_candidate(motion: dict, center_start: int, center_end: int) -> tuple[bool, float]:
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    trans = float(motion["translation_m"])
    gap = int(motion["frame_gap"])
    if not (50 <= gap <= 520):
        return False, 0.0
    if not (-58.0 <= ry <= -30.0):
        return False, 0.0
    if abs(ry) < 1.25 * max(abs(rx), abs(rz), 1e-6):
        return False, 0.0
    if trans > 0.90:
        return False, 0.0
    start = int(motion["start_frame_index"])
    end = int(motion["end_frame_index"])
    center_penalty = (abs(start - center_start) + abs(end - center_end)) / 60.0
    target_penalty = abs(abs(ry) - 45.0)
    leakage = max(abs(rx), abs(rz)) + trans * 10.0
    score = 120.0 - target_penalty * 5.0 - leakage * 5.0 - center_penalty * 1.2
    return True, score


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 56), (255, 255, 255), -1)
    cv2.putText(out, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 220), 2, cv2.LINE_AA)
    return out


def main() -> None:
    scene_dir = DATA_ROOT / SCENE_ID
    scene = base.load_scene_pose(scene_dir)
    if scene is None:
        raise SystemExit(f"scene not usable: {scene_dir}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    out_dir = OUTPUT_ROOT / "action_inference"
    base.ensure_dir(out_dir)

    center_start_slot = nearest_slot(scene.frame_ids, CENTER_START_FRAME)
    center_end_slot = nearest_slot(scene.frame_ids, CENTER_END_FRAME)
    candidates = []
    seen = set()

    for start in range(max(0, center_start_slot - 24), min(len(scene.frame_ids) - 2, center_start_slot + 25)):
        for gap in [6, 8, 10, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 48]:
            end = start + gap
            if end >= len(scene.frame_ids):
                continue
            motion = base.motion_between(scene, start, end)
            ok, score = score_candidate(motion, CENTER_START_FRAME, CENTER_END_FRAME)
            if not ok:
                continue
            key = (motion["start_frame_index"], motion["end_frame_index"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append((score, start, end, motion))

    # Always include the exact item-6 pair first if it passes the same filter.
    exact_motion = base.motion_between(scene, center_start_slot, center_end_slot)
    exact_ok, exact_score = score_candidate(exact_motion, CENTER_START_FRAME, CENTER_END_FRAME)
    rows = []
    selected = []
    if exact_ok:
        selected.append((float("inf"), center_start_slot, center_end_slot, exact_motion))
        seen.add((exact_motion["start_frame_index"], exact_motion["end_frame_index"]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for item in candidates:
        key = (item[3]["start_frame_index"], item[3]["end_frame_index"])
        if key in {(x[3]["start_frame_index"], x[3]["end_frame_index"]) for x in selected}:
            continue
        selected.append(item)
        if len(selected) >= 16:
            break

    panels = []
    report = [
        "# ARKit item 6 ry- neighborhood",
        "",
        f"Base item: scene {SCENE_ID}, frames {CENTER_START_FRAME} -> {CENTER_END_FRAME}.",
        "All rows are ry-negative dominant candidates near that interval; labels are not final.",
    ]
    for idx, (score, start, end, motion) in enumerate(selected, 1):
        frame_paths = [
            scene.scene_dir / "color" / f"{motion['start_frame_index']:06d}.jpg",
            scene.scene_dir / "color" / f"{motion['end_frame_index']:06d}.jpg",
        ]
        q = base.image_quality(frame_paths)
        if q is None:
            continue
        qa_id = f"arkit_item6_ryminus_{idx:03d}"
        paths = base.save_pair(out_dir / qa_id, q["frames"])
        rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
        answer = "ry-候选"
        text = f"{idx}. {answer} ry={ry:.1f} rx={rx:.1f} rz={rz:.1f}"
        contact = cv2.imread(paths["contact_sheet"])
        if contact is not None:
            panels.append(put_label(cv2.resize(contact, (960, 360)), text))
        row = {
            "qa_id": qa_id,
            "task_type": "action_inference",
            "dataset": "arkitscenes",
            "scene": SCENE_ID,
            "question": "图A到图B之间，这个 ry- 候选视觉上像什么转动？",
            "answer": answer,
            "answer_with_value": f"ry- 约{abs(ry):.1f}度",
            "options": ["左转", "右转", "上转", "下转", "看不出来"],
            "label_source": "item6_neighborhood_for_visual_judgment",
            "input": {
                "frame_paths": [paths["frame_A"], paths["frame_B"]],
                "frame_A": paths["frame_A"],
                "frame_B": paths["frame_B"],
                "contact_sheet": paths["contact_sheet"],
            },
            "gt": {
                "motion": motion,
                "approx_action": {"text": f"ry- 约{abs(ry):.1f}度", "value": round(abs(ry), 3), "unit": "deg"},
                "quality": {
                    "pose_score": score if np.isfinite(score) else exact_score,
                    "sharpness_min": q["sharpness_min"],
                    "pair_gray_mad": q["pair_gray_mad"],
                },
            },
        }
        rows.append(row)
        report.extend([
            "",
            f"## {idx}. {answer} ry={ry:.3f}",
            f"- frames: {motion['start_frame_index']} -> {motion['end_frame_index']}, gap={motion['frame_gap']}",
            f"- translation_m: {motion['translation_m']:.4f}",
            f"- local_delta_camera_xyz: {[round(x, 4) for x in motion['local_delta_camera_xyz']]}",
            f"- relative_rotvec_deg_xyz: {[round(x, 3) for x in motion['relative_rotvec_deg_xyz']]}",
            f"- contact_sheet: {paths['contact_sheet']}",
        ])

    if panels:
        lines = []
        for i in range(0, len(panels), 2):
            pair = panels[i : i + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            lines.append(np.concatenate(pair, axis=1))
        cv2.imwrite(str(out_dir / "overview_contact_sheet.png"), np.concatenate(lines, axis=0))

    base.write_json(out_dir / "qa_data.json", rows)
    (out_dir / "item6_ryminus_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "dataset": "arkitscenes",
        "rule_name": "item6_ryminus_neighborhood",
        "scene": SCENE_ID,
        "base_frames": [CENTER_START_FRAME, CENTER_END_FRAME],
        "count": len(rows),
        "overview": str(out_dir / "overview_contact_sheet.png"),
        "report": str(out_dir / "item6_ryminus_report.md"),
        "qa_data": str(out_dir / "qa_data.json"),
    }
    base.write_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
