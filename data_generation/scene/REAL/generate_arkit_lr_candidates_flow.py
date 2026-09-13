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


OUTPUT_ROOT = Path("/path/to/workspace/DATA/ARKIT_LR_CANDIDATES")
LABELS = ["左移候选", "右移候选", "左转候选", "右转候选"]


def flow_stats(path_a: Path, path_b: Path) -> dict | None:
    image_a = cv2.imread(str(path_a))
    image_b = cv2.imread(str(path_b))
    if image_a is None or image_b is None:
        return None
    image_a = cv2.resize(image_a, (320, 240))
    image_b = cv2.resize(image_b, (320, 240))
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    fx = flow[..., 0]
    fy = flow[..., 1]
    h, w = fx.shape
    yy, xx = np.mgrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rx = xx - cx
    ry = yy - cy
    rr = np.sqrt(rx * rx + ry * ry) + 1e-6
    radial = (fx * rx + fy * ry) / rr
    mag = np.sqrt(fx * fx + fy * fy)
    mask = mag < 80
    mask[:16, :] = False
    mask[-16:, :] = False
    mask[:, :16] = False
    mask[:, -16:] = False
    if mask.sum() < 1000:
        return None
    return {
        "mean_fx": float(np.mean(fx[mask])),
        "mean_fy": float(np.mean(fy[mask])),
        "mean_radial": float(np.mean(radial[mask])),
        "median_mag": float(np.median(mag[mask])),
    }


def classify_turn(motion: dict) -> tuple[str | None, str | None, float, float]:
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    arx, ary, arz = abs(rx), abs(ry), abs(rz)
    trans = float(motion["translation_m"])
    gap = int(motion["frame_gap"])
    if 40 <= gap <= 360 and 22.0 <= arz <= 55.0 and arz >= 2.0 * max(arx, ary, 1e-6) and trans <= 0.60:
        # User confirmed this left/right turn sign is correct.
        label = "右转候选" if rz > 0 else "左转候选"
        score = arz * 4.0 - max(arx, ary) * 8.0 - trans * 20.0
        return label, "rotation", arz, score
    return None, None, 0.0, 0.0


def collect(scenes: list[base.ScenePose]) -> dict[str, list[base.Candidate]]:
    per_label: dict[str, list[base.Candidate]] = defaultdict(list)
    gaps = [4, 5, 6, 8, 10, 12, 15, 18, 22, 26]
    for scene in scenes:
        local: list[base.Candidate] = []
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(3, gap_slots)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = base.motion_between(scene, start, end, local_rule="none")
                rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
                dx, dy, dz = [float(x) for x in motion["local_delta_hypothesis_xyz"]]
                frame_a = scene.scene_dir / "color" / f"{motion['start_frame_index']:06d}.jpg"
                frame_b = scene.scene_dir / "color" / f"{motion['end_frame_index']:06d}.jpg"

                # Keep confirmed z-axis turn candidates.
                label, family, mag, score = classify_turn(motion)
                if label:
                    motion["predicted_family"] = family
                    motion["predicted_magnitude"] = float(mag)
                    local.append(base.Candidate(scene.scene, scene.scene_dir, start, end, label, family, mag, motion, score))

                # For left/right translation, use visual flow instead of uncertain ARKit local axes.
                if max(abs(rx), abs(ry), abs(rz)) > 5.0:
                    continue
                if float(motion["translation_m"]) < 0.25:
                    continue
                fs = flow_stats(frame_a, frame_b)
                if fs is None:
                    continue
                horiz = abs(fs["mean_fx"])
                vert = abs(fs["mean_fy"])
                radial = abs(fs["mean_radial"])
                if horiz < 3.0 or horiz < 1.8 * max(vert, radial, 1e-6):
                    continue
                # Optical flow sign is image motion; camera motion is opposite.
                label = "左移候选" if fs["mean_fx"] > 0 else "右移候选"
                mag = float(motion["translation_m"])
                motion = dict(motion)
                motion["flow_filter"] = fs
                motion["predicted_family"] = "translation_flow"
                motion["predicted_magnitude"] = mag
                score = horiz * 10.0 - vert * 4.0 - radial * 5.0 - max(abs(rx), abs(ry), abs(rz)) * 3.0
                # Penalize pairs whose pose translation is dominated by the already bad front/back axis.
                score -= abs(dx) * 4.0 + abs(dz) * 4.0 - abs(dy) * 2.0
                local.append(base.Candidate(scene.scene, scene.scene_dir, start, end, label, "translation", mag, motion, score))

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
        print("[flow-lr-scan]", scene.scene, {k: len(per_label[k]) for k in LABELS}, flush=True)
        if all(per_label[k] for k in LABELS):
            break
    return per_label


def build_rows(per_label: dict[str, list[base.Candidate]], out: Path) -> list[dict]:
    rows: list[dict] = []
    for label in LABELS:
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
                "label_source": "left_right_flow_candidate_needs_user_validation",
                "input": {"frame_paths": [paths["frame_A"], paths["frame_B"]], "frame_A": paths["frame_A"], "frame_B": paths["frame_B"], "contact_sheet": paths["contact_sheet"]},
                "gt": {"motion": m, "approx_action": {"text": base.action_value_text(label, cand.family, cand.magnitude), "value": round(cand.magnitude, 3), "unit": "deg" if cand.family == "rotation" else "m"}, "quality": {"pose_score": cand.score, **q}},
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=base.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=120)
    args = parser.parse_args()
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    base.ensure_dir(args.output_root / "action_inference")
    scenes = []
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
        "rule_name": "lr_flow_translation_rz_turn",
        "count": len(rows),
        "label_counts": {label: sum(1 for r in rows if r["answer"] == label) for label in LABELS},
        "candidate_counts": {label: len(per_label[label]) for label in LABELS},
    }
    base.write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
