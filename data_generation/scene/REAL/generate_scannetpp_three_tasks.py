#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DATA_ROOT = Path("/path/to/workspace/Scannet++数据集/data")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/SCANNETPP")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]
TASKS = ["action_inference", "movement_degree_comparison", "movement_sequence_sorting"]


@dataclass
class MotionCandidate:
    scene: str
    start_slot: int
    end_slot: int
    label: str
    family: str
    magnitude: float
    motion: dict
    score: float


@dataclass
class ScenePose:
    scene: str
    frame_ids: np.ndarray
    mats: np.ndarray


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def retry_eagain(fn, *, attempts: int = 10, delay_s: float = 0.5):
    for attempt in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scene_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in retry_eagain(lambda: sorted(root.iterdir()))
        if retry_eagain(lambda path=path: path.is_dir())
        and retry_eagain(lambda path=path: (path / "iphone" / "rgb.mkv").exists())
        and retry_eagain(lambda path=path: (path / "iphone" / "pose_intrinsic_imu.json").exists())
    ]


def load_scene_pose(scene_dir: Path) -> ScenePose | None:
    pose_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
    try:
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    frame_ids = []
    mats = []
    for key in sorted(payload):
        raw = payload[key]
        mat = np.asarray(raw.get("aligned_pose") or raw.get("pose"), dtype=np.float64)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            continue
        frame_ids.append(int(key.split("_")[-1]))
        mats.append(mat)
    if len(mats) < 100:
        return None
    return ScenePose(scene=scene_dir.name, frame_ids=np.asarray(frame_ids, dtype=np.int32), mats=np.stack(mats, axis=0))


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


def motion_between(scene: ScenePose, start: int, end: int) -> dict:
    mat_a = scene.mats[start]
    mat_b = scene.mats[end]
    rel_rot = mat_a[:3, :3].T @ mat_b[:3, :3]
    rel_rotvec = rotvec_deg(rel_rot)
    local_delta = mat_a[:3, :3].T @ (mat_b[:3, 3] - mat_a[:3, 3])
    translation = float(np.linalg.norm(mat_b[:3, 3] - mat_a[:3, 3]))
    return {
        "start_frame_index": int(scene.frame_ids[start]),
        "end_frame_index": int(scene.frame_ids[end]),
        "frame_gap": int(scene.frame_ids[end] - scene.frame_ids[start]),
        "translation_m": translation,
        "local_delta": [float(x) for x in local_delta],
        "relative_rotvec_deg_xyz": [float(x) for x in rel_rotvec],
        "abs_rotation_deg": float(np.linalg.norm(rel_rotvec)),
    }


def classify_motion(motion: dict) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    gap = int(motion["frame_gap"])
    translation = float(motion["translation_m"])
    rot_primary = max(abs(rx), abs(ry))
    rot_other = max(min(abs(rx), abs(ry)), abs(rz))

    if 50 <= gap <= 220 and translation <= 0.75 and 30.0 <= rot_primary <= 60.0 and rot_primary >= 1.7 * max(rot_other, 1e-6) and abs(rz) <= 14.0:
        if abs(rx) >= abs(ry):
            label = "向上转动" if rx > 0 else "向下转动"
            magnitude = abs(rx)
        else:
            label = "向右转动" if ry > 0 else "向左转动"
            magnitude = abs(ry)
        score = magnitude * 2.0 + 0.04 * gap - translation * 10.0 - rot_other
        return label, "rotation", magnitude, score

    if 50 <= gap <= 220 and translation >= 0.75 and abs(dy) <= 0.45 and rot_primary <= 12.0 and abs(rz) <= 8.0:
        if abs(dx) >= abs(dz):
            primary = abs(dx)
            secondary = abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary = abs(dz)
            secondary = abs(dx)
            label = "向前移动" if dz > 0 else "向后移动"
        if not (0.85 <= primary <= 3.2 and primary >= 2.7 * max(secondary, 1e-6)):
            return None, None, 0.0, 0.0
        score = primary * 25.0 + primary / max(max(secondary, abs(dy)), 1e-6) * 2.0 - rot_primary * 1.5 - abs(rz)
        return label, "translation", primary, score

    return None, None, 0.0, 0.0


def collect_motion_candidates(
    scenes: list[ScenePose],
    gaps: Sequence[int],
    per_scene_label_cap: int = 5,
    per_label_cap: int = 80,
    stop_min_per_label: int = 20,
) -> dict[str, list[MotionCandidate]]:
    per_label: dict[str, list[MotionCandidate]] = defaultdict(list)
    for scene in scenes:
        local: list[MotionCandidate] = []
        for gap in gaps:
            if gap >= len(scene.frame_ids):
                continue
            step = max(15, gap // 3)
            for start in range(0, len(scene.frame_ids) - gap, step):
                end = start + gap
                motion = motion_between(scene, start, end)
                label, family, magnitude, score = classify_motion(motion)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(
                    MotionCandidate(
                        scene=scene.scene,
                        start_slot=start,
                        end_slot=end,
                        label=label,
                        family=family,
                        magnitude=float(magnitude),
                        motion=motion,
                        score=float(score),
                    )
                )
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if per_scene_label_cap > 0 and scene_counts[cand.label] >= per_scene_label_cap:
                continue
            per_label[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label[label].sort(key=lambda item: item.score, reverse=True)
            if per_label_cap > 0:
                del per_label[label][per_label_cap:]
        if stop_min_per_label > 0 and all(len(per_label[label]) >= stop_min_per_label for label in ACTION_OPTIONS):
            break
    return per_label


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def load_frames(scene: str, frame_indices: Sequence[int]) -> tuple[list[np.ndarray], dict] | None:
    cap = cv2.VideoCapture(str(DATA_ROOT / scene / "iphone" / "rgb.mkv"))
    if not cap.isOpened():
        return None
    frames = []
    grays = []
    sharpness = []
    try:
        for frame_index in frame_indices:
            frame = read_frame(cap, int(frame_index))
            if frame is None:
                return None
            frames.append(frame)
            small = cv2.resize(frame, (480, 360))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    finally:
        cap.release()
    pair_mads = []
    for i in range(len(grays) - 1):
        pair_mads.append(float(np.mean(np.abs(grays[i].astype(np.float32) - grays[i + 1].astype(np.float32)))))
    return frames, {"sharpness_min": min(sharpness), "pair_gray_mad": pair_mads}


def save_frames(sample_dir: Path, names_and_frames: Sequence[tuple[str, np.ndarray]]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs = {}
    smalls = []
    for name, frame in names_and_frames:
        path = sample_dir / f"{name}.png"
        cv2.imwrite(str(path), frame)
        outputs[name] = str(path)
        smalls.append(cv2.resize(frame, (480, 360)))
    contact = sample_dir / "contact_sheet.png"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1))
    outputs["contact_sheet"] = str(contact)
    return outputs


def action_value_text(label: str, family: str, magnitude: float) -> str:
    if family == "rotation":
        return f"{label}约{magnitude:.1f}度"
    return f"{label}约{magnitude:.2f}米"


def build_action_rows(per_label: dict[str, list[MotionCandidate]], target: int, output_root: Path) -> list[dict]:
    rows = []
    used_segments: set[tuple[str, int, int]] = set()
    label_order = ACTION_OPTIONS[:]
    idx = 1
    while target <= 0 or len(rows) < target:
        progressed = False
        for label in label_order:
            if target > 0 and len(rows) >= target:
                break
            for cand in per_label.get(label, []):
                key = (cand.scene, cand.start_slot, cand.end_slot)
                if key in used_segments:
                    continue
                loaded = load_frames(cand.scene, [cand.motion["start_frame_index"], cand.motion["end_frame_index"]])
                if loaded is None:
                    continue
                frames, quality = loaded
                if quality["sharpness_min"] < 8 or quality["pair_gray_mad"][0] < 20:
                    continue
                qa_id = f"scannetpp_action_inference_{idx:04d}"
                sample_dir = output_root / "action_inference" / qa_id
                paths = save_frames(sample_dir, [("frame_A", frames[0]), ("frame_B", frames[1])])
                rows.append(
                    {
                        "qa_id": qa_id,
                        "task_type": "action_inference",
                        "dataset": "scannetpp",
                        "scene": cand.scene,
                        "pose_source": "scannetpp_aligned_pose_relative_local_rule_v1_human_verified",
                        "question": "图A到图B之间，相机主要执行了什么动作？",
                        "answer": cand.label,
                        "answer_with_value": action_value_text(cand.label, cand.family, cand.magnitude),
                        "options": ACTION_OPTIONS,
                        "label_source": "predicted_by_confirmed_scannetpp_pose_rule",
                        "input": {
                            "frame_paths": [paths["frame_A"], paths["frame_B"]],
                            "frame_A": paths["frame_A"],
                            "frame_B": paths["frame_B"],
                            "contact_sheet": paths["contact_sheet"],
                        },
                        "gt": {
                            "motion": cand.motion,
                            "approx_action": {
                                "text": action_value_text(cand.label, cand.family, cand.magnitude),
                                "value": round(cand.magnitude, 3),
                                "unit": "deg" if cand.family == "rotation" else "m",
                            },
                            "quality": {"pose_score": cand.score, **quality},
                        },
                    }
                )
                used_segments.add(key)
                idx += 1
                progressed = True
                break
        if not progressed:
            break
    return rows


def collect_degree_candidates(scenes: list[ScenePose], target: int, output_root: Path) -> list[dict]:
    rows = []
    rng = random.Random(20260513)
    gaps = [50, 60, 80, 100, 120, 150, 180, 220]
    candidates = []
    for scene in scenes:
        for start in range(0, max(0, len(scene.frame_ids) - 230), 35):
            local = []
            for gap in gaps:
                end = start + gap
                if end >= len(scene.frame_ids):
                    continue
                motion = motion_between(scene, start, end)
                label, family, magnitude, score = classify_motion(motion)
                if label and family:
                    motion["predicted_family"] = family
                    motion["predicted_magnitude"] = float(magnitude)
                    local.append((label, family, magnitude, score, end, motion))
            for i in range(len(local)):
                for j in range(i + 1, len(local)):
                    a = local[i]
                    b = local[j]
                    if a[0] != b[0] or a[1] != b[1]:
                        continue
                    gap_mag = abs(a[2] - b[2])
                    min_gap = 0.45 if a[1] == "translation" else 14.0
                    if gap_mag < min_gap:
                        continue
                    score = gap_mag + 0.01 * (a[3] + b[3])
                    candidates.append((score, scene, start, a, b))
        if target > 0 and len(candidates) > target * 20:
            break
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_starts: set[tuple[str, int]] = set()
    for _score, scene, start, cand_a, cand_b in candidates:
        if target > 0 and len(rows) >= target:
            break
        if (scene.scene, start) in used_starts:
            continue
        left, right = cand_a, cand_b
        if rng.random() < 0.5:
            left, right = right, left
        larger = "A" if left[2] > right[2] else "B"
        frame_indices = [int(scene.frame_ids[start]), left[5]["end_frame_index"], right[5]["end_frame_index"]]
        loaded = load_frames(scene.scene, frame_indices)
        if loaded is None:
            continue
        frames, quality = loaded
        if quality["sharpness_min"] < 8 or min(quality["pair_gray_mad"]) < 18:
            continue
        qa_id = f"scannetpp_movement_degree_comparison_{len(rows) + 1:04d}"
        sample_dir = output_root / "movement_degree_comparison" / qa_id
        paths = save_frames(sample_dir, [("frame_start", frames[0]), ("frame_A", frames[1]), ("frame_B", frames[2])])
        family_unit = "度" if left[1] == "rotation" else "米"
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "movement_degree_comparison",
                "dataset": "scannetpp",
                "scene": scene.scene,
                "pose_source": "scannetpp_aligned_pose_relative_local_rule_v1_human_verified",
                "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
                "answer": f"候选图{larger}更大。",
                "answer_with_value": f"候选图A约{left[2]:.1f}{family_unit}，候选图B约{right[2]:.1f}{family_unit}。" if left[1] == "rotation" else f"候选图A约{left[2]:.2f}{family_unit}，候选图B约{right[2]:.2f}{family_unit}。",
                "input": {
                    "frame_paths": [paths["frame_start"], paths["frame_A"], paths["frame_B"]],
                    "start_frame": paths["frame_start"],
                    "frame_A": paths["frame_A"],
                    "frame_B": paths["frame_B"],
                    "contact_sheet": paths["contact_sheet"],
                },
                "gt": {
                    "comparison_label": left[0],
                    "comparison_family": left[1],
                    "larger_clip": larger,
                    "start_frame_index": int(scene.frame_ids[start]),
                    "frame_A": {"slot_end": int(left[4]), "magnitude": float(left[2]), "motion": left[5]},
                    "frame_B": {"slot_end": int(right[4]), "magnitude": float(right[2]), "motion": right[5]},
                    "quality": quality,
                },
            }
        )
        used_starts.add((scene.scene, start))
    return rows


def collect_sorting_rows(scenes: list[ScenePose], target: int, output_root: Path) -> list[dict]:
    rows = []
    rng = random.Random(20260514)
    candidates = []
    for scene in scenes:
        for gap in [25, 30, 35, 40, 45, 50, 60]:
            span = 3 * gap
            if span >= len(scene.frame_ids):
                continue
            for start in range(0, len(scene.frame_ids) - span, max(12, gap // 2)):
                slots = [start, start + gap, start + 2 * gap, start + 3 * gap]
                motions = [motion_between(scene, slots[i], slots[i + 1]) for i in range(3)]
                ok = True
                total_forward = 0.0
                total_lateral = 0.0
                total_rotation = 0.0
                for motion in motions:
                    dx, dy, dz = [float(x) for x in motion["local_delta"]]
                    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
                    translation = float(motion["translation_m"])
                    # Sequence sorting should be a coherent approach trajectory:
                    # small/medium forward motion, low lateral drift, and low rotation.
                    if not (
                        0.22 <= dz <= 0.85
                        and 0.25 <= translation <= 0.95
                        and abs(dx) <= max(0.18, 0.45 * abs(dz))
                        and abs(dy) <= 0.20
                        and max(abs(rx), abs(ry), abs(rz)) <= 7.0
                        and float(motion["abs_rotation_deg"]) <= 9.0
                    ):
                        ok = False
                        break
                    total_forward += dz
                    total_lateral += abs(dx)
                    total_rotation += float(motion["abs_rotation_deg"])
                if not ok:
                    continue
                score = total_forward * 10.0 - total_lateral * 4.0 - total_rotation * 0.5 - abs(gap - 40) * 0.03
                candidates.append((score, scene, slots, motions, total_forward, total_rotation))
        if target > 0 and len(candidates) > target * 15:
            break
    candidates.sort(key=lambda item: item[0], reverse=True)
    used: set[tuple[str, int]] = set()
    for _score, scene, slots, motions, total_forward, total_rotation in candidates:
        if target > 0 and len(rows) >= target:
            break
        if (scene.scene, slots[0]) in used:
            continue
        frame_indices = [int(scene.frame_ids[slot]) for slot in slots]
        loaded = load_frames(scene.scene, frame_indices)
        if loaded is None:
            continue
        frames, quality = loaded
        if quality["sharpness_min"] < 8 or min(quality["pair_gray_mad"]) < 18:
            continue
        labels = ["A", "B", "C"]
        candidates_labels = list(zip(labels, slots[1:], frames[1:]))
        rng.shuffle(candidates_labels)
        correct = [label for label, slot, _frame in sorted(candidates_labels, key=lambda item: item[1])]
        qa_id = f"scannetpp_movement_sequence_sorting_{len(rows) + 1:04d}"
        sample_dir = output_root / "movement_sequence_sorting" / qa_id
        named_frames = [("frame_start", frames[0])] + [(f"frame_{label}", frame) for label, _slot, frame in candidates_labels]
        paths = save_frames(sample_dir, named_frames)
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "movement_sequence_sorting",
                "dataset": "scannetpp",
                "scene": scene.scene,
                "pose_source": "scannetpp_aligned_pose_relative_local_rule_v1_human_verified",
                "question": "已知第一张图是起始帧。请将其余三张候选图按真实视频中的时间先后排序。",
                "answer": " -> ".join(correct),
                "input": {
                    "frame_paths": [paths["frame_start"]] + [paths[f"frame_{label}"] for label, _slot, _frame in candidates_labels],
                    "first_frame": paths["frame_start"],
                    "candidate_frames": {label: paths[f"frame_{label}"] for label, _slot, _frame in candidates_labels},
                    "contact_sheet": paths["contact_sheet"],
                },
                "gt": {
                    "start_slot": int(slots[0]),
                    "start_frame_index": int(scene.frame_ids[slots[0]]),
                    "candidate_slots": {label: int(slot) for label, slot, _frame in candidates_labels},
                    "candidate_frame_indices": {label: int(scene.frame_ids[slot]) for label, slot, _frame in candidates_labels},
                    "correct_order": correct,
                    "trajectory_type": "small_forward_approach_low_rotation",
                    "total_forward_m": float(total_forward),
                    "total_rotation_deg": float(total_rotation),
                    "motions_between_consecutive_true_frames": motions,
                    "quality": quality,
                },
            }
        )
        used.add((scene.scene, slots[0]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ScanNet++ motion QA with confirmed relative-local pose rule.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--target-per-task", type=int, default=0, help="Maximum rows per task. Use 0 for no cap.")
    parser.add_argument("--max-scenes", type=int, default=0, help="Maximum scenes to scan. Use 0 for all scenes.")
    parser.add_argument("--tasks", default="action_inference,movement_degree_comparison,movement_sequence_sorting")
    parser.add_argument("--action-per-scene-label-cap", type=int, default=5, help="Use 0 for no per-scene per-label cap.")
    parser.add_argument("--action-per-label-cap", type=int, default=80, help="Use 0 for no global per-label cap.")
    parser.add_argument("--action-stop-min-per-label", type=int, default=20, help="Use 0 to scan all scenes for action candidates.")
    args = parser.parse_args()
    selected_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    random.seed(20260513)
    if args.output_root.exists() and len(selected_tasks) == len(TASKS):
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    poses = []
    selected_scene_dirs = scene_dirs(args.data_root)
    if args.max_scenes > 0:
        selected_scene_dirs = selected_scene_dirs[: args.max_scenes]
    for scene_dir in selected_scene_dirs:
        pose = load_scene_pose(scene_dir)
        if pose is not None:
            poses.append(pose)
    task_rows = {}
    if "action_inference" in selected_tasks:
        gaps = [50, 60, 80, 100, 120, 150, 180, 220]
        per_label = collect_motion_candidates(
            poses,
            gaps,
            per_scene_label_cap=args.action_per_scene_label_cap,
            per_label_cap=args.action_per_label_cap,
            stop_min_per_label=args.action_stop_min_per_label,
        )
        task_rows["action_inference"] = build_action_rows(per_label, args.target_per_task, args.output_root)
    if "movement_degree_comparison" in selected_tasks:
        task_rows["movement_degree_comparison"] = collect_degree_candidates(poses, args.target_per_task, args.output_root)
    if "movement_sequence_sorting" in selected_tasks:
        task_rows["movement_sequence_sorting"] = collect_sorting_rows(poses, args.target_per_task, args.output_root)
    for task, rows in task_rows.items():
        write_json(args.output_root / task / "qa_data.json", rows)
        write_jsonl(args.output_root / task / "qa_data.jsonl", rows)

    summary = {
        "dataset": "scannetpp",
        "output_root": str(args.output_root),
        "target_per_task": args.target_per_task,
        "counts": {task: len(rows) for task, rows in task_rows.items()},
        "pose_rule": {
            "source": "iphone/pose_intrinsic_imu.json aligned_pose, treated as camera-to-world",
            "relative_rotation": "R_rel = R_A.T @ R_B, Rodrigues rotvec in degrees",
            "rotation_labels": "x>0 up, x<0 down, y>0 right-turn, y<0 left-turn",
            "translation_labels": "local_delta = R_A.T @ (t_B - t_A): x>0 right, x<0 left, z>0 forward, z<0 backward",
            "rotation_range_deg": "30-60 for action labels",
            "frame_gap_min": 50,
        },
        "action_label_counts": {
            label: sum(1 for row in task_rows.get("action_inference", []) if row.get("answer") == label)
            for label in ACTION_OPTIONS
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
