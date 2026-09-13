#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DATA_ROOT = Path("/path/to/workspace/Scannet++数据集/data")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/SCANNETPP_ACTION_PREVIEW")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动"]


@dataclass
class Candidate:
    scene: str
    label: str
    start: int
    end: int
    motion: dict
    pose_score: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wrap_deg(deg: np.ndarray | float) -> np.ndarray | float:
    return (deg + 180.0) % 360.0 - 180.0


def load_scene(scene_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    frame_ids = []
    mats = []
    for key in sorted(payload):
        mat = np.asarray(payload[key].get("aligned_pose") or payload[key].get("pose"), dtype=np.float64)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            continue
        frame_ids.append(int(key.split("_")[-1]))
        mats.append(mat)
    c2w = np.stack(mats, axis=0)
    centers = c2w[:, :3, 3]
    forward = c2w[:, :3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = np.degrees(np.arctan2(forward[:, 0], np.where(np.abs(forward[:, 2]) > 1e-8, forward[:, 2], 1e-8)))
    horiz = np.sqrt(forward[:, 0] ** 2 + forward[:, 2] ** 2)
    pitch = np.degrees(np.arctan2(forward[:, 1], np.maximum(horiz, 1e-8)))
    return np.asarray(frame_ids, dtype=np.int32), c2w, centers, yaw, pitch


def motion_at(frame_ids: np.ndarray, c2w: np.ndarray, centers: np.ndarray, yaw: np.ndarray, pitch: np.ndarray, start: int, end: int) -> dict:
    delta_world = centers[end] - centers[start]
    delta_local = c2w[start, :3, :3].T @ delta_world
    yaw_delta = float(wrap_deg(float(yaw[end] - yaw[start])))
    pitch_delta = float(wrap_deg(float(pitch[end] - pitch[start])))
    return {
        "start_frame_index": int(frame_ids[start]),
        "end_frame_index": int(frame_ids[end]),
        "frame_gap": int(frame_ids[end] - frame_ids[start]),
        "translation_m": float(np.linalg.norm(delta_world)),
        "yaw_deg": yaw_delta,
        "pitch_deg": pitch_delta,
        "local_delta": [float(delta_local[0]), float(delta_local[1]), float(delta_local[2])],
    }


def classify_motion(motion: dict) -> tuple[str | None, float]:
    right, up, forward = [float(x) for x in motion["local_delta"]]
    trans = float(motion["translation_m"])
    yaw = float(motion["yaw_deg"])
    pitch = float(motion["pitch_deg"])
    gap = int(motion["frame_gap"])
    primary = max(abs(right), abs(forward))
    secondary = min(abs(right), abs(forward))
    ratio = primary / max(secondary, 1e-8)
    straight = primary / max(trans, 1e-8)

    if (
        1.8 <= trans <= 6.0
        and 18 <= gap <= 180
        and abs(yaw) <= 12.0
        and abs(pitch) <= 8.0
        and abs(up) <= 0.35
        and primary >= 1.55
        and ratio >= 2.0
        and straight >= 0.78
    ):
        if abs(forward) >= abs(right):
            label = "向前移动" if forward > 0 else "向后移动"
        else:
            label = "向右移动" if right > 0 else "向左移动"
        score = 20.0 * trans + gap * 0.05 + min(ratio, 8.0) * 2.0 - abs(yaw) - abs(pitch) * 0.5
        return label, score

    if (
        80.0 <= abs(yaw) <= 170.0
        and 18 <= gap <= 220
        and trans <= 0.45
        and abs(pitch) <= 10.0
        and abs(up) <= 0.25
        and abs(right) <= 0.35
        and abs(forward) <= 0.35
    ):
        label = "向右转动" if yaw > 0 else "向左转动"
        score = abs(yaw) + gap * 0.05 - trans * 30.0 - abs(pitch)
        return label, score

    return None, 0.0


def scene_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / "iphone" / "rgb.mkv").exists()
        and (path / "iphone" / "pose_intrinsic_imu.json").exists()
    ]


def gather_pose_candidates(root: Path, max_scenes: int, per_label_limit: int) -> dict[str, list[Candidate]]:
    per_label: dict[str, list[Candidate]] = defaultdict(list)
    gaps = list(range(20, 221, 5))
    for scene_dir in scene_dirs(root)[:max_scenes]:
        try:
            frame_ids, c2w, centers, yaw, pitch = load_scene(scene_dir)
        except Exception:
            continue
        if len(frame_ids) < 50:
            continue
        scene = scene_dir.name
        local_found: list[Candidate] = []
        for gap in gaps:
            if gap >= len(frame_ids):
                continue
            for start in range(0, len(frame_ids) - gap, max(1, gap // 4)):
                end = start + gap
                motion = motion_at(frame_ids, c2w, centers, yaw, pitch, start, end)
                label, score = classify_motion(motion)
                if not label:
                    continue
                local_found.append(Candidate(scene=scene, label=label, start=start, end=end, motion=motion, pose_score=score))
        local_found.sort(key=lambda c: c.pose_score, reverse=True)
        scene_label_count: dict[str, int] = defaultdict(int)
        for cand in local_found:
            if scene_label_count[cand.label] >= 4:
                continue
            per_label[cand.label].append(cand)
            scene_label_count[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label[label].sort(key=lambda c: c.pose_score, reverse=True)
            del per_label[label][per_label_limit:]
        if all(len(per_label[label]) >= 8 for label in ACTION_OPTIONS):
            break
    return per_label


def read_gray(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def visual_metrics(root: Path, cand: Candidate) -> dict | None:
    video_path = root / cand.scene / "iphone" / "rgb.mkv"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        a = read_gray(cap, cand.motion["start_frame_index"])
        b = read_gray(cap, cand.motion["end_frame_index"])
    finally:
        cap.release()
    if a is None or b is None:
        return None
    mad = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    sharp_a = float(cv2.Laplacian(a, cv2.CV_64F).var())
    sharp_b = float(cv2.Laplacian(b, cv2.CV_64F).var())
    return {
        "gray_mad": mad,
        "frame_A_sharpness": sharp_a,
        "frame_B_sharpness": sharp_b,
        "sharpness_min": min(sharp_a, sharp_b),
        "visual_score": mad + 0.04 * min(sharp_a, sharp_b),
    }


def select_candidates(root: Path, per_label: dict[str, list[Candidate]], target_total: int) -> list[tuple[Candidate, dict]]:
    scored: dict[str, list[tuple[Candidate, dict]]] = defaultdict(list)
    for label in ACTION_OPTIONS:
        for cand in per_label.get(label, []):
            metrics = visual_metrics(root, cand)
            if metrics is None:
                continue
            if metrics["gray_mad"] < 45.0 or metrics["sharpness_min"] < 20.0:
                continue
            scored[label].append((cand, metrics))
        scored[label].sort(key=lambda item: (item[1]["visual_score"], item[0].pose_score), reverse=True)

    selected: list[tuple[Candidate, dict]] = []
    used: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def overlaps(cand: Candidate) -> bool:
        for a, b in used[cand.scene]:
            if not (cand.end < a - 30 or cand.start > b + 30):
                return True
        return False

    for label in ACTION_OPTIONS:
        for cand, metrics in scored[label]:
            if overlaps(cand):
                continue
            selected.append((cand, metrics))
            used[cand.scene].append((cand.start, cand.end))
            break

    pool = [item for rows in scored.values() for item in rows]
    pool.sort(key=lambda item: (item[1]["visual_score"], item[0].pose_score), reverse=True)
    for cand, metrics in pool:
        if len(selected) >= target_total:
            break
        if any(x.scene == cand.scene and x.start == cand.start and x.end == cand.end for x, _ in selected):
            continue
        if overlaps(cand):
            continue
        selected.append((cand, metrics))
        used[cand.scene].append((cand.start, cand.end))
    return selected[:target_total]


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed_to_read_frame:{video_path}:{frame_index}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"failed_to_write_frame:{output_path}")
    finally:
        cap.release()


def render(root: Path, output_root: Path, selected: list[tuple[Candidate, dict]]) -> list[dict]:
    task_root = output_root / "action_inference"
    if task_root.exists():
        shutil.rmtree(task_root)
    ensure_dir(task_root)

    rows = []
    for idx, (cand, metrics) in enumerate(selected, start=1):
        qa_id = f"scannetpp_action_preview_{idx:03d}"
        sample_dir = task_root / qa_id
        frame_a = sample_dir / "frame_A.png"
        frame_b = sample_dir / "frame_B.png"
        video_path = root / cand.scene / "iphone" / "rgb.mkv"
        extract_frame(video_path, cand.motion["start_frame_index"], frame_a)
        extract_frame(video_path, cand.motion["end_frame_index"], frame_b)
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "action_inference",
                "dataset": "scannetpp",
                "scene": cand.scene,
                "pose_source": "scannetpp_pose_intrinsic_imu_aligned_pose",
                "question": "图A到图B之间，相机主要执行了什么动作？",
                "answer": cand.label,
                "options": ACTION_OPTIONS,
                "input": {"frame_paths": [str(frame_a), str(frame_b)], "frame_A": str(frame_a), "frame_B": str(frame_b)},
                "gt": {
                    "motion": cand.motion,
                    "quality": {
                        "pose_score": cand.pose_score,
                        **metrics,
                    },
                },
            }
        )
    write_json(task_root / "qa_data.json", rows)
    with (task_root / "qa_data.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=500)
    parser.add_argument("--per-label-limit", type=int, default=80)
    parser.add_argument("--target-total", type=int, default=10)
    args = parser.parse_args()

    per_label = gather_pose_candidates(args.data_root, args.max_scenes, args.per_label_limit)
    selected = select_candidates(args.data_root, per_label, args.target_total)
    rows = render(args.data_root, args.output_root, selected)
    summary = {
        "generated_total": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in ACTION_OPTIONS},
        "scenes": sorted({row["scene"] for row in rows}),
        "selection_rule": "large_motion_obvious_preview",
        "thresholds": {
            "translation_m": "1.8 to 6.0",
            "rotation_yaw_deg": "80 to 170",
            "gray_mad_min": 45.0,
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        motion = row["gt"]["motion"]
        quality = row["gt"]["quality"]
        print(
            row["qa_id"],
            row["answer"],
            row["scene"],
            "gap",
            motion["frame_gap"],
            "trans",
            round(motion["translation_m"], 3),
            "yaw",
            round(motion["yaw_deg"], 3),
            "mad",
            round(quality["gray_mad"], 2),
        )


if __name__ == "__main__":
    main()
