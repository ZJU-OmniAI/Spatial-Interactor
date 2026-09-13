#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DATA_ROOT = Path("/path/to/workspace/Scannet++数据集/data")
DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/DATA/SCANNETPP_ACTION_PREVIEW")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动"]


@dataclass
class PoseItem:
    frame_index: int
    c2w: np.ndarray


@dataclass
class SegmentCandidate:
    scene_id: str
    label: str
    start_idx: int
    end_idx: int
    motion: dict
    score: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def heading_pitch_from_c2w(c2w: np.ndarray) -> tuple[float, float]:
    forward = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2]) if abs(float(forward[2])) > 1e-8 else 1e-8))
    horiz = math.sqrt(float(forward[0]) ** 2 + float(forward[2]) ** 2)
    pitch = math.degrees(math.atan2(float(forward[1]), max(horiz, 1e-8)))
    return yaw, pitch


def motion_between(items: list[PoseItem], start_idx: int, end_idx: int) -> dict:
    start = items[start_idx]
    end = items[end_idx]
    delta_world = end.c2w[:3, 3] - start.c2w[:3, 3]
    delta_local = start.c2w[:3, :3].T @ delta_world
    right_m = float(delta_local[0])
    up_m = float(delta_local[1])
    forward_m = float(delta_local[2])
    trans_m = float(np.linalg.norm(delta_world))
    yaw_a, pitch_a = heading_pitch_from_c2w(start.c2w)
    yaw_b, pitch_b = heading_pitch_from_c2w(end.c2w)
    yaw_delta = wrap_deg(yaw_b - yaw_a)
    pitch_delta = wrap_deg(pitch_b - pitch_a)
    return {
        "start_frame_index": start.frame_index,
        "end_frame_index": end.frame_index,
        "translation_m": trans_m,
        "yaw_deg": yaw_delta,
        "pitch_deg": pitch_delta,
        "local_delta": [right_m, up_m, forward_m],
        "abs_yaw_deg": abs(yaw_delta),
        "abs_pitch_deg": abs(pitch_delta),
    }


def dominant_axis(local_delta: list[float]) -> tuple[str | None, float, float]:
    right = abs(float(local_delta[0]))
    forward = abs(float(local_delta[2]))
    primary = max(right, forward)
    secondary = min(right, forward)
    if primary < 1e-8:
        return None, 0.0, 0.0
    return ("x" if right >= forward else "z"), primary, secondary


def load_scannetpp_scene(scene_dir: Path) -> list[PoseItem]:
    pose_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    items: list[PoseItem] = []
    for key in sorted(payload.keys()):
        raw = payload[key]
        mat = np.asarray(raw.get("aligned_pose") or raw.get("pose"), dtype=np.float64)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
        if mat.shape != (4, 4):
            continue
        if not np.isfinite(mat).all():
            continue
        items.append(PoseItem(frame_index=int(key.split("_")[-1]), c2w=mat))
    return items


def segment_purity(items: list[PoseItem], start_idx: int, end_idx: int) -> dict:
    step_motions = [motion_between(items, i, i + 1) for i in range(start_idx, end_idx)]
    if not step_motions:
        return {"step_count": 0}
    step_trans = np.array([m["translation_m"] for m in step_motions], dtype=np.float64)
    step_yaws = np.array([m["yaw_deg"] for m in step_motions], dtype=np.float64)
    step_pitch = np.array([m["pitch_deg"] for m in step_motions], dtype=np.float64)
    step_local = np.array([m["local_delta"] for m in step_motions], dtype=np.float64)
    summed_local = step_local.sum(axis=0)
    straightness = float(np.linalg.norm(summed_local[[0, 2]]) / max(np.abs(step_local[:, [0, 2]]).sum(), 1e-8))
    return {
        "step_count": len(step_motions),
        "step_trans_sum": float(step_trans.sum()),
        "step_trans_max": float(step_trans.max()),
        "step_abs_yaw_sum": float(np.abs(step_yaws).sum()),
        "step_abs_yaw_max": float(np.abs(step_yaws).max()),
        "step_abs_pitch_max": float(np.abs(step_pitch).max()),
        "step_local_abs_sum": np.abs(step_local).sum(axis=0).tolist(),
        "step_local_sum": summed_local.tolist(),
        "yaw_same_sign_ratio": float(np.mean(np.sign(step_yaws[step_yaws != 0.0]) == np.sign(step_yaws[step_yaws != 0.0][0])) if np.any(step_yaws != 0.0) else 1.0),
        "straightness_xz": straightness,
    }


def classify_translation(motion: dict, purity: dict) -> tuple[str | None, float]:
    axis, primary, secondary = dominant_axis(motion["local_delta"])
    if axis is None:
        return None, 0.0
    right, up, forward = [float(x) for x in motion["local_delta"]]
    ratio = primary / max(secondary, 1e-8)
    agg_straight = primary / max(float(motion["translation_m"]), 1e-8)
    purity_ratio = purity["step_trans_sum"] / max(float(motion["translation_m"]), 1e-8)

    if not (
        0.22 <= float(motion["translation_m"]) <= 0.75
        and abs(float(motion["yaw_deg"])) <= 3.0
        and abs(float(motion["pitch_deg"])) <= 3.0
        and abs(up) <= 0.08
        and primary >= 0.20
        and ratio >= 2.4
        and agg_straight >= 0.82
        and purity["step_abs_yaw_max"] <= 1.4
        and purity["step_abs_pitch_max"] <= 1.6
        and purity["straightness_xz"] >= 0.84
        and purity_ratio <= 1.18
    ):
        return None, 0.0

    if axis == "z":
        label = "向前移动" if forward > 0 else "向后移动"
    else:
        label = "向右移动" if right > 0 else "向左移动"

    score = (
        float(motion["translation_m"]) * 3.0
        + min(ratio, 8.0)
        + min(agg_straight, 1.0)
        + purity["straightness_xz"]
        - 0.15 * abs(float(motion["yaw_deg"]))
        - 0.20 * abs(float(motion["pitch_deg"]))
    )
    return label, score


def classify_rotation(motion: dict, purity: dict) -> tuple[str | None, float]:
    yaw = float(motion["yaw_deg"])
    trans = float(motion["translation_m"])
    right, up, forward = [abs(float(x)) for x in motion["local_delta"]]
    purity_ratio = purity["step_abs_yaw_sum"] / max(abs(yaw), 1e-8)

    if not (
        12.0 <= abs(yaw) <= 28.0
        and trans <= 0.05
        and right <= 0.04
        and forward <= 0.04
        and up <= 0.04
        and abs(float(motion["pitch_deg"])) <= 2.5
        and purity["step_trans_max"] <= 0.025
        and purity["step_abs_pitch_max"] <= 1.5
        and purity["yaw_same_sign_ratio"] >= 0.95
        and purity_ratio <= 1.18
    ):
        return None, 0.0

    label = "向右转动" if yaw > 0 else "向左转动"
    score = abs(yaw) + 10.0 * (0.06 - trans) - 0.8 * abs(float(motion["pitch_deg"]))
    return label, score


def classify_segment(items: list[PoseItem], start_idx: int, end_idx: int) -> tuple[str | None, dict, float]:
    motion = motion_between(items, start_idx, end_idx)
    purity = segment_purity(items, start_idx, end_idx)
    motion["segment_length"] = end_idx - start_idx
    motion["purity"] = purity

    label_t, score_t = classify_translation(motion, purity)
    label_r, score_r = classify_rotation(motion, purity)
    if label_t and not label_r:
        return label_t, motion, score_t
    if label_r and not label_t:
        return label_r, motion, score_r
    return None, motion, 0.0


def iter_scene_dirs(root: Path, scene_ids: set[str] | None = None) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if scene_ids is not None and path.name not in scene_ids:
            continue
        if (path / "iphone" / "rgb.mkv").exists() and (path / "iphone" / "pose_intrinsic_imu.json").exists():
            yield path


def gather_candidates(
    root: Path,
    max_scenes: int,
    min_gap: int,
    max_gap: int,
    scene_ids: set[str] | None = None,
) -> dict[str, list[SegmentCandidate]]:
    per_label: dict[str, list[SegmentCandidate]] = defaultdict(list)
    for scene_idx, scene_dir in enumerate(iter_scene_dirs(root, scene_ids), start=1):
        if scene_idx > max_scenes:
            break
        items = load_scannetpp_scene(scene_dir)
        if len(items) < 10:
            continue
        scene_id = scene_dir.name
        for start in range(len(items) - 1):
            for gap in range(max(2, min_gap), max_gap + 1):
                end = start + gap
                if end >= len(items):
                    break
                label, motion, score = classify_segment(items, start, end)
                if not label:
                    continue
                per_label[label].append(
                    SegmentCandidate(
                        scene_id=scene_id,
                        label=label,
                        start_idx=start,
                        end_idx=end,
                        motion=motion,
                        score=score,
                    )
                )
    for label in per_label:
        per_label[label].sort(key=lambda x: x.score, reverse=True)
    return per_label


def select_candidates(per_label: dict[str, list[SegmentCandidate]], target_total: int, max_per_scene: int) -> list[SegmentCandidate]:
    chosen: list[SegmentCandidate] = []
    used_frames: dict[str, list[tuple[int, int]]] = defaultdict(list)
    per_scene_count: dict[str, int] = defaultdict(int)

    def conflict(scene_id: str, start_idx: int, end_idx: int) -> bool:
        for a, b in used_frames[scene_id]:
            if not (end_idx < a - 8 or start_idx > b + 8):
                return True
        return False

    for label in ACTION_OPTIONS:
        for cand in per_label.get(label, []):
            if per_scene_count[cand.scene_id] >= max_per_scene:
                continue
            if conflict(cand.scene_id, cand.start_idx, cand.end_idx):
                continue
            chosen.append(cand)
            used_frames[cand.scene_id].append((cand.start_idx, cand.end_idx))
            per_scene_count[cand.scene_id] += 1
            break

    pool = []
    for label in ACTION_OPTIONS:
        pool.extend(per_label.get(label, []))
    pool.sort(key=lambda x: x.score, reverse=True)
    for cand in pool:
        if len(chosen) >= target_total:
            break
        if per_scene_count[cand.scene_id] >= max_per_scene:
            continue
        if any(c.scene_id == cand.scene_id and c.start_idx == cand.start_idx and c.end_idx == cand.end_idx for c in chosen):
            continue
        if conflict(cand.scene_id, cand.start_idx, cand.end_idx):
            continue
        chosen.append(cand)
        used_frames[cand.scene_id].append((cand.start_idx, cand.end_idx))
        per_scene_count[cand.scene_id] += 1
    return chosen[:target_total]


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed_to_read_frame:{video_path}:{frame_index}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"failed_to_write_frame:{output_path}")
    finally:
        cap.release()


def frame_sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def render_preview(output_root: Path, selected: list[SegmentCandidate]) -> list[dict]:
    preview_root = output_root / "action_inference"
    if preview_root.exists():
        shutil.rmtree(preview_root)
    ensure_dir(preview_root)

    rows = []
    for idx, cand in enumerate(selected, start=1):
        qa_id = f"scannetpp_action_preview_{idx:03d}"
        sample_dir = preview_root / qa_id
        ensure_dir(sample_dir)
        video_path = DATA_ROOT / cand.scene_id / "iphone" / "rgb.mkv"
        frame_a = sample_dir / "frame_A.png"
        frame_b = sample_dir / "frame_B.png"
        extract_frame(video_path, cand.motion["start_frame_index"], frame_a)
        extract_frame(video_path, cand.motion["end_frame_index"], frame_b)
        sharp_a = frame_sharpness(frame_a)
        sharp_b = frame_sharpness(frame_b)
        row = {
            "qa_id": qa_id,
            "task_type": "action_inference",
            "dataset": "scannetpp",
            "scene": cand.scene_id,
            "pose_source": "scannetpp_pose_intrinsic_imu_aligned_pose",
            "question": "图A到图B之间，相机主要执行了什么动作？",
            "answer": cand.label,
            "options": ACTION_OPTIONS,
            "input": {
                "frame_paths": [str(frame_a), str(frame_b)],
                "frame_A": str(frame_a),
                "frame_B": str(frame_b),
            },
            "gt": {
                "motion": cand.motion,
                "quality": {
                    "candidate_score": cand.score,
                    "frame_A_sharpness": sharp_a,
                    "frame_B_sharpness": sharp_b,
                },
            },
        }
        rows.append(row)

    write_json(preview_root / "qa_data.json", rows)
    with (preview_root / "qa_data.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small high-quality ScanNet++ action inference preview.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-total", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=200)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--max-gap", type=int, default=20)
    parser.add_argument("--max-per-scene", type=int, default=6)
    parser.add_argument("--scene-ids", type=str, default="", help="Comma-separated scene ids")
    args = parser.parse_args()

    scene_ids = {x.strip() for x in args.scene_ids.split(",") if x.strip()} or None
    per_label = gather_candidates(args.data_root, args.max_scenes, args.min_gap, args.max_gap, scene_ids)
    selected = select_candidates(per_label, args.target_total, args.max_per_scene)
    rows = render_preview(args.output_root, selected)

    summary = {
        "target_total": args.target_total,
        "generated_total": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in ACTION_OPTIONS},
        "scenes": sorted({row["scene"] for row in rows}),
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
