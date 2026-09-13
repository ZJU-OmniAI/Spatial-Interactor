#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np


DEFAULT_ALIGNED_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/aligned_frames")
DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/motion_qa")

ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动"]
FAMILY_OPTIONS = ["平移为主", "转动为主", "平移和转动都明显"]


@dataclass
class PoseItem:
    frame_path: Path
    frame_index: int
    pose_index: int
    c2w: np.ndarray
    raw: dict


@dataclass
class MotionScene:
    dataset: str
    scene_id: str
    metadata_path: Path
    video_path: Path
    items: list[PoseItem]

    @property
    def axis(self) -> dict:
        if self.dataset == "arkitscenes":
            return {"forward_sign": -1.0}
        return {"forward_sign": 1.0}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def axis_angle_to_matrix(axis_angle: Sequence[float]) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rot.astype(np.float64)


def parse_matrix_lines(lines: Sequence[str]) -> np.ndarray:
    mat = np.asarray([[float(x) for x in line.split()] for line in lines], dtype=np.float64)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected pose matrix shape: {mat.shape}")
    return mat


def parse_arkit_pose(raw_values: Sequence[float]) -> np.ndarray:
    if len(raw_values) != 6:
        raise ValueError(f"Unexpected ARKit pose values: {raw_values}")
    rot = axis_angle_to_matrix(raw_values[:3])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(raw_values[3:6], dtype=np.float64)
    return mat


def load_motion_scene(metadata_path: Path) -> MotionScene:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    scene_id = payload["scene_id"]
    items = []
    for item in payload.get("items", []):
        frame_path = Path(item["frame_path"])
        if not frame_path.exists():
            continue
        if dataset == "scannet":
            c2w = parse_matrix_lines(item["pose_matrix"])
        elif dataset == "arkitscenes":
            c2w = parse_arkit_pose(item["pose_raw_values"])
        else:
            continue
        if not np.isfinite(c2w).all():
            continue
        items.append(
            PoseItem(
                frame_path=frame_path,
                frame_index=int(item["frame_index"]),
                pose_index=int(item["pose_index"]),
                c2w=c2w,
                raw=item,
            )
        )
    items.sort(key=lambda item: item.pose_index)
    return MotionScene(
        dataset=dataset,
        scene_id=scene_id,
        metadata_path=metadata_path,
        video_path=Path(payload["video_path"]),
        items=items,
    )


def load_scenes(aligned_root: Path, datasets: set[str]) -> list[MotionScene]:
    scenes = []
    for metadata_path in sorted(aligned_root.glob("**/metadata.json")):
        try:
            scene = load_motion_scene(metadata_path)
        except Exception:
            continue
        if scene.dataset not in datasets:
            continue
        if len(scene.items) < 4:
            continue
        scenes.append(scene)
    return scenes


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def heading_pitch_from_c2w(c2w: np.ndarray, forward_sign: float) -> tuple[float, float]:
    forward_cam = np.array([0.0, 0.0, forward_sign], dtype=np.float64)
    forward = c2w[:3, :3] @ forward_cam
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2]) if abs(float(forward[2])) > 1e-8 else 1e-8))
    horiz = math.sqrt(float(forward[0]) ** 2 + float(forward[2]) ** 2)
    pitch = math.degrees(math.atan2(float(forward[1]), max(horiz, 1e-8)))
    return yaw, pitch


def motion_between(scene: MotionScene, start_idx: int, end_idx: int) -> dict:
    start = scene.items[start_idx]
    end = scene.items[end_idx]
    forward_sign = scene.axis["forward_sign"]
    center_a = start.c2w[:3, 3]
    center_b = end.c2w[:3, 3]
    delta_world = center_b - center_a
    delta_local = start.c2w[:3, :3].T @ delta_world
    right_m = float(delta_local[0])
    forward_m = float(delta_local[2] * forward_sign)
    up_m = float(delta_local[1])
    trans_m = float(np.linalg.norm(delta_world))
    yaw_a, pitch_a = heading_pitch_from_c2w(start.c2w, forward_sign)
    yaw_b, pitch_b = heading_pitch_from_c2w(end.c2w, forward_sign)
    yaw_delta = wrap_deg(yaw_b - yaw_a)
    pitch_delta = wrap_deg(pitch_b - pitch_a)
    return {
        "start_slot": start_idx,
        "end_slot": end_idx,
        "start_pose_index": start.pose_index,
        "end_pose_index": end.pose_index,
        "start_frame_index": start.frame_index,
        "end_frame_index": end.frame_index,
        "translation_m": trans_m,
        "yaw_deg": yaw_delta,
        "pitch_deg": pitch_delta,
        "local_delta": [right_m, up_m, forward_m],
        "abs_yaw_deg": abs(yaw_delta),
        "abs_pitch_deg": abs(pitch_delta),
    }


def dominant_axis_ratio(local_delta: list[float]) -> tuple[str | None, float, float]:
    if len(local_delta) < 3:
        return None, 0.0, 0.0
    right = abs(float(local_delta[0]))
    forward = abs(float(local_delta[2]))
    primary = max(right, forward)
    secondary = min(right, forward)
    if primary < 1e-8:
        return None, 0.0, 0.0
    axis = "x" if right >= forward else "z"
    return axis, primary, primary / max(secondary, 1e-8)


def clean_translation_motion(motion: dict) -> bool:
    axis, primary, ratio = dominant_axis_ratio(motion["local_delta"])
    up = abs(float(motion["local_delta"][1]))
    return (
        axis is not None
        and 0.18 <= float(motion["translation_m"]) <= 0.85
        and abs(float(motion["yaw_deg"])) <= 5.0
        and abs(float(motion["pitch_deg"])) <= 4.0
        and primary >= 0.12
        and ratio >= 1.25
        and up <= 0.18
    )


def clean_rotation_motion(motion: dict) -> bool:
    return (
        12.0 <= abs(float(motion["yaw_deg"])) <= 38.0
        and float(motion["translation_m"]) <= 0.12
        and abs(float(motion["pitch_deg"])) <= 4.0
        and abs(float(motion["local_delta"][1])) <= 0.08
    )


def classify_action(motion: dict) -> dict | None:
    trans_score = motion["translation_m"] / 0.30
    rot_score = abs(motion["yaw_deg"]) / 15.0
    if trans_score < 0.75 and rot_score < 0.75:
        return None
    if trans_score >= rot_score * 1.60 and clean_translation_motion(motion):
        right, _up, forward = motion["local_delta"]
        if abs(forward) >= abs(right):
            label = "向前移动" if forward > 0 else "向后移动"
        else:
            label = "向右移动" if right > 0 else "向左移动"
        return {"label": label, "family": "translation", "dominance": trans_score / max(rot_score, 1e-6)}
    if rot_score >= trans_score * 1.60 and clean_rotation_motion(motion):
        label = "向右转动" if motion["yaw_deg"] > 0 else "向左转动"
        return {"label": label, "family": "rotation", "dominance": rot_score / max(trans_score, 1e-6)}
    return None


def movement_magnitude(motion: dict, family: str | None = None) -> float:
    if family == "translation":
        return motion["translation_m"]
    if family == "rotation":
        return abs(motion["yaw_deg"])
    return motion["translation_m"] + 0.015 * abs(motion["yaw_deg"]) + 0.01 * abs(motion["pitch_deg"])


def family_label(motion: dict) -> str | None:
    trans_score = motion["translation_m"] / 0.30
    rot_score = abs(motion["yaw_deg"]) / 15.0
    if trans_score < 0.8 and rot_score < 0.8:
        return None
    if trans_score >= rot_score * 1.35:
        return "平移为主"
    if rot_score >= trans_score * 1.35:
        return "转动为主"
    if trans_score >= 0.9 and rot_score >= 0.9:
        return "平移和转动都明显"
    return None


def copy_frame(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def scene_key(scene: MotionScene) -> str:
    return f"{scene.dataset}/{scene.scene_id}"


def make_id(task_type: str, idx: int) -> str:
    return f"{task_type}_{idx:06d}"


def save_task_rows(output_root: Path, task_name: str, rows: list[dict]) -> None:
    task_dir = output_root / task_name
    write_json(task_dir / "qa_data.json", rows)
    write_jsonl(task_dir / "qa_data.jsonl", rows)


def build_action_inference(scene: MotionScene, qa_id: str, sample_dir: Path) -> dict | None:
    candidates = []
    for i in range(len(scene.items) - 1):
        motion = motion_between(scene, i, i + 1)
        action = classify_action(motion)
        if action:
            candidates.append((action["dominance"], i, motion, action))
    if not candidates:
        return None
    _score, i, motion, action = max(candidates, key=lambda x: x[0])
    frame_a = sample_dir / "frame_A.png"
    frame_b = sample_dir / "frame_B.png"
    copy_frame(scene.items[i].frame_path, frame_a)
    copy_frame(scene.items[i + 1].frame_path, frame_b)
    return {
        "qa_id": qa_id,
        "task_type": "action_inference",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "图A到图B之间，相机主要执行了什么动作？",
        "answer": action["label"],
        "options": ACTION_OPTIONS,
        "input": {"frame_paths": [str(frame_a), str(frame_b)], "frame_A": str(frame_a), "frame_B": str(frame_b)},
        "gt": {"motion": motion, "action_family": action["family"], "pose_source": "official_pose_aligned"},
    }


def build_sequence_sorting(scene: MotionScene, qa_id: str, sample_dir: Path) -> dict | None:
    if len(scene.items) < 4:
        return None
    windows = []
    for i in range(len(scene.items) - 3):
        motions = [motion_between(scene, i + j, i + j + 1) for j in range(3)]
        total = sum(movement_magnitude(m) for m in motions)
        if total > 0.25:
            windows.append((total, i))
    if not windows:
        return None
    _score, start = max(windows, key=lambda x: x[0])
    slots = [start, start + 1, start + 2, start + 3]
    labels = ["A", "B", "C"]
    candidates = list(zip(labels, slots[1:]))
    random.Random(f"{scene_key(scene)}:{qa_id}").shuffle(candidates)
    correct = [label for label, slot in sorted(candidates, key=lambda item: item[1])]
    start_path = sample_dir / "frame_start.png"
    copy_frame(scene.items[slots[0]].frame_path, start_path)
    frame_paths = [start_path]
    candidate_frames = {}
    for label, slot in candidates:
        out = sample_dir / f"frame_{label}.png"
        copy_frame(scene.items[slot].frame_path, out)
        frame_paths.append(out)
        candidate_frames[label] = str(out)
    return {
        "qa_id": qa_id,
        "task_type": "movement_sequence_sorting",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "已知第一张图是起始帧。请将其余三张候选图按真实视频中的时间先后排序。",
        "answer": " -> ".join(correct),
        "input": {"frame_paths": [str(p) for p in frame_paths], "first_frame": str(start_path), "candidate_frames": candidate_frames},
        "gt": {
            "start_slot": slots[0],
            "candidate_slots": {label: slot for label, slot in candidates},
            "correct_order": correct,
            "pose_source": "official_pose_aligned",
        },
    }


def build_degree_comparison(scene: MotionScene, qa_id: str, sample_dir: Path) -> dict | None:
    candidates = []
    for start in range(len(scene.items) - 2):
        for a in range(start + 1, min(len(scene.items), start + 4)):
            for b in range(a + 1, min(len(scene.items), start + 5)):
                ma = motion_between(scene, start, a)
                mb = motion_between(scene, start, b)
                fa = "rotation" if abs(ma["yaw_deg"]) / 15.0 > ma["translation_m"] / 0.30 else "translation"
                fb = "rotation" if abs(mb["yaw_deg"]) / 15.0 > mb["translation_m"] / 0.30 else "translation"
                if fa != fb:
                    continue
                mag_a = movement_magnitude(ma, fa)
                mag_b = movement_magnitude(mb, fb)
                gap = abs(mag_a - mag_b)
                min_gap = 0.18 if fa == "translation" else 8.0
                if gap >= min_gap:
                    candidates.append((gap, start, a, b, fa, ma, mb))
    if not candidates:
        return None
    _gap, start, a, b, family, ma, mb = max(candidates, key=lambda x: x[0])
    if hash(scene_key(scene)) % 2:
        left_slot, right_slot = a, b
        left_motion, right_motion = ma, mb
    else:
        left_slot, right_slot = b, a
        left_motion, right_motion = mb, ma
    larger = "A" if movement_magnitude(left_motion, family) > movement_magnitude(right_motion, family) else "B"
    start_path = sample_dir / "frame_start.png"
    frame_a = sample_dir / "frame_A.png"
    frame_b = sample_dir / "frame_B.png"
    copy_frame(scene.items[start].frame_path, start_path)
    copy_frame(scene.items[left_slot].frame_path, frame_a)
    copy_frame(scene.items[right_slot].frame_path, frame_b)
    return {
        "qa_id": qa_id,
        "task_type": "movement_degree_comparison",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
        "answer": f"候选图{larger}更大。",
        "input": {"frame_paths": [str(start_path), str(frame_a), str(frame_b)], "start_frame": str(start_path), "frame_A": str(frame_a), "frame_B": str(frame_b)},
        "gt": {
            "comparison_family": family,
            "larger_clip": larger,
            "start_slot": start,
            "frame_A": {"slot_end": left_slot, "magnitude": movement_magnitude(left_motion, family), "motion": left_motion},
            "frame_B": {"slot_end": right_slot, "magnitude": movement_magnitude(right_motion, family), "motion": right_motion},
            "pose_source": "official_pose_aligned",
        },
    }


def build_motion_family(scene: MotionScene, qa_id: str, sample_dir: Path) -> dict | None:
    candidates = []
    for i in range(len(scene.items) - 1):
        motion = motion_between(scene, i, i + 1)
        label = family_label(motion)
        if label:
            strength = max(motion["translation_m"] / 0.30, abs(motion["yaw_deg"]) / 15.0)
            candidates.append((strength, i, motion, label))
    if not candidates:
        return None
    _strength, i, motion, label = max(candidates, key=lambda x: x[0])
    frame_a = sample_dir / "frame_A.png"
    frame_b = sample_dir / "frame_B.png"
    copy_frame(scene.items[i].frame_path, frame_a)
    copy_frame(scene.items[i + 1].frame_path, frame_b)
    return {
        "qa_id": qa_id,
        "task_type": "motion_family_discrimination",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "图A到图B之间，相机运动更接近哪一种类型？",
        "answer": label,
        "options": FAMILY_OPTIONS,
        "input": {"frame_paths": [str(frame_a), str(frame_b)], "frame_A": str(frame_a), "frame_B": str(frame_b)},
        "gt": {"motion": motion, "pose_source": "official_pose_aligned"},
    }


def build_distance_to_start(scene: MotionScene, qa_id: str, sample_dir: Path) -> dict | None:
    if len(scene.items) < 3:
        return None
    candidates = []
    for start in range(len(scene.items) - 2):
        motions = [(end, motion_between(scene, start, end)) for end in range(start + 1, len(scene.items))]
        if len(motions) < 2:
            continue
        motions.sort(key=lambda x: movement_magnitude(x[1]))
        small = motions[0]
        large = motions[-1]
        if movement_magnitude(large[1]) - movement_magnitude(small[1]) > 0.25:
            candidates.append((movement_magnitude(large[1]) - movement_magnitude(small[1]), start, small, large))
    if not candidates:
        return None
    _gap, start, small, large = max(candidates, key=lambda x: x[0])
    pair = [small, large]
    random.Random(f"distance:{scene_key(scene)}").shuffle(pair)
    labels = ["A", "B"]
    larger_label = labels[pair.index(large)]
    start_path = sample_dir / "frame_start.png"
    copy_frame(scene.items[start].frame_path, start_path)
    frame_paths = [start_path]
    candidate_frames = {}
    gt_candidates = {}
    for label, (slot, motion) in zip(labels, pair):
        out = sample_dir / f"frame_{label}.png"
        copy_frame(scene.items[slot].frame_path, out)
        frame_paths.append(out)
        candidate_frames[label] = str(out)
        gt_candidates[label] = {"slot_end": slot, "magnitude": movement_magnitude(motion), "motion": motion}
    return {
        "qa_id": qa_id,
        "task_type": "distance_to_start_comparison",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "以第一张图为起始视角，候选图A和候选图B中，哪一张对应的相机位置或朝向离起始视角更远？",
        "answer": f"候选图{larger_label}更远。",
        "input": {"frame_paths": [str(p) for p in frame_paths], "start_frame": str(start_path), "candidate_frames": candidate_frames},
        "gt": {"start_slot": start, "farther_candidate": larger_label, "candidates": gt_candidates, "pose_source": "official_pose_aligned"},
    }


def build_return_to_start(scene: MotionScene, qa_id: str, sample_dir: Path) -> dict | None:
    if len(scene.items) < 4:
        return None
    candidates = []
    for start in range(len(scene.items) - 3):
        later = [(end, motion_between(scene, start, end)) for end in range(start + 1, len(scene.items))]
        later.sort(key=lambda x: movement_magnitude(x[1]))
        closest = later[0]
        farthest = later[-1]
        if movement_magnitude(farthest[1]) - movement_magnitude(closest[1]) > 0.35:
            candidates.append((movement_magnitude(farthest[1]) - movement_magnitude(closest[1]), start, later[:3]))
    if not candidates:
        return None
    _gap, start, later = max(candidates, key=lambda x: x[0])
    # Keep the closest candidate and two farther distractors.
    chosen = later[:1] + later[-2:]
    random.Random(f"return:{scene_key(scene)}").shuffle(chosen)
    labels = ["A", "B", "C"]
    closest_slot = min(chosen, key=lambda x: movement_magnitude(x[1]))[0]
    answer_label = labels[[slot for slot, _motion in chosen].index(closest_slot)]
    start_path = sample_dir / "frame_start.png"
    copy_frame(scene.items[start].frame_path, start_path)
    frame_paths = [start_path]
    candidate_frames = {}
    gt_candidates = {}
    for label, (slot, motion) in zip(labels, chosen):
        out = sample_dir / f"frame_{label}.png"
        copy_frame(scene.items[slot].frame_path, out)
        frame_paths.append(out)
        candidate_frames[label] = str(out)
        gt_candidates[label] = {"slot_end": slot, "magnitude": movement_magnitude(motion), "motion": motion}
    return {
        "qa_id": qa_id,
        "task_type": "return_to_start_detection",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "以第一张图为起始视角，候选图A、B、C中，哪一张最接近回到起始视角？",
        "answer": f"候选图{answer_label}最接近起始视角。",
        "input": {"frame_paths": [str(p) for p in frame_paths], "start_frame": str(start_path), "candidate_frames": candidate_frames},
        "gt": {"start_slot": start, "closest_candidate": answer_label, "candidates": gt_candidates, "pose_source": "official_pose_aligned"},
    }


TASK_BUILDERS: list[tuple[str, Callable[[MotionScene, str, Path], dict | None]]] = [
    ("action_inference", build_action_inference),
    ("movement_sequence_sorting", build_sequence_sorting),
    ("movement_degree_comparison", build_degree_comparison),
    ("motion_family_discrimination", build_motion_family),
    ("distance_to_start_comparison", build_distance_to_start),
    ("return_to_start_detection", build_return_to_start),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate motion-understanding QA from aligned real-video frames and official poses.")
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", type=str, default="scannet,arkitscenes")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--per-task-limit", type=int, default=0)
    args = parser.parse_args()

    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    scenes = load_scenes(args.aligned_root, datasets)
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]

    all_rows = []
    overview = {"aligned_root": str(args.aligned_root), "num_scenes_loaded": len(scenes), "tasks": {}}
    task_counters = {task_name: 0 for task_name, _builder in TASK_BUILDERS}

    for scene in scenes:
        for task_name, builder in TASK_BUILDERS:
            if args.per_task_limit > 0 and task_counters[task_name] >= args.per_task_limit:
                continue
            qa_id = make_id(task_name, task_counters[task_name] + 1)
            sample_dir = args.output_root / task_name / qa_id
            row = builder(scene, qa_id, sample_dir)
            if row is None:
                continue
            task_counters[task_name] += 1
            all_rows.append(row)

    for task_name, _builder in TASK_BUILDERS:
        rows = [row for row in all_rows if row["task_type"] == task_name]
        save_task_rows(args.output_root, task_name, rows)
        overview["tasks"][task_name] = {
            "count": len(rows),
            "path": str(args.output_root / task_name / "qa_data.json"),
        }

    write_json(args.output_root / "all_qa_data.json", all_rows)
    write_jsonl(args.output_root / "all_qa_data.jsonl", all_rows)
    write_json(args.output_root / "overview.json", overview)
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
