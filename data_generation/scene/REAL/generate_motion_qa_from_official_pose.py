#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from extract_aligned_real_frames import (
    ARKIT_RAW_ROOT,
    VSI_ROOT,
    ensure_scannet_scene_unpacked,
    extract_frames_by_index,
    locate_arkit_scene,
    parse_arkit_traj_line,
    read_video_info,
)


OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/motion_qa_official")
ARKIT_CSV = Path("/path/to/workspace/DATA/REAL_OFFICIAL/manifests/arkitscenes_vsi_subset.csv")
SCANNET_IDS = Path("/path/to/workspace/DATA/REAL_OFFICIAL/manifests/scannet_scene_ids.txt")

ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动"]
FAMILY_OPTIONS = ["平移为主", "转动为主", "平移和转动都明显"]


@dataclass
class DensePoseItem:
    pose_index: int
    frame_index: int
    c2w: np.ndarray
    timestamp: float | None = None
    pose_path: str | None = None
    intrinsics_path: str | None = None


@dataclass
class DenseScene:
    dataset: str
    scene_id: str
    video_path: Path
    video_info: dict
    items: list[DensePoseItem]
    sharpness_cache: dict[int, float] = field(default_factory=dict)

    @property
    def forward_sign(self) -> float:
        return -1.0 if self.dataset == "arkitscenes" else 1.0


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def axis_angle_to_matrix(axis_angle: Sequence[float]) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rot.astype(np.float64)


def parse_scannet_matrix(path: Path) -> np.ndarray:
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected pose matrix in {path}: {mat.shape}")
    return mat


def load_scannet_scene(scene_id: str) -> DenseScene:
    scene_dir = ensure_scannet_scene_unpacked(scene_id)
    video_path = VSI_ROOT / "scannet" / f"{scene_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"missing_scannet_video:{video_path}")
    video_info = read_video_info(video_path)

    pose_dir = scene_dir / "pose"
    pose_files = sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem))
    pose_ids = [int(p.stem) for p in pose_files]
    if pose_ids != list(range(video_info["frame_count"])):
        raise RuntimeError(
            f"scannet_pose_video_mismatch:{scene_id}:video_frames={video_info['frame_count']}:pose_files={len(pose_files)}"
        )

    items = []
    for pose_idx, pose_path in enumerate(pose_files):
        c2w = parse_scannet_matrix(pose_path)
        if not np.isfinite(c2w).all():
            continue
        items.append(
            DensePoseItem(
                pose_index=pose_idx,
                frame_index=pose_idx,
                c2w=c2w,
                pose_path=str(pose_path),
            )
        )
    return DenseScene("scannet", scene_id, video_path, video_info, items)


def load_arkit_scene(video_id: str) -> DenseScene:
    scene_dir = locate_arkit_scene(video_id)
    video_path = VSI_ROOT / "arkitscenes" / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"missing_arkit_video:{video_path}")
    video_info = read_video_info(video_path)

    traj_path = scene_dir / "lowres_wide.traj"
    intr_dir = scene_dir / "lowres_wide_intrinsics"
    intr_files = sorted(intr_dir.glob("*.pincam"), key=lambda p: float(p.stem.split("_")[-1]))
    if not intr_files:
        raise FileNotFoundError(f"missing_arkit_intrinsics:{intr_dir}")

    traj_records = [parse_arkit_traj_line(line) for line in traj_path.read_text().splitlines() if line.strip()]
    intr_timestamps = [float(path.stem.split("_")[-1]) for path in intr_files]
    official_start = intr_timestamps[0]
    official_end = intr_timestamps[-1]
    official_span = official_end - official_start
    if official_span <= 0:
        raise RuntimeError(f"invalid_intrinsics_timespan:{video_id}")

    video_span = (video_info["frame_count"] - 1) / video_info["fps"]
    if abs(video_span - official_span) > 0.2:
        raise RuntimeError(
            f"arkit_time_span_mismatch:{video_id}:video_span={video_span:.6f}:official_span={official_span:.6f}"
        )

    items = []
    for pose_idx, record in enumerate(traj_records):
        rot = axis_angle_to_matrix(record["raw_values"][:3])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot
        c2w[:3, 3] = np.asarray(record["raw_values"][3:6], dtype=np.float64)
        if not np.isfinite(c2w).all():
            continue
        ts = float(record["timestamp"])
        frame_index = round((ts - official_start) / official_span * (video_info["frame_count"] - 1))
        frame_index = max(0, min(video_info["frame_count"] - 1, frame_index))
        intr_idx = min(
            range(len(intr_timestamps)),
            key=lambda i: abs(intr_timestamps[i] - (official_start + frame_index * official_span / (video_info["frame_count"] - 1))),
        )
        items.append(
            DensePoseItem(
                pose_index=pose_idx,
                frame_index=frame_index,
                c2w=c2w,
                timestamp=ts,
                pose_path=str(traj_path),
                intrinsics_path=str(intr_files[intr_idx]),
            )
        )
    return DenseScene("arkitscenes", video_id, video_path, video_info, items)


def load_scannet_ids() -> list[str]:
    return [line.strip() for line in SCANNET_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_arkit_ids() -> list[str]:
    ids = []
    with ARKIT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(row["video_id"])
    return ids


def heading_pitch_from_c2w(c2w: np.ndarray, forward_sign: float) -> tuple[float, float]:
    forward_cam = np.array([0.0, 0.0, forward_sign], dtype=np.float64)
    forward = c2w[:3, :3] @ forward_cam
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2]) if abs(float(forward[2])) > 1e-8 else 1e-8))
    horiz = math.sqrt(float(forward[0]) ** 2 + float(forward[2]) ** 2)
    pitch = math.degrees(math.atan2(float(forward[1]), max(horiz, 1e-8)))
    return yaw, pitch


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def motion_between(scene: DenseScene, i: int, j: int) -> dict:
    a = scene.items[i]
    b = scene.items[j]
    delta_world = b.c2w[:3, 3] - a.c2w[:3, 3]
    delta_local = a.c2w[:3, :3].T @ delta_world
    yaw_a, pitch_a = heading_pitch_from_c2w(a.c2w, scene.forward_sign)
    yaw_b, pitch_b = heading_pitch_from_c2w(b.c2w, scene.forward_sign)
    return {
        "start_slot": i,
        "end_slot": j,
        "start_pose_index": a.pose_index,
        "end_pose_index": b.pose_index,
        "start_frame_index": a.frame_index,
        "end_frame_index": b.frame_index,
        "translation_m": float(np.linalg.norm(delta_world)),
        "yaw_deg": wrap_deg(yaw_b - yaw_a),
        "pitch_deg": wrap_deg(pitch_b - pitch_a),
        "local_delta": [float(delta_local[0]), float(delta_local[1]), float(delta_local[2] * scene.forward_sign)],
        "pose_gap": b.pose_index - a.pose_index,
        "frame_gap": b.frame_index - a.frame_index,
    }


def classify_action(motion: dict) -> dict | None:
    trans_score = motion["translation_m"] / 0.12
    rot_score = abs(motion["yaw_deg"]) / 6.0
    if trans_score < 0.75 and rot_score < 0.75:
        return None
    if trans_score >= rot_score * 1.25:
        right, _up, forward = motion["local_delta"]
        if abs(forward) >= abs(right):
            label = "向前移动" if forward > 0 else "向后移动"
        else:
            label = "向右移动" if right > 0 else "向左移动"
        return {"label": label, "family": "translation", "dominance": trans_score / max(rot_score, 1e-6)}
    if rot_score >= trans_score * 1.25:
        label = "向右转动" if motion["yaw_deg"] > 0 else "向左转动"
        return {"label": label, "family": "rotation", "dominance": rot_score / max(trans_score, 1e-6)}
    return None


def movement_magnitude(motion: dict, family: str | None = None) -> float:
    if family == "translation":
        return motion["translation_m"]
    if family == "rotation":
        return abs(motion["yaw_deg"])
    return motion["translation_m"] + 0.02 * abs(motion["yaw_deg"]) + 0.01 * abs(motion["pitch_deg"])


def family_label(motion: dict) -> str | None:
    trans_score = motion["translation_m"] / 0.12
    rot_score = abs(motion["yaw_deg"]) / 6.0
    if trans_score < 0.8 and rot_score < 0.8:
        return None
    if trans_score >= rot_score * 1.35:
        return "平移为主"
    if rot_score >= trans_score * 1.35:
        return "转动为主"
    if trans_score >= 0.9 and rot_score >= 0.9:
        return "平移和转动都明显"
    return None


def extract_frames_for_items(scene: DenseScene, frame_indices: list[int], output_paths: list[Path]) -> None:
    extract_frames_by_index(scene.video_path, frame_indices, output_paths)


def scene_key(scene: DenseScene) -> str:
    return f"{scene.dataset}/{scene.scene_id}"


def allowed_frame_gap(scene: DenseScene, max_gap: int) -> int:
    if scene.dataset == "arkitscenes":
        return max(1, math.ceil(max_gap * scene.video_info["fps"] / 10.0) + 1)
    return max_gap


def visible_motion_score(scene: DenseScene, motion: dict) -> float:
    if scene.dataset == "arkitscenes":
        trans_unit = 0.20
        rot_unit = 6.0
    else:
        trans_unit = 0.05
        rot_unit = 8.0
    return max(float(motion["translation_m"]) / trans_unit, abs(float(motion["yaw_deg"])) / rot_unit)


def sequence_visibility_score(scene: DenseScene, motions: Sequence[dict]) -> float:
    return sum(visible_motion_score(scene, motion) for motion in motions)


def sharpness_threshold(scene: DenseScene) -> float:
    return 80.0 if scene.dataset == "scannet" else 60.0


def frame_sharpness(scene: DenseScene, frame_index: int) -> float:
    cached = scene.sharpness_cache.get(frame_index)
    if cached is not None:
        return cached

    cap = cv2.VideoCapture(str(scene.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{scene.video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        score = 0.0
    else:
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side > 320:
            scale = 320.0 / max_side
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    scene.sharpness_cache[frame_index] = score
    return score


def select_clear_candidate(scene: DenseScene, candidates: list[tuple], frame_index_getter) -> tuple | None:
    if not candidates:
        return None
    top = sorted(candidates, key=lambda item: item[0], reverse=True)[:12]
    threshold = sharpness_threshold(scene)
    clean: list[tuple[float, tuple]] = []
    fallback: list[tuple[float, tuple]] = []
    for candidate in top:
        frame_indices = sorted(set(frame_index_getter(candidate)))
        sharpnesses = [frame_sharpness(scene, frame_index) for frame_index in frame_indices]
        blur_count = sum(score < threshold for score in sharpnesses)
        sharp_bonus = 0.002 * sum(min(score, 500.0) for score in sharpnesses)
        rank = candidate[0] + sharp_bonus - 2.0 * blur_count
        payload = (rank, candidate)
        fallback.append(payload)
        if blur_count == 0:
            clean.append(payload)
    pool = clean if clean else fallback
    return max(pool, key=lambda item: item[0])[1]


def is_local_window(scene: DenseScene, slots: Sequence[int], min_gap: int, max_gap: int) -> bool:
    if len(slots) < 2:
        return False
    max_frame_gap = allowed_frame_gap(scene, max_gap)
    min_frame_gap = allowed_frame_gap(scene, min_gap)
    items = [scene.items[slot] for slot in slots]
    frame_indices = [item.frame_index for item in items]
    if sorted(frame_indices) != frame_indices:
        return False
    if len(set(frame_indices)) != len(frame_indices):
        return False
    pose_indices = [item.pose_index for item in items]
    if sorted(pose_indices) != pose_indices:
        return False
    if pose_indices[-1] - pose_indices[0] < min_gap or pose_indices[-1] - pose_indices[0] > max_gap:
        return False
    if frame_indices[-1] - frame_indices[0] < min_frame_gap or frame_indices[-1] - frame_indices[0] > max_frame_gap:
        return False
    for a, b in zip(pose_indices, pose_indices[1:]):
        if b <= a or b - a > max_gap:
            return False
    for a, b in zip(frame_indices, frame_indices[1:]):
        if b <= a or b - a > max_frame_gap:
            return False
    return True


def build_action_inference(scene: DenseScene, qa_id: str, sample_dir: Path, min_gap: int, max_gap: int) -> dict | None:
    candidates = []
    for i in range(len(scene.items) - 1):
        for j in range(i + 1, min(len(scene.items), i + max_gap + 1)):
            if not is_local_window(scene, [i, j], min_gap, max_gap):
                continue
            motion = motion_between(scene, i, j)
            action = classify_action(motion)
            if action is None:
                continue
            score = visible_motion_score(scene, motion) + 0.35 * action["dominance"] + 0.12 * motion["pose_gap"]
            candidates.append((score, i, j, motion, action))
    selected = select_clear_candidate(scene, candidates, lambda item: [scene.items[item[1]].frame_index, scene.items[item[2]].frame_index])
    if selected is None:
        return None
    _score, i, j, motion, action = selected
    frame_a = sample_dir / "frame_A.png"
    frame_b = sample_dir / "frame_B.png"
    return {
        "qa_id": qa_id,
        "task_type": "action_inference",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "图A到图B之间，相机主要执行了什么动作？",
        "answer": action["label"],
        "options": ACTION_OPTIONS,
        "input": {"frame_paths": [str(frame_a), str(frame_b)], "frame_A": str(frame_a), "frame_B": str(frame_b)},
        "_frame_requests": [
            (scene.items[i].frame_index, str(frame_a)),
            (scene.items[j].frame_index, str(frame_b)),
        ],
        "gt": {
            "motion": motion,
            "action_family": action["family"],
            "frame_gap_limit": allowed_frame_gap(scene, max_gap),
            "pose_source": "official_pose_local_window",
        },
    }


def build_sequence_sorting(scene: DenseScene, qa_id: str, sample_dir: Path, min_gap: int, max_gap: int) -> dict | None:
    if len(scene.items) < 4:
        return None
    candidates = []
    for start in range(len(scene.items) - 3):
        max_end = min(len(scene.items), start + max_gap + 1)
        for mid1 in range(start + 1, max_end):
            for mid2 in range(mid1 + 1, max_end):
                for end in range(mid2 + 1, max_end):
                    slots = [start, mid1, mid2, end]
                    if not is_local_window(scene, slots, min_gap, max_gap):
                        continue
                    motions = [motion_between(scene, slots[k], slots[k + 1]) for k in range(3)]
                    total = sum(movement_magnitude(m) for m in motions)
                    visibility = sequence_visibility_score(scene, motions)
                    if visibility >= 3.5:
                        spacing_bonus = 0.25 * (scene.items[end].pose_index - scene.items[start].pose_index)
                        candidates.append((visibility + 0.2 * total + spacing_bonus, slots, motions))
    selected = select_clear_candidate(
        scene,
        candidates,
        lambda item: [scene.items[slot].frame_index for slot in item[1]],
    )
    if selected is None:
        return None
    _score, slots, motions = selected
    labels = ["A", "B", "C"]
    candidates_labeled = list(zip(labels, slots[1:]))
    random.Random(f"seq:{scene_key(scene)}").shuffle(candidates_labeled)
    correct_order = [label for label, slot in sorted(candidates_labeled, key=lambda x: x[1])]
    outputs = [sample_dir / "frame_start.png"] + [sample_dir / f"frame_{label}.png" for label, _ in candidates_labeled]
    frame_indices = [scene.items[slots[0]].frame_index] + [scene.items[slot].frame_index for _label, slot in candidates_labeled]
    return {
        "qa_id": qa_id,
        "task_type": "movement_sequence_sorting",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "已知第一张图是起始帧。请将其余三张候选图按真实视频中的时间先后排序。",
        "answer": " -> ".join(correct_order),
        "input": {
            "frame_paths": [str(p) for p in outputs],
            "first_frame": str(outputs[0]),
            "candidate_frames": {label: str(sample_dir / f"frame_{label}.png") for label, _ in candidates_labeled},
        },
        "_frame_requests": [(frame_index, str(output)) for frame_index, output in zip(frame_indices, outputs)],
        "gt": {
            "start_slot": slots[0],
            "candidate_slots": {label: slot for label, slot in candidates_labeled},
            "correct_order": correct_order,
            "motions": motions,
            "frame_gap_limit": allowed_frame_gap(scene, max_gap),
            "pose_source": "official_pose_local_window",
        },
    }


def build_degree_comparison(scene: DenseScene, qa_id: str, sample_dir: Path, min_gap: int, max_gap: int) -> dict | None:
    candidates = []
    for start in range(len(scene.items) - 2):
        max_end = min(len(scene.items), start + max_gap + 1)
        for a in range(start + 1, max_end):
            for b in range(a + 1, max_end):
                if not is_local_window(scene, [start, a, b], min_gap, max_gap):
                    continue
                ma = motion_between(scene, start, a)
                mb = motion_between(scene, start, b)
                fa = "rotation" if abs(ma["yaw_deg"]) / 6.0 > ma["translation_m"] / 0.12 else "translation"
                fb = "rotation" if abs(mb["yaw_deg"]) / 6.0 > mb["translation_m"] / 0.12 else "translation"
                if fa != fb:
                    continue
                gap = abs(movement_magnitude(ma, fa) - movement_magnitude(mb, fb))
                min_mag_gap = 0.08 if fa == "translation" else 4.0
                if gap >= min_mag_gap:
                    score = gap + 0.15 * (visible_motion_score(scene, ma) + visible_motion_score(scene, mb))
                    candidates.append((score, start, a, b, fa, ma, mb))
    selected = select_clear_candidate(
        scene,
        candidates,
        lambda item: [scene.items[item[1]].frame_index, scene.items[item[2]].frame_index, scene.items[item[3]].frame_index],
    )
    if selected is None:
        return None
    _score, start, a, b, family, ma, mb = selected
    pair = [("A", a, ma), ("B", b, mb)]
    if random.Random(f"deg:{scene_key(scene)}").randrange(2):
        pair = [("A", b, mb), ("B", a, ma)]
    outputs = [sample_dir / "frame_start.png", sample_dir / "frame_A.png", sample_dir / "frame_B.png"]
    frame_indices = [scene.items[start].frame_index, scene.items[pair[0][1]].frame_index, scene.items[pair[1][1]].frame_index]
    larger = "A" if movement_magnitude(pair[0][2], family) > movement_magnitude(pair[1][2], family) else "B"
    return {
        "qa_id": qa_id,
        "task_type": "movement_degree_comparison",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
        "answer": f"候选图{larger}更大。",
        "input": {"frame_paths": [str(p) for p in outputs], "start_frame": str(outputs[0]), "frame_A": str(outputs[1]), "frame_B": str(outputs[2])},
        "_frame_requests": [(frame_index, str(output)) for frame_index, output in zip(frame_indices, outputs)],
        "gt": {
            "comparison_family": family,
            "larger_clip": larger,
            "start_slot": start,
            "frame_A": {"slot_end": pair[0][1], "magnitude": movement_magnitude(pair[0][2], family), "motion": pair[0][2]},
            "frame_B": {"slot_end": pair[1][1], "magnitude": movement_magnitude(pair[1][2], family), "motion": pair[1][2]},
            "frame_gap_limit": allowed_frame_gap(scene, max_gap),
            "pose_source": "official_pose_local_window",
        },
    }


def build_motion_family(scene: DenseScene, qa_id: str, sample_dir: Path, min_gap: int, max_gap: int) -> dict | None:
    candidates = []
    for i in range(len(scene.items) - 1):
        for j in range(i + 1, min(len(scene.items), i + max_gap + 1)):
            if not is_local_window(scene, [i, j], min_gap, max_gap):
                continue
            motion = motion_between(scene, i, j)
            label = family_label(motion)
            if label is None:
                continue
            score = visible_motion_score(scene, motion) + 0.08 * motion["pose_gap"]
            candidates.append((score, i, j, motion, label))
    selected = select_clear_candidate(scene, candidates, lambda item: [scene.items[item[1]].frame_index, scene.items[item[2]].frame_index])
    if selected is None:
        return None
    _score, i, j, motion, label = selected
    outputs = [sample_dir / "frame_A.png", sample_dir / "frame_B.png"]
    return {
        "qa_id": qa_id,
        "task_type": "motion_family_discrimination",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "图A到图B之间，相机运动更接近哪一种类型？",
        "answer": label,
        "options": FAMILY_OPTIONS,
        "input": {"frame_paths": [str(p) for p in outputs], "frame_A": str(outputs[0]), "frame_B": str(outputs[1])},
        "_frame_requests": [
            (scene.items[i].frame_index, str(outputs[0])),
            (scene.items[j].frame_index, str(outputs[1])),
        ],
        "gt": {
            "motion": motion,
            "frame_gap_limit": allowed_frame_gap(scene, max_gap),
            "pose_source": "official_pose_local_window",
        },
    }


def build_distance_to_start(scene: DenseScene, qa_id: str, sample_dir: Path, min_gap: int, max_gap: int) -> dict | None:
    candidates = []
    for start in range(len(scene.items) - 2):
        ends = list(range(start + 1, min(len(scene.items), start + max_gap + 1)))
        if len(ends) < 2:
            continue
        motions = []
        for end in ends:
            if not is_local_window(scene, [start, end], min_gap, max_gap):
                continue
            motions.append((end, motion_between(scene, start, end)))
        if len(motions) < 2:
            continue
        motions.sort(key=lambda x: movement_magnitude(x[1]))
        small, large = motions[0], motions[-1]
        if movement_magnitude(large[1]) - movement_magnitude(small[1]) > 0.10:
            score = movement_magnitude(large[1]) - movement_magnitude(small[1]) + 0.1 * visible_motion_score(scene, large[1])
            candidates.append((score, start, small, large))
    selected = select_clear_candidate(
        scene,
        candidates,
        lambda item: [scene.items[item[1]].frame_index, scene.items[item[2][0]].frame_index, scene.items[item[3][0]].frame_index],
    )
    if selected is None:
        return None
    _score, start, small, large = selected
    pair = [small, large]
    random.Random(f"dist:{scene_key(scene)}").shuffle(pair)
    labels = ["A", "B"]
    farther_label = labels[pair.index(large)]
    outputs = [sample_dir / "frame_start.png", sample_dir / "frame_A.png", sample_dir / "frame_B.png"]
    frame_indices = [scene.items[start].frame_index, scene.items[pair[0][0]].frame_index, scene.items[pair[1][0]].frame_index]
    gt_candidates = {
        label: {"slot_end": slot, "magnitude": movement_magnitude(motion), "motion": motion}
        for label, (slot, motion) in zip(labels, pair)
    }
    return {
        "qa_id": qa_id,
        "task_type": "distance_to_start_comparison",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "question": "以第一张图为起始视角，候选图A和候选图B中，哪一张离起始视角更远？",
        "answer": f"候选图{farther_label}更远。",
        "input": {"frame_paths": [str(p) for p in outputs], "start_frame": str(outputs[0]), "candidate_frames": {"A": str(outputs[1]), "B": str(outputs[2])}},
        "_frame_requests": [(frame_index, str(output)) for frame_index, output in zip(frame_indices, outputs)],
        "gt": {
            "start_slot": start,
            "farther_candidate": farther_label,
            "candidates": gt_candidates,
            "frame_gap_limit": allowed_frame_gap(scene, max_gap),
            "pose_source": "official_pose_local_window",
        },
    }


TASK_BUILDERS = [
    ("action_inference", build_action_inference),
    ("movement_sequence_sorting", build_sequence_sorting),
    ("movement_degree_comparison", build_degree_comparison),
    ("motion_family_discrimination", build_motion_family),
    ("distance_to_start_comparison", build_distance_to_start),
]


def make_id(task_name: str, idx: int) -> str:
    return f"{task_name}_{idx:06d}"


def save_task_rows(output_root: Path, task_name: str, rows: list[dict]) -> None:
    task_dir = output_root / task_name
    write_json(task_dir / "qa_data.json", rows)
    write_jsonl(task_dir / "qa_data.jsonl", rows)


def iter_scene_specs(selected: set[str]) -> Sequence[tuple[str, str]]:
    per_dataset: list[tuple[str, list[str]]] = []
    if "scannet" in selected:
        per_dataset.append(("scannet", load_scannet_ids()))
    if "arkitscenes" in selected:
        per_dataset.append(("arkitscenes", load_arkit_ids()))
    positions = [0] * len(per_dataset)
    specs: list[tuple[str, str]] = []
    while per_dataset:
        advanced = False
        for idx, (dataset, ids) in enumerate(per_dataset):
            pos = positions[idx]
            if pos >= len(ids):
                continue
            specs.append((dataset, ids[pos]))
            positions[idx] += 1
            advanced = True
        if not advanced:
            break
    return specs


def load_scene(dataset: str, scene_id: str) -> DenseScene:
    if dataset == "scannet":
        return load_scannet_scene(scene_id)
    if dataset == "arkitscenes":
        return load_arkit_scene(scene_id)
    raise ValueError(f"unsupported_dataset:{dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate small-gap motion QA directly from official pose assets and original videos.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--datasets", type=str, default="scannet,arkitscenes")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--per-task-limit", type=int, default=0)
    parser.add_argument("--min-gap-scannet", type=int, default=2)
    parser.add_argument("--min-gap-arkit", type=int, default=2)
    parser.add_argument("--max-gap-scannet", type=int, default=4)
    parser.add_argument("--max-gap-arkit", type=int, default=5)
    args = parser.parse_args()

    selected = {item.strip() for item in args.datasets.split(",") if item.strip()}
    scene_specs = iter_scene_specs(selected)
    if args.limit_scenes > 0:
        scene_specs = scene_specs[: args.limit_scenes]

    counters = {task_name: 0 for task_name, _builder in TASK_BUILDERS}
    all_rows = []
    loaded_scene_count = 0

    for scene_idx, (dataset, scene_id) in enumerate(scene_specs, start=1):
        if args.per_task_limit > 0 and all(count >= args.per_task_limit for count in counters.values()):
            break
        try:
            scene = load_scene(dataset, scene_id)
        except Exception as exc:
            print(f"[scene {scene_idx}/{len(scene_specs)}] {dataset}/{scene_id} skipped={exc}", flush=True)
            continue
        loaded_scene_count += 1
        min_gap = args.min_gap_arkit if scene.dataset == "arkitscenes" else args.min_gap_scannet
        max_gap = args.max_gap_arkit if scene.dataset == "arkitscenes" else args.max_gap_scannet
        scene_rows: list[dict] = []
        for task_name, builder in TASK_BUILDERS:
            if args.per_task_limit > 0 and counters[task_name] >= args.per_task_limit:
                continue
            qa_id = make_id(task_name, counters[task_name] + 1)
            sample_dir = args.output_root / task_name / qa_id
            row = builder(scene, qa_id, sample_dir, min_gap, max_gap)
            if row is None:
                continue
            counters[task_name] += 1
            scene_rows.append(row)
        if scene_rows:
            frame_indices = []
            output_paths = []
            for row in scene_rows:
                for frame_index, output_path in row.pop("_frame_requests", []):
                    frame_indices.append(frame_index)
                    output_paths.append(Path(output_path))
            if frame_indices:
                extract_frames_for_items(scene, frame_indices, output_paths)
            all_rows.extend(scene_rows)
        print(
            f"[scene {scene_idx}/{len(scene_specs)}] {scene.dataset}/{scene.scene_id} "
            f"generated={len(scene_rows)} counters={json.dumps(counters, ensure_ascii=False)}",
            flush=True,
        )

    overview = {
        "num_scenes_loaded": loaded_scene_count,
        "tasks": {},
        "min_gap_scannet": args.min_gap_scannet,
        "min_gap_arkit": args.min_gap_arkit,
        "max_gap_scannet": args.max_gap_scannet,
        "max_gap_arkit": args.max_gap_arkit,
    }
    for task_name, _builder in TASK_BUILDERS:
        rows = [row for row in all_rows if row["task_type"] == task_name]
        save_task_rows(args.output_root, task_name, rows)
        overview["tasks"][task_name] = {"count": len(rows), "path": str(args.output_root / task_name / "qa_data.json")}

    write_json(args.output_root / "all_qa_data.json", all_rows)
    write_jsonl(args.output_root / "all_qa_data.jsonl", all_rows)
    write_json(args.output_root / "overview.json", overview)
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
