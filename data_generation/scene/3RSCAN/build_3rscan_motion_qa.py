#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


DEFAULT_INPUT_ROOT = Path("/path/to/workspace/3RScan_sequence_only")
DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/DATA/3RSCAN")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动"]

DEFAULT_PER_TASK_LIMIT = 60
QUALITY_PROFILES = {
    "default": {
        "min_frame_std": 0.035,
        "min_frame_sharpness": 28.0,
        "min_brightness": 28.0,
        "max_brightness": 225.0,
        "max_start_floor_bias": 0.22,
        "action_min_pair_diff": 0.028,
        "sequence_min_adj_diff": 0.016,
        "sequence_min_mean_pair_diff": 0.028,
        "degree_min_pair_diff": 0.022,
        "min_anchor_matches": 14,
        "min_anchor_ratio": 0.045,
        "max_vertical_ratio_translation": 0.85,
        "max_vertical_abs_translation": 0.45,
        "max_vertical_abs_rotation": 0.12,
        "min_action_pose_gap": 1,
        "max_action_pose_gap": 5,
        "min_sequence_pose_step": 1,
        "max_sequence_pose_step": 4,
        "min_degree_pose_gap": 1,
        "max_degree_pose_gap": 7,
        "sequence_min_total_magnitude": 0.30,
        "sequence_min_translation_final_distance": 0.30,
        "sequence_min_rotation_final_yaw": 18.0,
    },
    "strict_high_quality": {
        "min_frame_std": 0.042,
        "min_frame_sharpness": 36.0,
        "min_brightness": 34.0,
        "max_brightness": 210.0,
        "max_start_floor_bias": 0.14,
        "action_min_pair_diff": 0.040,
        "sequence_min_adj_diff": 0.024,
        "sequence_min_mean_pair_diff": 0.040,
        "degree_min_pair_diff": 0.032,
        "min_anchor_matches": 22,
        "min_anchor_ratio": 0.085,
        "max_vertical_ratio_translation": 0.30,
        "max_vertical_abs_translation": 0.16,
        "max_vertical_abs_rotation": 0.05,
        "min_action_pose_gap": 2,
        "max_action_pose_gap": 6,
        "min_sequence_pose_step": 1,
        "max_sequence_pose_step": 4,
        "min_degree_pose_gap": 2,
        "max_degree_pose_gap": 6,
        "sequence_min_total_magnitude": 0.42,
        "sequence_min_translation_final_distance": 0.55,
        "sequence_min_rotation_final_yaw": 26.0,
    },
}
ACTIVE_PROFILE = QUALITY_PROFILES["default"]


@dataclass
class FrameItem:
    frame_index: int
    pose: np.ndarray
    color_name: str
    pose_name: str


@dataclass
class SceneData:
    dataset: str
    scene_id: str
    zip_path: Path
    items: list[FrameItem]
    gray_cache: dict[int, np.ndarray] = field(default_factory=dict)
    quality_cache: dict[int, dict[str, float]] = field(default_factory=dict)
    floor_bias_cache: dict[int, float] = field(default_factory=dict)
    feature_cache: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = field(default_factory=dict)
    anchor_cache: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)

    @property
    def axis(self) -> dict[str, float]:
        return {"forward_sign": 1.0}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_rng(*parts: object) -> random.Random:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:12], 16))


def parse_pose_matrix(text: str) -> np.ndarray:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    mat = np.asarray([[float(x) for x in line.split()] for line in lines], dtype=np.float64)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"unexpected_pose_shape:{mat.shape}")
    return mat


def extract_frame_index(name: str) -> int:
    stem = Path(name).name.split(".")[0]
    return int(stem.split("-")[-1])


def load_scene(zip_path: Path) -> SceneData | None:
    items: list[FrameItem] = []
    try:
        with ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            color_names = sorted(name for name in names if name.endswith(".color.jpg"))
            for color_name in color_names:
                frame_index = extract_frame_index(color_name)
                pose_name = f"frame-{frame_index:06d}.pose.txt"
                if pose_name not in names:
                    continue
                pose = parse_pose_matrix(zf.read(pose_name).decode("utf-8", errors="ignore"))
                if not np.isfinite(pose).all():
                    continue
                items.append(
                    FrameItem(
                        frame_index=frame_index,
                        pose=pose,
                        color_name=color_name,
                        pose_name=pose_name,
                    )
                )
    except Exception:
        return None

    if len(items) < 4:
        return None
    items.sort(key=lambda item: item.frame_index)
    return SceneData(dataset="3rscan", scene_id=zip_path.parent.name, zip_path=zip_path, items=items)


def load_scenes(input_root: Path, limit_scenes: int = 0) -> list[SceneData]:
    scenes: list[SceneData] = []
    for zip_path in sorted(input_root.glob("*/sequence.zip")):
        scene = load_scene(zip_path)
        if scene is None:
            continue
        scenes.append(scene)
        if limit_scenes > 0 and len(scenes) >= limit_scenes:
            break
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


def motion_between(scene: SceneData, start_idx: int, end_idx: int) -> dict:
    start = scene.items[start_idx]
    end = scene.items[end_idx]
    forward_sign = scene.axis["forward_sign"]
    center_a = start.pose[:3, 3]
    center_b = end.pose[:3, 3]
    delta_world = center_b - center_a
    delta_local = start.pose[:3, :3].T @ delta_world
    right_m = float(delta_local[0])
    up_m = float(delta_local[1])
    forward_m = float(delta_local[2] * forward_sign)
    trans_m = float(np.linalg.norm(delta_world))
    yaw_a, pitch_a = heading_pitch_from_c2w(start.pose, forward_sign)
    yaw_b, pitch_b = heading_pitch_from_c2w(end.pose, forward_sign)
    yaw_delta = wrap_deg(yaw_b - yaw_a)
    pitch_delta = wrap_deg(pitch_b - pitch_a)
    return {
        "start_slot": start_idx,
        "end_slot": end_idx,
        "start_frame_index": start.frame_index,
        "end_frame_index": end.frame_index,
        "translation_m": trans_m,
        "yaw_deg": yaw_delta,
        "pitch_deg": pitch_delta,
        "local_delta": [right_m, up_m, forward_m],
        "pose_gap": end_idx - start_idx,
        "frame_gap": end.frame_index - start.frame_index,
    }


def classify_action(motion: dict) -> dict | None:
    trans_score = float(motion["translation_m"]) / 0.30
    rot_score = abs(float(motion["yaw_deg"])) / 15.0
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
        return float(motion["translation_m"])
    if family == "rotation":
        return abs(float(motion["yaw_deg"]))
    return float(motion["translation_m"]) + 0.015 * abs(float(motion["yaw_deg"])) + 0.01 * abs(float(motion["pitch_deg"]))


def dominant_axis_ratio(local_delta: list[float]) -> tuple[str | None, float, float]:
    if len(local_delta) < 3:
        return None, 0.0, 0.0
    tx = abs(float(local_delta[0]))
    tz = abs(float(local_delta[2]))
    primary = max(tx, tz)
    secondary = min(tx, tz)
    if primary < 1e-8:
        return None, primary, 0.0
    axis = "x" if tx >= tz else "z"
    return axis, primary, primary / max(secondary, 1e-8)


def translation_direction_key(local_delta: list[float]) -> str | None:
    axis, primary, ratio = dominant_axis_ratio(local_delta)
    if axis is None or primary < 0.08 or ratio < 1.05:
        return None
    if axis == "x":
        return "right" if float(local_delta[0]) > 0 else "left"
    return "forward" if float(local_delta[2]) > 0 else "backward"


def vertical_translation_ratio(local_delta: list[float]) -> float:
    horizontal = max(abs(float(local_delta[0])), abs(float(local_delta[2])), 1e-8)
    return abs(float(local_delta[1])) / horizontal


def decode_and_save(zip_path: Path, member_name: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with ZipFile(zip_path) as zf:
        data = zf.read(member_name)
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed_to_decode:{zip_path}:{member_name}")
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed_to_write:{output_path}")


def load_gray_small(scene: SceneData, frame_index: int, max_side: int = 256) -> np.ndarray:
    cached = scene.gray_cache.get(frame_index)
    if cached is not None:
        return cached
    index_to_member = {item.frame_index: item.color_name for item in scene.items}
    member_name = index_to_member.get(frame_index)
    if member_name is None:
        raise KeyError(f"missing_frame_index:{scene.scene_id}:{frame_index}")
    with ZipFile(scene.zip_path) as zf:
        data = zf.read(member_name)
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed_to_decode_gray:{scene.scene_id}:{frame_index}")
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    scene.gray_cache[frame_index] = image
    return image


def frame_quality(scene: SceneData, frame_index: int) -> dict[str, float]:
    cached = scene.quality_cache.get(frame_index)
    if cached is not None:
        return cached
    image = load_gray_small(scene, frame_index)
    quality = {
        "std": float(np.std(image.astype(np.float32)) / 255.0),
        "sharpness": float(cv2.Laplacian(image, cv2.CV_64F).var()),
        "brightness": float(np.mean(image)),
    }
    scene.quality_cache[frame_index] = quality
    return quality


def pair_diff(scene: SceneData, frame_a: int, frame_b: int) -> float:
    a = load_gray_small(scene, frame_a)
    b = load_gray_small(scene, frame_b)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def start_floor_bias(scene: SceneData, frame_index: int) -> float:
    cached = scene.floor_bias_cache.get(frame_index)
    if cached is not None:
        return cached
    image = load_gray_small(scene, frame_index)
    h, _w = image.shape
    corners = cv2.goodFeaturesToTrack(image, maxCorners=240, qualityLevel=0.01, minDistance=4)
    if corners is None or len(corners) == 0:
        bias = 0.0
    else:
        ys = [float(item[0][1]) / h for item in corners]
        top_ratio = sum(value < 0.35 for value in ys) / len(ys)
        bottom_ratio = sum(value > 0.55 for value in ys) / len(ys)
        bias = float(bottom_ratio - top_ratio)
    scene.floor_bias_cache[frame_index] = bias
    return bias


def orb_features(scene: SceneData, frame_index: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    cached = scene.feature_cache.get(frame_index)
    if cached is not None:
        return cached
    image = load_gray_small(scene, frame_index, max_side=320)
    orb = cv2.ORB_create(nfeatures=800, fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    cached = (keypoints or [], descriptors)
    scene.feature_cache[frame_index] = cached
    return cached


def anchor_metrics(scene: SceneData, frame_a: int, frame_b: int) -> dict[str, float]:
    key = (min(frame_a, frame_b), max(frame_a, frame_b))
    cached = scene.anchor_cache.get(key)
    if cached is not None:
        return cached
    kp_a, des_a = orb_features(scene, frame_a)
    kp_b, des_b = orb_features(scene, frame_b)
    if des_a is None or des_b is None or len(kp_a) < 12 or len(kp_b) < 12:
        metrics = {"match_count": 0.0, "match_ratio": 0.0}
        scene.anchor_cache[key] = metrics
        return metrics
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_a, des_b, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    match_count = float(len(good))
    match_ratio = match_count / max(1.0, min(float(len(kp_a)), float(len(kp_b))))
    metrics = {"match_count": match_count, "match_ratio": match_ratio}
    scene.anchor_cache[key] = metrics
    return metrics


def frames_pass_basic_quality(scene: SceneData, frame_indices: list[int]) -> bool:
    for frame_index in frame_indices:
        quality = frame_quality(scene, frame_index)
        if quality["std"] < ACTIVE_PROFILE["min_frame_std"]:
            return False
        if quality["sharpness"] < ACTIVE_PROFILE["min_frame_sharpness"]:
            return False
        if not (ACTIVE_PROFILE["min_brightness"] <= quality["brightness"] <= ACTIVE_PROFILE["max_brightness"]):
            return False
    if start_floor_bias(scene, frame_indices[0]) > ACTIVE_PROFILE["max_start_floor_bias"]:
        return False
    return True


def pure_motion_family(motion: dict) -> str | None:
    trans_score = float(motion["translation_m"]) / 0.30
    rot_score = abs(float(motion["yaw_deg"])) / 15.0
    if trans_score >= rot_score * 1.8:
        return "translation"
    if rot_score >= trans_score * 1.8:
        return "rotation"
    return None


def clean_translation_motion(motion: dict) -> bool:
    axis, primary, ratio = dominant_axis_ratio(motion["local_delta"])
    if axis is None:
        return False
    if abs(float(motion["local_delta"][1])) > ACTIVE_PROFILE["max_vertical_abs_translation"]:
        return False
    if vertical_translation_ratio(motion["local_delta"]) > ACTIVE_PROFILE["max_vertical_ratio_translation"]:
        return False
    return primary >= 0.16 and ratio >= 1.35


def clean_rotation_motion(motion: dict) -> bool:
    return (
        float(motion["translation_m"]) <= 0.08
        and abs(float(motion["local_delta"][1])) <= ACTIVE_PROFILE["max_vertical_abs_rotation"]
    )


def passes_anchor_gate(scene: SceneData, frame_a: int, frame_b: int) -> bool:
    metrics = anchor_metrics(scene, frame_a, frame_b)
    return (
        metrics["match_count"] >= ACTIVE_PROFILE["min_anchor_matches"]
        and metrics["match_ratio"] >= ACTIVE_PROFILE["min_anchor_ratio"]
    )


def copy_requested_frames(scene: SceneData, requests: list[tuple[int, Path]]) -> None:
    index_to_member = {item.frame_index: item.color_name for item in scene.items}
    for frame_index, output_path in requests:
        member_name = index_to_member.get(frame_index)
        if member_name is None:
            raise KeyError(f"missing_frame_index:{scene.scene_id}:{frame_index}")
        decode_and_save(scene.zip_path, member_name, output_path)


def build_action_inference(scene: SceneData, qa_id: str, sample_dir: Path) -> dict | None:
    candidates = []
    max_jump = min(int(ACTIVE_PROFILE["max_action_pose_gap"]), len(scene.items) - 1)
    for i in range(len(scene.items) - 1):
        for j in range(i + 1, min(len(scene.items), i + max_jump + 1)):
            motion = motion_between(scene, i, j)
            if motion["pose_gap"] < ACTIVE_PROFILE["min_action_pose_gap"]:
                continue
            action = classify_action(motion)
            if action:
                if action["family"] == "translation":
                    if not (
                        0.22 <= float(motion["translation_m"]) <= 0.75
                        and abs(float(motion["yaw_deg"])) <= 10.0
                        and abs(float(motion["pitch_deg"])) <= 8.0
                    ):
                        continue
                    if not clean_translation_motion(motion):
                        continue
                    if pure_motion_family(motion) != "translation":
                        continue
                else:
                    if not (
                        15.0 <= abs(float(motion["yaw_deg"])) <= 34.0
                        and float(motion["translation_m"]) <= 0.08
                        and abs(float(motion["pitch_deg"])) <= min(6.0, abs(float(motion["yaw_deg"])) * 0.25)
                    ):
                        continue
                    if not clean_rotation_motion(motion):
                        continue
                    if pure_motion_family(motion) != "rotation":
                        continue
                frame_indices = [scene.items[i].frame_index, scene.items[j].frame_index]
                if not frames_pass_basic_quality(scene, frame_indices):
                    continue
                if pair_diff(scene, frame_indices[0], frame_indices[1]) < ACTIVE_PROFILE["action_min_pair_diff"]:
                    continue
                if not passes_anchor_gate(scene, frame_indices[0], frame_indices[1]):
                    continue
                score = action["dominance"] + 0.15 * movement_magnitude(motion, action["family"]) + 0.02 * motion["pose_gap"]
                candidates.append((score, i, j, motion, action))
    if not candidates:
        return None
    _score, i, j, motion, action = max(candidates, key=lambda x: x[0])
    frame_a = sample_dir / "frame_A.png"
    frame_b = sample_dir / "frame_B.png"
    copy_requested_frames(scene, [(scene.items[i].frame_index, frame_a), (scene.items[j].frame_index, frame_b)])
    return {
        "qa_id": qa_id,
        "task_type": "action_inference",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "pose_source": "official_3rscan_sequence_pose",
        "question": "图A到图B之间，相机主要执行了什么动作？",
        "answer": action["label"],
        "options": ACTION_OPTIONS,
        "input": {
            "frame_paths": [str(frame_a), str(frame_b)],
            "frame_A": str(frame_a),
            "frame_B": str(frame_b),
        },
        "gt": {
            "slot_A": i,
            "slot_B": j,
            "action_family": action["family"],
            "translation_m": motion["translation_m"],
            "yaw_deg": motion["yaw_deg"],
            "pitch_deg": motion["pitch_deg"],
            "local_delta": motion["local_delta"],
            "motion": motion,
        },
    }


def build_sequence_sorting(scene: SceneData, qa_id: str, sample_dir: Path) -> dict | None:
    if len(scene.items) < 4:
        return None
    candidates = []
    for start in range(len(scene.items) - 3):
        end_limit = min(len(scene.items), start + 8)
        for s1 in range(start + 1, end_limit):
            for s2 in range(s1 + 1, end_limit):
                for s3 in range(s2 + 1, end_limit):
                    motions = [
                        motion_between(scene, start, s1),
                        motion_between(scene, s1, s2),
                        motion_between(scene, s2, s3),
                    ]
                    from_start = [
                        motion_between(scene, start, s1),
                        motion_between(scene, start, s2),
                        motion_between(scene, start, s3),
                    ]
                    if any(
                        motion["pose_gap"] < ACTIVE_PROFILE["min_sequence_pose_step"]
                        or motion["pose_gap"] > ACTIVE_PROFILE["max_sequence_pose_step"]
                        for motion in motions
                    ):
                        continue
                    families = [pure_motion_family(m) for m in motions]
                    if any(family is None for family in families):
                        continue
                    if len(set(families)) != 1:
                        continue
                    if families[0] == "translation":
                        if any(
                            not (
                                0.14 <= float(m["translation_m"]) <= 0.70
                                and abs(float(m["yaw_deg"])) <= 10.0
                                and abs(float(m["pitch_deg"])) <= 8.0
                            )
                            for m in motions
                        ):
                            continue
                        if any(not clean_translation_motion(m) for m in motions):
                            continue
                        directions = [translation_direction_key(m["local_delta"]) for m in motions]
                        if any(direction is None for direction in directions) or len(set(directions)) != 1:
                            continue
                        dists = [float(m["translation_m"]) for m in motions]
                        if not (dists[1] >= dists[0] * 0.9 and dists[2] >= dists[1] * 0.9):
                            continue
                        start_dists = [float(m["translation_m"]) for m in from_start]
                        if not (
                            start_dists[1] - start_dists[0] >= 0.10
                            and start_dists[2] - start_dists[1] >= 0.12
                            and start_dists[2] >= ACTIVE_PROFILE["sequence_min_translation_final_distance"]
                        ):
                            continue
                    else:
                        if any(
                            not (
                                10.0 <= abs(float(m["yaw_deg"])) <= 26.0
                                and float(m["translation_m"]) <= 0.08
                                and abs(float(m["pitch_deg"])) <= 5.0
                            )
                            for m in motions
                        ):
                            continue
                        if any(not clean_rotation_motion(m) for m in motions):
                            continue
                        yaws = [abs(float(m["yaw_deg"])) for m in motions]
                        if not (yaws[1] >= yaws[0] * 0.9 and yaws[2] >= yaws[1] * 0.9):
                            continue
                        yaw_signs = [1 if float(m["yaw_deg"]) > 0 else -1 for m in motions]
                        if len(set(yaw_signs)) != 1:
                            continue
                        start_yaws = [abs(float(m["yaw_deg"])) for m in from_start]
                        if not (
                            start_yaws[1] - start_yaws[0] >= 4.0
                            and start_yaws[2] - start_yaws[1] >= 4.0
                            and start_yaws[2] >= ACTIVE_PROFILE["sequence_min_rotation_final_yaw"]
                        ):
                            continue
                    frame_indices = [scene.items[start].frame_index, scene.items[s1].frame_index, scene.items[s2].frame_index, scene.items[s3].frame_index]
                    if not frames_pass_basic_quality(scene, frame_indices):
                        continue
                    adj_diffs = [pair_diff(scene, frame_indices[k], frame_indices[k + 1]) for k in range(3)]
                    pair_diffs = [pair_diff(scene, frame_indices[i], frame_indices[j]) for i in range(4) for j in range(i + 1, 4)]
                    if min(adj_diffs) < ACTIVE_PROFILE["sequence_min_adj_diff"] or (sum(pair_diffs) / len(pair_diffs)) < ACTIVE_PROFILE["sequence_min_mean_pair_diff"]:
                        continue
                    if not all(
                        passes_anchor_gate(scene, frame_indices[k], frame_indices[k + 1])
                        for k in range(3)
                    ):
                        continue
                    total = sum(movement_magnitude(m) for m in motions)
                    if total < ACTIVE_PROFILE["sequence_min_total_magnitude"]:
                        continue
                    spacing = (scene.items[s3].frame_index - scene.items[start].frame_index) / 30.0
                    candidates.append((total + spacing, [start, s1, s2, s3], motions))
    if not candidates:
        return None
    _score, slots, motions = max(candidates, key=lambda x: x[0])
    labels = ["A", "B", "C"]
    labeled = list(zip(labels, slots[1:]))
    stable_rng("sequence", scene.scene_id, qa_id).shuffle(labeled)
    correct = [label for label, slot in sorted(labeled, key=lambda item: item[1])]
    start_path = sample_dir / "frame_start.png"
    copy_requested_frames(scene, [(scene.items[slots[0]].frame_index, start_path)])
    frame_paths = [str(start_path)]
    candidate_frames: dict[str, str] = {}
    frame_requests = []
    for label, slot in labeled:
        out = sample_dir / f"frame_{label}.png"
        frame_requests.append((scene.items[slot].frame_index, out))
        frame_paths.append(str(out))
        candidate_frames[label] = str(out)
    copy_requested_frames(scene, frame_requests)
    return {
        "qa_id": qa_id,
        "task_type": "movement_sequence_sorting",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "pose_source": "official_3rscan_sequence_pose",
        "question": "已知第一张图是这段真实视频中的起始帧。请将其余三张图按时间先后排序。",
        "answer": " -> ".join(correct),
        "input": {
            "frame_paths": frame_paths,
            "first_frame": str(start_path),
            "candidate_frames": candidate_frames,
        },
        "gt": {
            "start_slot": slots[0],
            "slot_order": {label: slot for label, slot in labeled},
            "correct_order": correct,
            "motions": motions,
        },
    }


def build_degree_comparison(scene: SceneData, qa_id: str, sample_dir: Path) -> dict | None:
    candidates = []
    for start in range(len(scene.items) - 2):
        end_limit = min(len(scene.items), start + int(ACTIVE_PROFILE["max_degree_pose_gap"]) + 1)
        for a in range(start + 1, end_limit):
            for b in range(a + 1, end_limit):
                ma = motion_between(scene, start, a)
                mb = motion_between(scene, start, b)
                if (
                    ma["pose_gap"] < ACTIVE_PROFILE["min_degree_pose_gap"]
                    or mb["pose_gap"] < ACTIVE_PROFILE["min_degree_pose_gap"]
                ):
                    continue
                fa = pure_motion_family(ma)
                fb = pure_motion_family(mb)
                if fa is None or fb is None:
                    continue
                if fa != fb:
                    continue
                mag_a = movement_magnitude(ma, fa)
                mag_b = movement_magnitude(mb, fb)
                gap = abs(mag_a - mag_b)
                min_gap = 0.22 if fa == "translation" else 10.0
                if gap < min_gap:
                    continue
                if fa == "translation":
                    dir_a = translation_direction_key(ma["local_delta"])
                    dir_b = translation_direction_key(mb["local_delta"])
                    if dir_a is None or dir_b is None or dir_a != dir_b:
                        continue
                    if not clean_translation_motion(ma) or not clean_translation_motion(mb):
                        continue
                    if not (
                        0.20 <= float(ma["translation_m"]) <= 0.90
                        and 0.30 <= float(mb["translation_m"]) <= 1.20
                        and 0.22 <= gap <= 0.80
                        and abs(float(ma["yaw_deg"])) <= 10.0
                        and abs(float(mb["yaw_deg"])) <= 10.0
                        and abs(float(ma["pitch_deg"])) <= 8.0
                        and abs(float(mb["pitch_deg"])) <= 8.0
                        ):
                        continue
                else:
                    if not clean_rotation_motion(ma) or not clean_rotation_motion(mb):
                        continue
                    if not (
                        12.0 <= abs(float(ma["yaw_deg"])) <= 28.0
                        and 18.0 <= abs(float(mb["yaw_deg"])) <= 38.0
                        and 10.0 <= gap <= 22.0
                        and float(ma["translation_m"]) <= 0.08
                        and float(mb["translation_m"]) <= 0.08
                        and abs(float(ma["pitch_deg"])) <= 5.0
                        and abs(float(mb["pitch_deg"])) <= 5.0
                    ):
                        continue
                frame_indices = [scene.items[start].frame_index, scene.items[a].frame_index, scene.items[b].frame_index]
                if not frames_pass_basic_quality(scene, frame_indices):
                    continue
                pair_diffs = [
                    pair_diff(scene, frame_indices[0], frame_indices[1]),
                    pair_diff(scene, frame_indices[0], frame_indices[2]),
                    pair_diff(scene, frame_indices[1], frame_indices[2]),
                ]
                if min(pair_diffs) < ACTIVE_PROFILE["degree_min_pair_diff"]:
                    continue
                if not passes_anchor_gate(scene, frame_indices[0], frame_indices[1]):
                    continue
                if not passes_anchor_gate(scene, frame_indices[0], frame_indices[2]):
                    continue
                score = gap + 0.05 * (a - start + b - start)
                candidates.append((score, start, a, b, fa, ma, mb))
    if not candidates:
        return None
    _score, start, a, b, family, ma, mb = max(candidates, key=lambda x: x[0])
    pair = [("A", a, ma), ("B", b, mb)]
    if stable_rng("degree", scene.scene_id, qa_id).randrange(2):
        pair = [("A", b, mb), ("B", a, ma)]
    start_path = sample_dir / "frame_start.png"
    frame_a = sample_dir / "frame_A.png"
    frame_b = sample_dir / "frame_B.png"
    copy_requested_frames(
        scene,
        [
            (scene.items[start].frame_index, start_path),
            (scene.items[pair[0][1]].frame_index, frame_a),
            (scene.items[pair[1][1]].frame_index, frame_b),
        ],
    )
    larger = "A" if movement_magnitude(pair[0][2], family) > movement_magnitude(pair[1][2], family) else "B"
    unit = "米" if family == "translation" else "度"
    return {
        "qa_id": qa_id,
        "task_type": "movement_degree_comparison",
        "dataset": scene.dataset,
        "scene": scene.scene_id,
        "pose_source": "official_3rscan_sequence_pose",
        "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
        "answer": f"候选图{larger}更大。",
        "input": {
            "frame_paths": [str(start_path), str(frame_a), str(frame_b)],
            "start_frame": str(start_path),
            "frame_A": str(frame_a),
            "frame_B": str(frame_b),
        },
        "gt": {
            "comparison_family": family,
            "larger_clip": larger,
            "start_slot": start,
            "frame_A": {
                "slot_end": pair[0][1],
                "magnitude": movement_magnitude(pair[0][2], family),
                "unit": unit,
                "yaw_deg": pair[0][2]["yaw_deg"],
                "pitch_deg": pair[0][2]["pitch_deg"],
                "local_delta": pair[0][2]["local_delta"],
                "motion": pair[0][2],
            },
            "frame_B": {
                "slot_end": pair[1][1],
                "magnitude": movement_magnitude(pair[1][2], family),
                "unit": unit,
                "yaw_deg": pair[1][2]["yaw_deg"],
                "pitch_deg": pair[1][2]["pitch_deg"],
                "local_delta": pair[1][2]["local_delta"],
                "motion": pair[1][2],
            },
        },
    }


TASK_BUILDERS = [
    ("action_inference", build_action_inference),
    ("movement_sequence_sorting", build_sequence_sorting),
    ("movement_degree_comparison", build_degree_comparison),
]


def save_task_rows(output_root: Path, task_name: str, rows: list[dict]) -> None:
    task_dir = output_root / task_name
    write_json(task_dir / "qa_data.json", rows)
    write_jsonl(task_dir / "qa_data.jsonl", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build REAL-style 3RScan motion QA from sequence.zip.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--per-task-limit", type=int, default=DEFAULT_PER_TASK_LIMIT)
    parser.add_argument("--quality-profile", choices=sorted(QUALITY_PROFILES.keys()), default="default")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global ACTIVE_PROFILE
    ACTIVE_PROFILE = dict(QUALITY_PROFILES[args.quality_profile])
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    scenes = load_scenes(args.input_root, limit_scenes=args.limit_scenes)
    all_rows: list[dict] = []
    task_counts = {task: 0 for task, _builder in TASK_BUILDERS}
    scene_stats = {"loaded_scene_count": len(scenes), "used_scene_count": 0}

    for scene in scenes:
        if args.per_task_limit > 0 and all(task_counts[task] >= args.per_task_limit for task, _builder in TASK_BUILDERS):
            break
        used = False
        for task_name, builder in TASK_BUILDERS:
            if args.per_task_limit > 0 and task_counts[task_name] >= args.per_task_limit:
                continue
            qa_id = f"{task_name}_{task_counts[task_name] + 1:06d}"
            sample_dir = args.output_root / task_name / qa_id
            try:
                row = builder(scene, qa_id, sample_dir)
            except Exception:
                row = None
            if row is None:
                continue
            all_rows.append(row)
            task_counts[task_name] += 1
            used = True
        if used:
            scene_stats["used_scene_count"] += 1
        if used and scene_stats["used_scene_count"] % 50 == 0:
            print(
                json.dumps(
                    {
                        "progress_scenes_used": scene_stats["used_scene_count"],
                        "loaded_scene_count": scene_stats["loaded_scene_count"],
                        "task_counts": task_counts,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for task_name, _builder in TASK_BUILDERS:
        rows = [row for row in all_rows if row["task_type"] == task_name]
        save_task_rows(args.output_root, task_name, rows)

    qa_data_all = {"qa_count": len(all_rows), "data": all_rows}
    summary = {
        "root": str(args.output_root),
        "input_root": str(args.input_root),
        "quality_profile": args.quality_profile,
        "quality_thresholds": ACTIVE_PROFILE,
        "task_counts": task_counts,
        "kept_total": len(all_rows),
        "scene_stats": scene_stats,
    }
    write_json(args.output_root / "qa_data_all.json", qa_data_all)
    write_jsonl(args.output_root / "qa_data_all.jsonl", all_rows)
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
