from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pose_rule_probe import default_official_path, load_da3_bundle, parse_arkitscenes_traj, parse_scannet_pose_dir, sample_sequence


OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/pose_gt_samples")
POSE_CACHE_ROOT = Path("/path/to/workspace/Depth-Anything-3/workspace/vsibench_pose_cache")
SCENE_COUNT_PER_DATASET = int(os.environ.get("REAL_SCENE_COUNT_PER_DATASET", "50"))
USE_OFFICIAL_POSE = os.environ.get("REAL_USE_OFFICIAL_POSE", "0") == "1"

ACTION_OPTIONS = [
    "向前移动",
    "向后移动",
    "向左移动",
    "向右移动",
    "向左转动",
    "向右转动",
]
FAMILY_OPTIONS = ["平移为主", "转动为主", "平移和转动都明显"]


def sample_evenly(scene_ids: Sequence[str], limit: int) -> List[str]:
    ids = list(scene_ids)
    if limit <= 0 or len(ids) <= limit:
        return ids
    positions = np.linspace(0, len(ids) - 1, num=limit, dtype=np.int32)
    return [ids[int(pos)] for pos in positions]


def available_scannet_scenes(limit: int) -> List[str]:
    video_root = Path("/path/to/workspace/DATA/VSI-590K/scannet")
    videos = {path.stem for path in video_root.glob("*.mp4")}
    if USE_OFFICIAL_POSE:
        pose_root = Path("/path/to/workspace/DATA/REAL_OFFICIAL/scannet_pose_intrinsic_cache")
        poses = {path.name for path in pose_root.iterdir() if path.is_dir() and (path / "pose").is_dir()}
    else:
        pose_root = POSE_CACHE_ROOT / "scannet"
        poses = {path.name for path in pose_root.iterdir() if path.is_dir()}
    return sample_evenly(sorted(videos & poses), limit)


def available_arkitscenes_scenes(limit: int) -> List[str]:
    video_root = Path("/path/to/workspace/DATA/VSI-590K/arkitscenes") if USE_OFFICIAL_POSE else Path("/path/to/workspace/.cache/huggingface/vsibench/arkitscenes")
    pose_root = None if USE_OFFICIAL_POSE else POSE_CACHE_ROOT / "arkitscenes"
    official_root = Path("/path/to/workspace/DATA/REAL_OFFICIAL/arkitscenes/raw") if USE_OFFICIAL_POSE else Path("/path/to/workspace/datasets/arkitscenes_gt/raw")
    videos = {path.stem for path in video_root.glob("*.mp4")}
    poses = videos if pose_root is None else {path.name for path in pose_root.iterdir() if path.is_dir()}
    official = set()
    for split in ("Training", "Validation"):
        split_root = official_root / split
        if not split_root.exists():
            continue
        for scene_dir in split_root.iterdir():
            if scene_dir.is_dir() and (scene_dir / "lowres_wide.traj").exists():
                official.add(scene_dir.name)
    return sample_evenly(sorted(videos & poses & official), limit)


def build_scene_specs() -> List[dict]:
    specs: List[dict] = []
    for scene in available_scannet_scenes(SCENE_COUNT_PER_DATASET):
        specs.append({"dataset": "scannet", "scene": scene, "pose_source": "official_scannet_pose" if USE_OFFICIAL_POSE else "da3_reference"})
    for scene in available_arkitscenes_scenes(SCENE_COUNT_PER_DATASET):
        specs.append({"dataset": "arkitscenes", "scene": scene, "pose_source": "official_traj"})
    return specs


SCENE_SPECS = build_scene_specs()


@dataclass
class PoseScene:
    dataset: str
    scene: str
    pose_source: str
    video_path: Path
    sampled_frames_dir: Optional[Path]
    extr_w2c: np.ndarray
    frame_count: int
    frame_indices_32: List[int]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_sample_dir(sample_dir: Path) -> None:
    if not sample_dir.exists():
        return
    for path in sample_dir.iterdir():
        if path.is_file():
            path.unlink()


def resolve_video_path(dataset: str, scene: str) -> Path:
    candidates = []
    if dataset == "scannet":
        candidates = [
            Path(f"/path/to/workspace/.cache/huggingface/vsibench/scannet/{scene}.mp4"),
            Path(f"/path/to/workspace/DATA/VSI-590K/scannet/{scene}.mp4"),
        ]
    elif dataset == "scannetpp":
        candidates = [
            Path(f"/path/to/workspace/.cache/huggingface/vsibench/scannetpp/{scene}.mp4"),
            Path(f"/path/to/workspace/DATA/VSI-590K/scannetppv2/{scene}.mp4"),
        ]
    elif dataset == "arkitscenes":
        candidates = [
            Path(f"/path/to/workspace/.cache/huggingface/vsibench/arkitscenes/{scene}.mp4"),
            Path(f"/path/to/workspace/DATA/VSI-590K/arkitscenes/{scene}.mp4"),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Video not found for {dataset}/{scene}")


def load_video_info(video_path: Path) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return frame_count, fps
    finally:
        cap.release()


def read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def save_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    frame = read_frame(video_path, frame_index)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write frame: {output_path}")


def uniform_frame_indices(frame_count: int, total_slots: int = 32) -> List[int]:
    if frame_count <= 1:
        return [0] * total_slots
    return [
        min(frame_count - 1, max(0, int(round(i * (frame_count - 1) / (total_slots - 1)))))
        for i in range(total_slots)
    ]


def to_homogeneous(ext: np.ndarray) -> np.ndarray:
    if ext.shape[-2:] == (4, 4):
        return ext.astype(np.float64)
    if ext.shape[-2:] == (3, 4):
        out = np.repeat(np.eye(4, dtype=np.float64)[None], ext.shape[0], axis=0)
        out[:, :3, :] = ext
        return out
    raise ValueError(f"Unexpected extrinsic shape: {ext.shape}")


def camera_centers_from_w2c(ext_w2c: np.ndarray) -> np.ndarray:
    r = ext_w2c[:, :3, :3]
    t = ext_w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)


def heading_elevation_series(ext_w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c2w = np.linalg.inv(ext_w2c)
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    headings = []
    elevations = []
    for mat in c2w:
        forward = mat[:3, :3] @ ez
        headings.append(math.atan2(float(forward[0]), float(forward[2]) if abs(float(forward[2])) > 1e-8 else 1e-8))
        elevations.append(math.asin(float(np.clip(np.dot(forward, up), -1.0, 1.0))))
    return np.degrees(np.unwrap(np.asarray(headings))), np.degrees(np.unwrap(np.asarray(elevations)))


def load_pose_scene(dataset: str, scene: str, pose_source: str) -> PoseScene:
    video_path = resolve_video_path(dataset, scene)
    frame_count, _fps = load_video_info(video_path)
    da3_ext = None
    sampled_frames_dir = None
    target_slots = 32
    try:
        bundle, da3_ext = load_da3_bundle(dataset, scene)
        target_slots = len(da3_ext)
        scene_cache_dir = POSE_CACHE_ROOT / dataset / scene
        sampled_frames_dir = scene_cache_dir / f"frames{len(da3_ext)}"
        if not sampled_frames_dir.exists():
            sampled_frames_dir = scene_cache_dir / "frames32"
        if not sampled_frames_dir.exists():
            sampled_frames_dir = None
    except FileNotFoundError:
        bundle = None

    if pose_source == "official_traj":
        official_path = default_official_path(dataset, scene)
        if official_path is None or not official_path.exists():
            raise FileNotFoundError(f"Official traj not found for {dataset}/{scene}")
        official = parse_arkitscenes_traj(official_path)
        target_slots = min(target_slots, len(official))
        ext_w2c = sample_sequence(official, target_slots)
    elif pose_source == "official_scannet_pose":
        official_path = default_official_path(dataset, scene)
        if official_path is None or not official_path.exists():
            raise FileNotFoundError(f"Official ScanNet pose dir not found for {dataset}/{scene}")
        official = parse_scannet_pose_dir(official_path)
        target_slots = min(target_slots, len(official))
        ext_w2c = sample_sequence(official, target_slots)
    else:
        if da3_ext is None:
            raise FileNotFoundError(f"DA3 pose bundle not found for {dataset}/{scene}")
        ext_w2c = da3_ext
    if da3_ext is None or len(ext_w2c) != len(da3_ext):
        sampled_frames_dir = None
    return PoseScene(
        dataset=dataset,
        scene=scene,
        pose_source=pose_source,
        video_path=video_path,
        sampled_frames_dir=sampled_frames_dir,
        extr_w2c=to_homogeneous(ext_w2c),
        frame_count=frame_count,
        frame_indices_32=uniform_frame_indices(frame_count, total_slots=len(ext_w2c)),
    )


def classify_step(ext_w2c: np.ndarray, step_idx: int) -> Optional[dict]:
    centers = camera_centers_from_w2c(ext_w2c)
    heading, elevation = heading_elevation_series(ext_w2c)
    r = ext_w2c[:, :3, :3]
    delta_world = centers[step_idx + 1] - centers[step_idx]
    delta_local = r[step_idx] @ delta_world
    tx = float(delta_local[0])
    tz = float(delta_local[2])
    trans_mag = float(np.linalg.norm(delta_world))
    dyaw = float(heading[step_idx + 1] - heading[step_idx])
    dpitch = float(elevation[step_idx + 1] - elevation[step_idx])
    ry = -dyaw

    if 0.20 <= trans_mag <= 0.65 and abs(ry) <= 10.0 and abs(dpitch) <= 8.0:
        if abs(tz) >= max(0.16, abs(tx) * 1.45):
            label = "向前移动" if tz > 0 else "向后移动"
            target = 0.36
        elif abs(tx) >= max(0.16, abs(tz) * 1.45):
            label = "向右移动" if tx > 0 else "向左移动"
            target = 0.28
        else:
            return None
        score = max(
            0.0,
            1.8
            - abs(trans_mag - target) / max(target, 1e-6)
            - 0.04 * abs(ry)
            - 0.05 * abs(dpitch),
        )
        family = "translation"
    elif 15.0 <= abs(ry) <= 30.0 and trans_mag <= 0.14 and abs(dpitch) <= 6.0:
        label = "向左转动" if ry > 0 else "向右转动"
        score = max(
            0.0,
            1.8
            - abs(abs(ry) - 22.0) / 12.0
            - trans_mag / 0.14
            - abs(dpitch) / 6.0,
        )
        family = "rotation"
    else:
        return None
    return {
        "label": label,
        "family": family,
        "score": float(score),
        "translation_m": trans_mag,
        "yaw_deg": ry,
        "pitch_deg": dpitch,
        "local_delta": [tx, float(delta_local[1]), tz],
    }


def choose_action_step(scene: PoseScene) -> Tuple[int, dict]:
    candidates = []
    for i in range(len(scene.extr_w2c) - 1):
        info = classify_step(scene.extr_w2c, i)
        if info is not None:
            candidates.append((i, info))
    if not candidates:
        raise RuntimeError(f"No action candidate found for {scene.dataset}/{scene.scene}")
    candidates.sort(key=lambda item: item[1]["score"], reverse=True)
    return candidates[0]


def choose_sequence_slots(scene: PoseScene) -> List[int]:
    best = None
    fallback = None
    for start in range(0, len(scene.extr_w2c) - 3):
        slots = [start, start + 1, start + 2, start + 3]
        infos = [segment_family(scene.extr_w2c, slots[i], slots[i + 1]) for i in range(3)]
        smooth_enough = True
        window_score = 0.0
        fallback_score = 0.0
        for info in infos:
            if info["family"] == "translation":
                if info["magnitude"] > 1.10 or abs(info["yaw_deg"]) > 55.0 or abs(info["pitch_deg"]) > 55.0:
                    smooth_enough = False
                    break
                fallback_score += min(info["magnitude"] / 0.45, 1.2)
                if 0.18 <= info["magnitude"] <= 0.75 and abs(info["yaw_deg"]) <= 30.0 and abs(info["pitch_deg"]) <= 30.0:
                    window_score += 1.2 - abs(info["magnitude"] - 0.38) / 0.38
            else:
                if info["magnitude"] > 60.0 or abs(info["pitch_deg"]) > 55.0:
                    smooth_enough = False
                    break
                fallback_score += min(info["magnitude"] / 22.0, 1.2)
                if 8.0 <= info["magnitude"] <= 38.0 and abs(info["pitch_deg"]) <= 30.0:
                    window_score += 1.2 - abs(info["magnitude"] - 20.0) / 20.0
        if not smooth_enough:
            continue
        if fallback is None or fallback_score > fallback[0]:
            fallback = (fallback_score, slots)
        if window_score < 1.6:
            continue
        if best is None or window_score > best[0]:
            best = (window_score, slots)
    if best is None:
        if fallback is not None:
            return fallback[1]
        return [0, 1, 2, 3]
    return best[1]


def segment_family(ext_w2c: np.ndarray, start: int, end: int) -> dict:
    centers = camera_centers_from_w2c(ext_w2c)
    heading, elevation = heading_elevation_series(ext_w2c)
    delta_world = centers[end] - centers[start]
    r = ext_w2c[start, :3, :3]
    delta_local = r @ delta_world
    trans_mag = float(np.linalg.norm(delta_world))
    yaw_delta = float(-(heading[end] - heading[start]))
    pitch_delta = float(elevation[end] - elevation[start])
    if trans_mag >= max(0.25, abs(yaw_delta) * 0.01):
        return {
            "family": "translation",
            "magnitude": trans_mag,
            "local_delta": [float(delta_local[0]), float(delta_local[1]), float(delta_local[2])],
            "yaw_deg": yaw_delta,
            "pitch_deg": pitch_delta,
        }
    return {
        "family": "rotation",
        "magnitude": abs(yaw_delta),
        "local_delta": [float(delta_local[0]), float(delta_local[1]), float(delta_local[2])],
        "yaw_deg": yaw_delta,
        "pitch_deg": pitch_delta,
    }


def segment_motion(scene: PoseScene, start: int, end: int) -> dict:
    info = segment_family(scene.extr_w2c, start, end)
    centers = camera_centers_from_w2c(scene.extr_w2c)
    delta_world = centers[end] - centers[start]
    return {
        "start_slot": start + 1,
        "end_slot": end + 1,
        "start_frame_index": scene.frame_indices_32[start],
        "end_frame_index": scene.frame_indices_32[end],
        "translation_m": float(np.linalg.norm(delta_world)),
        "yaw_deg": info["yaw_deg"],
        "pitch_deg": info["pitch_deg"],
        "local_delta": info["local_delta"],
        "pose_gap": end - start,
        "frame_gap": scene.frame_indices_32[end] - scene.frame_indices_32[start],
        "family": info["family"],
        "magnitude": info["magnitude"],
    }


def movement_metric(motion: dict) -> float:
    return (
        float(motion["translation_m"])
        + 0.02 * abs(float(motion["yaw_deg"]))
        + 0.01 * abs(float(motion["pitch_deg"]))
    )


def choose_degree_triplet(scene: PoseScene) -> Tuple[int, int, int, str]:
    candidates = []
    for start in range(len(scene.extr_w2c) - 2):
        for end_a in range(start + 1, min(len(scene.extr_w2c), start + 3)):
            for end_b in range(end_a + 1, min(len(scene.extr_w2c), start + 4)):
                info_a = segment_family(scene.extr_w2c, start, end_a)
                info_b = segment_family(scene.extr_w2c, start, end_b)
                if info_a["family"] != info_b["family"]:
                    continue
                family = info_a["family"]
                if family == "translation":
                    if not (
                        0.18 <= info_a["magnitude"] <= 0.90
                        and 0.25 <= info_b["magnitude"] <= 1.25
                        and abs(info_a["yaw_deg"]) <= 35.0
                        and abs(info_b["yaw_deg"]) <= 45.0
                        and abs(info_a["pitch_deg"]) <= 35.0
                        and abs(info_b["pitch_deg"]) <= 40.0
                    ):
                        continue
                    gap = abs(info_a["magnitude"] - info_b["magnitude"])
                    if gap < 0.10:
                        continue
                    penalty = (
                        abs(info_a["magnitude"] - 0.40)
                        + abs(info_b["magnitude"] - 0.65)
                        + 0.01 * (abs(info_a["yaw_deg"]) + abs(info_b["yaw_deg"]))
                        + 0.01 * (abs(info_a["pitch_deg"]) + abs(info_b["pitch_deg"]))
                    )
                else:
                    if not (
                        8.0 <= info_a["magnitude"] <= 35.0
                        and 15.0 <= info_b["magnitude"] <= 55.0
                        and abs(info_a["pitch_deg"]) <= 30.0
                        and abs(info_b["pitch_deg"]) <= 30.0
                    ):
                        continue
                    gap = abs(info_a["magnitude"] - info_b["magnitude"])
                    if gap < 8.0:
                        continue
                    penalty = (
                        abs(info_a["magnitude"] - 20.0) * 0.03
                        + abs(info_b["magnitude"] - 34.0) * 0.03
                        + 0.01 * (abs(info_a["pitch_deg"]) + abs(info_b["pitch_deg"]))
                    )
                candidates.append((gap - penalty, start, end_a, end_b, family))
    if candidates:
        _, start, end_a, end_b, family = max(candidates, key=lambda item: item[0])
        return start, end_a, end_b, family

    relaxed = []
    for start in range(len(scene.extr_w2c) - 2):
        for end_a in range(start + 1, min(len(scene.extr_w2c), start + 3)):
            for end_b in range(end_a + 1, min(len(scene.extr_w2c), start + 4)):
                info_a = segment_family(scene.extr_w2c, start, end_a)
                info_b = segment_family(scene.extr_w2c, start, end_b)
                if info_a["family"] != info_b["family"]:
                    continue
                family = info_a["family"]
                gap = abs(info_a["magnitude"] - info_b["magnitude"])
                if family == "translation":
                    if not (
                        0.20 <= info_a["magnitude"] <= 1.10
                        and 0.20 <= info_b["magnitude"] <= 1.10
                        and abs(info_a["yaw_deg"]) <= 45.0
                        and abs(info_b["yaw_deg"]) <= 45.0
                        and abs(info_a["pitch_deg"]) <= 40.0
                        and abs(info_b["pitch_deg"]) <= 40.0
                        and gap >= 0.12
                    ):
                        continue
                    penalty = (
                        abs(info_a["magnitude"] - 0.45)
                        + abs(info_b["magnitude"] - 0.75)
                        + 0.01 * (abs(info_a["yaw_deg"]) + abs(info_b["yaw_deg"]))
                    )
                else:
                    if not (
                        15.0 <= info_a["magnitude"] <= 80.0
                        and 20.0 <= info_b["magnitude"] <= 80.0
                        and abs(info_a["pitch_deg"]) <= 45.0
                        and abs(info_b["pitch_deg"]) <= 45.0
                        and gap >= 10.0
                    ):
                        continue
                    penalty = (
                        abs(info_a["magnitude"] - 30.0) * 0.03
                        + abs(info_b["magnitude"] - 55.0) * 0.03
                        + 0.01 * (abs(info_a["pitch_deg"]) + abs(info_b["pitch_deg"]))
                    )
                relaxed.append((gap - penalty, start, end_a, end_b, family))
    if relaxed:
        _, start, end_a, end_b, family = max(relaxed, key=lambda item: item[0])
        return start, end_a, end_b, family

    fallback = []
    for start in range(len(scene.extr_w2c) - 2):
        for end_a in range(start + 1, min(len(scene.extr_w2c), start + 3)):
            for end_b in range(end_a + 1, min(len(scene.extr_w2c), start + 4)):
                info_a = segment_family(scene.extr_w2c, start, end_a)
                info_b = segment_family(scene.extr_w2c, start, end_b)
                if info_a["family"] != info_b["family"]:
                    continue
                gap = abs(info_a["magnitude"] - info_b["magnitude"])
                fallback.append((gap, start, end_a, end_b, info_a["family"]))
    if fallback:
        _, start, end_a, end_b, family = max(fallback, key=lambda item: item[0])
        return start, end_a, end_b, family
    raise RuntimeError(f"No degree comparison triplet found for {scene.dataset}/{scene.scene}")


def choose_motion_family_pair(scene: PoseScene) -> Tuple[int, int, str, dict]:
    candidates = []
    for start in range(len(scene.extr_w2c) - 1):
        for end in range(start + 1, min(len(scene.extr_w2c), start + 5)):
            motion = segment_motion(scene, start, end)
            trans_score = motion["translation_m"] / 0.12
            rot_score = abs(motion["yaw_deg"]) / 6.0
            label = None
            if trans_score >= rot_score * 1.35 and 0.22 <= motion["translation_m"] <= 1.0 and abs(motion["pitch_deg"]) <= 28.0:
                label = "平移为主"
                score = trans_score - 0.4 * rot_score
            elif rot_score >= trans_score * 1.35 and 12.0 <= abs(motion["yaw_deg"]) <= 55.0 and abs(motion["pitch_deg"]) <= 25.0:
                label = "转动为主"
                score = rot_score - 0.4 * trans_score
            elif (
                trans_score >= 1.0
                and rot_score >= 1.0
                and 0.18 <= motion["translation_m"] <= 0.90
                and 8.0 <= abs(motion["yaw_deg"]) <= 40.0
                and abs(motion["pitch_deg"]) <= 25.0
            ):
                label = "平移和转动都明显"
                score = min(trans_score, rot_score)
            else:
                continue
            candidates.append((score, start, end, label, motion))
    if not candidates:
        raise RuntimeError(f"No motion family pair found for {scene.dataset}/{scene.scene}")
    return max(candidates, key=lambda item: item[0])[1:]


def choose_distance_triplet(scene: PoseScene) -> Tuple[int, int, int, dict, dict]:
    candidates = []
    for start in range(len(scene.extr_w2c) - 2):
        for end_a in range(start + 1, min(len(scene.extr_w2c), start + 4)):
            for end_b in range(end_a + 1, min(len(scene.extr_w2c), start + 5)):
                motion_a = segment_motion(scene, start, end_a)
                motion_b = segment_motion(scene, start, end_b)
                mag_a = movement_metric(motion_a)
                mag_b = movement_metric(motion_b)
                diff = abs(mag_a - mag_b)
                if diff < 0.12:
                    continue
                penalty = 0.006 * (abs(motion_a["pitch_deg"]) + abs(motion_b["pitch_deg"]))
                score = diff + 0.15 * max(mag_a, mag_b) - penalty
                candidates.append((score, start, end_a, end_b, motion_a, motion_b))
    if not candidates:
        raise RuntimeError(f"No distance triplet found for {scene.dataset}/{scene.scene}")
    return max(candidates, key=lambda item: item[0])[1:]


def save_slot_frame(scene: PoseScene, slot: int, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if scene.sampled_frames_dir is not None:
        sampled_frame = scene.sampled_frames_dir / f"frame_{slot:03d}.png"
        if sampled_frame.exists():
            shutil.copy2(sampled_frame, out_path)
            return
    frame_idx = scene.frame_indices_32[slot]
    save_frame(scene.video_path, frame_idx, out_path)


def generate_action_samples(scenes: Sequence[PoseScene]) -> List[dict]:
    rows = []
    task_root = OUTPUT_ROOT / "action_inference"
    for idx, scene in enumerate(scenes, start=1):
        try:
            step_idx, step = choose_action_step(scene)
        except RuntimeError as exc:
            print(f"[WARN] skip action_inference for {scene.dataset}/{scene.scene}: {exc}")
            continue
        qa_id = f"action_inference_{idx:03d}"
        sample_dir = task_root / qa_id
        frame_a = sample_dir / "frame_A.png"
        frame_b = sample_dir / "frame_B.png"
        save_slot_frame(scene, step_idx, frame_a)
        save_slot_frame(scene, step_idx + 1, frame_b)
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "action_inference",
                "dataset": scene.dataset,
                "scene": scene.scene,
                "pose_source": scene.pose_source,
                "question": "图A到图B之间，相机主要执行了什么动作？",
                "answer": step["label"],
                "options": ACTION_OPTIONS,
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b)],
                    "frame_A": str(frame_a),
                    "frame_B": str(frame_b),
                },
                "gt": {
                    "slot_A": step_idx + 1,
                    "slot_B": step_idx + 2,
                    "action_family": step["family"],
                    "translation_m": step["translation_m"],
                    "yaw_deg": step["yaw_deg"],
                    "pitch_deg": step["pitch_deg"],
                    "local_delta": step["local_delta"],
                    "motion": {
                        "start_slot": step_idx + 1,
                        "end_slot": step_idx + 2,
                        "start_frame_index": scene.frame_indices_32[step_idx],
                        "end_frame_index": scene.frame_indices_32[step_idx + 1],
                        "translation_m": step["translation_m"],
                        "yaw_deg": step["yaw_deg"],
                        "pitch_deg": step["pitch_deg"],
                        "local_delta": step["local_delta"],
                        "pose_gap": 1,
                        "frame_gap": scene.frame_indices_32[step_idx + 1] - scene.frame_indices_32[step_idx],
                    },
                },
            }
        )
    return rows


def generate_sequence_samples(scenes: Sequence[PoseScene]) -> List[dict]:
    rows = []
    task_root = OUTPUT_ROOT / "movement_sequence_sorting"
    for idx, scene in enumerate(scenes, start=1):
        try:
            slots = choose_sequence_slots(scene)
        except RuntimeError as exc:
            print(f"[WARN] skip movement_sequence_sorting for {scene.dataset}/{scene.scene}: {exc}")
            continue
        qa_id = f"movement_sequence_sorting_{idx:03d}"
        sample_dir = task_root / qa_id
        clear_sample_dir(sample_dir)
        first_slot = slots[0]
        remaining = slots[1:]
        labels = ["A", "B", "C"]
        shuffled = list(zip(labels, remaining))
        random.Random(f"{scene.dataset}:{scene.scene}:{idx}").shuffle(shuffled)
        correct_order = [label for label, _slot in sorted(shuffled, key=lambda item: item[1])]
        if correct_order == labels:
            shuffled[0], shuffled[1] = (shuffled[1][0], shuffled[0][1]), (shuffled[0][0], shuffled[1][1])
            correct_order = [label for label, _slot in sorted(shuffled, key=lambda item: item[1])]
        first_frame = sample_dir / "frame_start.png"
        save_slot_frame(scene, first_slot, first_frame)
        input_paths = [first_frame]
        for label, slot in shuffled:
            out = sample_dir / f"frame_{label}.png"
            save_slot_frame(scene, slot, out)
            input_paths.append(out)
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "movement_sequence_sorting",
                "dataset": scene.dataset,
                "scene": scene.scene,
                "pose_source": scene.pose_source,
                "question": "已知第一张图是这段真实视频中的起始帧。请将其余三张图按时间先后排序。",
                "answer": " -> ".join(correct_order),
                "input": {
                    "frame_paths": [str(p) for p in input_paths],
                    "first_frame": str(first_frame),
                    "candidate_frames": {label: str(sample_dir / f"frame_{label}.png") for label, _slot in shuffled},
                },
                "gt": {
                    "start_slot": first_slot + 1,
                    "slot_order": {label: slot + 1 for label, slot in shuffled},
                    "correct_order": correct_order,
                    "motions": [segment_motion(scene, slots[i], slots[i + 1]) for i in range(3)],
                },
            }
        )
    return rows


def generate_degree_samples(scenes: Sequence[PoseScene]) -> List[dict]:
    rows = []
    task_root = OUTPUT_ROOT / "movement_degree_comparison"
    for idx, scene in enumerate(scenes, start=1):
        try:
            start_slot, end_a, end_b, family = choose_degree_triplet(scene)
        except RuntimeError as exc:
            print(f"[WARN] skip movement_degree_comparison for {scene.dataset}/{scene.scene}: {exc}")
            continue
        qa_id = f"movement_degree_comparison_{idx:03d}"
        sample_dir = task_root / qa_id
        clear_sample_dir(sample_dir)
        outputs = {
            "start": sample_dir / "frame_start.png",
            "A": sample_dir / "frame_A.png",
            "B": sample_dir / "frame_B.png",
        }
        info_1 = segment_family(scene.extr_w2c, start_slot, end_a)
        info_2 = segment_family(scene.extr_w2c, start_slot, end_b)
        flip = (idx % 2 == 0)
        if not flip:
            save_slot_frame(scene, start_slot, outputs["start"])
            save_slot_frame(scene, end_a, outputs["A"])
            save_slot_frame(scene, end_b, outputs["B"])
            info_a = info_1
            info_b = info_2
            slot_a = end_a
            slot_b = end_b
        else:
            save_slot_frame(scene, start_slot, outputs["start"])
            save_slot_frame(scene, end_b, outputs["A"])
            save_slot_frame(scene, end_a, outputs["B"])
            info_a = info_2
            info_b = info_1
            slot_a = end_b
            slot_b = end_a
        larger_clip = "A" if info_a["magnitude"] >= info_b["magnitude"] else "B"
        unit = "米" if family == "translation" else "度"
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "movement_degree_comparison",
                "dataset": scene.dataset,
                "scene": scene.scene,
                "pose_source": scene.pose_source,
                "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
                "answer": f"候选图{larger_clip}更大。",
                "input": {
                    "frame_paths": [str(outputs["start"]), str(outputs["A"]), str(outputs["B"])],
                    "start_frame": str(outputs["start"]),
                    "frame_A": str(outputs["A"]),
                    "frame_B": str(outputs["B"]),
                },
                "gt": {
                    "comparison_family": family,
                    "larger_clip": larger_clip,
                    "start_slot": start_slot + 1,
                    "frame_A": {
                        "slot_end": slot_a + 1,
                        "magnitude": info_a["magnitude"],
                        "unit": unit,
                        "yaw_deg": info_a["yaw_deg"],
                        "pitch_deg": info_a["pitch_deg"],
                        "local_delta": info_a["local_delta"],
                        "motion": segment_motion(scene, start_slot, slot_a),
                    },
                    "frame_B": {
                        "slot_end": slot_b + 1,
                        "magnitude": info_b["magnitude"],
                        "unit": unit,
                        "yaw_deg": info_b["yaw_deg"],
                        "pitch_deg": info_b["pitch_deg"],
                        "local_delta": info_b["local_delta"],
                        "motion": segment_motion(scene, start_slot, slot_b),
                    },
                },
            }
        )
    return rows


def generate_motion_family_samples(scenes: Sequence[PoseScene]) -> List[dict]:
    rows = []
    task_root = OUTPUT_ROOT / "motion_family_discrimination"
    for idx, scene in enumerate(scenes, start=1):
        try:
            start_slot, end_slot, label, motion = choose_motion_family_pair(scene)
        except RuntimeError as exc:
            print(f"[WARN] skip motion_family_discrimination for {scene.dataset}/{scene.scene}: {exc}")
            continue
        qa_id = f"motion_family_discrimination_{idx:03d}"
        sample_dir = task_root / qa_id
        clear_sample_dir(sample_dir)
        frame_a = sample_dir / "frame_A.png"
        frame_b = sample_dir / "frame_B.png"
        save_slot_frame(scene, start_slot, frame_a)
        save_slot_frame(scene, end_slot, frame_b)
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "motion_family_discrimination",
                "dataset": scene.dataset,
                "scene": scene.scene,
                "pose_source": scene.pose_source,
                "question": "图A到图B之间，相机运动更接近哪一种类型？",
                "answer": label,
                "options": FAMILY_OPTIONS,
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b)],
                    "frame_A": str(frame_a),
                    "frame_B": str(frame_b),
                },
                "gt": {
                    "start_slot": start_slot + 1,
                    "end_slot": end_slot + 1,
                    "motion": motion,
                },
            }
        )
    return rows


def generate_distance_samples(scenes: Sequence[PoseScene]) -> List[dict]:
    rows = []
    task_root = OUTPUT_ROOT / "distance_to_start_comparison"
    for idx, scene in enumerate(scenes, start=1):
        try:
            start_slot, end_a, end_b, motion_a, motion_b = choose_distance_triplet(scene)
        except RuntimeError as exc:
            print(f"[WARN] skip distance_to_start_comparison for {scene.dataset}/{scene.scene}: {exc}")
            continue
        qa_id = f"distance_to_start_comparison_{idx:03d}"
        sample_dir = task_root / qa_id
        clear_sample_dir(sample_dir)
        outputs = {
            "start": sample_dir / "frame_start.png",
            "A": sample_dir / "frame_A.png",
            "B": sample_dir / "frame_B.png",
        }
        save_slot_frame(scene, start_slot, outputs["start"])
        save_slot_frame(scene, end_a, outputs["A"])
        save_slot_frame(scene, end_b, outputs["B"])
        mag_a = movement_metric(motion_a)
        mag_b = movement_metric(motion_b)
        farther = "A" if mag_a >= mag_b else "B"
        rows.append(
            {
                "qa_id": qa_id,
                "task_type": "distance_to_start_comparison",
                "dataset": scene.dataset,
                "scene": scene.scene,
                "pose_source": scene.pose_source,
                "question": "以第一张图为起始视角，候选图A和候选图B中，哪一张离起始视角更远？",
                "answer": f"候选图{farther}更远。",
                "input": {
                    "frame_paths": [str(outputs["start"]), str(outputs["A"]), str(outputs["B"])],
                    "start_frame": str(outputs["start"]),
                    "candidate_frames": {"A": str(outputs["A"]), "B": str(outputs["B"])},
                },
                "gt": {
                    "start_slot": start_slot + 1,
                    "farther_candidate": farther,
                    "candidates": {
                        "A": {
                            "slot_end": end_a + 1,
                            "magnitude": mag_a,
                            "motion": motion_a,
                        },
                        "B": {
                            "slot_end": end_b + 1,
                            "magnitude": mag_b,
                            "motion": motion_b,
                        },
                    },
                },
            }
        )
    return rows


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    ensure_dir(OUTPUT_ROOT)
    scenes = [load_pose_scene(**spec) for spec in SCENE_SPECS]
    include_distance = os.environ.get("REAL_INCLUDE_DISTANCE", "1") != "0"
    tasks = {
        "action_inference": generate_action_samples(scenes),
        "movement_sequence_sorting": generate_sequence_samples(scenes),
        "movement_degree_comparison": generate_degree_samples(scenes),
        "motion_family_discrimination": generate_motion_family_samples(scenes),
    }
    if include_distance:
        tasks["distance_to_start_comparison"] = generate_distance_samples(scenes)
    all_rows = []
    overview = {"tasks": {}}
    for task_name, rows in tasks.items():
        task_dir = OUTPUT_ROOT / task_name
        write_json(task_dir / "qa_data.json", rows)
        write_jsonl(task_dir / "qa_data.jsonl", rows)
        overview["tasks"][task_name] = {
            "count": len(rows),
            "path": str(task_dir / "qa_data.json"),
        }
        all_rows.extend(rows)
    write_json(OUTPUT_ROOT / "all_qa_data.json", all_rows)
    write_jsonl(OUTPUT_ROOT / "all_qa_data.jsonl", all_rows)
    write_json(OUTPUT_ROOT / "overview.json", overview)


if __name__ == "__main__":
    main()
