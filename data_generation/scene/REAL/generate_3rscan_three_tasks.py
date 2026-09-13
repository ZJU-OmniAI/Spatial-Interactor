#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SEQUENCE_ROOT = Path("/path/to/workspace/3RScan_sequence_only")
EXTRACTED_SAMPLE_ROOT = Path("/path/to/workspace/3RScan_public_sample_rgb_pose")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/THREERSCAN")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]


@dataclass
class SceneData:
    scene: str
    frame_ids: np.ndarray
    poses: np.ndarray
    source: str
    zip_path: Path | None = None
    seq_dir: Path | None = None


@dataclass
class Candidate:
    scene: str
    start_slot: int
    end_slot: int
    label: str
    family: str
    magnitude: float
    motion: dict
    score: float
    source: str


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


def parse_pose_text(text: str) -> np.ndarray | None:
    try:
        mat = np.asarray([[float(x) for x in line.split()] for line in text.splitlines() if line.strip()], dtype=np.float64)
    except Exception:
        return None
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        return None
    return mat


def load_extracted_scene(scene_dir: Path) -> SceneData | None:
    seq_dir = scene_dir / "sequence"
    if not seq_dir.is_dir():
        return None
    frame_ids = []
    poses = []
    for pose_path in sorted(seq_dir.glob("frame-*.pose.txt")):
        stem = pose_path.name.split(".")[0]
        frame_id = int(stem.split("-")[-1])
        color_path = seq_dir / f"frame-{frame_id:06d}.color.jpg"
        if not color_path.exists():
            continue
        mat = parse_pose_text(pose_path.read_text(encoding="utf-8", errors="ignore"))
        if mat is None:
            continue
        frame_ids.append(frame_id)
        poses.append(mat)
    if len(poses) < 80:
        return None
    return SceneData(scene=scene_dir.name, frame_ids=np.asarray(frame_ids, dtype=np.int32), poses=np.stack(poses), source="extracted", seq_dir=seq_dir)


def load_zipped_scene(scene_dir: Path) -> SceneData | None:
    zip_path = scene_dir / "sequence.zip"
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            pose_names = sorted(name for name in names if name.startswith("frame-") and name.endswith(".pose.txt"))
            frame_ids = []
            poses = []
            for name in pose_names:
                frame_id = int(Path(name).name.split(".")[0].split("-")[-1])
                color_name = f"frame-{frame_id:06d}.color.jpg"
                if color_name not in names:
                    continue
                mat = parse_pose_text(zf.read(name).decode("utf-8", errors="ignore"))
                if mat is None:
                    continue
                frame_ids.append(frame_id)
                poses.append(mat)
    except Exception:
        return None
    if len(poses) < 80:
        return None
    return SceneData(scene=scene_dir.name, frame_ids=np.asarray(frame_ids, dtype=np.int32), poses=np.stack(poses), source="zip", zip_path=zip_path)


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


def motion_between(scene: SceneData, start: int, end: int) -> dict:
    pose_a = scene.poses[start]
    pose_b = scene.poses[end]
    rel_rot = pose_a[:3, :3].T @ pose_b[:3, :3]
    rel_rotvec = rotvec_deg(rel_rot)
    local_delta = pose_a[:3, :3].T @ (pose_b[:3, 3] - pose_a[:3, 3])
    translation = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
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
    trans = float(motion["translation_m"])
    rot_primary = max(abs(rx), abs(ry))
    rot_other = max(min(abs(rx), abs(ry)), abs(rz))

    # 3RScan user-validated partial rule:
    # x-axis rotation maps to left/right turn, y-axis rotation maps to up/down turn.
    if 18 <= gap <= 55 and trans <= 0.22 and 26.0 <= rot_primary <= 42.0 and rot_primary >= 2.3 * max(rot_other, 1e-6) and abs(rz) <= 7.0:
        if abs(rx) >= abs(ry):
            label = "向右转动" if rx > 0 else "向左转动"
            magnitude = abs(rx)
        else:
            label = "向下转动" if ry > 0 else "向上转动"
            magnitude = abs(ry)
        score = magnitude * 2.2 - trans * 28.0 - rot_other * 2.4 - abs(rz) * 2.6 - abs(dx) * 22.0 - abs(dy) * 18.0 - abs(dz) * 22.0
        return label, "rotation", magnitude, score

    # 3RScan translation rule after user validation:
    # dy>0 left, dy<0 right, dz>0 forward, dz<0 backward. dx is treated as vertical leakage.
    if 18 <= gap <= 80 and 0.25 <= trans <= 1.30 and abs(dx) <= 0.16 and rot_primary <= 8.0 and abs(rz) <= 6.0:
        if abs(dy) >= abs(dz):
            primary = abs(dy)
            secondary = abs(dz)
            label = "向左移动" if dy > 0 else "向右移动"
        else:
            primary = abs(dz)
            secondary = abs(dy)
            label = "向前移动" if dz > 0 else "向后移动"
        if label in {"向左移动", "向右移动"}:
            if not (0.30 <= primary <= 1.20 and primary >= 3.0 * max(secondary, abs(dx), 1e-6)):
                return None, None, 0.0, 0.0
        else:
            if not (0.25 <= primary <= 0.95 and primary >= 2.4 * max(secondary, abs(dx), 1e-6)):
                return None, None, 0.0, 0.0
        score = primary * 30.0 + primary / max(max(secondary, abs(dx)), 1e-6) - rot_primary * 4.0 - abs(rz) * 3.5 - abs(dx) * 28.0
        if label in {"向左移动", "向右移动"}:
            score -= abs(dz) * 70.0 + abs(dx) * 80.0
        else:
            score -= abs(dy) * 40.0 + abs(dx) * 50.0
        return label, "translation", primary, score

    return None, None, 0.0, 0.0


def collect_scenes(max_scenes: int) -> list[SceneData]:
    scenes: list[SceneData] = []
    for scene_dir in sorted(EXTRACTED_SAMPLE_ROOT.iterdir()) if EXTRACTED_SAMPLE_ROOT.exists() else []:
        if not scene_dir.is_dir():
            continue
        loaded = load_extracted_scene(scene_dir)
        if loaded:
            scenes.append(loaded)
    for scene_dir in sorted(SEQUENCE_ROOT.iterdir()):
        if max_scenes > 0 and len(scenes) >= max_scenes:
            break
        if not scene_dir.is_dir():
            continue
        loaded = load_zipped_scene(scene_dir)
        if loaded:
            scenes.append(loaded)
    return scenes


def collect_candidates(scenes: list[SceneData], per_scene_label_cap: int = 5, per_label_cap: int = 200) -> dict[str, list[Candidate]]:
    per_label: dict[str, list[Candidate]] = defaultdict(list)
    gaps = [18, 20, 24, 25, 30, 35, 40, 45, 50, 60, 70, 80]
    for scene in scenes:
        local: list[Candidate] = []
        for gap in gaps:
            if gap >= len(scene.frame_ids):
                continue
            step = max(5, gap // 3)
            for start in range(0, len(scene.frame_ids) - gap, step):
                end = start + gap
                motion = motion_between(scene, start, end)
                label, family, magnitude, score = classify_motion(motion)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(Candidate(scene=scene.scene, start_slot=start, end_slot=end, label=label, family=family, magnitude=float(magnitude), motion=motion, score=float(score), source=scene.source))
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
    return per_label


def read_frame_from_scene(scene: SceneData, frame_index: int) -> np.ndarray | None:
    if scene.source == "extracted" and scene.seq_dir is not None:
        return cv2.imread(str(scene.seq_dir / f"frame-{frame_index:06d}.color.jpg"))
    if scene.source == "zip" and scene.zip_path is not None:
        try:
            with zipfile.ZipFile(scene.zip_path) as zf:
                data = zf.read(f"frame-{frame_index:06d}.color.jpg")
        except Exception:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


def rotate_clockwise_90(frame: np.ndarray) -> np.ndarray:
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def frame_content_metrics(frame: np.ndarray) -> dict:
    small = cv2.resize(frame, (480, 270))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = gray[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    lower = gray[(2 * h) // 3 :, :]
    upper = gray[: h // 3, :]
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "gray_std": float(gray.std()),
        "edge_density": edge_density(gray),
        "center_edge_density": edge_density(center),
        "upper_edge_density": edge_density(upper),
        "lower_edge_density": edge_density(lower),
    }


def visual_quality(frame_a: np.ndarray, frame_b: np.ndarray) -> dict:
    small_a = cv2.resize(frame_a, (480, 270))
    small_b = cv2.resize(frame_b, (480, 270))
    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)
    content_a = frame_content_metrics(frame_a)
    content_b = frame_content_metrics(frame_b)
    return {
        "gray_mad": float(np.mean(np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32)))),
        "sharpness_min": min(content_a["sharpness"], content_b["sharpness"]),
        "gray_std_min": min(content_a["gray_std"], content_b["gray_std"]),
        "edge_density_min": min(content_a["edge_density"], content_b["edge_density"]),
        "center_edge_density_min": min(content_a["center_edge_density"], content_b["center_edge_density"]),
        "upper_edge_density_min": min(content_a["upper_edge_density"], content_b["upper_edge_density"]),
        "lower_edge_density_max": max(content_a["lower_edge_density"], content_b["lower_edge_density"]),
    }


def apparent_motion_after_rotation(frame_a: np.ndarray, frame_b: np.ndarray) -> dict | None:
    rot_a = rotate_clockwise_90(frame_a)
    rot_b = rotate_clockwise_90(frame_b)
    gray_a = cv2.cvtColor(cv2.resize(rot_a, (320, 240)), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(cv2.resize(rot_b, (320, 240)), cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray_a, maxCorners=300, qualityLevel=0.01, minDistance=7)
    if points is None or len(points) < 20:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, winSize=(21, 21), maxLevel=3)
    if tracked is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    if int(valid.sum()) < 20:
        return None
    delta = (tracked[valid] - points[valid]).reshape(-1, 2)
    median = np.median(delta, axis=0)
    return {
        "apparent_flow_x_px": float(median[0]),
        "apparent_flow_y_px": float(median[1]),
        "tracked_points": int(valid.sum()),
    }


def save_sample(sample_dir: Path, frame_a: np.ndarray, frame_b: np.ndarray) -> dict[str, str]:
    ensure_dir(sample_dir)
    frame_a = rotate_clockwise_90(frame_a)
    frame_b = rotate_clockwise_90(frame_b)
    path_a = sample_dir / "frame_A.png"
    path_b = sample_dir / "frame_B.png"
    contact = sample_dir / "contact_sheet_A_B.png"
    cv2.imwrite(str(path_a), frame_a)
    cv2.imwrite(str(path_b), frame_b)
    cv2.imwrite(str(contact), np.concatenate([cv2.resize(frame_a, (480, 270)), cv2.resize(frame_b, (480, 270))], axis=1))
    return {"frame_A": str(path_a), "frame_B": str(path_b), "contact_sheet": str(contact)}


def save_frames(sample_dir: Path, names_and_frames: Sequence[tuple[str, np.ndarray]]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs = {}
    smalls = []
    for name, frame in names_and_frames:
        frame = rotate_clockwise_90(frame)
        path = sample_dir / f"{name}.png"
        cv2.imwrite(str(path), frame)
        outputs[name] = str(path)
        smalls.append(cv2.resize(frame, (480, 270)))
    contact = sample_dir / "contact_sheet.png"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1))
    outputs["contact_sheet"] = str(contact)
    return outputs


def load_frames(scene: SceneData, frame_indices: Sequence[int]) -> tuple[list[np.ndarray], dict] | None:
    frames = []
    grays = []
    sharpness = []
    stds = []
    centers = []
    for frame_index in frame_indices:
        frame = read_frame_from_scene(scene, int(frame_index))
        if frame is None:
            return None
        frames.append(frame)
        small = cv2.resize(frame, (480, 270))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        stds.append(float(gray.std()))
        h, w = gray.shape
        centers.append(edge_density(gray[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]))
    pair_mads = [
        float(np.mean(np.abs(grays[i].astype(np.float32) - grays[i + 1].astype(np.float32))))
        for i in range(len(grays) - 1)
    ]
    return frames, {
        "sharpness_min": min(sharpness),
        "gray_std_min": min(stds),
        "center_edge_density_min": min(centers),
        "pair_gray_mad": pair_mads,
    }


def action_text(label: str, family: str, magnitude: float) -> str:
    if family == "rotation":
        return f"{label}约{magnitude:.1f}度"
    return f"{label}约{magnitude:.2f}米"


def combined_visual_pose_score(cand: Candidate, quality: dict) -> float:
    return (
        cand.score
        + quality["gray_mad"] * 0.25
        + quality["sharpness_min"] * 0.06
        + quality["gray_std_min"] * 0.15
        + quality["edge_density_min"] * 180.0
        + quality["center_edge_density_min"] * 240.0
        + quality["upper_edge_density_min"] * 120.0
        - quality["lower_edge_density_max"] * 60.0
    )


def passes_visual_quality(label: str, quality: dict) -> bool:
    if quality["gray_mad"] < 18:
        return False
    if quality["sharpness_min"] < 12:
        return False
    if quality["gray_std_min"] < 12:
        return False
    if quality["edge_density_min"] < 0.004:
        return False
    if quality["center_edge_density_min"] < 0.004:
        return False
    if quality["upper_edge_density_min"] < 0.003:
        return False
    if quality["lower_edge_density_max"] > 0.20 and quality["center_edge_density_min"] < 0.010:
        return False
    if label in {"向左移动", "向右移动"}:
        if quality["sharpness_min"] < 35:
            return False
        if quality["gray_std_min"] < 24:
            return False
        if quality["edge_density_min"] < 0.012:
            return False
        if quality["center_edge_density_min"] < 0.012:
            return False
        if "apparent_flow_x_px" not in quality or "apparent_flow_y_px" not in quality:
            return False
        if abs(quality["apparent_flow_x_px"]) < 2.0:
            return False
        if abs(quality["apparent_flow_x_px"]) < 1.15 * abs(quality["apparent_flow_y_px"]):
            return False
    return True


def pair_quality_ok(quality: dict, min_mad: float = 18.0) -> bool:
    return (
        quality["sharpness_min"] >= 18
        and quality["gray_std_min"] >= 16
        and quality["center_edge_density_min"] >= 0.006
        and min(quality["pair_gray_mad"]) >= min_mad
    )


def build_action_rows(per_label: dict[str, list[Candidate]], scenes: list[SceneData], target: int, output_root: Path) -> list[dict]:
    rows = []
    scene_by_id = {scene.scene: scene for scene in scenes}
    used = set()
    while target <= 0 or len(rows) < target:
        progressed = False
        for label in ACTION_OPTIONS:
            if target > 0 and len(rows) >= target:
                break
            for cand in per_label.get(label, []):
                key = (cand.scene, cand.start_slot, cand.end_slot)
                if key in used:
                    continue
                scene = scene_by_id[cand.scene]
                loaded = load_frames(scene, [cand.motion["start_frame_index"], cand.motion["end_frame_index"]])
                if loaded is None:
                    used.add(key)
                    continue
                frames, quality = loaded
                if not pair_quality_ok(quality, 18):
                    used.add(key)
                    continue
                qa_id = f"threerscan_action_inference_{len(rows) + 1:04d}"
                sample_dir = output_root / "action_inference" / qa_id
                paths = save_frames(sample_dir, [("frame_A", frames[0]), ("frame_B", frames[1])])
                rows.append(
                    {
                        "qa_id": qa_id,
                        "task_type": "action_inference",
                        "dataset": "3rscan",
                        "scene": cand.scene,
                        "pose_source": "3rscan_sequence_pose_txt_relative_local_rule_human_verified_v3",
                        "question": "图A到图B之间，相机主要执行了什么动作？",
                        "answer": cand.label,
                        "answer_with_value": action_text(cand.label, cand.family, cand.magnitude),
                        "options": ACTION_OPTIONS,
                        "label_source": "predicted_by_confirmed_3rscan_pose_rule",
                        "input": {
                            "frame_paths": [paths["frame_A"], paths["frame_B"]],
                            "frame_A": paths["frame_A"],
                            "frame_B": paths["frame_B"],
                            "contact_sheet": paths["contact_sheet"],
                        },
                        "gt": {
                            "motion": cand.motion,
                            "approx_action": {
                                "text": action_text(cand.label, cand.family, cand.magnitude),
                                "value": round(cand.magnitude, 3),
                                "unit": "deg" if cand.family == "rotation" else "m",
                            },
                            "quality": {"pose_score": cand.score, **quality},
                        },
                    }
                )
                used.add(key)
                progressed = True
                break
        if not progressed:
            break
    return rows


def collect_degree_rows(scenes: list[SceneData], target: int, output_root: Path) -> list[dict]:
    rows = []
    rng = random.Random(20260513)
    gaps = [18, 20, 24, 25, 30, 35, 40, 45, 50, 60, 70, 80]
    used_starts: set[tuple[str, int]] = set()
    for scene in scenes:
        scene_rows = 0
        max_scene_rows = 0 if target <= 0 else max(2, target // max(len(scenes), 1) + 2)
        for start in range(0, max(0, len(scene.frame_ids) - max(gaps) - 1), 8):
            if target > 0 and len(rows) >= target:
                return rows
            if max_scene_rows > 0 and scene_rows >= max_scene_rows:
                break
            local = []
            for gap in gaps:
                end = start + gap
                if end >= len(scene.frame_ids):
                    continue
                motion = motion_between(scene, start, end)
                label, family, magnitude, score = classify_motion(motion)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append((label, family, magnitude, score, end, motion))
            pairs = []
            for i in range(len(local)):
                for j in range(i + 1, len(local)):
                    a, b = local[i], local[j]
                    if a[0] != b[0] or a[1] != b[1]:
                        continue
                    gap_mag = abs(a[2] - b[2])
                    min_gap = 0.25 if a[1] == "translation" else 9.0
                    if gap_mag < min_gap:
                        continue
                    pairs.append((gap_mag + 0.01 * (a[3] + b[3]), a, b))
            if not pairs or (scene.scene, start) in used_starts:
                continue
            pairs.sort(key=lambda item: item[0], reverse=True)
            for _score, cand_a, cand_b in pairs[:3]:
                left, right = cand_a, cand_b
                if rng.random() < 0.5:
                    left, right = right, left
                larger = "A" if left[2] > right[2] else "B"
                frame_indices = [int(scene.frame_ids[start]), left[5]["end_frame_index"], right[5]["end_frame_index"]]
                loaded = load_frames(scene, frame_indices)
                if loaded is None:
                    continue
                frames, quality = loaded
                if not pair_quality_ok(quality, 16):
                    continue
                qa_id = f"threerscan_movement_degree_comparison_{len(rows) + 1:04d}"
                sample_dir = output_root / "movement_degree_comparison" / qa_id
                paths = save_frames(sample_dir, [("frame_start", frames[0]), ("frame_A", frames[1]), ("frame_B", frames[2])])
                unit = "度" if left[1] == "rotation" else "米"
                rows.append(
                    {
                        "qa_id": qa_id,
                        "task_type": "movement_degree_comparison",
                        "dataset": "3rscan",
                        "scene": scene.scene,
                        "pose_source": "3rscan_sequence_pose_txt_relative_local_rule_human_verified_v3",
                        "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
                        "answer": f"候选图{larger}更大。",
                        "answer_with_value": (
                            f"候选图A约{left[2]:.1f}{unit}，候选图B约{right[2]:.1f}{unit}。"
                            if left[1] == "rotation"
                            else f"候选图A约{left[2]:.2f}{unit}，候选图B约{right[2]:.2f}{unit}。"
                        ),
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
                scene_rows += 1
                break
    return rows


def collect_sorting_rows(scenes: list[SceneData], target: int, output_root: Path) -> list[dict]:
    rows = []
    rng = random.Random(20260514)
    for scene in scenes:
        scene_rows = 0
        max_scene_rows = 0 if target <= 0 else max(2, target // max(len(scenes), 1) + 2)
        for gap in [10, 12, 15, 18, 20, 24, 25, 30, 35, 40]:
            span = 3 * gap
            if span >= len(scene.frame_ids):
                continue
            for start in range(0, len(scene.frame_ids) - span, max(6, gap // 3)):
                if target > 0 and len(rows) >= target:
                    return rows
                if max_scene_rows > 0 and scene_rows >= max_scene_rows:
                    break
                slots = [start, start + gap, start + 2 * gap, start + 3 * gap]
                motions = [motion_between(scene, slots[i], slots[i + 1]) for i in range(3)]
                ok = True
                axis_signs = []
                total_primary = 0.0
                total_rotation = 0.0
                for motion in motions:
                    dx, dy, dz = [float(x) for x in motion["local_delta"]]
                    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
                    translation = float(motion["translation_m"])
                    if abs(dy) >= abs(dz):
                        axis = "y"
                        primary = dy
                        orthogonal = abs(dz)
                    else:
                        axis = "z"
                        primary = dz
                        orthogonal = abs(dy)
                    if not (
                        0.14 <= abs(primary) <= 1.20
                        and 0.16 <= translation <= 1.25
                        and orthogonal <= max(0.14, 0.50 * abs(primary))
                        and abs(dx) <= 0.18
                        and max(abs(rx), abs(ry), abs(rz)) <= 14.0
                        and float(motion["abs_rotation_deg"]) <= 18.0
                    ):
                        ok = False
                        break
                    axis_signs.append((axis, 1 if primary > 0 else -1))
                    total_primary += abs(primary)
                    total_rotation += float(motion["abs_rotation_deg"])
                if not ok or len(set(axis_signs)) != 1:
                    continue
                frame_indices = [int(scene.frame_ids[slot]) for slot in slots]
                loaded = load_frames(scene, frame_indices)
                if loaded is None:
                    continue
                frames, quality = loaded
                if not pair_quality_ok(quality, 16):
                    continue
                labels = ["A", "B", "C"]
                candidate_frames = list(zip(labels, slots[1:], frames[1:]))
                rng.shuffle(candidate_frames)
                correct = [label for label, slot, _frame in sorted(candidate_frames, key=lambda item: item[1])]
                qa_id = f"threerscan_movement_sequence_sorting_{len(rows) + 1:04d}"
                sample_dir = output_root / "movement_sequence_sorting" / qa_id
                named_frames = [("frame_start", frames[0])] + [(f"frame_{label}", frame) for label, _slot, frame in candidate_frames]
                paths = save_frames(sample_dir, named_frames)
                rows.append(
                    {
                        "qa_id": qa_id,
                        "task_type": "movement_sequence_sorting",
                        "dataset": "3rscan",
                        "scene": scene.scene,
                        "pose_source": "3rscan_sequence_pose_txt_relative_local_rule_human_verified_v3",
                        "question": "已知第一张图是起始帧。请将其余三张候选图按真实视频中的时间先后排序。",
                        "answer": " -> ".join(correct),
                        "input": {
                            "frame_paths": [paths["frame_start"]] + [paths[f"frame_{label}"] for label, _slot, _frame in candidate_frames],
                            "first_frame": paths["frame_start"],
                            "candidate_frames": {label: paths[f"frame_{label}"] for label, _slot, _frame in candidate_frames},
                            "contact_sheet": paths["contact_sheet"],
                        },
                        "gt": {
                            "start_slot": int(slots[0]),
                            "start_frame_index": int(scene.frame_ids[slots[0]]),
                            "candidate_slots": {label: int(slot) for label, slot, _frame in candidate_frames},
                            "candidate_frame_indices": {label: int(scene.frame_ids[slot]) for label, slot, _frame in candidate_frames},
                            "correct_order": correct,
                            "trajectory_type": "same_direction_translation_low_rotation",
                            "dominant_axis": axis_signs[0][0],
                            "dominant_axis_sign": axis_signs[0][1],
                            "total_primary_motion_m": float(total_primary),
                            "total_rotation_deg": float(total_rotation),
                            "motions_between_consecutive_true_frames": motions,
                            "quality": quality,
                        },
                    }
                )
                scene_rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3RScan motion QA with human-verified pose rules.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=0, help="Maximum scenes to scan. Use 0 for all scenes.")
    parser.add_argument("--target-per-task", type=int, default=0, help="Maximum rows per task. Use 0 for no explicit cap.")
    parser.add_argument("--tasks", default="action_inference,movement_degree_comparison,movement_sequence_sorting")
    parser.add_argument("--action-per-scene-label-cap", type=int, default=5, help="Use 0 for no per-scene per-label cap.")
    parser.add_argument("--action-per-label-cap", type=int, default=200, help="Use 0 for no global per-label cap.")
    args = parser.parse_args()

    selected_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    random.seed(20260513)

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    scenes = collect_scenes(args.max_scenes)
    task_rows: dict[str, list[dict]] = {}
    if "action_inference" in selected_tasks:
        per_label = collect_candidates(
            scenes,
            per_scene_label_cap=args.action_per_scene_label_cap,
            per_label_cap=args.action_per_label_cap,
        )
        task_rows["action_inference"] = build_action_rows(per_label, scenes, args.target_per_task, args.output_root)
    if "movement_degree_comparison" in selected_tasks:
        task_rows["movement_degree_comparison"] = collect_degree_rows(scenes, args.target_per_task, args.output_root)
    if "movement_sequence_sorting" in selected_tasks:
        task_rows["movement_sequence_sorting"] = collect_sorting_rows(scenes, args.target_per_task, args.output_root)

    for task, rows in task_rows.items():
        write_json(args.output_root / task / "qa_data.json", rows)
        write_jsonl(args.output_root / task / "qa_data.jsonl", rows)

    summary = {
        "dataset": "3rscan",
        "output_root": str(args.output_root),
        "scenes_loaded": len(scenes),
        "target_per_task": args.target_per_task,
        "counts": {task: len(rows) for task, rows in task_rows.items()},
        "pose_rule": {
            "source": "sequence/frame-xxxxxx.pose.txt, treated as camera-to-world",
            "relative_rotation": "R_rel = R_A.T @ R_B, Rodrigues rotvec in degrees",
            "rotation_labels": "x>0 right-turn, x<0 left-turn, y>0 down-turn, y<0 up-turn",
            "translation_labels": "local_delta[1]>0 left, local_delta[1]<0 right, local_delta[2]>0 forward, local_delta[2]<0 backward; local_delta[0] is vertical/leakage",
            "image_orientation": "saved frames are rotated 90 degrees clockwise",
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
