#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
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
OUTPUT_ROOT = Path("/path/to/workspace/DATA/THREERSCAN_POSE_RULE_CANDIDATES")
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


def collect_candidates(scenes: list[SceneData]) -> dict[str, list[Candidate]]:
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
            if scene_counts[cand.label] >= 5:
                continue
            per_label[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label[label].sort(key=lambda item: item.score, reverse=True)
            del per_label[label][200:]
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
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=80)
    parser.add_argument("--target", type=int, default=8)
    parser.add_argument("--labels", default=",".join(ACTION_OPTIONS), help="comma-separated labels to generate")
    parser.add_argument("--use-apparent-flow", action="store_true", help="Optional visual-only filter; labels still come from pose.")
    args = parser.parse_args()
    selected_labels = [label.strip() for label in args.labels.split(",") if label.strip()]

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root / "action_inference")

    scenes = collect_scenes(args.max_scenes)
    scene_by_id = {scene.scene: scene for scene in scenes}
    per_label = collect_candidates(scenes)

    rows = []
    used = set()
    for label in selected_labels:
        best_choice = None
        for cand in per_label.get(label, []):
            key = (cand.scene, cand.start_slot, cand.end_slot)
            if key in used:
                continue
            scene = scene_by_id[cand.scene]
            frame_a = read_frame_from_scene(scene, cand.motion["start_frame_index"])
            frame_b = read_frame_from_scene(scene, cand.motion["end_frame_index"])
            if frame_a is None or frame_b is None:
                continue
            quality = visual_quality(frame_a, frame_b)
            if args.use_apparent_flow and label in {"向左移动", "向右移动"}:
                apparent = apparent_motion_after_rotation(frame_a, frame_b)
                if apparent:
                    quality.update(apparent)
            if not passes_visual_quality(label, quality):
                continue
            total_score = combined_visual_pose_score(cand, quality)
            if best_choice is None or total_score > best_choice["total_score"]:
                best_choice = {
                    "cand": cand,
                    "quality": quality,
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "key": key,
                    "total_score": total_score,
                }
        if best_choice is not None:
            cand = best_choice["cand"]
            quality = best_choice["quality"]
            frame_a = best_choice["frame_a"]
            frame_b = best_choice["frame_b"]
            key = best_choice["key"]
            qa_id = f"threerscan_pose_candidate_{len(rows) + 1:03d}"
            sample_dir = args.output_root / "action_inference" / qa_id
            paths = save_sample(sample_dir, frame_a, frame_b)
            row = {
                "qa_id": qa_id,
                "task_type": "action_inference",
                "dataset": "3rscan",
                "scene": cand.scene,
                "pose_source": "3rscan_sequence_pose_txt_relative_local_rule_hypothesis_v2",
                "question": "图A到图B之间，相机主要执行了什么动作？",
                "answer": cand.label,
                "answer_with_value": action_text(cand.label, cand.family, cand.magnitude),
                "options": ACTION_OPTIONS,
                "label_source": "predicted_by_pose_rule_needs_human_correction",
                "input": {"frame_paths": [paths["frame_A"], paths["frame_B"]], **paths},
                "gt": {
                    "motion": cand.motion,
                    "approx_action": {"text": action_text(cand.label, cand.family, cand.magnitude), "value": round(cand.magnitude, 3), "unit": "deg" if cand.family == "rotation" else "m"},
                    "quality": {"pose_score": cand.score, **quality},
                    "pose_rule_hypothesis_v2": {
                        "pose_txt": "treated as camera-to-world",
                        "relative_rotation": "R_rel = R_A.T @ R_B, Rodrigues rotvec in degrees",
                        "rotation_rule": "user feedback v3: x>0 right-turn, x<0 left-turn, y>0 down-turn, y<0 up-turn",
                        "translation_rule": "user feedback v2: local_delta[1]>0 left, local_delta[1]<0 right, local_delta[2]>0 forward, local_delta[2]<0 backward; local_delta[0] is vertical/leakage",
                        "note": "Stricter single-action filtering plus rotated output. Labels still require user correction.",
                    },
                },
            }
            rows.append(row)
            used.add(key)
        if len(rows) >= args.target:
            break

    write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    with (args.output_root / "action_inference" / "qa_data.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "generated_total": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in selected_labels},
        "output": str(args.output_root / "action_inference" / "qa_data.json"),
        "rule_status": "hypothesis_needs_user_validation",
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        motion = row["gt"]["motion"]
        print(row["qa_id"], row["answer_with_value"], row["scene"], "gap", motion["frame_gap"], "rot", [round(x, 1) for x in motion["relative_rotvec_deg_xyz"]], "delta", [round(x, 2) for x in motion["local_delta"]])


if __name__ == "__main__":
    main()
