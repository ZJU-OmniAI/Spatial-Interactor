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


DATA_ROOT = Path("/path/to/workspace/Scannet-V2数据集/extracted_rgb_pose_stride10")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/SCANNETV2_ACTION_PREVIEW")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]


@dataclass
class ScenePose:
    scene: str
    scene_dir: Path
    frame_ids: np.ndarray
    pose_paths: list[Path]
    color_paths: list[Path]
    mats: np.ndarray


@dataclass
class MotionCandidate:
    scene: ScenePose
    start: int
    end: int
    label: str
    family: str
    magnitude: float
    score: float
    motion: dict


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


def path_exists_with_retry(path: Path, attempts: int = 8, delay_s: float = 0.5) -> bool:
    for attempt in range(attempts):
        try:
            return path.exists()
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))
    return False


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


def load_scene_pose(scene_dir: Path) -> ScenePose | None:
    pose_dir = scene_dir / "pose"
    color_dir = scene_dir / "color"
    if not pose_dir.is_dir() or not color_dir.is_dir():
        return None

    pose_paths = sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem))
    frame_ids: list[int] = []
    color_paths: list[Path] = []
    mats: list[np.ndarray] = []
    for pose_path in pose_paths:
        color_path = color_dir / f"{pose_path.stem}.jpg"
        try:
            mat = np.loadtxt(pose_path, dtype=np.float64)
        except Exception:
            continue
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            continue
        frame_ids.append(int(pose_path.stem))
        color_paths.append(color_path)
        mats.append(mat)

    if len(mats) < 80:
        return None
    return ScenePose(
        scene=scene_dir.name,
        scene_dir=scene_dir,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        pose_paths=pose_paths,
        color_paths=color_paths,
        mats=np.stack(mats, axis=0),
    )


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

    if 50 <= gap <= 260 and translation <= 0.85 and 30.0 <= rot_primary <= 60.0 and rot_primary >= 1.55 * max(rot_other, 1e-6) and abs(rz) <= 16.0:
        if abs(rx) >= abs(ry):
            label = "向上转动" if rx > 0 else "向下转动"
            magnitude = abs(rx)
        else:
            label = "向右转动" if ry > 0 else "向左转动"
            magnitude = abs(ry)
        score = magnitude * 2.0 + 0.03 * gap - translation * 8.0 - rot_other
        return label, "rotation", magnitude, score

    if 50 <= gap <= 260 and translation >= 0.55 and abs(dy) <= 0.5 and rot_primary <= 14.0 and abs(rz) <= 10.0:
        if abs(dx) >= abs(dz):
            primary = abs(dx)
            secondary = abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary = abs(dz)
            secondary = abs(dx)
            label = "向前移动" if dz > 0 else "向后移动"
        if not (0.6 <= primary <= 3.2 and primary >= 2.2 * max(secondary, 1e-6)):
            return None, None, 0.0, 0.0
        score = primary * 25.0 + primary / max(max(secondary, abs(dy)), 1e-6) * 2.0 - rot_primary * 1.5 - abs(rz)
        return label, "translation", primary, score

    return None, None, 0.0, 0.0


def image_quality(paths: Sequence[Path]) -> dict | None:
    frames = []
    grays = []
    sharpness = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        frames.append(frame)
        small = cv2.resize(frame, (480, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    pair_mad = float(np.mean(np.abs(grays[0].astype(np.float32) - grays[1].astype(np.float32))))
    return {"frames": frames, "sharpness_min": min(sharpness), "pair_gray_mad": pair_mad}


def save_pair(sample_dir: Path, frames: Sequence[np.ndarray]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs = {}
    for name, frame in zip(["frame_A", "frame_B"], frames):
        path = sample_dir / f"{name}.png"
        cv2.imwrite(str(path), frame)
        outputs[name] = str(path)
    smalls = [cv2.resize(frame, (480, 360)) for frame in frames]
    contact = sample_dir / "contact_sheet.png"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1))
    outputs["contact_sheet"] = str(contact)
    return outputs


def action_value_text(label: str, family: str, magnitude: float) -> str:
    if family == "rotation":
        return f"{label}约{magnitude:.1f}度"
    return f"{label}约{magnitude:.2f}米"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small ScanNetV2 action-inference preview for pose-rule checking.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--per-label", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root / "action_inference")

    scene_dirs = [p for p in sorted(args.data_root.iterdir()) if p.is_dir() and (p / "color").is_dir() and (p / "pose").is_dir()]
    random.shuffle(scene_dirs)
    scene_dirs = scene_dirs[: args.max_scenes]
    gaps = [5, 6, 8, 10, 12, 15, 18, 22, 26]

    rows: list[dict] = []
    label_counts: dict[str, int] = defaultdict(int)
    used: set[tuple[str, int, int]] = set()
    scenes_loaded = 0
    candidates_seen = 0

    for scene_dir in scene_dirs:
        if all(label_counts[label] >= args.per_label for label in ACTION_OPTIONS):
            break
        scene = load_scene_pose(scene_dir)
        if scene is None:
            continue
        scenes_loaded += 1
        local: list[MotionCandidate] = []
        for gap in gaps:
            if gap >= len(scene.frame_ids):
                continue
            step = max(3, gap // 2)
            for start in range(0, len(scene.frame_ids) - gap, step):
                end = start + gap
                motion = motion_between(scene, start, end)
                label, family, magnitude, score = classify_motion(motion)
                if label is None or family is None:
                    continue
                if label_counts[label] >= args.per_label:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(MotionCandidate(scene, start, end, label, family, float(magnitude), float(score), motion))
        local.sort(key=lambda item: item.score, reverse=True)
        for cand in local:
            if label_counts[cand.label] >= args.per_label:
                continue
            key = (cand.scene.scene, cand.start, cand.end)
            if key in used:
                continue
            quality = image_quality([cand.scene.color_paths[cand.start], cand.scene.color_paths[cand.end]])
            if quality is None or quality["sharpness_min"] < 8 or quality["pair_gray_mad"] < 14:
                continue
            candidates_seen += 1
            qa_id = f"scannetv2_action_preview_{len(rows) + 1:03d}"
            sample_dir = args.output_root / "action_inference" / qa_id
            paths = save_pair(sample_dir, quality["frames"])
            rows.append(
                {
                    "qa_id": qa_id,
                    "task_type": "action_inference",
                    "dataset": "scannetv2",
                    "scene": cand.scene.scene,
                    "pose_source": "scannetv2_official_sens_camera_to_world_candidate_rule",
                    "question": "图A到图B之间，相机主要执行了什么动作？",
                    "answer": cand.label,
                    "answer_with_value": action_value_text(cand.label, cand.family, cand.magnitude),
                    "options": ACTION_OPTIONS,
                    "label_source": "predicted_by_scannetv2_pose_rule_needs_human_check",
                    "input": {
                        "frame_paths": [paths["frame_A"], paths["frame_B"]],
                        "frame_A": paths["frame_A"],
                        "frame_B": paths["frame_B"],
                        "contact_sheet": paths["contact_sheet"],
                        "source_frame_A": str(cand.scene.color_paths[cand.start]),
                        "source_frame_B": str(cand.scene.color_paths[cand.end]),
                    },
                    "gt": {
                        "motion": cand.motion,
                        "approx_action": {
                            "text": action_value_text(cand.label, cand.family, cand.magnitude),
                            "value": round(cand.magnitude, 3),
                            "unit": "deg" if cand.family == "rotation" else "m",
                        },
                        "quality": {
                            "pose_score": cand.score,
                            "sharpness_min": quality["sharpness_min"],
                            "pair_gray_mad": quality["pair_gray_mad"],
                        },
                    },
                }
            )
            label_counts[cand.label] += 1
            used.add(key)
            print(f"[ok] {qa_id} {cand.label} {action_value_text(cand.label, cand.family, cand.magnitude)} scene={cand.scene.scene}", flush=True)

    write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    write_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", rows)
    summary = {
        "dataset": "scannetv2",
        "output_root": str(args.output_root),
        "data_root": str(args.data_root),
        "scenes_loaded": scenes_loaded,
        "candidates_saved": len(rows),
        "candidates_seen_after_quality": candidates_seen,
        "label_counts": {label: label_counts[label] for label in ACTION_OPTIONS},
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
