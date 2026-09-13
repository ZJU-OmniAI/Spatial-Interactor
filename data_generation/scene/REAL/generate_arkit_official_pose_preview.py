#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DATA_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/QUICK_EIGHT_ACTIONS/ARKIT_OFFICIAL_V5_HEADING_TURN")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]


def wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def heading_pitch_from_c2w(c2w: np.ndarray) -> tuple[float, float]:
    forward = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = np.degrees(np.arctan2(float(forward[0]), float(forward[2])))
    horiz = float(np.hypot(forward[0], forward[2]))
    pitch = np.degrees(np.arctan2(float(forward[1]), max(horiz, 1e-8)))
    return float(yaw), float(pitch)


@dataclass
class ScenePose:
    scene: str
    scene_dir: Path
    frame_ids: np.ndarray
    c2w: np.ndarray


@dataclass
class Candidate:
    scene: ScenePose
    start: int
    end: int
    label: str
    family: str
    magnitude: float
    motion: dict
    score: float


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


def scene_dirs(root: Path) -> list[Path]:
    return [
        p
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / ".done").exists() and (p / "manifest.json").exists() and (p / "color").is_dir()
    ]


def traj_line_to_c2w(line: str) -> tuple[float, np.ndarray] | None:
    values = [float(x) for x in line.split()]
    if len(values) != 7:
        return None
    timestamp = values[0]
    angle_axis = np.asarray(values[1:4], dtype=np.float64)
    translation = np.asarray(values[4:7], dtype=np.float64)
    r_w_to_cam, _ = cv2.Rodrigues(angle_axis.reshape(3, 1))
    w2cam = np.eye(4, dtype=np.float64)
    w2cam[:3, :3] = r_w_to_cam
    w2cam[:3, 3] = translation
    return timestamp, np.linalg.inv(w2cam)


def parse_traj(path: Path) -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = traj_line_to_c2w(line)
        if item is None:
            continue
        mats.append(item[1])
    return mats


def load_scene_pose(scene_dir: Path) -> ScenePose | None:
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    traj_path = Path(manifest["raw_assets"]["traj"])
    traj_mats = parse_traj(traj_path)
    frame_ids: list[int] = []
    mats: list[np.ndarray] = []
    for rec in manifest.get("frames", []):
        frame_id = int(rec["frame_index"])
        pose_idx = int(rec["pose_index"])
        if pose_idx < 0 or pose_idx >= len(traj_mats):
            continue
        if not (scene_dir / "color" / f"{frame_id:06d}.jpg").exists():
            continue
        frame_ids.append(frame_id)
        mats.append(traj_mats[pose_idx])
    if len(mats) < 80:
        return None
    return ScenePose(scene=scene_dir.name, scene_dir=scene_dir, frame_ids=np.asarray(frame_ids, dtype=np.int32), c2w=np.stack(mats))


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


def motion_between(scene: ScenePose, start: int, end: int) -> dict:
    mat_a = scene.c2w[start]
    mat_b = scene.c2w[end]
    rel_rot = mat_a[:3, :3].T @ mat_b[:3, :3]
    rel_rotvec = rotvec_deg(rel_rot)
    local_delta = mat_a[:3, :3].T @ (mat_b[:3, 3] - mat_a[:3, 3])
    translation = float(np.linalg.norm(mat_b[:3, 3] - mat_a[:3, 3]))
    yaw_a, pitch_a = heading_pitch_from_c2w(mat_a)
    yaw_b, pitch_b = heading_pitch_from_c2w(mat_b)
    return {
        "start_frame_index": int(scene.frame_ids[start]),
        "end_frame_index": int(scene.frame_ids[end]),
        "frame_gap": int(scene.frame_ids[end] - scene.frame_ids[start]),
        "translation_m": translation,
        "local_delta_camera_xyz": [float(x) for x in local_delta],
        "relative_rotvec_deg_xyz": [float(x) for x in rel_rotvec],
        "forward_heading_yaw_delta_deg": wrap_deg(yaw_b - yaw_a),
        "forward_pitch_delta_deg": wrap_deg(pitch_b - pitch_a),
        "abs_rotation_deg": float(np.linalg.norm(rel_rotvec)),
    }


def classify_motion(motion: dict) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta_camera_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    gap = int(motion["frame_gap"])
    trans = float(motion["translation_m"])

    arx, ary, arz = abs(rx), abs(ry), abs(rz)
    rot_primary = max(arx, ary)
    rot_other = max(min(arx, ary), arz)

    # Camera x labels matched user judgment; z front/back labels are flipped in v2.
    if (
        40 <= gap <= 360
        and 0.18 <= trans <= 1.60
        and abs(dy) <= 0.22
        and rot_primary <= 5.5
        and arz <= 5.0
    ):
        if abs(dx) >= abs(dz):
            primary = abs(dx)
            secondary = abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary = abs(dz)
            secondary = abs(dx)
            label = "向前移动" if dz > 0 else "向后移动"
        if primary < 0.24 or primary < 2.8 * max(secondary, abs(dy), 1e-6):
            return None, None, 0.0, 0.0
        score = primary * 80.0 + primary / max(secondary, abs(dy), 1e-6) * 2.5 - (arx + ary + arz) * 8.0
        return label, "translation", primary, score

    # User check: rx up/down is correct. Left/right turn is based on the
    # horizontal heading change of the camera forward vector, not rotvec axes.
    if (
        40 <= gap <= 360
        and trans <= 0.38
        and 24.0 <= arx <= 58.0
        and arx >= 2.4 * max(ary, arz, 1e-6)
        and arz <= 8.0
    ):
        label = "向上转动" if rx > 0 else "向下转动"
        mag = arx
        score = mag * 3.0 - trans * 35.0 - rot_other * 6.0 - arz * 4.0
        return label, "rotation", mag, score

    yaw_delta = float(motion["forward_heading_yaw_delta_deg"])
    pitch_delta = float(motion["forward_pitch_delta_deg"])
    if (
        40 <= gap <= 360
        and trans <= 0.40
        and 24.0 <= abs(yaw_delta) <= 58.0
        and abs(yaw_delta) >= 3.0 * max(abs(pitch_delta), 1e-6)
        and arx <= 10.0
    ):
        label = "向右转动" if yaw_delta > 0 else "向左转动"
        mag = abs(yaw_delta)
        score = mag * 3.0 - trans * 35.0 - abs(pitch_delta) * 8.0 - arx * 4.0
        return label, "rotation", mag, score

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
    mad = float(np.mean(np.abs(grays[0].astype(np.float32) - grays[1].astype(np.float32))))
    return {"frames": frames, "sharpness_min": min(sharpness), "pair_gray_mad": mad}


def save_pair(sample_dir: Path, frames: Sequence[np.ndarray]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs = {}
    smalls = []
    for name, frame in zip(["frame_A", "frame_B"], frames):
        path = sample_dir / f"{name}.png"
        cv2.imwrite(str(path), frame)
        outputs[name] = str(path)
        smalls.append(cv2.resize(frame, (480, 360)))
    contact = sample_dir / "contact_sheet.png"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1))
    outputs["contact_sheet"] = str(contact)
    return outputs


def action_value_text(label: str, family: str, magnitude: float) -> str:
    return f"{label}约{magnitude:.1f}度" if family == "rotation" else f"{label}约{magnitude:.2f}米"


def collect_candidates(scenes: list[ScenePose]) -> dict[str, list[Candidate]]:
    per_label: dict[str, list[Candidate]] = defaultdict(list)
    gaps = [4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36]
    for scene in scenes:
        local: list[Candidate] = []
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(2, gap_slots // 2)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = motion_between(scene, start, end)
                label, family, mag, score = classify_motion(motion)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(mag)
                local.append(Candidate(scene, start, end, label, family, float(mag), motion, float(score)))
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.label] >= 3:
                continue
            per_label[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label[label].sort(key=lambda item: item.score, reverse=True)
            del per_label[label][50:]
        print("[official-arkit-scan]", scene.scene, {k: len(per_label[k]) for k in ACTION_OPTIONS}, flush=True)
        if all(per_label[label] for label in ACTION_OPTIONS):
            break
    return per_label


def build_rows(per_label: dict[str, list[Candidate]], output_root: Path) -> list[dict]:
    rows: list[dict] = []
    used: set[tuple[str, int, int]] = set()
    for label in ACTION_OPTIONS:
        for cand in per_label[label]:
            m = cand.motion
            key = (cand.scene.scene, m["start_frame_index"], m["end_frame_index"])
            if key in used:
                continue
            frame_paths = [
                cand.scene.scene_dir / "color" / f"{m['start_frame_index']:06d}.jpg",
                cand.scene.scene_dir / "color" / f"{m['end_frame_index']:06d}.jpg",
            ]
            quality = image_quality(frame_paths)
            if quality is None or quality["sharpness_min"] < 10 or quality["pair_gray_mad"] < 8:
                continue
            qa_id = f"arkit_official_pose_{len(rows) + 1:03d}"
            paths = save_pair(output_root / "action_inference" / qa_id, quality["frames"])
            rows.append(
                {
                    "qa_id": qa_id,
                    "task_type": "action_inference",
                    "dataset": "arkitscenes",
                    "scene": cand.scene.scene,
                    "pose_source": "arkitscenes_official_traj_axis_angle_translation_inverted_to_c2w",
                    "question": "图A到图B之间，相机主要执行了什么动作？",
                    "answer": cand.label,
                    "answer_with_value": action_value_text(cand.label, cand.family, cand.magnitude),
                    "options": ACTION_OPTIONS,
                    "label_source": "predicted_by_official_pose_conversion_rule_needs_human_check",
                    "input": {
                        "frame_paths": [paths["frame_A"], paths["frame_B"]],
                        "frame_A": paths["frame_A"],
                        "frame_B": paths["frame_B"],
                        "contact_sheet": paths["contact_sheet"],
                    },
                    "gt": {
                        "motion": m,
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
                        "pose_rule_hypothesis": {
                            "traj_columns": "timestamp rx ry rz tx ty tz",
                            "official_conversion": "Rodrigues(axis_angle), put translation in w2cam, invert to camera_to_world",
                            "translation_rule_v2": "+x right, -x left, +z forward, -z backward after user check",
                            "rotation_rule_v5": "+rx up, -rx down; left/right turn from forward-heading yaw delta",
                        },
                    },
                }
            )
            used.add(key)
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=80)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root / "action_inference")

    scenes: list[ScenePose] = []
    for scene_dir in scene_dirs(args.data_root)[: args.max_scenes]:
        scene = load_scene_pose(scene_dir)
        if scene is not None:
            scenes.append(scene)

    per_label = collect_candidates(scenes)
    rows = build_rows(per_label, args.output_root)
    write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    write_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", rows)
    summary = {
        "dataset": "arkitscenes",
        "rule_name": "official_traj_axis_angle_translation_invert_c2w_v5_heading_turn",
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in ACTION_OPTIONS},
        "candidate_counts": {label: len(per_label[label]) for label in ACTION_OPTIONS},
        "sources": [
            "apple/ARKitScenes DATA.md: .traj columns are timestamp, axis-angle rotation, translation",
            "Apple tenFpsDataLoader pattern: Rodrigues(axis_angle), translation, invert extrinsics",
        ],
        "user_feedback_applied": [
            "front/back movement swapped from v1",
            "left/right movement kept from v1",
            "rx up/down rotation kept from v1",
            "left/right turn uses camera forward heading yaw because ry/rz samples looked like up/down",
        ],
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
