#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SCAN_ROOT = Path("/path/to/workspace/MultiScan/scans")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/MULTISCAN_POSE_RULE_PREVIEW")
CACHE_ROOT = Path("/path/to/workspace/MultiScan_preview_cache")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]


@dataclass
class ScenePose:
    scene: str
    zip_path: Path
    frame_ids: np.ndarray
    mats: np.ndarray


@dataclass
class Candidate:
    scene: str
    zip_path: Path
    start_slot: int
    end_slot: int
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


def matrix_from_multiscan_transform(values: Sequence[float]) -> np.ndarray | None:
    if len(values) != 16:
        return None
    # MultiScan/ARKit stores the matrix flattened with translation in the last row.
    # Transpose it to the standard homogeneous camera-to-world layout.
    mat = np.asarray(values, dtype=np.float64).reshape(4, 4).T
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        return None
    mat[3] = [0.0, 0.0, 0.0, 1.0]
    return mat


def load_scene_pose(zip_path: Path, max_frames: int = 0) -> ScenePose | None:
    scene = zip_path.stem
    jsonl_name = f"{scene}/{scene}.jsonl"
    frames: list[int] = []
    mats: list[np.ndarray] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(jsonl_name) as handle:
                for idx, raw_line in enumerate(handle):
                    if max_frames > 0 and idx >= max_frames:
                        break
                    try:
                        item = json.loads(raw_line)
                    except Exception:
                        continue
                    mat = matrix_from_multiscan_transform(item.get("transform") or [])
                    if mat is None:
                        continue
                    frames.append(idx)
                    mats.append(mat)
    except Exception as exc:
        print(f"[pose-skip] {zip_path.name}: {exc}", flush=True)
        return None
    if len(mats) < 180:
        return None
    return ScenePose(scene=scene, zip_path=zip_path, frame_ids=np.asarray(frames, dtype=np.int32), mats=np.stack(mats))


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


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
        "time_gap_sec_approx": float((scene.frame_ids[end] - scene.frame_ids[start]) / 60.0),
        "translation_m": translation,
        "local_delta_hypothesis_xyz": [float(x) for x in local_delta],
        "relative_rotvec_deg_xyz": [float(x) for x in rel_rotvec],
        "abs_rotation_deg": float(np.linalg.norm(rel_rotvec)),
    }


def classify_motion(motion: dict) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta_hypothesis_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    gap = int(motion["frame_gap"])
    translation = float(motion["translation_m"])
    rot_primary = max(abs(rx), abs(ry))
    rot_other = max(min(abs(rx), abs(ry)), abs(rz))

    if 30 <= gap <= 220 and translation <= 0.35 and abs(dy) <= 0.28 and 28.0 <= rot_primary <= 55.0 and rot_primary >= 2.4 * max(rot_other, 1e-6) and abs(rz) <= 8.0:
        if abs(rx) >= abs(ry):
            label = "向上转动" if rx > 0 else "向下转动"
            magnitude = abs(rx)
        else:
            label = "向左转动" if ry > 0 else "向右转动"
            magnitude = abs(ry)
        score = magnitude * 2.0 + 0.02 * gap - translation * 12.0 - rot_other
        return label, "rotation", magnitude, score

    if 45 <= gap <= 260 and 0.35 <= translation <= 1.8 and abs(dy) <= 0.25 and rot_primary <= 8.0 and abs(rz) <= 5.0:
        # Human-corrected MultiScan hypothesis: local +x=right, +z=backward.
        if abs(dx) >= abs(dz):
            primary = abs(dx)
            secondary = abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary = abs(dz)
            secondary = abs(dx)
            label = "向后移动" if dz > 0 else "向前移动"
        if not (0.35 <= primary <= 1.8 and primary >= 3.0 * max(secondary, abs(dy), 1e-6)):
            return None, None, 0.0, 0.0
        score = primary * 30.0 + primary / max(secondary, abs(dy), 1e-6) * 2.0 - rot_primary * 2.0 - abs(rz)
        return label, "translation", primary, score
    return None, None, 0.0, 0.0


def extract_mp4(zip_path: Path, cache_root: Path) -> Path | None:
    ensure_dir(cache_root)
    scene = zip_path.stem
    out = cache_root / f"{scene}.mp4"
    if out.exists() and out.stat().st_size > 1024 * 1024:
        return out
    tmp = out.with_suffix(".mp4.tmp")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            member = f"{scene}/{scene}.mp4"
            with zf.open(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        tmp.replace(out)
        return out
    except Exception as exc:
        print(f"[mp4-skip] {zip_path.name}: {exc}", flush=True)
        tmp.unlink(missing_ok=True)
        return None


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def load_frames(zip_path: Path, frame_indices: Sequence[int], cache_root: Path) -> tuple[list[np.ndarray], dict] | None:
    mp4 = extract_mp4(zip_path, cache_root)
    if mp4 is None:
        return None
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    sharpness: list[float] = []
    try:
        for frame_index in frame_indices:
            frame = read_frame(cap, int(frame_index))
            if frame is None:
                return None
            frames.append(frame)
            small = cv2.resize(frame, (480, 640))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    finally:
        cap.release()
    pair_mads = [
        float(np.mean(np.abs(grays[i].astype(np.float32) - grays[i + 1].astype(np.float32))))
        for i in range(len(grays) - 1)
    ]
    return frames, {"sharpness_min": min(sharpness), "pair_gray_mad": pair_mads}


def save_frames(sample_dir: Path, names_and_frames: Sequence[tuple[str, np.ndarray]]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs: dict[str, str] = {}
    smalls: list[np.ndarray] = []
    for name, frame in names_and_frames:
        path = sample_dir / f"{name}.png"
        cv2.imwrite(str(path), frame)
        outputs[name] = str(path)
        smalls.append(cv2.resize(frame, (360, 480)))
    contact = sample_dir / "contact_sheet.png"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1))
    outputs["contact_sheet"] = str(contact)
    return outputs


def action_value_text(label: str, family: str, magnitude: float) -> str:
    if family == "rotation":
        return f"{label}约{magnitude:.1f}度"
    return f"{label}约{magnitude:.2f}米"


def collect_candidates(scan_root: Path, max_scenes: int, max_frames_per_scene: int) -> dict[str, list[Candidate]]:
    per_label: dict[str, list[Candidate]] = defaultdict(list)
    zips = [p for p in sorted(scan_root.glob("scene_*.zip")) if p.stat().st_size >= 1024 * 1024]
    gaps = [30, 45, 60, 80, 100, 130, 160, 200, 240]
    loaded = 0
    for zip_path in zips:
        if loaded >= max_scenes:
            break
        scene = load_scene_pose(zip_path, max_frames=max_frames_per_scene)
        if scene is None:
            continue
        loaded += 1
        local: list[Candidate] = []
        for gap in gaps:
            if gap >= len(scene.frame_ids):
                continue
            step = max(12, gap // 2)
            for start in range(0, len(scene.frame_ids) - gap, step):
                end = start + gap
                motion = motion_between(scene, start, end)
                label, family, magnitude, score = classify_motion(motion)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(Candidate(scene.scene, zip_path, start, end, label, family, magnitude, motion, score))
        local.sort(key=lambda x: x.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.label] >= 3:
                continue
            per_label[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label[label].sort(key=lambda x: x.score, reverse=True)
            del per_label[label][25:]
        print("[scan] %s labels=%s" % (scene.scene, {k: len(per_label[k]) for k in ACTION_OPTIONS}), flush=True)
        if all(per_label[label] for label in ACTION_OPTIONS):
            break
    return per_label


def build_preview(per_label: dict[str, list[Candidate]], output_root: Path, cache_root: Path) -> list[dict]:
    rows: list[dict] = []
    used: set[tuple[str, int, int]] = set()
    for label in ACTION_OPTIONS:
        for cand in per_label[label]:
            key = (cand.scene, cand.motion["start_frame_index"], cand.motion["end_frame_index"])
            if key in used:
                continue
            loaded = load_frames(cand.zip_path, [cand.motion["start_frame_index"], cand.motion["end_frame_index"]], cache_root)
            if loaded is None:
                continue
            frames, quality = loaded
            if quality["sharpness_min"] < 20 or quality["pair_gray_mad"][0] < 8:
                continue
            qa_id = f"multiscan_pose_rule_{len(rows) + 1:03d}"
            sample_dir = output_root / "action_inference" / qa_id
            paths = save_frames(sample_dir, [("frame_A", frames[0]), ("frame_B", frames[1])])
            rows.append(
                {
                    "qa_id": qa_id,
                    "task_type": "action_inference",
                    "dataset": "multiscan",
                    "scene": cand.scene,
                    "pose_source": "multiscan_jsonl_transform_transposed_camera_to_world_hypothesis_v1",
                    "question": "图A到图B之间，相机主要执行了什么动作？",
                    "answer": cand.label,
                    "answer_with_value": action_value_text(cand.label, cand.family, cand.magnitude),
                    "options": ACTION_OPTIONS,
                    "label_source": "predicted_by_pose_rule_needs_human_correction",
                    "input": {
                        "frame_paths": [paths["frame_A"], paths["frame_B"]],
                        "frame_A": paths["frame_A"],
                        "frame_B": paths["frame_B"],
                        "contact_sheet": paths["contact_sheet"],
                    },
                    "gt": {
                        "motion": cand.motion,
                        "approx_action": {
                            "text": action_value_text(cand.label, cand.family, cand.magnitude),
                            "value": round(cand.magnitude, 3),
                            "unit": "deg" if cand.family == "rotation" else "m",
                        },
                        "quality": {"pose_score": cand.score, **quality},
                        "pose_rule_hypothesis": {
                            "transform_layout": "np.array(transform).reshape(4,4).T",
                            "matrix_type": "camera_to_world_hypothesis",
                            "local_delta": "R_a.T @ (t_b - t_a)",
                            "relative_rotation": "R_a.T @ R_b",
                            "translation_label": "+x right, -x left, +z backward, -z forward",
                            "rotation_label": "+rx up, -rx down, +ry left, -ry right",
                        },
                    },
                }
            )
            used.add(key)
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=SCAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--max-scenes", type=int, default=35)
    parser.add_argument("--max-frames-per-scene", type=int, default=4500)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root / "action_inference")
    per_label = collect_candidates(args.scan_root, args.max_scenes, args.max_frames_per_scene)
    rows = build_preview(per_label, args.output_root, args.cache_root)
    write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    write_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", rows)
    summary = {
        "dataset": "multiscan",
        "output_root": str(args.output_root),
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in ACTION_OPTIONS},
        "candidate_counts": {label: len(per_label[label]) for label in ACTION_OPTIONS},
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
