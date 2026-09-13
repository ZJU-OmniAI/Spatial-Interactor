#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DATA_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")
OUTPUT_ROOT = Path("/path/to/workspace/DATA/QUICK_EIGHT_ACTIONS/ARKIT_RY_RZ_LARGE_SWEEP")
BUCKETS = ["ry+", "ry-", "rz+", "rz-"]


def retry_eagain(fn, *, attempts: int = 10, delay_s: float = 0.5):
    for attempt in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))


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
    bucket: str
    axis_value: float
    motion: dict
    score: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scene_dirs(root: Path) -> list[Path]:
    return [
        p
        for p in retry_eagain(lambda: sorted(root.iterdir()))
        if retry_eagain(lambda p=p: p.is_dir())
        and retry_eagain(lambda p=p: (p / ".done").exists())
        and retry_eagain(lambda p=p: (p / "manifest.json").exists())
        and retry_eagain(lambda p=p: (p / "color").is_dir())
    ]


def traj_line_to_c2w(line: str) -> np.ndarray | None:
    values = [float(x) for x in line.split()]
    if len(values) != 7:
        return None
    angle_axis = np.asarray(values[1:4], dtype=np.float64)
    translation = np.asarray(values[4:7], dtype=np.float64)
    r_w_to_cam, _ = cv2.Rodrigues(angle_axis.reshape(3, 1))
    w2cam = np.eye(4, dtype=np.float64)
    w2cam[:3, :3] = r_w_to_cam
    w2cam[:3, 3] = translation
    return np.linalg.inv(w2cam)


def parse_traj(path: Path) -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        mat = traj_line_to_c2w(line)
        if mat is not None:
            mats.append(mat)
    return mats


def load_scene_pose(scene_dir: Path) -> ScenePose | None:
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    traj_mats = parse_traj(Path(manifest["raw_assets"]["traj"]))
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
    return ScenePose(scene_dir.name, scene_dir, np.asarray(frame_ids, dtype=np.int32), np.stack(mats))


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


def motion_between(scene: ScenePose, start: int, end: int) -> dict:
    mat_a = scene.c2w[start]
    mat_b = scene.c2w[end]
    rel_rot = mat_a[:3, :3].T @ mat_b[:3, :3]
    rel_rotvec = rotvec_deg(rel_rot)
    local_delta = mat_a[:3, :3].T @ (mat_b[:3, 3] - mat_a[:3, 3])
    return {
        "start_frame_index": int(scene.frame_ids[start]),
        "end_frame_index": int(scene.frame_ids[end]),
        "frame_gap": int(scene.frame_ids[end] - scene.frame_ids[start]),
        "translation_m": float(np.linalg.norm(mat_b[:3, 3] - mat_a[:3, 3])),
        "local_delta_camera_xyz": [float(x) for x in local_delta],
        "relative_rotvec_deg_xyz": [float(x) for x in rel_rotvec],
    }


def classify_axis_candidate(motion: dict) -> tuple[str | None, float, float]:
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    trans = float(motion["translation_m"])
    gap = int(motion["frame_gap"])
    arx, ary, arz = abs(rx), abs(ry), abs(rz)
    if not (40 <= gap <= 420 and trans <= 0.75):
        return None, 0.0, 0.0

    best_axis = "ry" if ary >= arz else "rz"
    axis_value = ry if best_axis == "ry" else rz
    axis_abs = abs(axis_value)
    other = max(arx, arz if best_axis == "ry" else ary)
    if not (28.0 <= axis_abs <= 95.0):
        return None, 0.0, 0.0
    if axis_abs < 1.35 * max(other, 1e-6):
        return None, 0.0, 0.0

    bucket = f"{best_axis}{'+' if axis_value > 0 else '-'}"
    dx, dy, dz = [abs(float(x)) for x in motion["local_delta_camera_xyz"]]
    leakage = max(dx, dy, dz)
    score = axis_abs * 4.0 - other * 8.0 - trans * 35.0 - leakage * 8.0
    return bucket, axis_abs, score


def image_quality(paths: Sequence[Path]) -> dict | None:
    frames = []
    sharpness = []
    grays = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        frames.append(frame)
        small = cv2.resize(frame, (480, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        grays.append(gray)
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


def collect_candidates(scenes: list[ScenePose], per_bucket: int) -> dict[str, list[Candidate]]:
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    gaps = [6, 8, 10, 12, 15, 18, 24, 30, 36, 44]
    for scene in scenes:
        local: list[Candidate] = []
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(2, gap_slots // 2)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = motion_between(scene, start, end)
                bucket, axis_abs, score = classify_axis_candidate(motion)
                if not bucket:
                    continue
                local.append(Candidate(scene, start, end, bucket, axis_abs, motion, score))
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.bucket] >= 2:
                continue
            buckets[cand.bucket].append(cand)
            scene_counts[cand.bucket] += 1
        for bucket in BUCKETS:
            buckets[bucket].sort(key=lambda item: item.score, reverse=True)
            del buckets[bucket][per_bucket * 5 :]
        print("[arkit-ry-rz]", scene.scene, {k: len(buckets[k]) for k in BUCKETS}, flush=True)
        if all(len(buckets[k]) >= per_bucket * 3 for k in BUCKETS):
            break
    return buckets


def make_overview(rows: list[dict], out: Path) -> None:
    panels = []
    for idx, row in enumerate(rows, 1):
        img = cv2.imread(row["input"]["contact_sheet"])
        if img is None:
            continue
        img = cv2.resize(img, (960, 360))
        text = f"{idx}. {row['answer']} {row['answer_with_value']}"
        cv2.putText(img, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
        panels.append(img)
    if not panels:
        return
    lines = []
    for i in range(0, len(panels), 2):
        pair = panels[i : i + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        lines.append(np.concatenate(pair, axis=1))
    cv2.imwrite(str(out / "action_inference" / "overview_contact_sheet.png"), np.concatenate(lines, axis=0))


def write_report(rows: list[dict], out: Path) -> None:
    lines = ["# ARKit ry/rz large rotation sweep", ""]
    lines.append("Official traj conversion: axis-angle+translation as w2cam, then inverted to c2w.")
    lines.append("Buckets are raw candidate signs, not final left/right labels.")
    for idx, row in enumerate(rows, 1):
        m = row["gt"]["motion"]
        lines.extend([
            "",
            f"## {idx}. {row['answer']} {row['answer_with_value']}",
            f"- scene: {row['scene']}",
            f"- frames: {m['start_frame_index']} -> {m['end_frame_index']}, gap={m['frame_gap']}",
            f"- translation_m: {m['translation_m']:.4f}",
            f"- local_delta_camera_xyz: {[round(x, 4) for x in m['local_delta_camera_xyz']]}",
            f"- relative_rotvec_deg_xyz: {[round(x, 3) for x in m['relative_rotvec_deg_xyz']]}",
            f"- contact_sheet: {row['input']['contact_sheet']}",
        ])
    (out / "action_inference" / "ry_rz_candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rows(buckets: dict[str, list[Candidate]], out: Path, per_bucket: int) -> list[dict]:
    rows: list[dict] = []
    used: set[tuple[str, int, int]] = set()
    for bucket in BUCKETS:
        kept = 0
        for cand in buckets[bucket]:
            m = cand.motion
            key = (cand.scene.scene, m["start_frame_index"], m["end_frame_index"])
            if key in used:
                continue
            frame_paths = [
                cand.scene.scene_dir / "color" / f"{m['start_frame_index']:06d}.jpg",
                cand.scene.scene_dir / "color" / f"{m['end_frame_index']:06d}.jpg",
            ]
            q = image_quality(frame_paths)
            if q is None or q["sharpness_min"] < 10 or q["pair_gray_mad"] < 6:
                continue
            qa_id = f"arkit_ry_rz_{len(rows) + 1:03d}"
            paths = save_pair(out / "action_inference" / qa_id, q["frames"])
            rows.append({
                "qa_id": qa_id,
                "task_type": "action_inference",
                "dataset": "arkitscenes",
                "scene": cand.scene.scene,
                "pose_source": "arkitscenes_official_traj_axis_angle_translation_inverted_to_c2w",
                "question": "图A到图B之间，候选旋转轴是哪一种？",
                "answer": bucket,
                "answer_with_value": f"{bucket} 约{cand.axis_value:.1f}度",
                "options": BUCKETS,
                "label_source": "ry_rz_large_axis_candidate_needs_visual_judgment",
                "input": {
                    "frame_paths": [paths["frame_A"], paths["frame_B"]],
                    "frame_A": paths["frame_A"],
                    "frame_B": paths["frame_B"],
                    "contact_sheet": paths["contact_sheet"],
                },
                "gt": {
                    "motion": m,
                    "approx_action": {"text": f"{bucket} 约{cand.axis_value:.1f}度", "value": round(cand.axis_value, 3), "unit": "deg"},
                    "quality": {
                        "pose_score": cand.score,
                        "sharpness_min": q["sharpness_min"],
                        "pair_gray_mad": q["pair_gray_mad"],
                    },
                    "pose_rule_hypothesis": {
                        "rotation_candidate": "only ry/rz dominant large rotations are included",
                        "not_final_action_label": "bucket sign is shown so visual left/right can be judged manually",
                    },
                },
            })
            used.add(key)
            kept += 1
            if kept >= per_bucket:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=50)
    parser.add_argument("--per-bucket", type=int, default=4)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root / "action_inference")

    scenes: list[ScenePose] = []
    for idx, scene_dir in enumerate(scene_dirs(args.data_root)[: args.max_scenes], 1):
        scene = load_scene_pose(scene_dir)
        if scene is not None:
            scenes.append(scene)
        print(f"[load] {idx}/{args.max_scenes} {scene_dir.name} usable={scene is not None}", flush=True)

    buckets = collect_candidates(scenes, args.per_bucket)
    rows = build_rows(buckets, args.output_root, args.per_bucket)
    write_json(args.output_root / "action_inference" / "qa_data.json", rows)
    make_overview(rows, args.output_root)
    write_report(rows, args.output_root)
    summary = {
        "dataset": "arkitscenes",
        "rule_name": "official_ry_rz_large_rotation_sweep",
        "count": len(rows),
        "label_counts": {bucket: sum(1 for row in rows if row["answer"] == bucket) for bucket in BUCKETS},
        "candidate_counts": {bucket: len(buckets[bucket]) for bucket in BUCKETS},
        "output": str(args.output_root / "action_inference" / "qa_data.json"),
        "overview": str(args.output_root / "action_inference" / "overview_contact_sheet.png"),
        "report": str(args.output_root / "action_inference" / "ry_rz_candidate_report.md"),
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
