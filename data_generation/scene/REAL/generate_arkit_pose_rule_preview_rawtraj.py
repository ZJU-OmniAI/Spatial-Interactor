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
OUTPUT_ROOT = Path("/path/to/workspace/DATA/ARKIT_ACTION_PREVIEW")
ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]


@dataclass
class ScenePose:
    scene: str
    scene_dir: Path
    frame_ids: np.ndarray
    mats: np.ndarray


@dataclass
class Candidate:
    scene: str
    scene_dir: Path
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


def scene_dirs(root: Path) -> list[Path]:
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and (p / ".done").exists() and (p / "manifest.json").exists() and (p / "color").is_dir()
    ]


def parse_traj(path: Path, traj_order: str = "tx_ty_tz_rx_ry_rz") -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) != 7:
            continue
        if traj_order == "tx_ty_tz_rx_ry_rz":
            trans = np.asarray(values[1:4], dtype=np.float64)
            rotvec = np.asarray(values[4:7], dtype=np.float64)
        elif traj_order == "rx_ry_rz_tx_ty_tz":
            rotvec = np.asarray(values[1:4], dtype=np.float64)
            trans = np.asarray(values[4:7], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported traj_order: {traj_order}")
        rot, _ = cv2.Rodrigues(rotvec.reshape(3, 1))
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = trans
        mats.append(mat)
    return mats


def load_scene_pose(scene_dir: Path, inverse: bool = False, traj_order: str = "tx_ty_tz_rx_ry_rz") -> ScenePose | None:
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    traj_path = Path(manifest["raw_assets"]["traj"])
    traj_mats = parse_traj(traj_path, traj_order=traj_order)
    frame_ids: list[int] = []
    mats: list[np.ndarray] = []
    for rec in manifest.get("frames", []):
        frame_id = int(rec["frame_index"])
        pose_idx = int(rec["pose_index"])
        if pose_idx < 0 or pose_idx >= len(traj_mats):
            continue
        if not (scene_dir / "color" / f"{frame_id:06d}.jpg").exists():
            continue
        mat = traj_mats[pose_idx]
        if inverse:
            mat = np.linalg.inv(mat)
        frame_ids.append(frame_id)
        mats.append(mat)
    if len(mats) < 80:
        return None
    print(f"[pose] loaded {scene_dir.name} frames={len(mats)} inverse={inverse}", flush=True)
    return ScenePose(scene=scene_dir.name, scene_dir=scene_dir, frame_ids=np.asarray(frame_ids, dtype=np.int32), mats=np.stack(mats))


def rotvec_deg(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation)
    return np.degrees(rotvec.reshape(3))


def motion_between(scene: ScenePose, start: int, end: int, local_rule: str) -> dict:
    mat_a = scene.mats[start]
    mat_b = scene.mats[end]
    rel_rot = mat_a[:3, :3].T @ mat_b[:3, :3]
    rel_rotvec = rotvec_deg(rel_rot)
    local_delta = mat_a[:3, :3].T @ (mat_b[:3, 3] - mat_a[:3, 3])
    if local_rule == "flip_z":
        local_delta = local_delta.copy()
        local_delta[2] *= -1.0
    elif local_rule == "flip_x":
        local_delta = local_delta.copy()
        local_delta[0] *= -1.0
    translation = float(np.linalg.norm(mat_b[:3, 3] - mat_a[:3, 3]))
    return {
        "start_frame_index": int(scene.frame_ids[start]),
        "end_frame_index": int(scene.frame_ids[end]),
        "frame_gap": int(scene.frame_ids[end] - scene.frame_ids[start]),
        "translation_m": translation,
        "local_delta_hypothesis_xyz": [float(x) for x in local_delta],
        "relative_rotvec_deg_xyz": [float(x) for x in rel_rotvec],
        "abs_rotation_deg": float(np.linalg.norm(rel_rotvec)),
    }


def classify_motion(motion: dict, rot_rule: str) -> tuple[str | None, str | None, float, float]:
    dx, dy, dz = [float(x) for x in motion["local_delta_hypothesis_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    if rot_rule == "flip_yaw":
        ry = -ry
    if rot_rule == "flip_pitch":
        rx = -rx
    gap = int(motion["frame_gap"])
    translation = float(motion["translation_m"])
    rot_primary = max(abs(rx), abs(ry))
    rot_other = max(min(abs(rx), abs(ry)), abs(rz))

    if 50 <= gap <= 320 and translation <= 0.45 and abs(dy) <= 0.35 and 25.0 <= rot_primary <= 60.0 and rot_primary >= 2.2 * max(rot_other, 1e-6) and abs(rz) <= 10.0:
        if abs(rx) >= abs(ry):
            label = "向上转动" if rx > 0 else "向下转动"
            magnitude = abs(rx)
        else:
            label = "向右转动" if ry > 0 else "向左转动"
            magnitude = abs(ry)
        score = magnitude * 2.0 - translation * 15.0 - rot_other + 0.01 * gap
        return label, "rotation", magnitude, score

    if 50 <= gap <= 360 and 0.35 <= translation <= 2.0 and abs(dy) <= 0.35 and rot_primary <= 10.0 and abs(rz) <= 7.0:
        if abs(dx) >= abs(dz):
            primary = abs(dx)
            secondary = abs(dz)
            label = "向右移动" if dx > 0 else "向左移动"
        else:
            primary = abs(dz)
            secondary = abs(dx)
            label = "向前移动" if dz > 0 else "向后移动"
        if not (0.35 <= primary <= 2.0 and primary >= 2.6 * max(secondary, abs(dy), 1e-6)):
            return None, None, 0.0, 0.0
        score = primary * 30.0 + primary / max(secondary, abs(dy), 1e-6) * 2.0 - rot_primary * 2.0 - abs(rz)
        return label, "translation", primary, score
    return None, None, 0.0, 0.0


def image_quality(paths: Sequence[Path]) -> dict | None:
    grays: list[np.ndarray] = []
    sharpness: list[float] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            return None
        small = cv2.resize(image, (480, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    pair_mads = [
        float(np.mean(np.abs(grays[i].astype(np.float32) - grays[i + 1].astype(np.float32))))
        for i in range(len(grays) - 1)
    ]
    return {"sharpness_min": min(sharpness), "pair_gray_mad": pair_mads}


def save_frames(sample_dir: Path, frame_paths: Sequence[Path]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs: dict[str, str] = {}
    smalls: list[np.ndarray] = []
    for name, src in zip(["frame_A", "frame_B"], frame_paths):
        image = cv2.imread(str(src))
        if image is None:
            raise RuntimeError(f"bad image: {src}")
        dst = sample_dir / f"{name}.png"
        cv2.imwrite(str(dst), image)
        outputs[name] = str(dst)
        smalls.append(cv2.resize(image, (480, 360)))
    contact = sample_dir / "contact_sheet.png"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1))
    outputs["contact_sheet"] = str(contact)
    return outputs


def action_value_text(label: str, family: str, magnitude: float) -> str:
    return f"{label}约{magnitude:.1f}度" if family == "rotation" else f"{label}约{magnitude:.2f}米"


def collect_candidates(scenes: list[ScenePose], local_rule: str, rot_rule: str) -> dict[str, list[Candidate]]:
    per_label: dict[str, list[Candidate]] = defaultdict(list)
    gaps = [5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36]
    for scene in scenes:
        local: list[Candidate] = []
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(3, gap_slots // 2)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = motion_between(scene, start, end, local_rule=local_rule)
                label, family, magnitude, score = classify_motion(motion, rot_rule=rot_rule)
                if not label or not family:
                    continue
                motion["predicted_family"] = family
                motion["predicted_magnitude"] = float(magnitude)
                local.append(Candidate(scene.scene, scene.scene_dir, start, end, label, family, magnitude, motion, score))
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.label] >= 3:
                continue
            per_label[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label[label].sort(key=lambda item: item.score, reverse=True)
            del per_label[label][30:]
        print("[scan] %s labels=%s" % (scene.scene, {k: len(per_label[k]) for k in ACTION_OPTIONS}), flush=True)
        if all(per_label[label] for label in ACTION_OPTIONS):
            break
    return per_label


def build_rows(per_label: dict[str, list[Candidate]], output_root: Path, rule_name: str, local_rule: str, rot_rule: str, inverse: bool) -> list[dict]:
    rows: list[dict] = []
    used: set[tuple[str, int, int]] = set()
    for label in ACTION_OPTIONS:
        for cand in per_label[label]:
            key = (cand.scene, cand.motion["start_frame_index"], cand.motion["end_frame_index"])
            if key in used:
                continue
            frame_paths = [
                cand.scene_dir / "color" / f"{cand.motion['start_frame_index']:06d}.jpg",
                cand.scene_dir / "color" / f"{cand.motion['end_frame_index']:06d}.jpg",
            ]
            quality = image_quality(frame_paths)
            if quality is None or quality["sharpness_min"] < 10 or quality["pair_gray_mad"][0] < 8:
                continue
            qa_id = f"arkit_{rule_name}_{len(rows) + 1:03d}"
            sample_dir = output_root / "action_inference" / qa_id
            paths = save_frames(sample_dir, frame_paths)
            rows.append(
                {
                    "qa_id": qa_id,
                    "task_type": "action_inference",
                    "dataset": "arkitscenes",
                    "scene": cand.scene,
                    "pose_source": f"arkitscenes_raw_traj_tx_ty_tz_rx_ry_rz_{rule_name}",
                    "question": "图A到图B之间，相机主要执行了什么动作？",
                    "answer": cand.label,
                    "answer_with_value": action_value_text(cand.label, cand.family, cand.magnitude),
                    "options": ACTION_OPTIONS,
                    "label_source": "predicted_by_pose_rule_needs_human_correction",
                    "input": {"frame_paths": [paths["frame_A"], paths["frame_B"]], "frame_A": paths["frame_A"], "frame_B": paths["frame_B"], "contact_sheet": paths["contact_sheet"]},
                    "gt": {
                        "motion": cand.motion,
                        "approx_action": {"text": action_value_text(cand.label, cand.family, cand.magnitude), "value": round(cand.magnitude, 3), "unit": "deg" if cand.family == "rotation" else "m"},
                        "quality": {"pose_score": cand.score, **quality},
                        "pose_rule_hypothesis": {"traj_order": "timestamp tx ty tz rx ry rz", "inverse_matrix": inverse, "local_rule": local_rule, "rot_rule": rot_rule},
                    },
                }
            )
            used.add(key)
            break
    return rows


def make_overview(output_root: Path) -> None:
    rows = json.loads((output_root / "action_inference" / "qa_data.json").read_text(encoding="utf-8"))
    imgs = []
    for i, row in enumerate(rows, 1):
        image = cv2.imread(row["input"]["contact_sheet"])
        if image is None:
            continue
        image = cv2.resize(image, (960, 360))
        value = row["gt"]["approx_action"]["value"]
        unit = row["gt"]["approx_action"]["unit"]
        cv2.putText(image, f"{i}. {row['answer']} {value}{unit}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        imgs.append(image)
    if not imgs:
        return
    rows_img = []
    for idx in range(0, len(imgs), 2):
        pair = imgs[idx: idx + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows_img.append(np.concatenate(pair, axis=1))
    cv2.imwrite(str(output_root / "action_inference" / "overview_contact_sheet.png"), np.concatenate(rows_img, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--local-rule", choices=["none", "flip_z", "flip_x"], default="flip_z")
    parser.add_argument("--rot-rule", choices=["none", "flip_yaw", "flip_pitch"], default="flip_yaw")
    parser.add_argument("--traj-order", choices=["tx_ty_tz_rx_ry_rz", "rx_ry_rz_tx_ty_tz"], default="tx_ty_tz_rx_ry_rz")
    parser.add_argument("--rule-name", default="")
    args = parser.parse_args()

    rule_name = args.rule_name or f"inv{int(args.inverse)}_{args.local_rule}_{args.rot_rule}"
    output_root = args.output_root
    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root / "action_inference")

    loaded: list[ScenePose] = []
    for scene_dir in scene_dirs(args.data_root)[: args.max_scenes]:
        scene = load_scene_pose(scene_dir, inverse=args.inverse, traj_order=args.traj_order)
        if scene is not None:
            loaded.append(scene)
    per_label = collect_candidates(loaded, local_rule=args.local_rule, rot_rule=args.rot_rule)
    rows = build_rows(per_label, output_root, rule_name, args.local_rule, args.rot_rule, args.inverse)
    write_json(output_root / "action_inference" / "qa_data.json", rows)
    write_jsonl(output_root / "action_inference" / "qa_data.jsonl", rows)
    make_overview(output_root)
    summary = {
        "dataset": "arkitscenes",
        "rule_name": rule_name,
        "output_root": str(output_root),
        "scenes_loaded": len(loaded),
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in ACTION_OPTIONS},
        "candidate_counts": {label: len(per_label[label]) for label in ACTION_OPTIONS},
        "inverse": args.inverse,
        "local_rule": args.local_rule,
        "rot_rule": args.rot_rule,
        "traj_order": args.traj_order,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
