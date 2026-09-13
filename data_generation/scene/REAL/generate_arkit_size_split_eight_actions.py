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

import generate_arkit_ry_rz_large_sweep as base


DATA_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")
ROOT_OUTPUT = Path("/path/to/workspace/DATA/QUICK_EIGHT_ACTIONS")


@dataclass
class SizedScene:
    scene: base.ScenePose
    width: int
    height: int


@dataclass
class SizedCandidate:
    scene: base.ScenePose
    start: int
    end: int
    label: str
    family: str
    magnitude: float
    motion: dict
    score: float


ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]
EXCLUDE_1920_LR_TURN_SCENES = {"41007589", "40958737"}


def scene_size(scene_dir: Path) -> tuple[int, int]:
    man = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    return int(man["width"]), int(man["height"])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def load_scenes(data_root: Path, max_scenes: int | None = None) -> list[SizedScene]:
    scenes: list[SizedScene] = []
    for idx, scene_dir in enumerate(base.scene_dirs(data_root), 1):
        scene = base.load_scene_pose(scene_dir)
        if scene is None:
            continue
        w, h = scene_size(scene_dir)
        scenes.append(SizedScene(scene, w, h))
        print(f"[load] {idx} {scene.scene} {w}x{h}", flush=True)
        if max_scenes is not None and len(scenes) >= max_scenes:
            break
    return scenes


def classify_armed(size: tuple[int, int], motion: dict) -> tuple[str | None, str | None, float, float]:
    width, height = size
    dx, dy, dz = [float(x) for x in motion["local_delta_camera_xyz"]]
    rx, ry, rz = [float(x) for x in motion["relative_rotvec_deg_xyz"]]
    gap = int(motion["frame_gap"])
    trans = float(motion["translation_m"])
    arx, ary, arz = abs(rx), abs(ry), abs(rz)
    rot_primary = max(arx, ary)
    rot_other = max(min(arx, ary), arz)

    if width > height:
        if 40 <= gap <= 360 and 0.18 <= trans <= 1.60 and abs(dy) <= 0.22 and rot_primary <= 6.0 and arz <= 8.0:
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
            score = primary * 80.0 + primary / max(max(secondary, abs(dy)), 1e-6) * 2.5 - (arx + ary + arz) * 6.0
            return label, "translation", primary, score
    else:
        if 40 <= gap <= 360 and 0.18 <= trans <= 1.60 and abs(dx) <= 0.26 and rot_primary <= 6.0 and arz <= 8.0:
            if abs(dy) >= abs(dz):
                primary = abs(dy)
                secondary = abs(dz)
                label = "向右移动" if dy > 0 else "向左移动"
            else:
                primary = abs(dz)
                secondary = abs(dy)
                label = "向前移动" if dz > 0 else "向后移动"
            if primary < 0.24 or primary < 2.4 * max(secondary, abs(dx), 1e-6):
                return None, None, 0.0, 0.0
            score = primary * 80.0 + primary / max(max(secondary, abs(dx)), 1e-6) * 2.5 - (arx + ary + arz) * 6.0
            return label, "translation", primary, score

    if width > height:
        if 40 <= gap <= 360 and trans <= 0.45 and 34.0 <= ary <= 58.0 and ary >= 2.5 * max(arx, arz, 1e-6):
            label = "向左转动" if ry < 0 else "向右转动"
            score = 160.0 - abs(ary - 45.0) * 5.0 - trans * 35.0 - rot_other * 6.0
            return label, "rotation", ary, score
        if 40 <= gap <= 360 and trans <= 0.45 and 34.0 <= arx <= 58.0 and arx >= 2.5 * max(ary, arz, 1e-6):
            label = "向上转动" if rx > 0 else "向下转动"
            score = 160.0 - abs(arx - 45.0) * 3.0 - trans * 35.0 - rot_other * 6.0
            return label, "rotation", arx, score
    else:
        if 40 <= gap <= 360 and trans <= 0.45 and 34.0 <= ary <= 58.0 and ary >= 2.5 * max(arx, arz, 1e-6):
            label = "向上转动" if ry < 0 else "向下转动"
            score = 160.0 - abs(ary - 45.0) * 3.0 - trans * 35.0 - rot_other * 6.0
            return label, "rotation", ary, score
        if 40 <= gap <= 360 and trans <= 0.45 and 34.0 <= arx <= 58.0 and arx >= 2.5 * max(ary, arz, 1e-6):
            label = "向左转动" if rx < 0 else "向右转动"
            score = 160.0 - abs(arx - 45.0) * 3.0 - trans * 35.0 - rot_other * 6.0
            return label, "rotation", arx, score

    return None, None, 0.0, 0.0


def collect_for_size(scenes: list[SizedScene], width: int, height: int, per_label: int) -> dict[str, list[SizedCandidate]]:
    per_label_cands: dict[str, list[SizedCandidate]] = defaultdict(list)
    gaps = [6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 42]
    for sized in scenes:
        if sized.width != width or sized.height != height:
            continue
        local: list[SizedCandidate] = []
        scene = sized.scene
        for gap_slots in gaps:
            if gap_slots >= len(scene.frame_ids):
                continue
            step = max(2, gap_slots // 2)
            for start in range(0, len(scene.frame_ids) - gap_slots, step):
                end = start + gap_slots
                motion = base.motion_between(scene, start, end)
                label, family, mag, score = classify_armed((width, height), motion)
                if not label:
                    continue
                if width == 1920 and height == 1440 and label in {"向左转动", "向右转动"} and scene.scene in EXCLUDE_1920_LR_TURN_SCENES:
                    continue
                local.append(SizedCandidate(scene, start, end, label, family, float(mag), motion, float(score)))
        local.sort(key=lambda item: item.score, reverse=True)
        scene_counts: dict[str, int] = defaultdict(int)
        for cand in local:
            if scene_counts[cand.label] >= 1:
                continue
            per_label_cands[cand.label].append(cand)
            scene_counts[cand.label] += 1
        for label in ACTION_OPTIONS:
            per_label_cands[label].sort(key=lambda item: item.score, reverse=True)
            del per_label_cands[label][per_label * 4 :]
        print(f"[scan] {scene.scene} {width}x{height}", {k: len(per_label_cands[k]) for k in ACTION_OPTIONS}, flush=True)
        if all(len(per_label_cands[label]) >= per_label for label in ACTION_OPTIONS):
            break
    return per_label_cands


def build_rows(per_label: dict[str, list[SizedCandidate]], out: Path, size_tag: str, per_label_limit: int) -> list[dict]:
    rows: list[dict] = []
    used: set[tuple[str, int, int]] = set()
    for label in ACTION_OPTIONS:
        kept = 0
        for cand in per_label[label]:
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
            qa_id = f"arkit_{size_tag}_{len(rows) + 1:03d}"
            paths = save_pair(out / "action_inference" / qa_id, q["frames"])
            rows.append(
                {
                    "qa_id": qa_id,
                    "task_type": "action_inference",
                    "dataset": "arkitscenes",
                    "scene": cand.scene.scene,
                    "pose_source": "arkitscenes_official_traj_axis_angle_translation_inverted_to_c2w",
                    "question": "图A到图B之间，相机主要执行了什么动作？",
                    "answer": label,
                    "answer_with_value": (
                        f"{label}约{cand.magnitude:.1f}度" if cand.family == "rotation" else f"{label}约{cand.magnitude:.2f}米"
                    ),
                    "options": ACTION_OPTIONS,
                    "label_source": "predicted_by_size_split_rule_needs_human_check",
                    "input": {
                        "frame_paths": [paths["frame_A"], paths["frame_B"]],
                        "frame_A": paths["frame_A"],
                        "frame_B": paths["frame_B"],
                        "contact_sheet": paths["contact_sheet"],
                    },
                    "gt": {
                        "motion": m,
                        "approx_action": {
                            "text": (
                                f"{label}约{cand.magnitude:.1f}度" if cand.family == "rotation" else f"{label}约{cand.magnitude:.2f}米"
                            ),
                            "value": round(cand.magnitude, 3),
                            "unit": "deg" if cand.family == "rotation" else "m",
                        },
                        "quality": {
                            "pose_score": cand.score,
                            "sharpness_min": q["sharpness_min"],
                            "pair_gray_mad": q["pair_gray_mad"],
                        },
                        "pose_rule_hypothesis": {
                            "size_tag": size_tag,
                            "rotation_mapping": "size-specific; see summary.json",
                            "translation_mapping": "dx/dz kept as before",
                        },
                    },
                }
            )
            used.add(key)
            kept += 1
            if kept >= per_label_limit:
                break
    return rows


def make_overview(rows: list[dict], out: Path) -> None:
    panels = []
    for idx, row in enumerate(rows, 1):
        img = cv2.imread(row["input"]["contact_sheet"])
        if img is None:
            continue
        img = cv2.resize(img, (960, 360))
        text = f"{idx}. {row['answer']} {row['answer_with_value']}"
        cv2.putText(img, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 220), 2, cv2.LINE_AA)
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


def run_split(size_tag: str, width: int, height: int, scenes: list[SizedScene], per_label: int) -> dict:
    out = ROOT_OUTPUT / f"ARKIT_SIZE_RULE_V2_{size_tag}"
    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out / "action_inference")
    per_label_cands = collect_for_size(scenes, width, height, per_label)
    rows = build_rows(per_label_cands, out, size_tag, per_label)
    write_json(out / "action_inference" / "qa_data.json", rows)
    make_overview(rows, out)
    summary = {
        "dataset": "arkitscenes",
        "size_tag": size_tag,
        "width": width,
        "height": height,
        "count": len(rows),
        "label_counts": {label: sum(1 for row in rows if row["answer"] == label) for label in ACTION_OPTIONS},
        "candidate_counts": {label: len(per_label_cands[label]) for label in ACTION_OPTIONS},
        "excluded_1920_lr_turn_scenes": sorted(EXCLUDE_1920_LR_TURN_SCENES) if width == 1920 and height == 1440 else [],
        "overview": str(out / "action_inference" / "overview_contact_sheet.png"),
        "qa_data": str(out / "action_inference" / "qa_data.json"),
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--max-scenes", type=int, default=140)
    parser.add_argument("--per-label", type=int, default=1)
    args = parser.parse_args()

    scenes = load_scenes(args.data_root, args.max_scenes)
    summaries = {
        "ARKIT_SIZE_RULE_V2_1920x1440": run_split("1920x1440", 1920, 1440, scenes, args.per_label),
        "ARKIT_SIZE_RULE_V2_1440x1920": run_split("1440x1920", 1440, 1920, scenes, args.per_label),
    }
    write_json(ROOT_OUTPUT / "ARKIT_SIZE_RULE_V2_SUMMARY.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
