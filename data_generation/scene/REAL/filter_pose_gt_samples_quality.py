#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None


ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/pose_gt_samples")
DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/pose_gt_samples_quality_filtered_v1")
DEFAULT_MODEL = Path("/path/to/workspace/models/Qwen3.5-9B")
TASKS = [
    "action_inference",
    "movement_sequence_sorting",
    "movement_degree_comparison",
    "motion_family_discrimination",
]

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from generate_pose_gt_samples import heading_elevation_series, load_pose_scene, segment_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter REAL first-stage pose_gt_samples with deterministic quality rules.")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--min-sharpness", type=float, default=28.0)
    parser.add_argument("--min-std", type=float, default=0.035)
    parser.add_argument("--min-action-pair-diff", type=float, default=0.018)
    parser.add_argument("--min-sequence-adj-diff", type=float, default=0.012)
    parser.add_argument("--min-degree-pair-diff", type=float, default=0.018)
    parser.add_argument("--min-family-pair-diff", type=float, default=0.016)
    parser.add_argument("--hard-floor-elevation-deg", type=float, default=-50.0)
    parser.add_argument("--hard-floor-bias", type=float, default=0.45)
    parser.add_argument("--suspect-floor-elevation-deg", type=float, default=-22.0)
    parser.add_argument("--suspect-floor-bias", type=float, default=0.28)
    parser.add_argument("--use-vlm-start-audit", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=1)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gray(path: str, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def image_std(image: np.ndarray) -> float:
    return float(np.std(image.astype(np.float32)) / 255.0)


def image_sharpness(path: str) -> float:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def dominant_axis_ratio(local_delta: list[float]) -> tuple[str | None, float, float]:
    if len(local_delta) < 3:
        return None, 0.0, 0.0
    tx = abs(float(local_delta[0]))
    tz = abs(float(local_delta[2]))
    primary = max(tx, tz)
    secondary = min(tx, tz)
    ratio = primary / max(secondary, 1e-6)
    if primary < 1e-6:
        return None, primary, ratio
    axis = "x" if tx >= tz else "z"
    return axis, primary, ratio


def translation_direction_key(local_delta: list[float]) -> str | None:
    axis, primary, ratio = dominant_axis_ratio(local_delta)
    if axis is None or primary < 0.10 or ratio < 1.12:
        return None
    if axis == "x":
        return "right" if float(local_delta[0]) > 0 else "left"
    return "forward" if float(local_delta[2]) > 0 else "backward"


def start_frame_path(row: dict[str, Any]) -> str:
    data = row["input"]
    for key in ["first_frame", "start_frame", "frame_A"]:
        value = data.get(key)
        if isinstance(value, str):
            return value
    frame_paths = data.get("frame_paths") or []
    if frame_paths:
        return str(frame_paths[0])
    raise KeyError(f"no_start_frame:{row.get('qa_id')}")


def start_slot_1based(row: dict[str, Any]) -> int:
    gt = row["gt"]
    if "start_slot" in gt:
        return int(gt["start_slot"])
    if "slot_A" in gt:
        return int(gt["slot_A"])
    raise KeyError(f"no_start_slot:{row.get('qa_id')}")


def replace_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [replace_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, path_map) for key, item in value.items()}
    return value


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


@dataclass
class SceneCacheEntry:
    scene_obj: Any
    elevations: list[float]


class SceneCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], SceneCacheEntry] = {}

    def elevations(self, dataset: str, scene: str, pose_source: str) -> list[float]:
        return self.load(dataset, scene, pose_source).elevations

    def scene_obj(self, dataset: str, scene: str, pose_source: str) -> Any:
        return self.load(dataset, scene, pose_source).scene_obj

    def load(self, dataset: str, scene: str, pose_source: str) -> SceneCacheEntry:
        key = (dataset, scene, pose_source)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        scene_obj = load_pose_scene(dataset, scene, pose_source)
        _heading, elevation = heading_elevation_series(scene_obj.extr_w2c)
        entry = SceneCacheEntry(scene_obj=scene_obj, elevations=[float(x) for x in elevation])
        self._cache[key] = entry
        return entry


def start_elevation_deg(row: dict[str, Any], scene_cache: SceneCache) -> float:
    slot = start_slot_1based(row) - 1
    values = scene_cache.elevations(str(row["dataset"]), str(row["scene"]), str(row["pose_source"]))
    return float(values[slot])


def frame_metrics(frame_paths: list[str]) -> dict[str, Any]:
    arrays = [load_gray(path) for path in frame_paths]
    stds = [image_std(array) for array in arrays]
    sharpness = [image_sharpness(path) for path in frame_paths]
    adj_diffs = [mad(arrays[idx], arrays[idx + 1]) for idx in range(len(arrays) - 1)]
    pair_diffs = [mad(arrays[i], arrays[j]) for i in range(len(arrays)) for j in range(i + 1, len(arrays))]
    return {
        "stds": stds,
        "sharpness": sharpness,
        "adj_diffs": adj_diffs,
        "pair_diffs": pair_diffs,
        "min_std": min(stds) if stds else 0.0,
        "min_sharpness": min(sharpness) if sharpness else 0.0,
        "min_adj_diff": min(adj_diffs) if adj_diffs else 0.0,
        "min_pair_diff": min(pair_diffs) if pair_diffs else 0.0,
        "mean_pair_diff": (sum(pair_diffs) / len(pair_diffs)) if pair_diffs else 0.0,
    }


def floor_bias_score(start_path: str) -> tuple[float, dict[str, float]]:
    image = load_gray(start_path, size=(160, 120))
    h, _w = image.shape
    corners = cv2.goodFeaturesToTrack(image, maxCorners=240, qualityLevel=0.01, minDistance=4)
    if corners is None or len(corners) == 0:
        return 0.0, {"top_ratio": 0.0, "bottom_ratio": 0.0}
    ys = [float(item[0][1]) / h for item in corners]
    top_ratio = sum(value < 0.35 for value in ys) / len(ys)
    bottom_ratio = sum(value > 0.55 for value in ys) / len(ys)
    bias = bottom_ratio - top_ratio
    return float(bias), {"top_ratio": round(top_ratio, 4), "bottom_ratio": round(bottom_ratio, 4)}


def stage2_overlap_ids(root: Path) -> dict[str, set[str]]:
    overlap: dict[str, set[str]] = {}
    bench_root = root.parent / "vsibench_style_motion_benchmark"
    for task in TASKS:
        path = bench_root / task / "qa_data.json"
        if path.exists():
            overlap[task] = {str(row["qa_id"]) for row in read_rows(path)}
        else:
            overlap[task] = set()
    return overlap


def action_reasons(row: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    gt = row["gt"]
    motion = gt["motion"]
    reasons: list[str] = []
    if metrics["min_pair_diff"] < 0.018:
        reasons.append("weak_visual_change")
    if gt["action_family"] == "translation":
        trans = float(motion["translation_m"])
        yaw = abs(float(motion["yaw_deg"]))
        pitch = abs(float(motion["pitch_deg"]))
        axis, primary, ratio = dominant_axis_ratio(motion["local_delta"])
        if not (0.18 <= trans <= 0.85):
            reasons.append("translation_out_of_range")
        if yaw > 16.0:
            reasons.append("translation_has_large_yaw")
        if pitch > 12.0:
            reasons.append("translation_has_large_pitch")
        if axis is None or primary < 0.12 or ratio < 1.22:
            reasons.append("translation_not_directionally_clean")
    else:
        trans = float(motion["translation_m"])
        yaw = abs(float(motion["yaw_deg"]))
        pitch = abs(float(motion["pitch_deg"]))
        if not (12.0 <= yaw <= 42.0):
            reasons.append("rotation_yaw_out_of_range")
        if trans > 0.24:
            reasons.append("rotation_has_large_translation")
        if pitch > 10.0 or pitch > max(6.0, yaw * 0.35):
            reasons.append("rotation_has_large_pitch")
    return reasons


def sequence_reasons(row: dict[str, Any], metrics: dict[str, Any], scene_cache: SceneCache) -> list[str]:
    reasons: list[str] = []
    if metrics["min_adj_diff"] < 0.010 or metrics["mean_pair_diff"] < 0.020:
        reasons.append("sequence_visual_change_too_small")
    gt = row["gt"]
    start = int(gt["start_slot"]) - 1
    slot_order = {key: int(value) - 1 for key, value in gt["slot_order"].items()}
    correct_order = gt["correct_order"]
    chronological_slots = [start] + [slot_order[label] for label in correct_order]
    scene = scene_cache.scene_obj(str(row["dataset"]), str(row["scene"]), str(row["pose_source"]))
    steps = [segment_motion(scene, chronological_slots[idx], chronological_slots[idx + 1]) for idx in range(3)]
    from_start = [segment_motion(scene, start, end_slot) for end_slot in chronological_slots[1:]]
    directions = []
    for step in steps:
        if step["family"] == "translation":
            directions.append(translation_direction_key(step["local_delta"]))
        else:
            directions.append("left" if float(step["yaw_deg"]) > 0 else "right")
    if any(direction is None for direction in directions):
        reasons.append("sequence_direction_ambiguous")
    elif len(set(directions)) != 1:
        reasons.append("sequence_direction_inconsistent")
    families = {str(step["family"]) for step in steps}
    if len(families) != 1:
        reasons.append("sequence_motion_family_mixed")
    else:
        family = next(iter(families))
        if family == "translation":
            for step in steps:
                if not (
                    0.08 <= float(step["translation_m"]) <= 1.80
                    and abs(float(step["yaw_deg"])) <= 30.0
                    and abs(float(step["pitch_deg"])) <= 20.0
                ):
                    reasons.append("sequence_step_translation_unstable")
                    break
                axis, primary, ratio = dominant_axis_ratio(step["local_delta"])
                if axis is None or primary < 0.08 or ratio < 1.05:
                    reasons.append("sequence_step_not_clean_translation")
                    break
            dists = [float(item["translation_m"]) for item in from_start]
            if not (dists[1] - dists[0] >= 0.05 and dists[2] - dists[1] >= 0.06 and dists[2] >= 0.30):
                reasons.append("sequence_progression_too_weak")
        else:
            for step in steps:
                yaw = abs(float(step["yaw_deg"]))
                if not (
                    6.0 <= yaw <= 42.0
                    and float(step["translation_m"]) <= 0.50
                    and abs(float(step["pitch_deg"])) <= min(14.0, yaw * 0.45)
                ):
                    reasons.append("sequence_step_rotation_unstable")
                    break
            yaws = [abs(float(item["yaw_deg"])) for item in from_start]
            if not (yaws[1] - yaws[0] >= 3.0 and yaws[2] - yaws[1] >= 4.0 and yaws[2] >= 14.0):
                reasons.append("sequence_progression_too_weak")
    if min(abs(start_elevation_deg(row, scene_cache)), 90.0) > 70.0:
        reasons.append("sequence_start_view_extreme")
    return reasons


def degree_reasons(row: dict[str, Any], metrics: dict[str, Any], scene_cache: SceneCache) -> list[str]:
    reasons: list[str] = []
    if metrics["min_pair_diff"] < 0.014:
        reasons.append("degree_visual_gap_too_small")
    gt = row["gt"]
    start = int(gt["start_slot"]) - 1
    end_a = int(gt["frame_A"]["slot_end"]) - 1
    end_b = int(gt["frame_B"]["slot_end"]) - 1
    scene = scene_cache.scene_obj(str(row["dataset"]), str(row["scene"]), str(row["pose_source"]))
    motion_a = segment_motion(scene, start, end_a)
    motion_b = segment_motion(scene, start, end_b)
    if motion_a["family"] != motion_b["family"]:
        reasons.append("degree_family_mismatch")
        return reasons
    family = str(motion_a["family"])
    if abs(float(motion_a["pitch_deg"])) > 12.0 or abs(float(motion_b["pitch_deg"])) > 12.0:
        reasons.append("degree_pitch_too_large")
    if family == "translation":
        gap = abs(float(motion_a["translation_m"]) - float(motion_b["translation_m"]))
        dir_a = translation_direction_key(motion_a["local_delta"])
        dir_b = translation_direction_key(motion_b["local_delta"])
        if dir_a is None or dir_b is None or dir_a != dir_b:
            reasons.append("degree_direction_inconsistent")
        if not (
            0.14 <= float(motion_a["translation_m"]) <= 1.40
            and 0.18 <= float(motion_b["translation_m"]) <= 1.80
            and 0.10 <= gap <= 0.90
            and abs(float(motion_a["yaw_deg"])) <= 18.0
            and abs(float(motion_b["yaw_deg"])) <= 18.0
        ):
            reasons.append("degree_translation_not_clean")
    else:
        mag_a = abs(float(motion_a["yaw_deg"]))
        mag_b = abs(float(motion_b["yaw_deg"]))
        gap = abs(mag_a - mag_b)
        if not (
            6.0 <= mag_a <= 36.0
            and 10.0 <= mag_b <= 50.0
            and 4.0 <= gap <= 30.0
            and float(motion_a["translation_m"]) <= 0.24
            and float(motion_b["translation_m"]) <= 0.24
        ):
            reasons.append("degree_rotation_not_clean")
    if min(abs(start_elevation_deg(row, scene_cache)), 90.0) > 70.0:
        reasons.append("degree_start_view_extreme")
    return reasons


def family_reasons(row: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if metrics["min_pair_diff"] < 0.016:
        reasons.append("family_visual_change_too_small")
    gt = row["gt"]
    motion = gt["motion"]
    pitch = abs(float(motion["pitch_deg"]))
    yaw = abs(float(motion["yaw_deg"]))
    trans = float(motion["translation_m"])
    family = str(motion["family"])
    if pitch > 7.0:
        reasons.append("family_pitch_too_large")
        return reasons
    if family == "translation":
        direction = translation_direction_key(motion["local_delta"])
        _axis, primary, ratio = dominant_axis_ratio(motion["local_delta"])
        if direction is None:
            reasons.append("family_translation_direction_ambiguous")
        if not (0.16 <= trans <= 0.75 and yaw <= 14.0 and primary >= 0.12 and ratio >= 1.12):
            reasons.append("family_translation_not_clean")
    else:
        if not (12.0 <= yaw <= 36.0 and trans <= 0.18 and pitch <= min(7.0, yaw * 0.30)):
            reasons.append("family_rotation_not_clean")
    return reasons


def split_reasons(reasons: list[str], metrics: dict[str, Any], start_elev: float | None) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    hard_exact = {
        "floor_dominant_start",
        "poster_or_sign_on_floor",
        "closeup_ground_object_start",
        "image_read_error:FileNotFoundError",
        "image_read_error:OSError",
        "start_view_metric_error:FileNotFoundError",
        "degree_family_mismatch",
    }
    hard_prefixes = ("image_read_error:", "start_view_metric_error:")
    for reason in reasons:
        if reason in hard_exact or reason.startswith(hard_prefixes):
            hard.append(reason)
        else:
            soft.append(reason)
    if metrics.get("min_std", 1.0) < 0.020:
        hard.append("extremely_low_texture_frame")
    elif metrics.get("min_std", 1.0) < 0.030:
        soft.append("low_texture_frame")
    if metrics.get("min_sharpness", 999.0) < 12.0:
        hard.append("extremely_blurry_frame")
    elif metrics.get("min_sharpness", 999.0) < 22.0:
        soft.append("blurry_frame")
    if metrics.get("min_pair_diff", 1.0) < 0.006 or metrics.get("min_adj_diff", 1.0) < 0.006:
        hard.append("near_duplicate_or_imperceptible_change")
    return sorted(set(hard)), sorted(set(soft))


def build_start_audit_prompt(sample: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "<|im_start|>system\n"
        "You are a strict image-start-frame filter for embodied QA. "
        "Do not output reasoning, markdown, or <think>. Output only a compact JSON object."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Judge only the first image. Delete it if the camera is clearly pointed down at the floor/ground, "
        "or if the foreground is dominated by a poster/sign/paper/magazine/object lying on the floor, "
        "or if this is an unsuitable close-up ground-start frame. "
        "Keep normal indoor views.\n"
        "Return JSON with keys: decision, tags, reason.\n"
        "decision must be keep or delete.\n"
        "tags must be a JSON array chosen from: [\"floor_dominant_start\", \"poster_or_sign_on_floor\", \"closeup_ground_object_start\", \"normal_start\"].\n"
        "reason must be one short sentence under 16 words.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": [sample["start_frame"]]}}


def parse_start_audit_output(text: str) -> tuple[str, list[str], str]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            decision = str(payload.get("decision") or "keep").strip().lower()
            tags = [str(item) for item in payload.get("tags") or [] if item]
            reason = str(payload.get("reason") or "").strip()
            if decision not in {"keep", "delete"}:
                decision = "keep"
            return decision, tags, reason
        except Exception:
            pass
    lowered = text.lower()
    if "delete" in lowered:
        return "delete", [], text[:80]
    return "keep", [], text[:80]


def run_start_frame_audit(
    suspects: list[dict[str, Any]],
    model_path: Path,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    limit_mm_per_prompt: int,
) -> dict[str, dict[str, Any]]:
    if not suspects or LLM is None or SamplingParams is None:
        return {}
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": limit_mm_per_prompt},
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=100)
    outputs = llm.generate([build_start_audit_prompt(item) for item in suspects], sampling_params=sampling, use_tqdm=True)
    results: dict[str, dict[str, Any]] = {}
    for item, output in zip(suspects, outputs):
        text = output.outputs[0].text if output.outputs else ""
        decision, tags, reason = parse_start_audit_output(text)
        results[item["qa_id"]] = {"decision": decision, "tags": tags, "reason": reason, "raw_text": text}
    return results


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    scene_cache = SceneCache()
    overlap_ids = stage2_overlap_ids(args.input_root)
    kept_rows_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    rejected_records: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    per_task_counts = defaultdict(lambda: {"total": 0, "keep": 0, "reject": 0})
    reason_counts: Counter[str] = Counter()

    for task in TASKS:
        rows = read_rows(args.input_root / task / "qa_data.json")
        for row in rows:
            qa_id = str(row["qa_id"])
            per_task_counts[task]["total"] += 1
            reasons: list[str] = []
            try:
                metrics = frame_metrics(list(row["input"]["frame_paths"]))
            except Exception as exc:
                reasons = [f"image_read_error:{type(exc).__name__}"]
                metrics = {}
            start_elev = None
            floor_bias = None
            bias_parts: dict[str, float] = {}
            if not reasons:
                try:
                    start_elev = start_elevation_deg(row, scene_cache)
                    floor_bias, bias_parts = floor_bias_score(start_frame_path(row))
                except Exception as exc:
                    reasons.append(f"start_view_metric_error:{type(exc).__name__}")

            if not reasons:
                if task == "action_inference":
                    if metrics["min_pair_diff"] < args.min_action_pair_diff:
                        reasons.append("weak_visual_change")
                    reasons.extend(action_reasons(row, metrics))
                elif task == "movement_sequence_sorting":
                    if metrics["min_adj_diff"] < args.min_sequence_adj_diff:
                        reasons.append("weak_visual_change")
                    reasons.extend(sequence_reasons(row, metrics, scene_cache))
                elif task == "movement_degree_comparison":
                    if metrics["min_pair_diff"] < args.min_degree_pair_diff:
                        reasons.append("weak_visual_change")
                    reasons.extend(degree_reasons(row, metrics, scene_cache))
                elif task == "motion_family_discrimination":
                    if metrics["min_pair_diff"] < args.min_family_pair_diff:
                        reasons.append("weak_visual_change")
                    reasons.extend(family_reasons(row, metrics))

            audit_suspect = False
            if start_elev is not None and floor_bias is not None:
                if start_elev <= args.hard_floor_elevation_deg and floor_bias >= args.hard_floor_bias:
                    reasons.append("floor_dominant_start")
                elif start_elev <= args.suspect_floor_elevation_deg and floor_bias >= args.suspect_floor_bias:
                    audit_suspect = True

            overlap_keep = qa_id in overlap_ids.get(task, set())
            hard_reasons, soft_reasons = split_reasons(reasons, metrics, start_elev)
            reject = False
            if hard_reasons:
                reject = True
            elif len(soft_reasons) >= 4:
                reject = True
            reasons = sorted(set(hard_reasons + soft_reasons))

            record = {
                "qa_id": qa_id,
                "task": task,
                "dataset": row["dataset"],
                "scene": row["scene"],
                "overlap_stage2": overlap_keep,
                "start_elevation_deg": round(float(start_elev), 3) if start_elev is not None else None,
                "floor_bias": round(float(floor_bias), 4) if floor_bias is not None else None,
                "floor_bias_parts": bias_parts,
                "metrics": {
                    "min_std": round(float(metrics.get("min_std", 0.0)), 5),
                    "min_sharpness": round(float(metrics.get("min_sharpness", 0.0)), 3),
                    "min_adj_diff": round(float(metrics.get("min_adj_diff", 0.0)), 5),
                    "min_pair_diff": round(float(metrics.get("min_pair_diff", 0.0)), 5),
                    "mean_pair_diff": round(float(metrics.get("mean_pair_diff", 0.0)), 5),
                },
                "reasons": sorted(set(reasons)),
                "hard_reasons": hard_reasons,
                "soft_reasons": soft_reasons,
            }

            if audit_suspect and not reject:
                audit_candidates.append({"qa_id": qa_id, "task": task, "row": row, "record": record, "start_frame": start_frame_path(row)})
                pending.append({"task": task, "row": row, "record": record})
                continue

            if reject:
                per_task_counts[task]["reject"] += 1
                for reason in record["reasons"]:
                    reason_counts[reason] += 1
                rejected_records.append(record)
            else:
                per_task_counts[task]["keep"] += 1
                kept_rows_by_task[task].append(row)

    audit_results = {}
    if args.use_vlm_start_audit and audit_candidates:
        audit_results = run_start_frame_audit(
            audit_candidates,
            model_path=args.model,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            limit_mm_per_prompt=args.limit_mm_per_prompt,
        )

    for item in pending:
        task = item["task"]
        row = item["row"]
        record = item["record"]
        audit = audit_results.get(record["qa_id"])
        if audit and audit["decision"] == "delete":
            tags = [tag for tag in audit.get("tags") or [] if tag != "normal_start"]
            record["hard_reasons"] = sorted(set(record.get("hard_reasons") or []) | set(tags or ["floor_dominant_start"]))
            record["reasons"] = sorted(set(record["reasons"]) | set(record["hard_reasons"]))
            record["start_audit"] = audit
            per_task_counts[task]["reject"] += 1
            for reason in record["reasons"]:
                reason_counts[reason] += 1
            rejected_records.append(record)
        else:
            if audit:
                record["start_audit"] = audit
            per_task_counts[task]["keep"] += 1
            kept_rows_by_task[task].append(row)

    kept_all: list[dict[str, Any]] = []
    copied_counts = {}
    for task in TASKS:
        output_task_dir = args.output_root / task
        output_task_dir.mkdir(parents=True, exist_ok=True)
        rewritten_rows: list[dict[str, Any]] = []
        for row in kept_rows_by_task[task]:
            qa_id = str(row["qa_id"])
            sample_dir = output_task_dir / qa_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            old_paths = list(row["input"]["frame_paths"])
            path_map: dict[str, str] = {}
            for old_path in old_paths:
                src = Path(old_path)
                dst = sample_dir / src.name
                copy_or_link(src, dst, args.copy_mode)
                path_map[str(src)] = str(dst)
            new_row = deepcopy(row)
            new_row["input"] = replace_paths(new_row["input"], path_map)
            rewritten_rows.append(new_row)
            kept_all.append(new_row)
        write_json(output_task_dir / "qa_data.json", rewritten_rows)
        copied_counts[task] = len(rewritten_rows)

    kept_payload = {
        "tag": args.tag,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "qa_count": sum(per_task_counts[task]["total"] for task in TASKS),
        "kept_count": len(kept_all),
        "rejected_count": len(rejected_records),
        "data": kept_all,
    }

    summary = {
        "tag": args.tag,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "qa_count": sum(per_task_counts[task]["total"] for task in TASKS),
        "kept_count": len(kept_all),
        "rejected_count": len(rejected_records),
        "kept_ratio": round(len(kept_all) / max(sum(per_task_counts[task]["total"] for task in TASKS), 1), 6),
        "per_task_counts": per_task_counts,
        "reason_counts": dict(reason_counts.most_common()),
        "stage2_overlap_counts": {task: len(overlap_ids.get(task, set())) for task in TASKS},
        "vlm_start_audit_enabled": bool(args.use_vlm_start_audit),
        "vlm_start_audit_candidate_count": len(audit_candidates),
        "copied_counts": copied_counts,
        "outputs": {
            "kept_all_json": str(args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_kept_all.json"),
            "rejected_jsonl": str(args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_rejected.jsonl"),
            "summary_json": str(args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_summary.json"),
            "keep_qa_ids": str(args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_keep_qa_ids.txt"),
            "reject_qa_ids": str(args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_reject_qa_ids.txt"),
        },
    }

    kept_path = args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_kept_all.json"
    rejected_path = args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_rejected.jsonl"
    summary_path = args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_summary.json"
    keep_ids_path = args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_keep_qa_ids.txt"
    reject_ids_path = args.output_root / f"pose_gt_samples_quality_filter_{args.tag}_reject_qa_ids.txt"

    write_json(kept_path, kept_payload)
    write_json(summary_path, summary)
    write_jsonl(rejected_path, rejected_records)
    keep_ids_path.write_text("\n".join(row["qa_id"] for row in kept_all) + ("\n" if kept_all else ""), encoding="utf-8")
    reject_ids_path.write_text("\n".join(item["qa_id"] for item in rejected_records) + ("\n" if rejected_records else ""), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
