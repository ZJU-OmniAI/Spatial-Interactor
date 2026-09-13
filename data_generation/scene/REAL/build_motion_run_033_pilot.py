#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_motion_run_index as base  # noqa: E402


OUT_ROOT = Path("/path/to/workspace/DATA/MOTION_RUNS_033_PILOT")


TARGET_FRAME_GAP = {
    "scannetv2": 10,
    "scannetpp": 10,
    "multiscan": 20,
    "3rscan": 10,
    "arkit": 20,
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resample_slots(frame_ids: np.ndarray, target_gap: int) -> list[int]:
    slots = [0]
    while True:
        target = int(frame_ids[slots[-1]]) + target_gap
        nxt = int(np.searchsorted(frame_ids, target, side="left"))
        if nxt >= len(frame_ids):
            break
        if nxt <= slots[-1]:
            nxt = slots[-1] + 1
        if nxt >= len(frame_ids):
            break
        slots.append(nxt)
    return slots


def step_axes(loaded: base.LoadedScene, slots: list[int], args: argparse.Namespace) -> list[dict]:
    out = []
    for a, b in zip(slots, slots[1:]):
        motion = loaded.motion_between(loaded.scene, a, b)
        axes = loaded.canonical_axes(motion, loaded.scene)
        label, family, _mag, norm = base.step_label(
            axes,
            args.trans_unit_m,
            args.rot_unit_deg,
            args.min_step_norm,
            args.step_ratio,
        )
        out.append(
            {
                "start_slot": int(a),
                "end_slot": int(b),
                "start_frame_index": int(loaded.frame_ids[a]),
                "end_frame_index": int(loaded.frame_ids[b]),
                "label": label,
                "family": family,
                "axes": axes,
                "norm": norm,
            }
        )
    return out


def label_from_axis(axis: str, sign: int) -> tuple[str, str, str]:
    if axis == "forward_m":
        return ("向前移动" if sign > 0 else "向后移动", "translation", "m")
    if axis == "right_m":
        return ("向右移动" if sign > 0 else "向左移动", "translation", "m")
    if axis == "right_turn_deg":
        return ("向右转动" if sign > 0 else "向左转动", "rotation", "deg")
    if axis == "up_turn_deg":
        return ("向上转动" if sign > 0 else "向下转动", "rotation", "deg")
    raise ValueError(axis)


def evaluate_window(loaded: base.LoadedScene, steps: list[dict], start: int, end: int, args: argparse.Namespace) -> dict | None:
    total_norm = 0.0
    component_sums = {"forward_m": 0.0, "right_m": 0.0, "right_turn_deg": 0.0, "up_turn_deg": 0.0}
    component_abs = {k: 0.0 for k in component_sums}
    pos = {k: 0.0 for k in component_sums}
    neg = {k: 0.0 for k in component_sums}
    step_labels = Counter()
    for step in steps[start:end]:
        if step["label"]:
            step_labels[step["label"]] += 1
        for key, value in step["axes"].items():
            component_sums[key] += value
            component_abs[key] += abs(value)
            if value >= 0:
                pos[key] += value
            else:
                neg[key] += -value
        total_norm += abs(step["axes"]["forward_m"]) / args.trans_unit_m
        total_norm += abs(step["axes"]["right_m"]) / args.trans_unit_m
        total_norm += abs(step["axes"]["right_turn_deg"]) / args.rot_unit_deg
        total_norm += abs(step["axes"]["up_turn_deg"]) / args.rot_unit_deg
    if total_norm <= 1e-8:
        return None

    candidates = []
    for axis in component_sums:
        for sign in (1, -1):
            unit_scale = args.trans_unit_m if axis.endswith("_m") else args.rot_unit_deg
            value = pos[axis] if sign > 0 else neg[axis]
            label, family, unit = label_from_axis(axis, sign)
            candidates.append((value / unit_scale, value, axis, sign, label, family, unit))
    primary_norm, primary_value, axis, sign, label, family, unit = max(candidates, key=lambda x: x[0])
    opposite = neg[axis] if sign > 0 else pos[axis]
    unit_scale = args.trans_unit_m if unit == "m" else args.rot_unit_deg
    opposite_norm = opposite / unit_scale
    purity = primary_norm / total_norm
    monotonicity = primary_norm / max(primary_norm + opposite_norm, 1e-8)
    min_value = args.min_translation_m if unit == "m" else args.min_rotation_deg
    if primary_value < min_value:
        return None
    if purity < args.min_purity or monotonicity < args.min_monotonicity:
        return None

    start_slot = steps[start]["start_slot"]
    end_slot = steps[end - 1]["end_slot"]
    frame_indices = [int(loaded.frame_ids[s]) for s in range(start_slot, end_slot + 1)]
    return {
        "dataset": loaded.dataset,
        "scene": loaded.scene_id,
        "label": label,
        "family": family,
        "unit": unit,
        "cumulative_value": float(primary_value),
        "purity": float(purity),
        "monotonicity": float(monotonicity),
        "num_steps_033": int(end - start),
        "slot_start": int(start_slot),
        "slot_end": int(end_slot),
        "start_frame_index": int(loaded.frame_ids[start_slot]),
        "end_frame_index": int(loaded.frame_ids[end_slot]),
        "frame_gap": int(loaded.frame_ids[end_slot] - loaded.frame_ids[start_slot]),
        "frame_indices": frame_indices,
        "component_sums": {k: float(v) for k, v in component_sums.items()},
        "component_abs_sums": {k: float(v) for k, v in component_abs.items()},
        "step_label_counts": dict(step_labels),
        "target_frame_gap_per_step": TARGET_FRAME_GAP[loaded.dataset],
        "rule_metadata": loaded.metadata,
    }


def iou(a: dict, b: dict) -> float:
    if a["scene"] != b["scene"]:
        return 0.0
    left = max(a["slot_start"], b["slot_start"])
    right = min(a["slot_end"], b["slot_end"])
    inter = max(0, right - left)
    union = max(a["slot_end"], b["slot_end"]) - min(a["slot_start"], b["slot_start"])
    return inter / max(union, 1)


def runs_for_scene(loaded: base.LoadedScene, args: argparse.Namespace) -> tuple[list[dict], dict]:
    target_gap = TARGET_FRAME_GAP[loaded.dataset]
    slots = resample_slots(loaded.frame_ids, target_gap)
    steps = step_axes(loaded, slots, args)
    candidates = []
    for start in range(0, len(steps), args.start_stride):
        max_end = min(len(steps), start + args.max_run_steps)
        for end in range(start + args.min_steps, max_end + 1, args.end_stride):
            run = evaluate_window(loaded, steps, start, end, args)
            if run:
                candidates.append(run)
    candidates.sort(key=lambda r: (r["purity"], r["monotonicity"], r["cumulative_value"], r["num_steps_033"]), reverse=True)
    kept = []
    for cand in candidates:
        if any(iou(cand, old) >= args.overlap_iou for old in kept):
            continue
        kept.append(cand)
        if len(kept) >= args.max_runs_per_scene:
            break
    stats = {
        "raw_slots": int(len(loaded.frame_ids)),
        "resampled_slots": int(len(slots)),
        "resampled_steps": int(len(steps)),
        "target_frame_gap": int(target_gap),
    }
    return kept, stats


def dataset_iter(name: str, args: argparse.Namespace):
    if name == "scannetv2":
        return base.iter_scannetv2(args.scenes_per_dataset)
    if name == "scannetpp":
        return base.iter_scannetpp(args.scenes_per_dataset)
    if name == "multiscan":
        return base.iter_multiscan(args.scenes_per_dataset)
    if name == "3rscan":
        return base.iter_3rscan(args.scenes_per_dataset)
    if name == "arkit":
        return base.iter_arkit(args.scenes_per_dataset)
    raise ValueError(name)


def build_dataset(name: str, args: argparse.Namespace) -> dict:
    runs = []
    scene_stats = []
    label_counts = Counter()
    scene_count = 0
    for loaded in dataset_iter(name, args):
        scene_count += 1
        scene_runs, stats = runs_for_scene(loaded, args)
        stats.update({"scene": loaded.scene_id, "runs_kept": len(scene_runs)})
        scene_stats.append(stats)
        for run in scene_runs:
            label_counts[run["label"]] += 1
        runs.extend(scene_runs)
        print(f"[033] {name}/{loaded.scene_id}: {len(scene_runs)} runs, steps={stats['resampled_steps']}", flush=True)
    runs.sort(key=lambda r: (r["purity"], r["monotonicity"], r["cumulative_value"]), reverse=True)
    out_dir = args.output_root / name
    write_jsonl(out_dir / "runs.jsonl", runs)
    summary = {
        "dataset": name,
        "scenes_scanned": scene_count,
        "runs_kept": len(runs),
        "label_counts": dict(label_counts),
        "target_frame_gap": TARGET_FRAME_GAP[name],
        "approx_step_seconds": "about 0.33s",
        "thresholds": {
            "min_purity": args.min_purity,
            "min_monotonicity": args.min_monotonicity,
            "min_translation_m": args.min_translation_m,
            "min_rotation_deg": args.min_rotation_deg,
        },
        "scene_stats": scene_stats,
        "runs_jsonl": str(out_dir / "runs.jsonl"),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--scenes-per-dataset", type=int, default=10)
    parser.add_argument("--trans-unit-m", type=float, default=0.05)
    parser.add_argument("--rot-unit-deg", type=float, default=4.0)
    parser.add_argument("--min-step-norm", type=float, default=0.12)
    parser.add_argument("--step-ratio", type=float, default=1.15)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--max-run-steps", type=int, default=90)
    parser.add_argument("--start-stride", type=int, default=1)
    parser.add_argument("--end-stride", type=int, default=1)
    parser.add_argument("--max-runs-per-scene", type=int, default=8)
    parser.add_argument("--overlap-iou", type=float, default=0.5)
    parser.add_argument("--min-purity", type=float, default=0.76)
    parser.add_argument("--min-monotonicity", type=float, default=0.94)
    parser.add_argument("--min-translation-m", type=float, default=0.35)
    parser.add_argument("--min-rotation-deg", type=float, default=30.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in ["scannetv2", "scannetpp", "multiscan", "3rscan", "arkit"]:
        summaries[name] = build_dataset(name, args)
    write_json(args.output_root / "summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
