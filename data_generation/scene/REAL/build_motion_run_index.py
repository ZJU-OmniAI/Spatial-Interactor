#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import errno
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_3rscan_three_tasks as threed  # noqa: E402
import generate_arkit_ry_rz_large_sweep as arkit  # noqa: E402
import generate_multiscan_pose_rule_preview as multiscan  # noqa: E402
import generate_scannetpp_three_tasks as scannetpp  # noqa: E402
import generate_scannetv2_action_preview as scannetv2  # noqa: E402


OUT_ROOT = Path("/path/to/workspace/DATA/MOTION_RUNS")
ACTION_LABELS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]
TRANSLATION_LABELS = {"向前移动", "向后移动", "向左移动", "向右移动"}


def retry_eagain(fn, *, attempts: int = 10, delay_s: float = 0.5):
    for attempt in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))


@dataclass
class LoadedScene:
    dataset: str
    scene_id: str
    frame_ids: np.ndarray
    scene: object
    motion_between: Callable[[object, int, int], dict]
    canonical_axes: Callable[[dict, object], dict[str, float]]
    metadata: dict


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def rotvec(motion: dict) -> tuple[float, float, float]:
    return tuple(float(x) for x in motion["relative_rotvec_deg_xyz"])


def local_delta(motion: dict, key: str = "local_delta") -> tuple[float, float, float]:
    return tuple(float(x) for x in motion[key])


def axes_scannet(motion: dict, _scene: object) -> dict[str, float]:
    dx, _dy, dz = local_delta(motion)
    rx, ry, _rz = rotvec(motion)
    return {"forward_m": dz, "right_m": dx, "right_turn_deg": ry, "up_turn_deg": rx}


def axes_scannetpp(motion: dict, _scene: object) -> dict[str, float]:
    return axes_scannet(motion, _scene)


def axes_multiscan(motion: dict, _scene: object) -> dict[str, float]:
    dx, _dy, dz = local_delta(motion, "local_delta_hypothesis_xyz")
    rx, ry, _rz = rotvec(motion)
    return {"forward_m": -dz, "right_m": dx, "right_turn_deg": -ry, "up_turn_deg": rx}


def axes_3rscan(motion: dict, _scene: object) -> dict[str, float]:
    _dx, dy, dz = local_delta(motion)
    rx, ry, _rz = rotvec(motion)
    return {"forward_m": dz, "right_m": -dy, "right_turn_deg": rx, "up_turn_deg": -ry}


def arkit_size(scene: object) -> tuple[int, int]:
    manifest = json.loads((scene.scene_dir / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["width"]), int(manifest["height"])


def axes_arkit(motion: dict, scene: object) -> dict[str, float]:
    dx, dy, dz = local_delta(motion, "local_delta_camera_xyz")
    rx, ry, _rz = rotvec(motion)
    width, height = arkit_size(scene)
    if width > height:
        return {"forward_m": dz, "right_m": dx, "right_turn_deg": ry, "up_turn_deg": rx}
    return {"forward_m": dz, "right_m": dy, "right_turn_deg": rx, "up_turn_deg": -ry}


def label_from_axis(axis: str, value: float) -> tuple[str, str, float]:
    if axis == "forward_m":
        return ("向前移动" if value >= 0 else "向后移动", "translation", abs(value))
    if axis == "right_m":
        return ("向右移动" if value >= 0 else "向左移动", "translation", abs(value))
    if axis == "right_turn_deg":
        return ("向右转动" if value >= 0 else "向左转动", "rotation", abs(value))
    if axis == "up_turn_deg":
        return ("向上转动" if value >= 0 else "向下转动", "rotation", abs(value))
    raise ValueError(axis)


def axis_for_label(label: str) -> tuple[str, int, str]:
    if label == "向前移动":
        return "forward_m", 1, "m"
    if label == "向后移动":
        return "forward_m", -1, "m"
    if label == "向右移动":
        return "right_m", 1, "m"
    if label == "向左移动":
        return "right_m", -1, "m"
    if label == "向右转动":
        return "right_turn_deg", 1, "deg"
    if label == "向左转动":
        return "right_turn_deg", -1, "deg"
    if label == "向上转动":
        return "up_turn_deg", 1, "deg"
    if label == "向下转动":
        return "up_turn_deg", -1, "deg"
    raise ValueError(label)


def step_label(
    axes: dict[str, float],
    trans_unit: float,
    rot_unit: float,
    min_step_norm: float,
    step_ratio: float,
) -> tuple[str | None, str | None, float, dict]:
    norm = {
        "forward_m": axes["forward_m"] / trans_unit,
        "right_m": axes["right_m"] / trans_unit,
        "right_turn_deg": axes["right_turn_deg"] / rot_unit,
        "up_turn_deg": axes["up_turn_deg"] / rot_unit,
    }
    ordered = sorted(norm.items(), key=lambda kv: abs(kv[1]), reverse=True)
    best_axis, best_value = ordered[0]
    second = abs(ordered[1][1])
    if abs(best_value) < min_step_norm:
        return None, None, 0.0, norm
    if second > 1e-8 and abs(best_value) / second < step_ratio:
        return None, None, 0.0, norm
    label, family, magnitude = label_from_axis(best_axis, best_value)
    return label, family, magnitude, norm


def finalize_run(
    loaded: LoadedScene,
    steps: list[dict],
    start_step: int,
    end_step_exclusive: int,
    label: str,
    args: argparse.Namespace,
) -> dict | None:
    if end_step_exclusive - start_step < args.min_steps:
        return None
    axis, sign, unit = axis_for_label(label)
    primary_norm = 0.0
    opposite_norm = 0.0
    total_norm = 0.0
    primary_value = 0.0
    component_sums = {"forward_m": 0.0, "right_m": 0.0, "right_turn_deg": 0.0, "up_turn_deg": 0.0}
    component_abs = {"forward_m": 0.0, "right_m": 0.0, "right_turn_deg": 0.0, "up_turn_deg": 0.0}
    step_labels = Counter()
    for step in steps[start_step:end_step_exclusive]:
        axes = step["axes"]
        norm = step["norm"]
        step_labels[step["label"]] += 1
        for key in component_sums:
            component_sums[key] += axes[key]
            component_abs[key] += abs(axes[key])
            total_norm += abs(norm[key])
        signed = sign * axes[axis]
        unit_scale = args.trans_unit_m if unit == "m" else args.rot_unit_deg
        if signed >= 0:
            primary_value += signed
            primary_norm += signed / unit_scale
        else:
            opposite_norm += -signed / unit_scale
    if total_norm <= 1e-8:
        return None
    purity = primary_norm / total_norm
    monotonicity = primary_norm / max(primary_norm + opposite_norm, 1e-8)
    min_value = args.min_translation_m if unit == "m" else args.min_rotation_deg
    if primary_value < min_value:
        return None
    if purity < args.min_purity or monotonicity < args.min_monotonicity:
        return None

    start_slot = start_step
    end_slot = end_step_exclusive
    frame_ids = [int(x) for x in loaded.frame_ids[start_slot : end_slot + 1]]
    return {
        "dataset": loaded.dataset,
        "scene": loaded.scene_id,
        "label": label,
        "family": "translation" if unit == "m" else "rotation",
        "unit": unit,
        "cumulative_value": float(primary_value),
        "purity": float(purity),
        "monotonicity": float(monotonicity),
        "num_steps": int(end_step_exclusive - start_step),
        "slot_start": int(start_slot),
        "slot_end": int(end_slot),
        "start_frame_index": int(loaded.frame_ids[start_slot]),
        "end_frame_index": int(loaded.frame_ids[end_slot]),
        "frame_gap": int(loaded.frame_ids[end_slot] - loaded.frame_ids[start_slot]),
        "frame_indices": frame_ids,
        "component_sums": {k: float(v) for k, v in component_sums.items()},
        "component_abs_sums": {k: float(v) for k, v in component_abs.items()},
        "step_label_counts": dict(step_labels),
        "rule_metadata": loaded.metadata,
    }


def build_runs_for_scene(loaded: LoadedScene, args: argparse.Namespace) -> list[dict]:
    steps: list[dict] = []
    for i in range(len(loaded.frame_ids) - 1):
        motion = loaded.motion_between(loaded.scene, i, i + 1)
        axes = loaded.canonical_axes(motion, loaded.scene)
        label, family, _mag, norm = step_label(axes, args.trans_unit_m, args.rot_unit_deg, args.min_step_norm, args.step_ratio)
        steps.append({"label": label, "family": family, "axes": axes, "norm": norm})

    # Adjacent pose deltas are often noisy.  Build candidate runs by evaluating
    # cumulative windows, then keep long high-purity non-overlapping windows.
    candidates: list[dict] = []
    n = len(steps)
    for start in range(0, n, args.start_stride):
        max_end = min(n, start + args.max_run_steps)
        for end in range(start + args.min_steps, max_end + 1, args.end_stride):
            candidate = best_window_run(loaded, steps, start, end, args)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda r: (r["cumulative_value"], r["num_steps"], r["purity"], r["monotonicity"]), reverse=True)
    kept: list[dict] = []
    for cand in candidates:
        if any(window_iou(cand, old) >= args.overlap_iou for old in kept):
            continue
        kept.append(cand)
        if len(kept) >= args.max_runs_per_scene:
            break
    return kept


def best_window_run(
    loaded: LoadedScene,
    steps: list[dict],
    start_step: int,
    end_step_exclusive: int,
    args: argparse.Namespace,
) -> dict | None:
    total_norm = 0.0
    component_sums = {"forward_m": 0.0, "right_m": 0.0, "right_turn_deg": 0.0, "up_turn_deg": 0.0}
    component_abs = {"forward_m": 0.0, "right_m": 0.0, "right_turn_deg": 0.0, "up_turn_deg": 0.0}
    pos = {k: 0.0 for k in component_sums}
    neg = {k: 0.0 for k in component_sums}
    step_labels = Counter()
    for step in steps[start_step:end_step_exclusive]:
        if step["label"] is not None:
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

    scored: list[tuple[float, str, int, str, str, float]] = []
    for axis in component_sums:
        unit = "m" if axis.endswith("_m") else "deg"
        unit_scale = args.trans_unit_m if unit == "m" else args.rot_unit_deg
        for sign in (1, -1):
            value = pos[axis] if sign > 0 else neg[axis]
            label, family, _ = label_from_axis(axis, float(sign))
            scored.append((value / unit_scale, axis, sign, unit, label, value))
    primary_norm, axis, sign, unit, label, primary_value = max(scored, key=lambda x: x[0])
    opposite_value = neg[axis] if sign > 0 else pos[axis]
    unit_scale = args.trans_unit_m if unit == "m" else args.rot_unit_deg
    opposite_norm = opposite_value / unit_scale
    purity = primary_norm / total_norm
    monotonicity = primary_norm / max(primary_norm + opposite_norm, 1e-8)
    min_value = args.min_translation_m if unit == "m" else args.min_rotation_deg
    if primary_value < min_value:
        return None
    if purity < args.min_purity or monotonicity < args.min_monotonicity:
        return None

    start_slot = start_step
    end_slot = end_step_exclusive
    frame_ids = [int(x) for x in loaded.frame_ids[start_slot : end_slot + 1]]
    return {
        "dataset": loaded.dataset,
        "scene": loaded.scene_id,
        "label": label,
        "family": "translation" if unit == "m" else "rotation",
        "unit": unit,
        "cumulative_value": float(primary_value),
        "purity": float(purity),
        "monotonicity": float(monotonicity),
        "num_steps": int(end_step_exclusive - start_step),
        "slot_start": int(start_slot),
        "slot_end": int(end_slot),
        "start_frame_index": int(loaded.frame_ids[start_slot]),
        "end_frame_index": int(loaded.frame_ids[end_slot]),
        "frame_gap": int(loaded.frame_ids[end_slot] - loaded.frame_ids[start_slot]),
        "frame_indices": frame_ids,
        "component_sums": {k: float(v) for k, v in component_sums.items()},
        "component_abs_sums": {k: float(v) for k, v in component_abs.items()},
        "step_label_counts": dict(step_labels),
        "rule_metadata": loaded.metadata,
    }


def window_iou(a: dict, b: dict) -> float:
    if a["scene"] != b["scene"]:
        return 0.0
    left = max(int(a["slot_start"]), int(b["slot_start"]))
    right = min(int(a["slot_end"]), int(b["slot_end"]))
    inter = max(0, right - left)
    union = max(int(a["slot_end"]), int(b["slot_end"])) - min(int(a["slot_start"]), int(b["slot_start"]))
    return inter / max(union, 1)


def iter_scannetv2(max_scenes: int, skip_scenes: int = 0) -> Iterable[LoadedScene]:
    count = 0
    seen_valid = 0
    for scene_dir in retry_eagain(lambda: sorted(scannetv2.DATA_ROOT.iterdir())):
        if max_scenes and count >= max_scenes:
            break
        if not retry_eagain(lambda p=scene_dir: p.is_dir()):
            continue
        scene = scannetv2.load_scene_pose(scene_dir)
        if scene is None:
            continue
        if seen_valid < skip_scenes:
            seen_valid += 1
            continue
        seen_valid += 1
        count += 1
        yield LoadedScene("scannetv2", scene.scene, scene.frame_ids, scene, scannetv2.motion_between, axes_scannet, {"pose_rule": "dx right, dz forward, ry right-turn, rx up-turn"})


def iter_scannetpp(max_scenes: int, skip_scenes: int = 0) -> Iterable[LoadedScene]:
    count = 0
    seen_valid = 0
    for scene_dir in scannetpp.scene_dirs(scannetpp.DATA_ROOT):
        if max_scenes and count >= max_scenes:
            break
        scene = scannetpp.load_scene_pose(scene_dir)
        if scene is None:
            continue
        if seen_valid < skip_scenes:
            seen_valid += 1
            continue
        seen_valid += 1
        count += 1
        yield LoadedScene("scannetpp", scene.scene, scene.frame_ids, scene, scannetpp.motion_between, axes_scannetpp, {"pose_rule": "dx right, dz forward, ry right-turn, rx up-turn"})


def iter_multiscan(max_scenes: int, skip_scenes: int = 0) -> Iterable[LoadedScene]:
    count = 0
    seen_valid = 0
    for zip_path in retry_eagain(lambda: sorted(multiscan.SCAN_ROOT.glob("*.zip"))):
        if max_scenes and count >= max_scenes:
            break
        scene = multiscan.load_scene_pose(zip_path, max_frames=4500)
        if scene is None:
            continue
        if seen_valid < skip_scenes:
            seen_valid += 1
            continue
        seen_valid += 1
        count += 1
        yield LoadedScene("multiscan", scene.scene, scene.frame_ids, scene, multiscan.motion_between, axes_multiscan, {"pose_rule": "dx right, -dz forward, -ry right-turn, rx up-turn"})


def iter_3rscan(max_scenes: int, skip_scenes: int = 0) -> Iterable[LoadedScene]:
    count = 0
    seen_valid = 0
    roots = []
    if retry_eagain(lambda: threed.EXTRACTED_SAMPLE_ROOT.exists()):
        roots.extend([p for p in retry_eagain(lambda: sorted(threed.EXTRACTED_SAMPLE_ROOT.iterdir())) if retry_eagain(lambda p=p: p.is_dir())])
    roots.extend([p for p in retry_eagain(lambda: sorted(threed.SEQUENCE_ROOT.iterdir())) if retry_eagain(lambda p=p: p.is_dir())])
    seen: set[str] = set()
    for scene_dir in roots:
        if max_scenes and count >= max_scenes:
            break
        if scene_dir.name in seen:
            continue
        scene = threed.load_extracted_scene(scene_dir) if (scene_dir / "sequence").is_dir() else threed.load_zipped_scene(scene_dir)
        if scene is None:
            continue
        seen.add(scene.scene)
        if seen_valid < skip_scenes:
            seen_valid += 1
            continue
        seen_valid += 1
        count += 1
        yield LoadedScene("3rscan", scene.scene, scene.frame_ids, scene, threed.motion_between, axes_3rscan, {"pose_rule": "-dy right, dz forward, rx right-turn, -ry up-turn", "source": scene.source})


def arkit_scene_dirs_fast() -> list[Path]:
    root = arkit.DATA_ROOT
    metadata_path = root.parents[1] / "raw_minimal" / "metadata.csv"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("fold") == root.name and row.get("video_id")]
        rows.sort(key=lambda row: (row["fold"], row["video_id"]))
        return [root / str(row["video_id"]) for row in rows]
    return arkit.scene_dirs(root)


def iter_arkit(max_scenes: int, skip_scenes: int = 0) -> Iterable[LoadedScene]:
    count = 0
    seen_valid = 0
    for scene_idx, scene_dir in enumerate(arkit_scene_dirs_fast()):
        if scene_idx < skip_scenes:
            continue
        if max_scenes and scene_idx >= skip_scenes + max_scenes:
            break
        if max_scenes and count >= max_scenes:
            break
        try:
            scene = arkit.load_scene_pose(scene_dir)
        except Exception:
            scene = None
        if scene is None:
            continue
        width, height = arkit_size(scene)
        seen_valid += 1
        count += 1
        yield LoadedScene(
            "arkit",
            scene.scene,
            scene.frame_ids,
            scene,
            arkit.motion_between,
            axes_arkit,
            {
                "pose_rule": "size split; 1920x1440: dx right dz forward ry right-turn rx up-turn; 1440x1920: dy right dz forward rx right-turn -ry up-turn",
                "width": width,
                "height": height,
            },
        )


def build_dataset(name: str, iterator: Iterable[LoadedScene], args: argparse.Namespace) -> dict:
    out_dir = args.output_root / name
    run_path = out_dir / "runs.jsonl"
    scene_count = 0
    all_runs: list[dict] = []
    label_counts = Counter()
    for loaded in iterator:
        scene_count += 1
        runs = build_runs_for_scene(loaded, args)
        for run in runs:
            label_counts[run["label"]] += 1
        all_runs.extend(runs)
        print(f"[runs] {name}/{loaded.scene_id}: {len(runs)} kept", flush=True)

    all_runs.sort(key=lambda r: (r["purity"], r["monotonicity"], r["cumulative_value"]), reverse=True)
    if args.max_runs_per_dataset > 0:
        all_runs = all_runs[: args.max_runs_per_dataset]
        label_counts = Counter(run["label"] for run in all_runs)
    append_jsonl(run_path, all_runs)
    summary = {
        "dataset": name,
        "scenes_scanned": scene_count,
        "runs_kept": len(all_runs),
        "label_counts": dict(label_counts),
        "min_purity": args.min_purity,
        "min_monotonicity": args.min_monotonicity,
        "min_translation_m": args.min_translation_m,
        "min_rotation_deg": args.min_rotation_deg,
        "runs_jsonl": str(run_path),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-purity continuous motion-run indexes from pose trajectories.")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=60)
    parser.add_argument("--max-scenes-arkit", type=int, default=120)
    parser.add_argument("--max-scenes-multiscan", type=int, default=35)
    parser.add_argument("--max-runs-per-dataset", type=int, default=300)
    parser.add_argument("--trans-unit-m", type=float, default=0.05)
    parser.add_argument("--rot-unit-deg", type=float, default=4.0)
    parser.add_argument("--min-step-norm", type=float, default=0.18)
    parser.add_argument("--step-ratio", type=float, default=1.35)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--max-run-steps", type=int, default=120)
    parser.add_argument("--start-stride", type=int, default=2)
    parser.add_argument("--end-stride", type=int, default=2)
    parser.add_argument("--max-runs-per-scene", type=int, default=12)
    parser.add_argument("--overlap-iou", type=float, default=0.50)
    parser.add_argument("--min-purity", type=float, default=0.82)
    parser.add_argument("--min-monotonicity", type=float, default=0.96)
    parser.add_argument("--min-translation-m", type=float, default=0.35)
    parser.add_argument("--min-rotation-deg", type=float, default=30.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = {
        "scannetv2": build_dataset("scannetv2", iter_scannetv2(args.max_scenes), args),
        "scannetpp": build_dataset("scannetpp", iter_scannetpp(args.max_scenes), args),
        "multiscan": build_dataset("multiscan", iter_multiscan(args.max_scenes_multiscan), args),
        "3rscan": build_dataset("3rscan", iter_3rscan(args.max_scenes), args),
        "arkit": build_dataset("arkit", iter_arkit(args.max_scenes_arkit), args),
    }
    write_json(args.output_root / "summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
