#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from ai2thor_env import create_controller
from scene_catalog import all_scene_names, normalize_scene_tokens, room_type_for_scene
from state_dedup import SceneStateDeduper
from state_quality import build_state_record


DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/output")


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _reachable_positions_with_retry(controller, scene: str, attempts: int = 2) -> List[dict]:
    last_metadata = None
    for attempt in range(1, attempts + 1):
        evt = controller.step(action="GetReachablePositions")
        last_metadata = dict(evt.metadata)
        action_return = evt.metadata.get("actionReturn")
        if bool(evt.metadata.get("lastActionSuccess", False)) and action_return is not None:
            return _sorted_positions(action_return)
        if attempt < attempts:
            controller.reset(scene=scene)
    raise RuntimeError(
        f"GetReachablePositions failed for scene={scene}: "
        f"{json.dumps(last_metadata, ensure_ascii=False)}"
    )


def _sorted_positions(positions: Iterable[dict]) -> List[dict]:
    return sorted(
        positions,
        key=lambda item: (
            round(float(item["x"]), 3),
            round(float(item["z"]), 3),
            round(float(item["y"]), 3),
        ),
    )


def _record_priority(record) -> tuple[float, float, float, float, float, float]:
    metrics = record.metrics
    return (
        float(metrics.get("salient_center_count", 0.0)),
        float(metrics.get("salient_prominent_count", 0.0)),
        float(metrics.get("good_bbox_count", 0.0)),
        float(metrics.get("brightness", 0.0)),
        float(metrics.get("far_count", 0.0)),
        -float(metrics.get("near_count", 0.0)),
    )


def _circular_yaw_delta(first: int, second: int) -> int:
    delta = abs(int(first) - int(second)) % 360
    return min(delta, 360 - delta)


def _position_distance(first: dict, second: dict) -> float:
    dx = float(first["x"]) - float(second["x"])
    dz = float(first["z"]) - float(second["z"])
    return (dx * dx + dz * dz) ** 0.5


def _position_group_score(records: List[object]) -> tuple[float, float, float]:
    priorities = [_record_priority(record) for record in records]
    best = priorities[0]
    diversity_bonus = 0.1 * len(records)
    second = priorities[1] if len(priorities) > 1 else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        best[0] + best[1] + 0.4 * best[2] + 0.02 * best[3] + diversity_bonus,
        second[0] + second[1] + 0.2 * second[2],
        best[4] - best[5],
    )


def _hotspot_group_score(records: List[object]) -> tuple[float, float, float]:
    priorities = [_record_priority(record) for record in records]
    metrics_list = [record.metrics for record in records]
    best = priorities[0]
    actionable = max(float(metrics.get("actionable_count", 0.0)) for metrics in metrics_list)
    pickupable = max(float(metrics.get("pickupable_count", 0.0)) for metrics in metrics_list)
    toggleable = max(float(metrics.get("toggleable_count", 0.0)) for metrics in metrics_list)
    openable = max(float(metrics.get("openable_count", 0.0)) for metrics in metrics_list)
    prominent = max(float(metrics.get("salient_prominent_count", 0.0)) for metrics in metrics_list)
    return (
        actionable + 0.8 * pickupable + 0.5 * openable + 0.4 * toggleable + 0.6 * prominent,
        best[0] + best[1] + 0.5 * best[2],
        best[4] - best[5],
    )


def _select_best_orientations(records: List[object], limit: int) -> List[object]:
    if limit <= 0 or not records:
        return []
    selected: List[object] = []
    for record in sorted(records, key=_record_priority, reverse=True):
        if any(_circular_yaw_delta(record.yaw, existing.yaw) < 90 for existing in selected):
            continue
        selected.append(record)
        if len(selected) >= limit:
            return selected
    for record in sorted(records, key=_record_priority, reverse=True):
        if any(record.state_id == existing.state_id for existing in selected):
            continue
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def _select_position_groups(
    candidates: List[dict],
    *,
    limit: int,
    separation: float,
    selected_groups: List[dict],
    selected_ids: set[str],
) -> tuple[List[dict], int]:
    picked: List[dict] = []
    skipped_close = 0
    if limit <= 0:
        return picked, skipped_close
    for group in candidates:
        if group["group_id"] in selected_ids:
            continue
        if any(
            _position_distance(group["position"], kept["position"]) < separation
            for kept in selected_groups
        ):
            skipped_close += 1
            continue
        selected_groups.append(group)
        selected_ids.add(group["group_id"])
        picked.append(group)
        if len(picked) >= limit:
            break
    return picked, skipped_close


def _probe_position_records(
    controller,
    *,
    scene: str,
    position: dict,
    probe_yaws: List[int],
    probe_horizon: int,
    width: int,
    height: int,
    state_index_start: int,
) -> tuple[List[object], int, int]:
    records: List[object] = []
    total_candidates = 0
    next_state_index = state_index_start
    for yaw in probe_yaws:
        evt = controller.step(
            action="Teleport",
            position=position,
            rotation={"x": 0, "y": int(yaw), "z": 0},
            horizon=int(probe_horizon),
            standing=True,
        )
        total_candidates += 1
        if not bool(evt.metadata.get("lastActionSuccess", False)):
            continue
        record = build_state_record(
            scene=scene,
            scene_id=int(scene.replace("FloorPlan", "")),
            state_index=next_state_index,
            position=position,
            yaw=int(yaw),
            horizon=int(probe_horizon),
            event=evt,
            image_width=width,
            image_height=height,
        )
        next_state_index += 1
        if not bool(record.metrics["is_good_state"]):
            continue
        records.append(record)
    return records, total_candidates, next_state_index


def _full_position_records(
    controller,
    *,
    scene: str,
    position: dict,
    yaws: List[int],
    horizons: List[int],
    width: int,
    height: int,
    state_index_start: int,
) -> tuple[List[object], int, int]:
    records: List[object] = []
    total_candidates = 0
    next_state_index = state_index_start
    for yaw in yaws:
        for horizon in horizons:
            evt = controller.step(
                action="Teleport",
                position=position,
                rotation={"x": 0, "y": int(yaw), "z": 0},
                horizon=int(horizon),
                standing=True,
            )
            total_candidates += 1
            if not bool(evt.metadata.get("lastActionSuccess", False)):
                continue
            record = build_state_record(
                scene=scene,
                scene_id=int(scene.replace("FloorPlan", "")),
                state_index=next_state_index,
                position=position,
                yaw=int(yaw),
                horizon=int(horizon),
                event=evt,
                image_width=width,
                image_height=height,
            )
            next_state_index += 1
            if not bool(record.metrics["is_good_state"]):
                continue
            records.append(record)
    return records, total_candidates, next_state_index


def build_scene_state_bank(args: argparse.Namespace, scene: str) -> dict:
    scene_dir = Path(args.output_root) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    state_bank_path = scene_dir / "state_bank.jsonl"
    summary_path = scene_dir / "state_bank_summary.json"

    yaws = _parse_csv_ints(args.yaws)
    horizons = _parse_csv_ints(args.horizons)
    probe_yaws = _parse_csv_ints(args.probe_yaws)
    probe_horizon = int(horizons[0] if horizons else 0)
    room_type = room_type_for_scene(scene)
    controller = create_controller(
        scene,
        width=args.width,
        height=args.height,
        grid_size=args.grid_size,
        rotate_step_degrees=min(max(1, min(yaws or [30])), 90),
        field_of_view=args.field_of_view,
        visibility_distance=args.visibility_distance,
        use_cloud_rendering=args.use_cloud_rendering,
        render_instance_segmentation=True,
    )

    try:
        reachable = _reachable_positions_with_retry(controller, scene=scene, attempts=2)
        if args.max_positions is not None:
            reachable = reachable[: args.max_positions]

        total_candidates = 0
        probe_candidate_states = 0
        final_candidate_states = 0
        quality_passed_states = 0
        probe_quality_passed_states = 0
        final_quality_passed_states = 0
        kept_states = 0
        skipped_duplicate_states = 0
        skipped_close_positions = 0
        kept_by_class = {str(idx): 0 for idx in range(1, 11)}
        deduper = None if args.disable_state_dedup else SceneStateDeduper()
        candidate_position_groups = []

        state_index = 0
        for pos_idx, position in enumerate(reachable):
            if pos_idx % 10 == 0:
                print(
                    f"[state_bank] scene={scene} position={pos_idx + 1}/{len(reachable)} "
                    f"kept={kept_states} probe_candidates={probe_candidate_states}",
                    flush=True,
                )
            probe_records, probe_count, state_index = _probe_position_records(
                controller,
                scene=scene,
                position=position,
                probe_yaws=probe_yaws,
                probe_horizon=probe_horizon,
                width=args.width,
                height=args.height,
                state_index_start=state_index,
            )
            probe_candidate_states += probe_count
            probe_quality_passed_states += len(probe_records)
            if not probe_records:
                continue
            candidate_position_groups.append(
                {
                    "group_id": f"{scene}__group_{len(candidate_position_groups):05d}",
                    "position": position,
                    "probe_records": probe_records,
                    "global_score": _position_group_score(probe_records),
                    "hotspot_score": _hotspot_group_score(probe_records),
                }
            )

        global_candidates = sorted(
            candidate_position_groups,
            key=lambda item: item["global_score"],
            reverse=True,
        )
        hotspot_candidates = sorted(
            candidate_position_groups,
            key=lambda item: item["hotspot_score"],
            reverse=True,
        )

        selected_groups: List[dict] = []
        selected_ids: set[str] = set()
        max_position_groups = args.max_position_groups
        if max_position_groups is None and args.max_states_per_position:
            max_position_groups = max(1, args.max_total_states // args.max_states_per_position)
        max_position_groups = max_position_groups or 0

        preferred_hotspots = args.hotspot_position_groups
        preferred_globals = max(0, max_position_groups - preferred_hotspots)
        preferred_globals = min(preferred_globals, args.global_position_groups)
        preferred_hotspots = min(preferred_hotspots, max(0, max_position_groups - preferred_globals))

        global_selected, skipped = _select_position_groups(
            global_candidates,
            limit=preferred_globals,
            separation=args.min_position_separation,
            selected_groups=selected_groups,
            selected_ids=selected_ids,
        )
        skipped_close_positions += skipped

        hotspot_selected, skipped = _select_position_groups(
            hotspot_candidates,
            limit=preferred_hotspots,
            separation=args.hotspot_min_position_separation,
            selected_groups=selected_groups,
            selected_ids=selected_ids,
        )
        skipped_close_positions += skipped

        remaining = max(0, max_position_groups - len(selected_groups))
        fallback_candidates = sorted(
            candidate_position_groups,
            key=lambda item: (item["hotspot_score"], item["global_score"]),
            reverse=True,
        )
        fallback_selected, skipped = _select_position_groups(
            fallback_candidates,
            limit=remaining,
            separation=args.hotspot_min_position_separation,
            selected_groups=selected_groups,
            selected_ids=selected_ids,
        )
        skipped_close_positions += skipped

        role_by_id = {group["group_id"]: "global" for group in global_selected}
        role_by_id.update({group["group_id"]: "hotspot" for group in hotspot_selected})
        role_by_id.update({group["group_id"]: "fallback" for group in fallback_selected})

        selected_full_groups = 0
        with state_bank_path.open("w", encoding="utf-8") as fout:
            for group in selected_groups:
                full_records, full_count, state_index = _full_position_records(
                    controller,
                    scene=scene,
                    position=group["position"],
                    yaws=yaws,
                    horizons=horizons,
                    width=args.width,
                    height=args.height,
                    state_index_start=state_index,
                )
                final_candidate_states += full_count
                final_quality_passed_states += len(full_records)
                full_records.sort(key=_record_priority, reverse=True)
                picked_records = _select_best_orientations(full_records, args.max_states_per_position)
                if not picked_records:
                    continue
                selected_full_groups += 1
                for record in picked_records:
                    if args.max_total_states is not None and kept_states >= args.max_total_states:
                        break
                    if deduper is not None and not deduper.keep(record):
                        skipped_duplicate_states += 1
                        continue
                    record.metrics["selection_group"] = role_by_id.get(group["group_id"], "fallback")
                    record.metrics["room_type"] = room_type
                    kept_states += 1
                    for class_id in range(1, 11):
                        if bool(record.suitability.get(f"class{class_id:02d}") or record.suitability.get(f"class{class_id}")):
                            kept_by_class[str(class_id)] += 1
                    fout.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                if args.max_total_states is not None and kept_states >= args.max_total_states:
                    break

        total_candidates = probe_candidate_states + final_candidate_states
        quality_passed_states = probe_quality_passed_states + final_quality_passed_states

        summary = {
            "scene": scene,
            "room_type": room_type,
            "reachable_positions": len(reachable),
            "total_candidate_states": total_candidates,
            "probe_candidate_states": probe_candidate_states,
            "final_candidate_states": final_candidate_states,
            "quality_passed_states": quality_passed_states,
            "probe_quality_passed_states": probe_quality_passed_states,
            "final_quality_passed_states": final_quality_passed_states,
            "candidate_position_groups": len(candidate_position_groups),
            "selected_position_groups": len(selected_groups),
            "selected_full_groups": selected_full_groups,
            "selected_global_groups": len(global_selected),
            "selected_hotspot_groups": len(hotspot_selected),
            "selected_fallback_groups": len(fallback_selected),
            "skipped_close_positions": skipped_close_positions,
            "skipped_duplicate_states": skipped_duplicate_states,
            "kept_states": kept_states,
            "kept_by_class": kept_by_class,
            "state_bank_path": str(state_bank_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        controller.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI2THOR state bank for fixed scenes.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes", type=str, default=None, help="Comma-separated scene ids or FloorPlan names")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--probe-yaws", type=str, default="0,90,180,270")
    parser.add_argument("--yaws", type=str, default="0,30,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--horizons", type=str, default="0")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-states-per-position", type=int, default=2)
    parser.add_argument("--max-total-states", type=int, default=40)
    parser.add_argument("--max-position-groups", type=int, default=20)
    parser.add_argument("--min-position-separation", type=float, default=0.75)
    parser.add_argument("--global-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-min-position-separation", type=float, default=0.35)
    parser.add_argument("--use-cloud-rendering", action="store_true")
    parser.add_argument("--disable-state-dedup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = all_scene_names() if args.scenes is None else normalize_scene_tokens(args.scenes.split(","))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [build_scene_state_bank(args, scene) for scene in scenes]
    merged = {
        "scenes": len(summaries),
        "total_candidates": sum(item["total_candidate_states"] for item in summaries),
        "total_kept_states": sum(item["kept_states"] for item in summaries),
        "scene_summaries": summaries,
    }
    (args.output_root / "state_bank_overview.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output_root / "state_bank_overview.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
