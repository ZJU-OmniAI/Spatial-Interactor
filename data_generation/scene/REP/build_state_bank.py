#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from hssd_env import create_runner, load_scene
from scene_catalog import all_scene_names, normalize_scene_tokens, scene_id
from state_dedup import SceneStateDeduper
from state_quality import build_state_record


DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/REP/output")


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _sorted_positions(positions: Iterable[Sequence[float]]) -> List[np.ndarray]:
    return sorted(
        (np.array(pos, dtype=np.float32) for pos in positions),
        key=lambda item: (
            round(float(item[0]), 3),
            round(float(item[2]), 3),
            round(float(item[1]), 3),
        ),
    )


def _dedupe_navmesh_vertices(vertices: Iterable[Sequence[float]]) -> List[np.ndarray]:
    deduped = {}
    for pos in vertices:
        key = (round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2))
        deduped.setdefault(key, np.array(pos, dtype=np.float32))
    return _sorted_positions(deduped.values())


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


def _position_distance(first: Sequence[float], second: Sequence[float]) -> float:
    dx = float(first[0]) - float(second[0])
    dz = float(first[2]) - float(second[2])
    return (dx * dx + dz * dz) ** 0.5


def _select_probe_positions(
    positions: List[np.ndarray],
    *,
    limit: int | None,
    separation: float,
) -> List[np.ndarray]:
    if limit is None or limit <= 0 or len(positions) <= limit:
        return list(positions)
    candidate_positions: List[np.ndarray] = []
    total = len(positions)
    if limit > 1:
        used_indices: set[int] = set()
        for step in range(limit):
            idx = round(step * (total - 1) / float(limit - 1))
            if idx in used_indices:
                continue
            used_indices.add(idx)
            candidate_positions.append(positions[idx])
    for position in positions:
        candidate_positions.append(position)
    selected: List[np.ndarray] = []
    for position in candidate_positions:
        if any(_position_distance(position, kept) < separation for kept in selected):
            continue
        selected.append(position)
        if len(selected) >= limit:
            return selected
    if len(selected) >= limit:
        return selected[:limit]
    for position in positions:
        if any(np.allclose(position, kept) for kept in selected):
            continue
        selected.append(position)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _prefilter_view_score(metrics: dict) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        1.0 if bool(metrics.get("is_good_state")) else 0.0,
        float(metrics.get("salient_center_count", 0.0)),
        float(metrics.get("good_bbox_count", 0.0)),
        float(metrics.get("actionable_count", 0.0)),
        float(metrics.get("meaningful_count", 0.0)),
        float(metrics.get("far_count", 0.0)) - 0.4 * float(metrics.get("near_count", 0.0)),
        float(metrics.get("visible_count", 0.0)),
        float(metrics.get("brightness", 0.0)),
    )


def _prefilter_position_score(metrics_list: List[dict]) -> tuple[float, float, float, float, float, float, float, float, float]:
    if not metrics_list:
        return (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    best = max((_prefilter_view_score(metrics) for metrics in metrics_list))
    count = float(len(metrics_list))
    good_views = float(sum(1 for metrics in metrics_list if bool(metrics.get("is_good_state"))))
    avg_salient_center = sum(float(metrics.get("salient_center_count", 0.0)) for metrics in metrics_list) / count
    avg_good_bbox = sum(float(metrics.get("good_bbox_count", 0.0)) for metrics in metrics_list) / count
    avg_actionable = sum(float(metrics.get("actionable_count", 0.0)) for metrics in metrics_list) / count
    avg_meaningful = sum(float(metrics.get("meaningful_count", 0.0)) for metrics in metrics_list) / count
    avg_visible = sum(float(metrics.get("visible_count", 0.0)) for metrics in metrics_list) / count
    avg_depth = sum(
        float(metrics.get("far_count", 0.0)) - 0.4 * float(metrics.get("near_count", 0.0))
        for metrics in metrics_list
    ) / count
    return (
        good_views,
        best[1],
        best[2],
        best[3],
        avg_salient_center,
        avg_good_bbox,
        avg_actionable,
        avg_meaningful + 0.2 * avg_visible,
        avg_depth + 0.01 * best[7],
    )


def _prefilter_position_metrics(
    runner,
    *,
    scene: str,
    position: np.ndarray,
    scene_objects: Sequence[dict],
    yaws: List[int],
    horizon: int,
    width: int,
    height: int,
) -> List[dict]:
    metrics_list: List[dict] = []
    for yaw in yaws:
        obs = runner.set_agent_state(position=position, yaw=int(yaw), pitch=int(horizon))
        brightness = runner.frame_brightness(obs)
        visible_objects = runner.visible_objects(scene_objects, obs["depth"])
        record = build_state_record(
            scene=scene,
            scene_id=scene_id(scene),
            state_index=0,
            position=position,
            yaw=int(yaw),
            horizon=int(horizon),
            scene_objects=scene_objects,
            visible_objects=visible_objects,
            brightness=brightness,
            image_width=width,
            image_height=height,
        )
        metrics_list.append(record.metrics)
    return metrics_list


def _select_top_scored_positions(
    candidates: List[dict],
    *,
    limit: int | None,
    separation: float,
) -> tuple[List[np.ndarray], int]:
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return [np.array(item["position"], copy=True) for item in candidates], 0
    selected: List[dict] = []
    skipped_close = 0
    for item in candidates:
        if any(_position_distance(item["position"], kept["position"]) < separation for kept in selected):
            skipped_close += 1
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return [np.array(item["position"], copy=True) for item in selected], skipped_close


def _prefilter_probe_positions(
    runner,
    *,
    args: argparse.Namespace,
    scene: str,
    scene_objects: Sequence[dict],
    reachable: List[np.ndarray],
    yaws: List[int],
    horizon: int,
) -> tuple[List[np.ndarray], int, int]:
    if args.disable_position_prefilter:
        return list(reachable), len(reachable), 0
    top_limit = int(args.prefilter_top_positions)
    if top_limit <= 0 or len(reachable) <= top_limit:
        return list(reachable), len(reachable), 0

    candidate_limit = int(args.prefilter_candidate_positions)
    candidate_limit = max(top_limit, candidate_limit)
    candidate_positions = _select_probe_positions(
        reachable,
        limit=candidate_limit,
        separation=args.prefilter_candidate_separation,
    )

    print(
        f"[state_bank] scene={scene} step=prefilter_pool source={len(reachable)} "
        f"candidate_positions={len(candidate_positions)} target={top_limit}",
        flush=True,
    )
    scored_candidates: List[dict] = []
    for idx, position in enumerate(candidate_positions):
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(candidate_positions):
            print(
                f"[state_bank] scene={scene} step=prefilter_score position={idx + 1}/{len(candidate_positions)}",
                flush=True,
            )
        metrics_list = _prefilter_position_metrics(
            runner,
            scene=scene,
            position=position,
            scene_objects=scene_objects,
            yaws=yaws,
            horizon=horizon,
            width=args.width,
            height=args.height,
        )
        scored_candidates.append(
            {
                "position": np.array(position, copy=True),
                "score": _prefilter_position_score(metrics_list),
            }
        )

    scored_candidates.sort(
        key=lambda item: (
            item["score"],
            round(float(item["position"][0]), 3),
            round(float(item["position"][2]), 3),
            round(float(item["position"][1]), 3),
        ),
        reverse=True,
    )
    selected_positions, skipped_close = _select_top_scored_positions(
        scored_candidates,
        limit=top_limit,
        separation=args.prefilter_top_separation,
    )
    print(
        f"[state_bank] scene={scene} step=prefilter_done candidate_positions={len(candidate_positions)} "
        f"selected_positions={len(selected_positions)} skipped_close={skipped_close}",
        flush=True,
    )
    return selected_positions, len(candidate_positions), skipped_close


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
    prominent = max(float(metrics.get("salient_prominent_count", 0.0)) for metrics in metrics_list)
    centered = max(float(metrics.get("salient_center_count", 0.0)) for metrics in metrics_list)
    good_bbox = max(float(metrics.get("good_bbox_count", 0.0)) for metrics in metrics_list)
    return (
        actionable + 0.8 * prominent + 0.6 * centered + 0.4 * good_bbox,
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
    runner,
    *,
    scene: str,
    position: np.ndarray,
    scene_objects: Sequence[dict],
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
        obs = runner.set_agent_state(position=position, yaw=int(yaw), pitch=int(probe_horizon))
        total_candidates += 1
        brightness = runner.frame_brightness(obs)
        visible_objects = runner.visible_objects(scene_objects, obs["depth"])
        record = build_state_record(
            scene=scene,
            scene_id=scene_id(scene),
            state_index=next_state_index,
            position=position,
            yaw=int(yaw),
            horizon=int(probe_horizon),
            scene_objects=scene_objects,
            visible_objects=visible_objects,
            brightness=brightness,
            image_width=width,
            image_height=height,
        )
        next_state_index += 1
        if not bool(record.metrics["is_good_state"]):
            continue
        records.append(record)
    return records, total_candidates, next_state_index


def _full_position_records(
    runner,
    *,
    scene: str,
    position: np.ndarray,
    scene_objects: Sequence[dict],
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
            obs = runner.set_agent_state(position=position, yaw=int(yaw), pitch=int(horizon))
            total_candidates += 1
            brightness = runner.frame_brightness(obs)
            visible_objects = runner.visible_objects(scene_objects, obs["depth"])
            record = build_state_record(
                scene=scene,
                scene_id=scene_id(scene),
                state_index=next_state_index,
                position=position,
                yaw=int(yaw),
                horizon=int(horizon),
                scene_objects=scene_objects,
                visible_objects=visible_objects,
                brightness=brightness,
                image_width=width,
                image_height=height,
            )
            next_state_index += 1
            if not bool(record.metrics["is_good_state"]):
                continue
            records.append(record)
    return records, total_candidates, next_state_index


def _build_runner(args: argparse.Namespace):
    return create_runner(
        width=args.width,
        height=args.height,
        field_of_view=args.field_of_view,
        sensor_height=args.sensor_height,
        agent_radius=args.agent_radius,
        seed=args.seed,
        dataset_root=args.dataset_root,
        enable_color_sensor=False,
        enable_depth_sensor=True,
    )


def _load_scene_with_runner(args: argparse.Namespace, scene: str):
    runner = _build_runner(args)
    try:
        scene_file, _, scene_objects = load_scene(runner, scene)
        navmesh_vertices = runner.sim.pathfinder.build_navmesh_vertices()
        reachable = _dedupe_navmesh_vertices(navmesh_vertices)
        return runner, scene_file, scene_objects, reachable
    except Exception:
        runner.shutdown()
        raise


def _iter_chunks(items: Sequence, chunk_size: int):
    if not items:
        return
    if chunk_size <= 0 or chunk_size >= len(items):
        yield 0, items
        return
    for start in range(0, len(items), chunk_size):
        yield start, items[start : start + chunk_size]


def _shutdown_runner(runner) -> None:
    if runner is None:
        return
    runner.shutdown()
    gc.collect()


def build_scene_state_bank(args: argparse.Namespace, scene: str) -> dict:
    scene_dir = Path(args.output_root) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    state_bank_path = scene_dir / "state_bank.jsonl"
    summary_path = scene_dir / "state_bank_summary.json"

    yaws = _parse_csv_ints(args.yaws)
    horizons = _parse_csv_ints(args.horizons)
    probe_yaws = _parse_csv_ints(args.probe_yaws)
    prefilter_yaws = _parse_csv_ints(args.prefilter_yaws)
    probe_horizon = int(horizons[0] if horizons else 0)
    runner = None
    try:
        print(f"[state_bank] scene={scene} step=load_scene", flush=True)
        runner, scene_file, scene_objects, reachable = _load_scene_with_runner(args, scene)
        prefiltered_positions, prefilter_candidate_positions, prefilter_skipped_close = _prefilter_probe_positions(
            runner,
            args=args,
            scene=scene,
            scene_objects=scene_objects,
            reachable=reachable,
            yaws=prefilter_yaws,
            horizon=probe_horizon,
        )
        probe_limit = args.max_probe_positions
        if args.max_positions is not None:
            probe_limit = (
                min(int(args.max_positions), int(probe_limit))
                if probe_limit is not None
                else int(args.max_positions)
            )
        probe_positions = _select_probe_positions(
            prefiltered_positions,
            limit=probe_limit,
            separation=args.probe_position_separation,
        )
        _shutdown_runner(runner)
        runner = None
        print(
            f"[state_bank] scene={scene} step=navmesh_done reachable={len(reachable)} "
            f"prefilter_positions={len(prefiltered_positions)} probe_positions={len(probe_positions)}",
            flush=True,
        )

        probe_candidate_states = 0
        final_candidate_states = 0
        probe_quality_passed_states = 0
        final_quality_passed_states = 0
        kept_states = 0
        skipped_duplicate_states = 0
        skipped_close_positions = 0
        kept_by_class = {str(idx): 0 for idx in range(1, 11)}
        deduper = None if args.disable_state_dedup else SceneStateDeduper()
        candidate_position_groups = []

        state_index = 0
        for chunk_start, position_chunk in _iter_chunks(probe_positions, args.probe_restart_interval):
            print(
                f"[state_bank] scene={scene} step=probe_chunk start={chunk_start + 1} "
                f"end={chunk_start + len(position_chunk)} total={len(probe_positions)}",
                flush=True,
            )
            runner, _, _, _ = _load_scene_with_runner(args, scene)
            try:
                for offset, position in enumerate(position_chunk):
                    pos_idx = chunk_start + offset
                    if pos_idx % 10 == 0:
                        print(
                            f"[state_bank] scene={scene} position={pos_idx + 1}/{len(probe_positions)} "
                            f"kept={kept_states} probe_candidates={probe_candidate_states}",
                            flush=True,
                        )
                    probe_records, probe_count, state_index = _probe_position_records(
                        runner,
                        scene=scene,
                        position=position,
                        scene_objects=scene_objects,
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
                            "position": np.array(position, copy=True),
                            "probe_records": probe_records,
                            "global_score": _position_group_score(probe_records),
                            "hotspot_score": _hotspot_group_score(probe_records),
                        }
                    )
            finally:
                _shutdown_runner(runner)
                runner = None

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
        if max_position_groups is None and args.max_states_per_position and args.max_total_states is not None:
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
            for chunk_start, group_chunk in _iter_chunks(selected_groups, args.full_restart_interval):
                print(
                    f"[state_bank] scene={scene} step=full_chunk start={chunk_start + 1} "
                    f"end={chunk_start + len(group_chunk)} total={len(selected_groups)}",
                    flush=True,
                )
                runner, _, _, _ = _load_scene_with_runner(args, scene)
                try:
                    for group in group_chunk:
                        full_records, full_count, state_index = _full_position_records(
                            runner,
                            scene=scene,
                            position=group["position"],
                            scene_objects=scene_objects,
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
                            kept_states += 1
                            for class_id_value in range(1, 11):
                                if bool(
                                    record.suitability.get(f"class{class_id_value:02d}")
                                    or record.suitability.get(f"class{class_id_value}")
                                ):
                                    kept_by_class[str(class_id_value)] += 1
                            fout.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        if args.max_total_states is not None and kept_states >= args.max_total_states:
                            break
                    if args.max_total_states is not None and kept_states >= args.max_total_states:
                        break
                finally:
                    _shutdown_runner(runner)
                    runner = None

        total_candidates = probe_candidate_states + final_candidate_states
        quality_passed_states = probe_quality_passed_states + final_quality_passed_states

        summary = {
            "scene": scene,
            "scene_file": scene_file,
            "reachable_positions": len(reachable),
            "prefilter_candidate_positions": prefilter_candidate_positions,
            "prefilter_selected_positions": len(prefiltered_positions),
            "prefilter_skipped_close_positions": prefilter_skipped_close,
            "probe_positions": len(probe_positions),
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
        if runner is not None:
            _shutdown_runner(runner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ReplicaCAD state bank for fixed scenes.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scenes", type=str, default=None, help="Comma-separated HSSD scene stems")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--sensor-height", type=float, default=1.6)
    parser.add_argument("--agent-radius", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefilter-yaws", type=str, default="0,90,180,270")
    parser.add_argument("--probe-yaws", type=str, default="0,90,180,270")
    parser.add_argument("--yaws", type=str, default="0,30,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--horizons", type=str, default="0")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-states-per-position", type=int, default=2)
    parser.add_argument("--max-total-states", type=int, default=40)
    parser.add_argument("--max-position-groups", type=int, default=20)
    parser.add_argument("--max-probe-positions", type=int, default=160)
    parser.add_argument("--prefilter-candidate-positions", type=int, default=200)
    parser.add_argument("--prefilter-top-positions", type=int, default=50)
    parser.add_argument("--prefilter-candidate-separation", type=float, default=1.0)
    parser.add_argument("--prefilter-top-separation", type=float, default=0.75)
    parser.add_argument("--min-position-separation", type=float, default=0.75)
    parser.add_argument("--probe-position-separation", type=float, default=0.5)
    parser.add_argument("--global-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-min-position-separation", type=float, default=0.35)
    parser.add_argument("--probe-restart-interval", type=int, default=0)
    parser.add_argument("--full-restart-interval", type=int, default=0)
    parser.add_argument("--disable-position-prefilter", action="store_true")
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
