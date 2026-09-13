#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import traceback

from build_state_bank import build_scene_state_bank
from enumerators.registry import enumerate_proposals
from scene_catalog import normalize_scene_tokens, room_type_for_scene
from schema import StateRecord, TaskProposal, VisibleObjectRecord


DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/output")
DEFAULT_MAX_PROPOSALS_PER_SUBCAT = 5

ROOM_SUBCAT_LIMITS = {
    "class01__two_image_single_action": 8,
    "class01__two_image_double_action": 8,
    "class01__multi_image_single_action_chain": 8,
    "class02__default": 15,
    "class03__default": 15,
    "class04__start_visible": 10,
    "class04__start_not_visible": 10,
    "class05__rotation": 8,
    "class05__translation": 8,
    "class06__rotate": 10,
    "class06__open_close": 10,
    "class06__toggle": 10,
    "class06__remove": 10,
    "class07__default": 15,
    "class08__distance_change": 15,
    "class08__occlusion_change": 15,
    "class09__default": 8,
    "class10__single_visibility": 10,
    "class10__two_frame_direction": 10,
    "class10__single_direction": 10,
    "class10__trend_disappear": 10,
}

ROOM_TYPE_CLASS_WEIGHTS = {
    "kitchen": {2: 1.12, 3: 1.14, 6: 1.10, 7: 1.14, 8: 1.14, 10: 1.08},
    "living_room": {1: 1.10, 4: 1.08, 5: 1.08, 7: 1.10, 9: 1.12, 10: 1.06},
    "bedroom": {1: 1.10, 4: 1.08, 5: 1.08, 6: 1.08, 9: 1.10, 10: 1.08, 7: 0.92, 8: 0.92},
    "bathroom": {1: 1.08, 4: 1.08, 5: 1.08, 6: 1.10, 9: 1.08, 10: 1.06, 3: 0.88, 7: 0.82, 8: 0.82},
}

OBJECT_HEAVY_CLASSES = {2, 3, 6, 7, 8, 10}
GLOBAL_VIEW_CLASSES = {1, 4, 5, 9}


def _state_from_dict(data: dict) -> StateRecord:
    visible_objects = [VisibleObjectRecord(**item) for item in data.get("visible_objects", [])]
    return StateRecord(
        scene=data["scene"],
        scene_id=int(data["scene_id"]),
        state_id=data["state_id"],
        position=data["position"],
        yaw=int(data["yaw"]),
        horizon=int(data["horizon"]),
        metrics=data.get("metrics", {}),
        visible_objects=visible_objects,
        suitability=data.get("suitability", {}),
    )


def _limit_state_proposals(
    proposals: list[TaskProposal],
    per_subcat_limit: int,
) -> list[TaskProposal]:
    if per_subcat_limit <= 0:
        return []
    grouped: dict[tuple[int, str], list[TaskProposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[(int(proposal.class_id), str(proposal.subcat))].append(proposal)

    kept: list[TaskProposal] = []
    for key in sorted(grouped.keys()):
        ranked = sorted(
            grouped[key],
            key=lambda proposal: (-float(proposal.score), str(proposal.proposal_id)),
        )
        kept.extend(ranked[:per_subcat_limit])

    kept.sort(
        key=lambda proposal: (
            int(proposal.class_id),
            str(proposal.subcat),
            -float(proposal.score),
            str(proposal.proposal_id),
        )
    )
    return kept


def _proposal_signature(proposal: TaskProposal) -> tuple[object, ...]:
    payload = dict(proposal.payload)
    class_id = int(proposal.class_id)
    subcat = str(proposal.subcat)

    if class_id == 1:
        actions = tuple(
            (
                str(action.get("action_api")),
                tuple(sorted(dict(action.get("action_params", {})).items())),
            )
            for action in payload.get("actions", [])
        )
        return (class_id, subcat, actions)
    if class_id == 2:
        pair = tuple(
            sorted(
                [
                    str(payload.get("reference_object_id", "")),
                    str(payload.get("target_object_id", "")),
                ]
            )
        )
        return (class_id, subcat, pair)
    if class_id == 3:
        return (
            class_id,
            subcat,
            str(payload.get("near_object_id", "")),
            str(payload.get("far_object_id", "")),
        )
    if class_id == 4:
        return (
            class_id,
            subcat,
            str(payload.get("target_object_id") or payload.get("anchor_object_id") or ""),
        )
    if class_id == 5:
        return (
            class_id,
            subcat,
            tuple(sorted(payload.items())),
        )
    if class_id == 6:
        return (class_id, subcat, str(payload.get("target_object_id", "")))
    if class_id == 7:
        pair = tuple(
            sorted(
                [
                    str(payload.get("first_object_id", "")),
                    str(payload.get("second_object_id", "")),
                ]
            )
        )
        return (class_id, subcat, pair)
    if class_id == 8:
        return (class_id, subcat, str(payload.get("target_object_id", "")))
    if class_id == 9:
        return (
            class_id,
            subcat,
            str(payload.get("anchor_object_id", "")),
            str(payload.get("target_object_id", "")),
        )
    if class_id == 10:
        return (class_id, subcat, str(payload.get("target_object_id", "")))
    return (class_id, subcat, tuple(sorted(payload.items())))


def _room_adjusted_score(
    proposal: TaskProposal,
    state: StateRecord,
    room_type: str,
) -> float:
    score = float(proposal.score)
    selection_group = str(state.metrics.get("selection_group", "fallback"))

    if int(proposal.class_id) in OBJECT_HEAVY_CLASSES:
        if selection_group == "hotspot":
            score += 0.30
        elif selection_group == "fallback":
            score += 0.08
    if int(proposal.class_id) in GLOBAL_VIEW_CLASSES and selection_group == "global":
        score += 0.15

    score *= ROOM_TYPE_CLASS_WEIGHTS.get(room_type, {}).get(int(proposal.class_id), 1.0)
    visible_count = float(state.metrics.get("visible_count", len(state.visible_objects)))
    if int(proposal.class_id) in OBJECT_HEAVY_CLASSES:
        score += 0.02 * min(visible_count, 12.0)
    return score


def _select_room_candidates(
    room_type: str,
    state_proposals: list[tuple[StateRecord, TaskProposal]],
    default_limit: int,
) -> tuple[list[TaskProposal], dict[str, int]]:
    grouped: dict[str, list[TaskProposal]] = defaultdict(list)
    for state, proposal in state_proposals:
        proposal.score = _room_adjusted_score(proposal, state, room_type)
        grouped[f"class{int(proposal.class_id):02d}__{proposal.subcat}"].append(proposal)

    selected: list[TaskProposal] = []
    selected_counts: dict[str, int] = {}
    for subcat_key in sorted(grouped.keys()):
        best_by_signature: dict[tuple[object, ...], TaskProposal] = {}
        for proposal in grouped[subcat_key]:
            signature = _proposal_signature(proposal)
            current = best_by_signature.get(signature)
            if current is None or float(proposal.score) > float(current.score):
                best_by_signature[signature] = proposal
        ranked = sorted(
            best_by_signature.values(),
            key=lambda item: (-float(item.score), str(item.proposal_id)),
        )
        limit = ROOM_SUBCAT_LIMITS.get(subcat_key, default_limit)
        kept = ranked[:limit]
        selected.extend(kept)
        selected_counts[subcat_key] = len(kept)

    selected.sort(
        key=lambda proposal: (
            int(proposal.class_id),
            str(proposal.subcat),
            -float(proposal.score),
            str(proposal.proposal_id),
        )
    )
    return selected, selected_counts


def process_scene(args: argparse.Namespace, scene: str) -> dict:
    print(f"[scene_worker] start scene={scene}", flush=True)
    scene_dir = Path(args.output_root) / scene
    room_type = room_type_for_scene(scene)
    scene_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene_summary = build_scene_state_bank(args, scene)
        state_bank_path = scene_dir / "state_bank.jsonl"
        proposals_path = scene_dir / "proposals.jsonl"

        proposal_count = 0
        per_class = {str(idx): 0 for idx in range(1, 11)}
        state_proposals: list[tuple[StateRecord, TaskProposal]] = []
        with state_bank_path.open("r", encoding="utf-8") as fin:
            for state_idx, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                state = _state_from_dict(json.loads(line))
                proposals = _limit_state_proposals(
                    list(enumerate_proposals(state)),
                    args.max_proposals_per_subcat,
                )
                for proposal in proposals:
                    state_proposals.append((state, proposal))
                if state_idx % 100 == 0:
                    print(
                        f"[scene_worker] scene={scene} proposal_states={state_idx} "
                        f"raw_proposal_count={len(state_proposals)}",
                        flush=True,
                    )

        selected_proposals, selected_by_subcat = _select_room_candidates(
            room_type,
            state_proposals,
            args.max_proposals_per_subcat,
        )
        with proposals_path.open("w", encoding="utf-8") as fout:
            for proposal in selected_proposals:
                fout.write(json.dumps(proposal.to_dict(), ensure_ascii=False) + "\n")
                proposal_count += 1
                per_class[str(proposal.class_id)] += 1

        result = {
            **scene_summary,
            "status": "ok",
            "room_type": room_type,
            "raw_proposal_count": len(state_proposals),
            "proposal_count": proposal_count,
            "proposal_count_by_class": per_class,
            "proposal_count_by_subcat": selected_by_subcat,
            "proposal_limit_per_subcat": args.max_proposals_per_subcat,
            "proposals_path": str(proposals_path),
        }
        print(
            f"[scene_worker] finished scene={scene} kept_states={scene_summary['kept_states']} "
            f"proposal_count={proposal_count}",
            flush=True,
        )
    except Exception as exc:
        result = {
            "scene": scene,
            "status": "failed",
            "room_type": room_type,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "state_bank_path": str(scene_dir / "state_bank.jsonl"),
            "proposals_path": str(scene_dir / "proposals.jsonl"),
            "raw_proposal_count": 0,
            "proposal_count": 0,
            "proposal_count_by_class": {str(idx): 0 for idx in range(1, 11)},
            "proposal_count_by_subcat": {},
            "proposal_limit_per_subcat": args.max_proposals_per_subcat,
            "probe_candidate_states": 0,
            "final_candidate_states": 0,
            "quality_passed_states": 0,
            "probe_quality_passed_states": 0,
            "final_quality_passed_states": 0,
            "kept_states": 0,
            "skipped_duplicate_states": 0,
            "skipped_close_positions": 0,
            "kept_by_class": {str(idx): 0 for idx in range(1, 11)},
        }
        print(
            f"[scene_worker] failed scene={scene} error={exc}",
            flush=True,
        )

    (scene_dir / "scene_pipeline_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process one or more AI2THOR scenes into state/proposal banks.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes", type=str, required=True)
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
    parser.add_argument("--max-proposals-per-subcat", type=int, default=DEFAULT_MAX_PROPOSALS_PER_SUBCAT)
    parser.add_argument("--min-position-separation", type=float, default=0.75)
    parser.add_argument("--global-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-min-position-separation", type=float, default=0.35)
    parser.add_argument("--use-cloud-rendering", action="store_true")
    parser.add_argument("--disable-state-dedup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = normalize_scene_tokens(args.scenes.split(","))
    summaries = [process_scene(args, scene) for scene in scenes]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
