#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ai2thor_env import create_controller
from schema import StateRecord, TaskProposal, VisibleObjectRecord


MIN_SHARED_OBJECTS = 1
MIN_CHAIN_FRAME_DIFF = 8.0


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


def _proposal_from_dict(data: dict) -> TaskProposal:
    return TaskProposal(
        proposal_id=data["proposal_id"],
        scene=data["scene"],
        state_id=data["state_id"],
        class_id=int(data["class_id"]),
        subcat=data["subcat"],
        score=float(data["score"]),
        payload=data.get("payload", {}),
        source=data.get("source", "enumerated"),
    )


def _save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame).save(path)


def _frame_diff_score(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def _visible_object_ids(event) -> set[str]:
    return {
        str(obj["objectId"])
        for obj in event.metadata.get("objects", [])
        if obj.get("visible", False)
    }


def _pair_has_enough_overlap(first_event, second_event) -> bool:
    first_ids = _visible_object_ids(first_event)
    second_ids = _visible_object_ids(second_event)
    return len(first_ids.intersection(second_ids)) >= MIN_SHARED_OBJECTS


def _events_have_enough_overlap(events: List[object]) -> bool:
    return all(_pair_has_enough_overlap(a, b) for a, b in zip(events, events[1:]))


def _chain_has_enough_differences(events: List[object]) -> bool:
    return all(_frame_diff_score(a.frame, b.frame) >= MIN_CHAIN_FRAME_DIFF for a, b in zip(events, events[1:]))


def _question_for_variant(variant: str) -> str:
    if variant == "two_image_single_action":
        return "给定首尾两张图，中间只执行了一个动作。这个动作是什么？"
    if variant == "two_image_double_action":
        return "给定首尾两张图，中间连续执行了两个动作。请按顺序回答这两个动作。"
    return "观察整段图像序列。每相邻两帧之间只执行了一个动作，请按顺序回答所有动作。"


def _answer_for_variant(variant: str, actions: List[dict]) -> str:
    texts = [str(action["action_text"]) for action in actions]
    if variant == "two_image_single_action":
        return texts[0]
    if variant == "two_image_double_action":
        return f"先{texts[0]}，再{texts[1]}"
    return "；".join(f"第{idx + 1}步{text}" for idx, text in enumerate(texts))


def _teleport_to_state(controller, state: StateRecord):
    return controller.step(
        action="Teleport",
        position=state.position,
        rotation={"x": 0, "y": state.yaw, "z": 0},
        horizon=state.horizon,
        standing=True,
    )


def _execute_action(controller, action: dict):
    return controller.step(action=str(action["action_api"]), **dict(action["action_params"]))


def _load_states(path: Path) -> Dict[str, StateRecord]:
    mapping: Dict[str, StateRecord] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            state = _state_from_dict(json.loads(line))
            mapping[state.state_id] = state
    return mapping


def _load_proposals(path: Path, *, subcats: set[str] | None, max_proposals: int | None) -> List[TaskProposal]:
    proposals: List[TaskProposal] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            proposal = _proposal_from_dict(json.loads(line))
            if proposal.class_id != 1:
                continue
            if subcats and proposal.subcat not in subcats:
                continue
            proposals.append(proposal)
    grouped: dict[str, List[TaskProposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[str(proposal.subcat)].append(proposal)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.score, reverse=True)

    ordered: List[TaskProposal] = []
    round_idx = 0
    while True:
        added = False
        for subcat in sorted(grouped):
            rows = grouped[subcat]
            if round_idx < len(rows):
                ordered.append(rows[round_idx])
                added = True
                if max_proposals is not None and len(ordered) >= max_proposals:
                    return ordered
        if not added:
            break
        round_idx += 1
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute class-01 proposals into final QA samples.")
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--use-cloud-rendering", action="store_true")
    parser.add_argument("--subcats", type=str, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--target-per-subcat", type=int, default=5)
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subcats = None if args.subcats is None else {part.strip() for part in args.subcats.split(",") if part.strip()}
    states = _load_states(args.state_bank)
    proposals = _load_proposals(args.proposals, subcats=subcats, max_proposals=args.max_proposals)

    if args.reset_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    image_root = args.output_root / "images"
    meta_root = args.output_root / "meta"
    image_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    records: List[dict] = []
    variant_success = Counter()
    controller_cache = {}

    try:
        for proposal_idx, proposal in enumerate(proposals, start=1):
            variant = str(proposal.subcat)
            if args.target_per_subcat is not None and variant_success[variant] >= int(args.target_per_subcat):
                records.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "status": "target_reached_skip",
                        "variant": variant,
                        "scene": proposal.scene,
                    }
                )
                continue
            state = states.get(proposal.state_id)
            if state is None:
                records.append({"proposal_id": proposal.proposal_id, "status": "missing_state"})
                continue

            controller = controller_cache.get(state.scene)
            if controller is None:
                controller = create_controller(
                    state.scene,
                    width=args.width,
                    height=args.height,
                    grid_size=args.grid_size,
                    rotate_step_degrees=30,
                    field_of_view=args.field_of_view,
                    visibility_distance=args.visibility_distance,
                    use_cloud_rendering=args.use_cloud_rendering,
                    render_instance_segmentation=False,
                )
                controller_cache[state.scene] = controller

            try:
                evt = _teleport_to_state(controller, state)
                if not bool(evt.metadata.get("lastActionSuccess", False)):
                    records.append({"proposal_id": proposal.proposal_id, "status": "teleport_failed"})
                    continue

                variant = str(proposal.payload["variant"])
                actions = list(proposal.payload["actions"])
                sample_dir = image_root / proposal.proposal_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                saved_events = [evt]
                frame_paths: List[str] = []
                start_path = sample_dir / "frame_000_start.png"
                _save_frame(evt.frame, start_path)
                frame_paths.append(str(start_path))

                current_event = evt
                for step_idx, action in enumerate(actions, start=1):
                    current_event = _execute_action(controller, action)
                    if not bool(current_event.metadata.get("lastActionSuccess", False)):
                        raise RuntimeError(f"action_failed:{action['action_api']}:{action['action_params']}")
                    should_save = variant == "multi_image_single_action_chain" or step_idx == len(actions)
                    if should_save:
                        frame_path = sample_dir / f"frame_{step_idx:03d}.png"
                        _save_frame(current_event.frame, frame_path)
                        frame_paths.append(str(frame_path))
                        saved_events.append(current_event)

                if variant == "multi_image_single_action_chain":
                    if not _chain_has_enough_differences(saved_events):
                        raise RuntimeError("chain_frame_diff_too_small")
                else:
                    if not _events_have_enough_overlap(saved_events):
                        raise RuntimeError("insufficient_overlap")

                row = {
                    "sample_id": proposal.proposal_id,
                    "task_type": "action_inference",
                    "sample_variant": variant,
                    "scene": state.scene,
                    "input": {
                        "frame_paths": frame_paths,
                        "start_frame": frame_paths[0],
                        "end_frame": frame_paths[-1],
                        "num_frames": len(frame_paths),
                    },
                    "question": _question_for_variant(variant),
                    "answer": _answer_for_variant(variant, actions),
                    "gt": {
                        "variant": variant,
                        "class_id": 1,
                        "proposal_id": proposal.proposal_id,
                        "state_id": state.state_id,
                        "actions": actions,
                        "source": proposal.source,
                    },
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                rows.append(row)
                variant_success[variant] += 1
                records.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "status": "success",
                        "variant": variant,
                        "scene": state.scene,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "status": "failed",
                        "variant": proposal.subcat,
                        "scene": state.scene,
                        "error": str(exc),
                    }
                )
            if proposal_idx % 500 == 0:
                print(
                    f"[class01_execute] processed={proposal_idx}/{len(proposals)} "
                    f"success={len(rows)}",
                    flush=True,
                )
    finally:
        for controller in controller_cache.values():
            controller.stop()

    (meta_root / "qa_data.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (meta_root / "qa_data.jsonl").open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "requested_proposals": len(proposals),
        "successful_samples": len(rows),
        "success_by_variant": dict(variant_success),
        "records": records,
    }
    (meta_root / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[class01_execute] finished requested={len(proposals)} successful={len(rows)}",
        flush=True,
    )
    print(meta_root / "stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
