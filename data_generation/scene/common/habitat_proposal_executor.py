from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import quaternion
from habitat_sim.agent import ActionSpec as HabitatActionSpec
from habitat_sim.agent.controls import ActuationSpec
from habitat_sim.utils.common import quat_rotate_vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from habitat_qa_generators import (
    HabitatDatasetContext,
    HabitatSceneRunner,
    _direction_from_local,
    _frame_diff_score,
    _horizontal_location,
)


DATASET_ROOT = "/path/to/workspace/habitat_data"

MOVE_KEY_PREFIX = {
    "MoveAhead": "move_forward",
    "MoveBack": "move_backward",
    "MoveLeft": "move_left",
    "MoveRight": "move_right",
}
ROTATE_KEY_PREFIX = {
    "RotateLeft": "turn_left",
    "RotateRight": "turn_right",
}
LOOK_KEY_PREFIX = {
    "LookUp": "look_up",
    "LookDown": "look_down",
}

MOVE_TEXT = {
    "MoveAhead": "向前移动",
    "MoveBack": "向后移动",
    "MoveLeft": "向左平移",
    "MoveRight": "向右平移",
}
ROTATE_TEXT = {
    "RotateLeft": "向左旋转",
    "RotateRight": "向右旋转",
}
LOOK_TEXT = {
    "LookUp": "向上抬头",
    "LookDown": "向下低头",
}

TASK_TYPE_BY_CLASS = {
    1: "action_inference",
    2: "multi_image_overlap_localization",
    3: "parallax_depth_inference",
    4: "movement_sequence_sorting",
    5: "movement_degree_comparison",
    6: "object_state_attribute_changes",
    7: "object_position_swapping",
    8: "dynamic_movement_occlusion",
    9: "imagined_perspective_taking",
    10: "imagined_movement_consequence",
}

MOVE_MAGS = (0.25, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.75, 0.8, 1.0)
ROTATE_DEGS = (15, 30, 45, 60, 75, 90)
LOOK_DEGS = (30, 45, 60)


@dataclass
class VisibleObjectRecord:
    object_id: str
    object_type: str
    name: str
    distance: float
    pickupable: bool
    moveable: bool
    openable: bool
    toggleable: bool
    receptacle: bool
    rotation_y: float | None
    bbox_area_ratio: float | None
    center_x_ratio: float | None
    center_y_ratio: float | None
    edge_margin_ratio: float | None


@dataclass
class StateRecord:
    scene: str
    scene_id: int
    state_id: str
    position: Dict[str, float]
    yaw: int
    horizon: int
    metrics: Dict[str, object]
    visible_objects: List[VisibleObjectRecord]
    suitability: Dict[str, object]


@dataclass
class TaskProposal:
    proposal_id: str
    scene: str
    state_id: str
    class_id: int
    subcat: str
    score: float
    payload: Dict[str, object]
    source: str = "enumerated"


@dataclass
class ActionSpecLite:
    action_key: str
    action_api: str
    action_params: Dict[str, float]
    action_text: str
    family: str


class ExtendedHabitatRunner(HabitatSceneRunner):
    def _build_action_space(self) -> Dict[str, HabitatActionSpec]:
        action_space: Dict[str, HabitatActionSpec] = {}
        for mag in MOVE_MAGS:
            for key_prefix in MOVE_KEY_PREFIX.values():
                key = f"{key_prefix}_{mag:.2f}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(mag)),
                )
        for deg in ROTATE_DEGS:
            for key_prefix in ROTATE_KEY_PREFIX.values():
                key = f"{key_prefix}_{int(deg)}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(deg)),
                )
        for deg in LOOK_DEGS:
            for key_prefix in LOOK_KEY_PREFIX.values():
                key = f"{key_prefix}_{int(deg)}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(deg)),
                )
        return action_space


def _state_from_dict(data: dict) -> StateRecord:
    visible_objects = [VisibleObjectRecord(**item) for item in data.get("visible_objects", [])]
    return StateRecord(
        scene=str(data["scene"]),
        scene_id=int(data["scene_id"]),
        state_id=str(data["state_id"]),
        position=dict(data["position"]),
        yaw=int(data["yaw"]),
        horizon=int(data["horizon"]),
        metrics=dict(data.get("metrics", {})),
        visible_objects=visible_objects,
        suitability=dict(data.get("suitability", {})),
    )


def _proposal_from_dict(data: dict) -> TaskProposal:
    return TaskProposal(
        proposal_id=str(data["proposal_id"]),
        scene=str(data["scene"]),
        state_id=str(data["state_id"]),
        class_id=int(data["class_id"]),
        subcat=str(data["subcat"]),
        score=float(data["score"]),
        payload=dict(data.get("payload", {})),
        source=str(data.get("source", "enumerated")),
    )


def _parse_csv_set(raw: str | None) -> Optional[set[str]]:
    if raw is None:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _load_states(path: Path) -> Dict[str, StateRecord]:
    states: Dict[str, StateRecord] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            state = _state_from_dict(json.loads(line))
            states[state.state_id] = state
    return states


def _load_proposals(path: Path, class_id: int, subcats: Optional[set[str]], max_proposals: Optional[int]) -> List[TaskProposal]:
    grouped: Dict[str, List[TaskProposal]] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            proposal = _proposal_from_dict(json.loads(line))
            if proposal.class_id != int(class_id):
                continue
            if subcats and proposal.subcat not in subcats:
                continue
            grouped.setdefault(proposal.subcat, []).append(proposal)
    for rows in grouped.values():
        rows.sort(key=lambda item: (-float(item.score), str(item.proposal_id)))
    ordered: List[TaskProposal] = []
    round_idx = 0
    while True:
        added = False
        for subcat in sorted(grouped):
            rows = grouped[subcat]
            if round_idx < len(rows):
                ordered.append(rows[round_idx])
                added = True
                if max_proposals is not None and len(ordered) >= int(max_proposals):
                    return ordered
        if not added:
            break
        round_idx += 1
    return ordered


def _pose_text(pose) -> str:
    return f"Pos=({float(pose.x):.3f}, {float(pose.y):.3f}, {float(pose.z):.3f}), Rot={float(pose.yaw):.1f}, Horizon={float(pose.pitch):.1f}"


def _pose_raw(pose) -> Dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw": float(pose.yaw),
        "pitch": float(pose.pitch),
    }


def _state_position(state: StateRecord) -> List[float]:
    return [
        float(state.position["x"]),
        float(state.position["y"]),
        float(state.position["z"]),
    ]


def _world_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _quat_from_wxyz(values: Sequence[float]) -> quaternion.quaternion:
    return quaternion.quaternion(float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _quat_to_wxyz(q: quaternion.quaternion) -> List[float]:
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def _yaw_quaternion(degrees: float) -> quaternion.quaternion:
    radians = math.radians(float(degrees))
    return quaternion.quaternion(math.cos(radians / 2.0), 0.0, math.sin(radians / 2.0), 0.0)


def _make_move_action(action_api: str, magnitude: float) -> ActionSpecLite:
    return ActionSpecLite(
        action_key=f"{MOVE_KEY_PREFIX[action_api]}_{float(magnitude):.2f}",
        action_api=action_api,
        action_params={"moveMagnitude": float(magnitude)},
        action_text=f"{MOVE_TEXT[action_api]}{float(magnitude):.2f}米",
        family="move",
    )


def _make_rotate_action(action_api: str, degrees: float) -> ActionSpecLite:
    return ActionSpecLite(
        action_key=f"{ROTATE_KEY_PREFIX[action_api]}_{int(degrees)}",
        action_api=action_api,
        action_params={"degrees": float(degrees)},
        action_text=f"{ROTATE_TEXT[action_api]}{int(degrees)}度",
        family="rotate",
    )


def _make_look_action(action_api: str, degrees: float) -> ActionSpecLite:
    return ActionSpecLite(
        action_key=f"{LOOK_KEY_PREFIX[action_api]}_{int(degrees)}",
        action_api=action_api,
        action_params={"degrees": float(degrees)},
        action_text=f"{LOOK_TEXT[action_api]}{int(degrees)}度",
        family="look",
    )


def _action_from_payload(data: dict) -> ActionSpecLite:
    action_api = str(data["action_api"])
    params = dict(data.get("action_params", {}))
    if action_api in MOVE_KEY_PREFIX:
        return _make_move_action(action_api, float(params["moveMagnitude"]))
    if action_api in ROTATE_KEY_PREFIX:
        return _make_rotate_action(action_api, float(params["degrees"]))
    if action_api in LOOK_KEY_PREFIX:
        return _make_look_action(action_api, float(params["degrees"]))
    raise RuntimeError(f"unsupported_action:{action_api}")


def _save_frame(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path)


def _sample_dir(image_root: Path, proposal: TaskProposal) -> Path:
    path = image_root / proposal.proposal_id
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ProposalExecutor:
    def __init__(
        self,
        args: argparse.Namespace,
        dataset_name: str,
        resolve_scene_path: Callable[[str], str],
        class_id: int,
    ):
        self.args = args
        self.dataset_name = dataset_name
        self.resolve_scene_path = resolve_scene_path
        self.class_id = int(class_id)
        self.states = _load_states(args.state_bank)
        self.proposals = _load_proposals(
            args.proposals,
            class_id=self.class_id,
            subcats=_parse_csv_set(args.subcats),
            max_proposals=args.max_proposals,
        )
        self.output_root = args.output_root
        self.image_root = self.output_root / "images"
        self.meta_root = self.output_root / "meta"
        self.rows: List[dict] = []
        self.records: List[dict] = []
        self.subcat_success: Dict[str, int] = {}
        self.ctx = HabitatDatasetContext(
            dataset=self.dataset_name,
            dataset_root=DATASET_ROOT,
            image_width=args.width,
            image_height=args.height,
            seed=42,
        )
        self.runner = ExtendedHabitatRunner(
            ctx=self.ctx,
            image_width=args.width,
            image_height=args.height,
            hfov=args.field_of_view,
            sensor_height=1.6,
            agent_radius=0.1,
            seed=42,
        )
        self._current_scene_path: Optional[str] = None
        self._source_scene_name: Optional[str] = None
        self._source_scene_path: Optional[str] = None
        self._source_scene_data: Optional[dict] = None
        self._source_scene_objects: Optional[List[dict]] = None

    def _prepare_output(self) -> None:
        if self.args.reset_output and self.output_root.exists():
            shutil.rmtree(self.output_root)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.meta_root.mkdir(parents=True, exist_ok=True)

    def _scene_assets(self, scene_name: str) -> Tuple[str, dict, List[dict]]:
        if self._source_scene_name == scene_name and self._source_scene_data is not None and self._source_scene_objects is not None:
            return self._source_scene_path, self._source_scene_data, self._source_scene_objects
        scene_path = self.resolve_scene_path(scene_name)
        scene_data = self.ctx.load_scene(scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)
        self._source_scene_name = scene_name
        self._source_scene_path = scene_path
        self._source_scene_data = scene_data
        self._source_scene_objects = scene_objects
        return scene_path, scene_data, scene_objects

    def _ensure_scene(self, scene_path: str, navmesh_key_path: str) -> None:
        if self._current_scene_path != scene_path:
            self.runner.build(scene_path, navmesh_key_path=navmesh_key_path)
            self._current_scene_path = scene_path

    def _start_obs(
        self,
        state: StateRecord,
        *,
        scene_path: Optional[str] = None,
        scene_objects: Optional[List[dict]] = None,
        navmesh_key_path: Optional[str] = None,
    ) -> Tuple[dict, List[dict]]:
        base_scene_path, _, base_scene_objects = self._scene_assets(state.scene)
        use_scene_path = scene_path or base_scene_path
        use_scene_objects = scene_objects or base_scene_objects
        self._ensure_scene(use_scene_path, navmesh_key_path or base_scene_path)
        obs = self.runner.set_agent_state(_state_position(state), float(state.yaw), float(state.horizon))
        return obs, use_scene_objects

    def _visible_map(self, scene_objects: List[dict], obs: dict) -> Dict[int, object]:
        visible = self.runner.visible_objects(scene_objects, obs["depth"])
        return {int(obj.index): obj for obj in visible}

    def _scene_object(self, scene_objects: List[dict], object_id: str) -> Optional[dict]:
        try:
            index = int(str(object_id))
        except Exception:
            return None
        for obj in scene_objects:
            if int(obj["index"]) == index:
                return obj
        return None

    def _visible_object_name(self, state: StateRecord, object_id: str, fallback: str = "物体") -> str:
        for obj in state.visible_objects:
            if str(obj.object_id) == str(object_id):
                return str(obj.name)
        return fallback

    def _record_step(self, step_id: int, action: ActionSpecLite, before_pose, after_pose) -> dict:
        return {
            "step_id": int(step_id),
            "action_api": str(action.action_api),
            "action_params": dict(action.action_params),
            "action_text": str(action.action_text),
            "success": True,
            "before_pose": _pose_raw(before_pose),
            "after_pose": _pose_raw(after_pose),
        }

    def _execute_actions(self, start_obs: dict, actions: Sequence[ActionSpecLite]) -> Tuple[List[dict], List[dict]]:
        observations = [start_obs]
        steps: List[dict] = []
        current = start_obs
        for step_idx, action in enumerate(actions, start=1):
            before_pose = current["pose"]
            current = self.runner.execute_action(action)
            steps.append(self._record_step(step_idx, action, before_pose, current["pose"]))
            observations.append(current)
        return observations, steps

    def _state_gt(self, pose) -> dict:
        return {"text": _pose_text(pose), "raw": _pose_raw(pose)}

    def _base_row(self, proposal: TaskProposal, scene_path: str) -> dict:
        return {
            "sample_id": proposal.proposal_id,
            "task_type": TASK_TYPE_BY_CLASS[self.class_id],
            "scene": os.path.basename(scene_path),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

    def _write_meta(self) -> None:
        qa_jsonl_path = self.meta_root / "qa_data.jsonl"
        qa_json_path = self.meta_root / "qa_data.json"
        records_path = self.meta_root / "records.json"
        stats_path = self.meta_root / "stats.json"

        with qa_jsonl_path.open("w", encoding="utf-8") as fout:
            for row in self.rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        qa_json_path.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8")
        records_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        stats = {
            "class_id": self.class_id,
            "task_type": TASK_TYPE_BY_CLASS[self.class_id],
            "requested_proposals": len(self.proposals),
            "generated_samples": len(self.rows),
            "success_by_subcat": self.subcat_success,
            "jsonl_path": str(qa_jsonl_path),
            "json_path": str(qa_json_path),
            "records_path": str(records_path),
            "image_root": str(self.image_root),
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> int:
        self._prepare_output()
        try:
            for proposal in self.proposals:
                if self.args.target_per_subcat is not None:
                    if int(self.subcat_success.get(proposal.subcat, 0)) >= int(self.args.target_per_subcat):
                        self.records.append(
                            {
                                "proposal_id": proposal.proposal_id,
                                "subcat": proposal.subcat,
                                "status": "target_reached_skip",
                            }
                        )
                        continue
                state = self.states.get(proposal.state_id)
                if state is None:
                    self.records.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "subcat": proposal.subcat,
                            "status": "missing_state",
                        }
                    )
                    continue
                try:
                    row = self._execute_one(state, proposal)
                    self.rows.append(row)
                    self.records.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "subcat": proposal.subcat,
                            "status": "success",
                        }
                    )
                    self.subcat_success[proposal.subcat] = int(self.subcat_success.get(proposal.subcat, 0)) + 1
                except Exception as exc:
                    self.records.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "subcat": proposal.subcat,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
        finally:
            self.runner.shutdown()
        self._write_meta()
        return 0 if (not self.proposals or self.rows) else 1

    def _execute_one(self, state: StateRecord, proposal: TaskProposal) -> dict:
        if self.class_id == 1:
            return self._execute_class01(state, proposal)
        if self.class_id == 2:
            return self._execute_class02(state, proposal)
        if self.class_id == 3:
            return self._execute_class03(state, proposal)
        if self.class_id == 4:
            return self._execute_class04(state, proposal)
        if self.class_id == 5:
            return self._execute_class05(state, proposal)
        if self.class_id == 6:
            return self._execute_class06(state, proposal)
        if self.class_id == 7:
            return self._execute_class07(state, proposal)
        if self.class_id == 8:
            return self._execute_class08(state, proposal)
        if self.class_id == 9:
            return self._execute_class09(state, proposal)
        if self.class_id == 10:
            return self._execute_class10(state, proposal)
        raise RuntimeError(f"unsupported_class:{self.class_id}")

    def _execute_class01(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        actions = [_action_from_payload(item) for item in proposal.payload.get("actions", [])]
        observations, steps = self._execute_actions(start_obs, actions)

        variant = str(proposal.payload["variant"])
        sample_dir = _sample_dir(self.image_root, proposal)
        frame_paths: List[str] = []
        start_path = sample_dir / "frame_000_start.png"
        _save_frame(observations[0]["rgb"], start_path)
        frame_paths.append(str(start_path))
        if variant == "multi_image_single_action_chain":
            for idx, obs in enumerate(observations[1:], start=1):
                frame_path = sample_dir / f"frame_{idx:03d}.png"
                _save_frame(obs["rgb"], frame_path)
                frame_paths.append(str(frame_path))
        else:
            end_path = sample_dir / f"frame_{len(actions):03d}.png"
            _save_frame(observations[-1]["rgb"], end_path)
            frame_paths.append(str(end_path))

        if variant == "two_image_single_action":
            question = "给定首尾两张图，中间只执行了一个动作。这个动作是什么？"
            answer = steps[0]["action_text"]
        elif variant == "two_image_double_action":
            question = "给定首尾两张图，中间连续执行了两个动作。请按顺序回答这两个动作。"
            answer = f"先{steps[0]['action_text']}，再{steps[1]['action_text']}"
        else:
            question = "观察整段图像序列。每相邻两帧之间只执行了一个动作，请按顺序回答所有动作。"
            answer = "；".join(f"第{idx + 1}步{step['action_text']}" for idx, step in enumerate(steps))

        row = self._base_row(proposal, scene_path)
        row["sample_variant"] = variant
        row["input"] = {
            "frame_paths": frame_paths,
            "start_frame": frame_paths[0],
            "end_frame": frame_paths[-1],
            "num_frames": len(frame_paths),
        }
        row["question"] = question
        row["answer"] = answer
        row["gt"] = {
            "variant": variant,
            "action_count": len(actions),
            "api_actions": [step["action_api"] for step in steps],
            "action_texts": [step["action_text"] for step in steps],
            "steps": steps,
            "start_state": self._state_gt(observations[0]["pose"]),
            "end_state": self._state_gt(observations[-1]["pose"]),
            "scene_path": scene_path,
            "scene_dataset_config": self.ctx.scene_dataset_config,
            "proposal_id": proposal.proposal_id,
            "state_id": proposal.state_id,
        }
        return row

    def _candidate_overlap_sequences(self, target_local_x: float) -> List[List[ActionSpecLite]]:
        turn_first = "RotateRight" if target_local_x >= 0 else "RotateLeft"
        turn_second = "RotateLeft" if turn_first == "RotateRight" else "RotateRight"
        sequences: List[List[ActionSpecLite]] = []
        for deg1 in (30, 45, 60):
            for deg2 in (15, 30, 45):
                sequences.append([_make_rotate_action(turn_first, deg1), _make_rotate_action(turn_first, deg2)])
                sequences.append([_make_rotate_action(turn_first, deg1), _make_rotate_action(turn_second, deg2)])
        for deg in (30, 45):
            for mag in (0.4, 0.6):
                sequences.append([_make_rotate_action(turn_first, deg), _make_move_action("MoveAhead", mag)])
                sequences.append([_make_move_action("MoveAhead", mag), _make_rotate_action(turn_first, deg)])
        return sequences

    def _execute_class02(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        reference_id = str(proposal.payload["reference_object_id"])
        target_id = str(proposal.payload["target_object_id"])
        reference_name = str(proposal.payload.get("reference_name") or self._visible_object_name(state, reference_id))
        target_name = str(proposal.payload.get("target_name") or self._visible_object_name(state, target_id))

        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        ref_obj = self._scene_object(scene_objects, reference_id)
        target_obj = self._scene_object(scene_objects, target_id)
        if ref_obj is None or target_obj is None:
            raise RuntimeError("missing_scene_object")

        dx = float(target_obj["position"][0]) - float(state.position["x"])
        dz = float(target_obj["position"][2]) - float(state.position["z"])
        angle = math.radians(float(state.yaw))
        target_local_x = math.cos(angle) * dx - math.sin(angle) * dz

        best = None
        for actions in self._candidate_overlap_sequences(target_local_x):
            start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
            observations, steps = self._execute_actions(start_obs, actions)
            vis_a = self._visible_map(scene_objects, observations[0])
            vis_b = self._visible_map(scene_objects, observations[1])
            vis_c = self._visible_map(scene_objects, observations[2])
            if int(target_id) not in vis_c:
                continue
            shared = len(set(vis_a).intersection(vis_b)) + len(set(vis_b).intersection(vis_c))
            diff = _frame_diff_score(observations[0]["rgb"], observations[2]["rgb"])
            score = float(shared) + 0.05 * diff
            if best is None or score > best[0]:
                best = (score, observations, steps)
        if best is None:
            raise RuntimeError("no_valid_overlap_sequence")

        _, observations, steps = best
        sample_dir = _sample_dir(self.image_root, proposal)
        path_a = sample_dir / "frame_000_A.png"
        path_b = sample_dir / "frame_001_B.png"
        path_c = sample_dir / "frame_002_C.png"
        _save_frame(observations[0]["rgb"], path_a)
        _save_frame(observations[1]["rgb"], path_b)
        _save_frame(observations[2]["rgb"], path_c)

        rel_dx = float(target_obj["position"][0]) - float(ref_obj["position"][0])
        rel_dz = float(target_obj["position"][2]) - float(ref_obj["position"][2])
        local_x = math.cos(angle) * rel_dx - math.sin(angle) * rel_dz
        local_z = math.sin(angle) * rel_dx + math.cos(angle) * rel_dz
        direction = _direction_from_local(local_x, local_z)

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a), str(path_b), str(path_c)],
            "frame_A": str(path_a),
            "frame_B": str(path_b),
            "frame_C": str(path_c),
        }
        row["question"] = (
            f"结合图A、图B、图C的重叠区域，并以图A的视角朝向为参考，"
            f"请判断图C中的【{target_name}】相对于图A中的【{reference_name}】在图A视角下更接近哪个水平方位？"
        )
        row["answer"] = direction
        row["gt"] = {
            "api_actions": [step["action_api"] for step in steps],
            "steps": steps,
            "start_state": self._state_gt(observations[0]["pose"]),
            "mid_state": self._state_gt(observations[1]["pose"]),
            "end_state": self._state_gt(observations[2]["pose"]),
            "reference_object_id": reference_id,
            "target_object_id": target_id,
            "direction_word": direction,
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _execute_class03(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        near_id = str(proposal.payload["near_object_id"])
        far_id = str(proposal.payload["far_object_id"])
        near_name = str(proposal.payload.get("near_name") or self._visible_object_name(state, near_id))
        far_name = str(proposal.payload.get("far_name") or self._visible_object_name(state, far_id))

        best = None
        candidate_actions = [
            _make_move_action("MoveAhead", 0.25),
            _make_move_action("MoveLeft", 0.25),
            _make_move_action("MoveRight", 0.25),
            _make_rotate_action("RotateLeft", 15),
            _make_rotate_action("RotateRight", 15),
        ]
        for action in candidate_actions:
            start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
            observations, steps = self._execute_actions(start_obs, [action])
            vis_b = self._visible_map(scene_objects, observations[1])
            score = _frame_diff_score(observations[0]["rgb"], observations[1]["rgb"])
            visible_bonus = int(int(near_id) in vis_b) + int(int(far_id) in vis_b)
            total = 10.0 * visible_bonus + score
            if best is None or total > best[0]:
                best = (total, observations, steps)
        if best is None:
            raise RuntimeError("no_valid_parallax_action")

        _, observations, steps = best
        path_a = _sample_dir(self.image_root, proposal) / "frame_000_A.png"
        path_b = path_a.parent / "frame_001_B.png"
        _save_frame(observations[0]["rgb"], path_a)
        _save_frame(observations[1]["rgb"], path_b)

        near_obj = self._scene_object(scene_objects, near_id)
        far_obj = self._scene_object(scene_objects, far_id)
        if near_obj is None or far_obj is None:
            raise RuntimeError("missing_scene_object")
        camera_pos = _state_position(state)
        near_dist = _world_distance(camera_pos, near_obj["position"])
        far_dist = _world_distance(camera_pos, far_obj["position"])
        answer = f"【{near_name}】距离更近。" if near_dist <= far_dist else f"【{far_name}】距离更近。"

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a), str(path_b)],
            "frame_A": str(path_a),
            "frame_B": str(path_b),
        }
        row["question"] = f"在这两张图像中，【{near_name}】和【{far_name}】哪个距离相机更近？"
        row["answer"] = answer
        row["gt"] = {
            "api_actions": [step["action_api"] for step in steps],
            "steps": steps,
            "start_state": self._state_gt(observations[0]["pose"]),
            "end_state": self._state_gt(observations[-1]["pose"]),
            "camera_position_A": {
                "x": float(state.position["x"]),
                "y": float(state.position["y"]),
                "z": float(state.position["z"]),
            },
            "objects": {
                "object_a": {
                    "id": int(near_id),
                    "name": near_name,
                    "position": {
                        "x": float(near_obj["position"][0]),
                        "y": float(near_obj["position"][1]),
                        "z": float(near_obj["position"][2]),
                    },
                    "distance_to_camera_A": near_dist,
                },
                "object_b": {
                    "id": int(far_id),
                    "name": far_name,
                    "position": {
                        "x": float(far_obj["position"][0]),
                        "y": float(far_obj["position"][1]),
                        "z": float(far_obj["position"][2]),
                    },
                    "distance_to_camera_A": far_dist,
                },
            },
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _route_actions(self, state: StateRecord, scene_objects: List[dict], target_id: str) -> List[ActionSpecLite]:
        target = self._scene_object(scene_objects, target_id)
        if target is None:
            raise RuntimeError("missing_scene_object")
        dx = float(target["position"][0]) - float(state.position["x"])
        dz = float(target["position"][2]) - float(state.position["z"])
        current_yaw = float(state.yaw)
        target_yaw = math.degrees(math.atan2(dx, dz))
        delta = ((target_yaw - current_yaw + 180.0) % 360.0) - 180.0
        turn_api = "RotateRight" if delta > 0 else "RotateLeft"
        turn_deg = min(90, max(30, int(round(abs(delta) / 15.0) * 15.0)))
        if turn_deg not in ROTATE_DEGS:
            turn_deg = min(ROTATE_DEGS, key=lambda item: abs(item - turn_deg))
        distance = math.hypot(dx, dz)
        move_mag = 0.8 if distance > 2.5 else 0.6
        return [
            _make_rotate_action(turn_api, turn_deg),
            _make_move_action("MoveAhead", move_mag),
            _make_move_action("MoveAhead", move_mag),
        ]

    def _execute_class04(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        target_id = str(proposal.payload.get("target_object_id") or proposal.payload.get("anchor_object_id"))
        target_name = self._visible_object_name(state, target_id)

        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        observations, steps = self._execute_actions(start_obs, self._route_actions(state, scene_objects, target_id))
        if len(observations) != 4:
            raise RuntimeError("invalid_route_length")

        rng = random.Random(proposal.proposal_id)
        labels = ["A", "B", "C", "D"]
        shuffled = labels[:]
        rng.shuffle(shuffled)
        sample_dir = _sample_dir(self.image_root, proposal)
        labeled_frames: Dict[str, str] = {}
        chronological_labels: List[str] = []
        for obs, label in zip(observations, shuffled):
            frame_path = sample_dir / f"frame_{label}.png"
            _save_frame(obs["rgb"], frame_path)
            labeled_frames[label] = str(frame_path)
            chronological_labels.append(label)

        route_type = "turn_then_advance"
        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [labeled_frames[label] for label in labels],
            "labeled_frames": labeled_frames,
        }
        row["question"] = (
            f"智能体正朝着【{target_name}】所在区域移动，目标不一定会一直出现在视野中。"
            f"请根据空间布局和相机位姿变化，给出这4张图片（A, B, C, D）正确的时序发生顺序。"
        )
        row["answer"] = f"正确的顺序是：{', '.join(chronological_labels)}"
        row["gt"] = {
            "target": {
                "id": int(target_id),
                "name": target_name,
            },
            "route_type": route_type,
            "api_actions": [step["action_api"] for step in steps],
            "steps": steps,
            "chronological_labels": chronological_labels,
            "start_state": self._state_gt(observations[0]["pose"]),
            "end_state": self._state_gt(observations[-1]["pose"]),
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _execute_class05(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        family = str(proposal.payload["action_family"])
        sample_dir = _sample_dir(self.image_root, proposal)
        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)

        rng = random.Random(proposal.proposal_id)
        if family == "rotation":
            api = rng.choice(["RotateLeft", "RotateRight"])
            large = float(proposal.payload["degrees"])
            small = max(30.0, large - 30.0)
            action_large = _make_rotate_action(api, large)
            action_small = _make_rotate_action(api, small)
            action_desc = "旋转"
        else:
            api = rng.choice(["MoveAhead", "MoveLeft", "MoveRight", "MoveBack"])
            large = float(proposal.payload["move_magnitude"])
            small = max(0.25, large - 0.20)
            action_large = _make_move_action(api, large)
            action_small = _make_move_action(api, small)
            action_desc = "平移" if api in {"MoveLeft", "MoveRight"} else "移动"

        self._start_obs(state, scene_objects=scene_objects)
        obs_large, step_large = self._execute_actions(start_obs, [action_large])
        self._start_obs(state, scene_objects=scene_objects)
        obs_small, step_small = self._execute_actions(start_obs, [action_small])

        bigger_label, smaller_label = ("B", "C") if rng.random() < 0.5 else ("C", "B")
        frame_paths = {
            "A": sample_dir / "frame_A.png",
            "B": sample_dir / "frame_B.png",
            "C": sample_dir / "frame_C.png",
        }
        _save_frame(start_obs["rgb"], frame_paths["A"])
        _save_frame(obs_large[-1]["rgb"] if bigger_label == "B" else obs_small[-1]["rgb"], frame_paths["B"])
        _save_frame(obs_large[-1]["rgb"] if bigger_label == "C" else obs_small[-1]["rgb"], frame_paths["C"])

        larger_image = bigger_label
        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(frame_paths["A"]), str(frame_paths["B"]), str(frame_paths["C"])],
            "labeled_frames": {key: str(value) for key, value in frame_paths.items()},
            "reference_frame": str(frame_paths["A"]),
        }
        row["question"] = (
            "图A是起始视角，图B和图C都表示智能体从图A出发，各执行了一次同类型但不同幅度的单动作后的结果。"
            "这个动作类型可能是前后左右移动，也可能是左右转动。请判断图B和图C里，哪一张对应的动作幅度更大？"
        )
        row["answer"] = f"【图{larger_image}】对应的动作幅度更大。"
        row["gt"] = {
            "start_pose": self._state_gt(start_obs["pose"]),
            "comparison_type": family,
            "action_api": api,
            "action_description": action_desc,
            "same_action_type_constraint": True,
            "larger_image": larger_image,
            "smaller_image": smaller_label,
            "frame_A_role": "shared_start",
            "frame_B": {
                "tag": "B",
                "family": family,
                "action_api": action_large.action_api if larger_image == "B" else action_small.action_api,
                "action_text": action_large.action_text if larger_image == "B" else action_small.action_text,
                "planned_value": large if larger_image == "B" else small,
                "actual_value": large if larger_image == "B" else small,
                "unit": "degrees" if family == "rotation" else "meters",
                "success": True,
                "start_pose": _pose_raw(start_obs["pose"]),
                "end_pose": _pose_raw((obs_large[-1] if larger_image == "B" else obs_small[-1])["pose"]),
            },
            "frame_C": {
                "tag": "C",
                "family": family,
                "action_api": action_large.action_api if larger_image == "C" else action_small.action_api,
                "action_text": action_large.action_text if larger_image == "C" else action_small.action_text,
                "planned_value": large if larger_image == "C" else small,
                "actual_value": large if larger_image == "C" else small,
                "unit": "degrees" if family == "rotation" else "meters",
                "success": True,
                "start_pose": _pose_raw(start_obs["pose"]),
                "end_pose": _pose_raw((obs_large[-1] if larger_image == "C" else obs_small[-1])["pose"]),
            },
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _modified_scene(self, scene_data: dict, task_id: int, sample_key: str) -> Tuple[str, dict, List[dict]]:
        scene_path = self._source_scene_path
        temp_path = self.ctx.save_temp_scene(scene_data, scene_path, task_id, abs(hash(sample_key)) % 1_000_000)
        return temp_path, scene_data, self.ctx.scene_objects(scene_data)

    def _render_modified_pair(
        self,
        state: StateRecord,
        proposal: TaskProposal,
        modified_scene_data: dict,
        *,
        task_id: int,
    ) -> Tuple[str, dict, List[dict], dict, dict]:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        modified_scene_path, _, modified_objects = self._modified_scene(modified_scene_data, task_id, proposal.proposal_id)
        modified_obs, _ = self._start_obs(
            state,
            scene_path=modified_scene_path,
            scene_objects=modified_objects,
            navmesh_key_path=scene_path,
        )
        return scene_path, scene_objects, modified_objects, start_obs, modified_obs

    def _execute_class06(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, scene_data, scene_objects = self._scene_assets(state.scene)
        target_id = str(proposal.payload["target_object_id"])
        target = self._scene_object(scene_objects, target_id)
        if target is None:
            raise RuntimeError("missing_scene_object")
        target_name = str(target["name"])
        target_index = int(target["index"])
        category = str(proposal.subcat)

        new_scene = copy.deepcopy(scene_data)
        target_inst = new_scene["object_instances"][target_index]
        if category == "remove":
            del new_scene["object_instances"][target_index]
            change_text = "被移走了"
            question = "场景中哪个物体被移走了？"
            answer = f"【{target_name}】被移走了。"
            action_api = "ModifyScene/RemoveObject"
            action_params = {"target_index": target_index}
        elif category == "rotate":
            current_q = _quat_from_wxyz(target_inst.get("rotation", [1.0, 0.0, 0.0, 0.0]))
            target_inst["rotation"] = _quat_to_wxyz(_yaw_quaternion(90.0) * current_q)
            change_text = "被旋转了"
            question = "场景中哪个物体被旋转了？"
            answer = f"【{target_name}】被旋转了。"
            action_api = "ModifyScene/RotateObject"
            action_params = {"target_index": target_index, "degrees": 90.0}
        else:
            raise RuntimeError(f"unsupported_class06_subcat:{category}")

        _, _, _, start_obs, modified_obs = self._render_modified_pair(state, proposal, new_scene, task_id=6)
        sample_dir = _sample_dir(self.image_root, proposal)
        path_a = sample_dir / "frame_000_A.png"
        path_b = sample_dir / "frame_001_B.png"
        _save_frame(start_obs["rgb"], path_a)
        _save_frame(modified_obs["rgb"], path_b)

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a), str(path_b)],
            "frame_A": str(path_a),
            "frame_B": str(path_b),
        }
        row["question"] = question
        row["answer"] = answer
        row["gt"] = {
            "target": {
                "index": target_index,
                "template_name": target["template_name"],
                "name": target_name,
                "position": {
                    "x": float(target["position"][0]),
                    "y": float(target["position"][1]),
                    "z": float(target["position"][2]),
                },
            },
            "change_category": category,
            "change_text": change_text,
            "action_api": action_api,
            "action_params": action_params,
            "target_state_before": {
                "exists": True,
                "visible": True,
                "position": {
                    "x": float(target["position"][0]),
                    "y": float(target["position"][1]),
                    "z": float(target["position"][2]),
                },
                "rotation": target.get("rotation"),
            },
            "target_state_after": {
                "exists": category != "remove",
                "visible": category == "rotate",
            },
            "start_state": self._state_gt(start_obs["pose"]),
            "end_state": self._state_gt(modified_obs["pose"]),
            "scene_path": scene_path,
            "scene_dataset_config": self.ctx.scene_dataset_config,
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _execute_class07(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, scene_data, scene_objects = self._scene_assets(state.scene)
        first_id = str(proposal.payload["first_object_id"])
        second_id = str(proposal.payload["second_object_id"])
        first = self._scene_object(scene_objects, first_id)
        second = self._scene_object(scene_objects, second_id)
        if first is None or second is None:
            raise RuntimeError("missing_scene_object")

        new_scene = copy.deepcopy(scene_data)
        obj_a = new_scene["object_instances"][int(first_id)]
        obj_b = new_scene["object_instances"][int(second_id)]
        obj_a["translation"], obj_b["translation"] = obj_b["translation"], obj_a["translation"]
        obj_a["rotation"], obj_b["rotation"] = obj_b["rotation"], obj_a["rotation"]

        _, _, modified_objects, start_obs, modified_obs = self._render_modified_pair(state, proposal, new_scene, task_id=7)
        sample_dir = _sample_dir(self.image_root, proposal)
        path_a = sample_dir / "frame_000_A.png"
        path_b = sample_dir / "frame_001_B.png"
        _save_frame(start_obs["rgb"], path_a)
        _save_frame(modified_obs["rgb"], path_b)

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a), str(path_b)],
            "frame_A": str(path_a),
            "frame_B": str(path_b),
        }
        row["question"] = "图A是原始场景，图B是交换两个物体位置后的场景。场景中哪两个物体交换了位置？"
        row["answer"] = f"【{first['name']}】和【{second['name']}】交换了位置。"
        row["gt"] = {
            "swap_pair": [
                {
                    "before": {
                        "exists": True,
                        "index": int(first["index"]),
                        "template_name": first["template_name"],
                        "name": first["name"],
                        "position": {
                            "x": float(first["position"][0]),
                            "y": float(first["position"][1]),
                            "z": float(first["position"][2]),
                        },
                    },
                    "after": {
                        "exists": True,
                        "index": int(first["index"]),
                        "template_name": first["template_name"],
                        "name": first["name"],
                        "position": {
                            "x": float(modified_objects[int(first_id)]["position"][0]),
                            "y": float(modified_objects[int(first_id)]["position"][1]),
                            "z": float(modified_objects[int(first_id)]["position"][2]),
                        },
                    },
                    "answer_name": first["name"],
                },
                {
                    "before": {
                        "exists": True,
                        "index": int(second["index"]),
                        "template_name": second["template_name"],
                        "name": second["name"],
                        "position": {
                            "x": float(second["position"][0]),
                            "y": float(second["position"][1]),
                            "z": float(second["position"][2]),
                        },
                    },
                    "after": {
                        "exists": True,
                        "index": int(second["index"]),
                        "template_name": second["template_name"],
                        "name": second["name"],
                        "position": {
                            "x": float(modified_objects[int(second_id)]["position"][0]),
                            "y": float(modified_objects[int(second_id)]["position"][1]),
                            "z": float(modified_objects[int(second_id)]["position"][2]),
                        },
                    },
                    "answer_name": second["name"],
                },
            ],
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _candidate_motion_offsets(self) -> List[Tuple[float, float]]:
        return [
            (0.00, -0.45),
            (0.00, 0.45),
            (-0.25, 0.00),
            (0.25, 0.00),
            (-0.18, -0.22),
            (0.18, -0.22),
            (-0.18, 0.22),
            (0.18, 0.22),
        ]

    def _execute_class08(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, scene_data, scene_objects = self._scene_assets(state.scene)
        target_id = str(proposal.payload["target_object_id"])
        target = self._scene_object(scene_objects, target_id)
        if target is None:
            raise RuntimeError("missing_scene_object")

        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        before_visible = self._visible_map(scene_objects, start_obs)
        if int(target_id) not in before_visible:
            raise RuntimeError("target_not_visible_before")
        before_target = before_visible[int(target_id)]
        best = None
        yaw_rad = math.radians(float(state.yaw))
        for local_x, local_z in self._candidate_motion_offsets():
            world_dx = math.cos(yaw_rad) * local_x + math.sin(yaw_rad) * local_z
            world_dz = -math.sin(yaw_rad) * local_x + math.cos(yaw_rad) * local_z
            new_scene = copy.deepcopy(scene_data)
            inst = new_scene["object_instances"][int(target_id)]
            pos = list(inst["translation"])
            pos[0] = float(pos[0]) + float(world_dx)
            pos[2] = float(pos[2]) + float(world_dz)
            inst["translation"] = pos
            _, _, modified_objects, _, modified_obs = self._render_modified_pair(state, proposal, new_scene, task_id=8)
            after_visible = self._visible_map(modified_objects, modified_obs)
            after_target = after_visible.get(int(target_id))
            if after_target is None:
                continue
            if proposal.subcat == "distance_change":
                delta = float(after_target.distance) - float(before_target.distance)
                score = abs(delta)
            else:
                delta = float(self.runner.target_prominence_score(after_target)) - float(
                    self.runner.target_prominence_score(before_target)
                )
                score = abs(delta)
            if best is None or score > best[0]:
                best = (score, new_scene, modified_objects, modified_obs, before_target, after_target)
        if best is None:
            raise RuntimeError("no_valid_motion_change")

        _, new_scene, modified_objects, modified_obs, before_target, after_target = best
        _, _, _, start_obs, modified_obs = self._render_modified_pair(state, proposal, new_scene, task_id=8)
        sample_dir = _sample_dir(self.image_root, proposal)
        path_a = sample_dir / "frame_A.png"
        path_b = sample_dir / "frame_B.png"
        _save_frame(start_obs["rgb"], path_a)
        _save_frame(modified_obs["rgb"], path_b)

        if proposal.subcat == "distance_change":
            relation = "更近" if float(after_target.distance) < float(before_target.distance) else "更远"
            question = "图A到图B中，发生移动的那个物体相对观察者是更近了还是更远了？"
            answer = relation
        else:
            relation = "更容易被遮挡" if self.runner.target_prominence_score(after_target) < self.runner.target_prominence_score(before_target) else "更不容易被遮挡"
            question = "图A到图B中，发生移动的那个物体变得更容易被遮挡，还是更不容易被遮挡？"
            answer = relation

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a), str(path_b)],
            "frame_A": str(path_a),
            "frame_B": str(path_b),
        }
        row["question"] = question
        row["answer"] = answer
        row["gt"] = {
            "question_family": proposal.subcat,
            "mover": {
                "before": {
                    "index": int(before_target.index),
                    "name": before_target.name,
                    "template_name": before_target.template_name,
                    "position": {
                        "x": float(before_target.position[0]),
                        "y": float(before_target.position[1]),
                        "z": float(before_target.position[2]),
                    },
                    "distance": float(before_target.distance),
                    "pixel": {"x": float(before_target.pixel_x), "y": float(before_target.pixel_y)},
                },
                "after": {
                    "index": int(after_target.index),
                    "name": after_target.name,
                    "position": list(after_target.position),
                    "distance": float(after_target.distance),
                },
            },
            "relation": relation,
            "distance_before": float(before_target.distance),
            "distance_after": float(after_target.distance),
            "distance_delta": float(after_target.distance) - float(before_target.distance),
            "start_pose": _pose_raw(start_obs["pose"]),
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _execute_class09(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        anchor_id = str(proposal.payload["anchor_object_id"])
        target_id = str(proposal.payload["target_object_id"])
        anchor = self._scene_object(scene_objects, anchor_id)
        target = self._scene_object(scene_objects, target_id)
        if anchor is None or target is None:
            raise RuntimeError("missing_scene_object")

        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        sample_dir = _sample_dir(self.image_root, proposal)
        path_a = sample_dir / "frame_A.png"
        _save_frame(start_obs["rgb"], path_a)

        anchor_q = _quat_from_wxyz(anchor["rotation"])
        offset = np.array(target["position"], dtype=np.float32) - np.array(anchor["position"], dtype=np.float32)
        local = quat_rotate_vector(anchor_q.conjugate(), offset)
        direction = _direction_from_local(float(local[0]), float(local[2]))

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a)],
            "frame_A": str(path_a),
        }
        row["question"] = (
            f"想象你现在位于图中的 [{anchor['name']}] 位置，并且把该物体当前朝向的正前方定义为你的正前方。"
            f"请问 [{target['name']}] 在你的什么方位？"
        )
        row["answer"] = direction
        row["gt"] = {
            "preferred_direction": direction,
            "object_A": {
                "index": int(anchor["index"]),
                "name": anchor["name"],
                "position": {
                    "x": float(anchor["position"][0]),
                    "y": float(anchor["position"][1]),
                    "z": float(anchor["position"][2]),
                },
                "rotation_quaternion_wxyz": anchor["rotation"],
            },
            "object_B": {
                "index": int(target["index"]),
                "name": target["name"],
                "position": {
                    "x": float(target["position"][0]),
                    "y": float(target["position"][1]),
                    "z": float(target["position"][2]),
                },
            },
            "pos_B_local": [float(local[0]), float(local[1]), float(local[2])],
            "axis_convention": f"Habitat/{'HSSD' if self.dataset_name == 'hssd' else 'ReplicaCAD'}局部坐标：Z为forward，X为right，Y为up",
            "direction_word": direction,
            "start_state": self._state_gt(start_obs["pose"]),
            "scene_path": scene_path,
            "scene_dataset_config": self.ctx.scene_dataset_config,
            "proposal_id": proposal.proposal_id,
        }
        return row

    def _candidate_consequence_sequences(self, subcat: str) -> List[List[ActionSpecLite]]:
        if subcat == "single_direction":
            return [
                [_make_rotate_action("RotateLeft", 30)],
                [_make_rotate_action("RotateRight", 30)],
                [_make_move_action("MoveAhead", 0.6)],
                [_make_move_action("MoveBack", 0.6)],
            ]
        if subcat == "two_frame_direction":
            return [
                [_make_rotate_action("RotateLeft", 30), _make_move_action("MoveAhead", 0.8)],
                [_make_rotate_action("RotateRight", 30), _make_move_action("MoveAhead", 0.8)],
                [_make_move_action("MoveBack", 0.8), _make_rotate_action("RotateLeft", 45)],
            ]
        if subcat == "trend_disappear":
            return [
                [_make_move_action("MoveBack", 1.0), _make_move_action("MoveBack", 1.0), _make_rotate_action("RotateRight", 60)],
                [_make_move_action("MoveBack", 1.0), _make_rotate_action("RotateLeft", 60)],
                [_make_rotate_action("RotateLeft", 60), _make_move_action("MoveAhead", 1.0)],
            ]
        return [
            [_make_rotate_action("RotateLeft", 30), _make_move_action("MoveAhead", 0.8)],
            [_make_rotate_action("RotateRight", 45), _make_move_action("MoveBack", 1.0)],
            [_make_move_action("MoveBack", 1.0), _make_move_action("MoveBack", 1.0), _make_rotate_action("RotateRight", 60)],
        ]

    def _execute_class10(self, state: StateRecord, proposal: TaskProposal) -> dict:
        scene_path, _, scene_objects = self._scene_assets(state.scene)
        target_id = str(proposal.payload["target_object_id"])
        target_name = self._visible_object_name(state, target_id)
        sample_dir = _sample_dir(self.image_root, proposal)
        start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
        start_visible = self._visible_map(scene_objects, start_obs)
        if int(target_id) not in start_visible:
            raise RuntimeError("target_not_visible_before")

        best = None
        for actions in self._candidate_consequence_sequences(proposal.subcat):
            start_obs, _ = self._start_obs(state, scene_objects=scene_objects)
            observations, _ = self._execute_actions(start_obs, actions)
            final_visible = self._visible_map(scene_objects, observations[-1])
            target_after = final_visible.get(int(target_id))
            if proposal.subcat == "trend_disappear" and target_after is not None:
                continue
            if proposal.subcat in {"single_direction", "two_frame_direction"} and target_after is None:
                continue
            score = _frame_diff_score(observations[0]["rgb"], observations[-1]["rgb"])
            if best is None or score > best[0]:
                best = (score, observations, actions, target_after)
        if best is None:
            raise RuntimeError("no_valid_consequence_sequence")

        _, observations, actions, target_after = best
        path_a = sample_dir / "frame_A.png"
        _save_frame(observations[0]["rgb"], path_a)
        action_text = "，然后".join(action.action_text for action in actions)
        if target_after is None:
            answer = "不能看见，它会移出视野。"
            result_visibility = False
            result_location = None
        else:
            answer = f"能看见，它会在{_horizontal_location(float(target_after.pixel_x), self.args.width)}。"
            result_visibility = True
            result_location = _horizontal_location(float(target_after.pixel_x), self.args.width)

        row = self._base_row(proposal, scene_path)
        row["input"] = {
            "frame_paths": [str(path_a)],
            "frame_A": str(path_a),
        }
        row["question"] = (
            f"如果我从当前视角依次{action_text}，那么【{target_name}】还能看见吗？"
            "如果还能看见，它会更靠左侧、中间还是右侧？"
        )
        row["answer"] = answer
        row["gt"] = {
            "question_family": proposal.subcat,
            "scene_path": scene_path,
            "scene_dataset_config": self.ctx.scene_dataset_config,
            "target": {
                "index": int(target_id),
                "name": target_name,
                "position": {
                    "x": float(self._scene_object(scene_objects, target_id)["position"][0]),
                    "y": float(self._scene_object(scene_objects, target_id)["position"][1]),
                    "z": float(self._scene_object(scene_objects, target_id)["position"][2]),
                },
                "distance": float(start_visible[int(target_id)].distance),
                "pixel": {
                    "x": float(start_visible[int(target_id)].pixel_x),
                    "y": float(start_visible[int(target_id)].pixel_y),
                },
            },
            "action_sequence": [
                {"api": action.action_api, **action.action_params, "text": action.action_text} for action in actions
            ],
            "action_text": action_text,
            "start_pose": _pose_raw(observations[0]["pose"]),
            "result_visibility": result_visibility,
            "result_location": result_location,
            "proposal_id": proposal.proposal_id,
        }
        return row


def parse_args(default_class_id: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Execute class-{default_class_id:02d} Habitat proposals into final QA samples.")
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
    parser.add_argument("--target-per-subcat", type=int, default=None)
    parser.add_argument("--max-episode-retry", type=int, default=1)
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def main_for_dataset(
    *,
    dataset_name: str,
    resolve_scene_path: Callable[[str], str],
    default_class_id: int,
) -> int:
    args = parse_args(default_class_id)
    executor = ProposalExecutor(args, dataset_name, resolve_scene_path, default_class_id)
    return executor.run()
