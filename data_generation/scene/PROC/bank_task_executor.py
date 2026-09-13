#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ai2thor_env import create_controller
from schema import StateRecord, TaskProposal, VisibleObjectRecord
from task_execution_specs import spec_for


MIN_SHARED_OBJECTS = 1
MIN_FRAME_DIFF = 8.0
DEFAULT_ROTATION_VALUES = (30, 45, 60, 75, 90)
DEFAULT_MOVE_VALUES = (0.25, 0.5, 0.75, 1.0)
CLASS08_VIEW_SEARCH_YAWS = tuple(range(0, 360, 15))
CLASS08_VIEW_SEARCH_HORIZONS = (-30, -15, 0, 15, 30)
CLASS08_LOCAL_MOVE_OFFSETS = (
    (0.00, -0.30),
    (0.00, 0.30),
    (-0.26, 0.00),
    (0.26, 0.00),
    (-0.20, -0.20),
    (0.20, -0.20),
    (-0.20, 0.20),
    (0.20, 0.20),
    (0.00, -0.24),
    (0.00, 0.24),
    (-0.18, 0.00),
    (0.18, 0.00),
)
CLASS08_MAX_CANDIDATE_ATTEMPTS = 14
MAX_TARGET_DISTANCE = 5.2
MIN_BBOX_AREA_RATIO = 0.008
MAX_BBOX_AREA_RATIO = 0.24
EXCLUDED_OBJECT_TYPES = {
    "Bathtub",
    "BathtubBasin",
    "Blinds",
    "Burner",
    "Cabinet",
    "Ceiling",
    "Counter",
    "CounterTop",
    "Countertop",
    "Curtains",
    "DeskLamp",
    "Door",
    "Doorway",
    "Drawer",
    "Faucet",
    "Floor",
    "LightSwitch",
    "Mirror",
    "Room",
    "Shelf",
    "ShelvingUnit",
    "ShowerCurtain",
    "ShowerDoor",
    "ShowerGlass",
    "SinkBasin",
    "StandardDoor",
    "StoveBurner",
    "StoveKnob",
    "Wall",
    "Window",
}
DISALLOWED_SUPPORT_TYPES = {
    "BathtubBasin",
    "Cabinet",
    "Drawer",
    "GarbageCan",
    "Microwave",
    "Safe",
    "SinkBasin",
    "Toilet",
}
PREFERRED_SUPPORT_TYPES = {
    "Bed",
    "CoffeeTable",
    "CounterTop",
    "Desk",
    "DiningTable",
    "Dresser",
    "Ottoman",
    "SideTable",
    "Sofa",
    "TVStand",
}
ROTATION_EXCLUDED_TYPES = {
    "Bed",
    "Chair",
    "CoffeeTable",
    "DiningTable",
    "Dresser",
    "Fridge",
    "Microwave",
    "Sofa",
    "Television",
    "TVStand",
}


@dataclass
class PoseState:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float

    def to_text(self) -> str:
        return (
            f"Pos=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}), "
            f"Rot={self.yaw:.1f}, Horizon={self.pitch:.1f}"
        )


@dataclass
class ActionSpec:
    action_api: str
    action_args: Dict[str, float]
    action_text: str


@dataclass
class StepRecord:
    step_id: int
    action_api: str
    action_args: Dict[str, float]
    success: bool
    before_pose: Dict[str, float]
    after_pose: Dict[str, float]


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


def _visible_objects(event) -> List[Dict]:
    return [
        obj
        for obj in event.metadata.get("objects", [])
        if obj.get("visible", False) and obj.get("position") is not None
    ]


def _visible_object_ids(event) -> set[str]:
    return {str(obj["objectId"]) for obj in event.metadata.get("objects", []) if obj.get("visible", False)}


def _find_object(event, object_id: str) -> Optional[Dict]:
    for obj in event.metadata.get("objects", []):
        if obj.get("objectId") == object_id:
            return obj
    return None


def _is_visible_object(obj: Optional[Dict]) -> bool:
    return obj is not None and bool(obj.get("visible", False)) and obj.get("position") is not None


def _pose_from_event(event) -> PoseState:
    agent = event.metadata["agent"]
    position = agent["position"]
    rotation = agent["rotation"]
    return PoseState(
        x=float(position["x"]),
        y=float(position["y"]),
        z=float(position["z"]),
        yaw=float(rotation["y"]),
        pitch=float(agent["cameraHorizon"]),
    )


def _pretty_name(text: str) -> str:
    if not text:
        return "object"
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = words.replace("_", " ").strip()
    words = re.sub(r"\s+", " ", words)
    return words.lower()


def _display_name(text: str) -> str:
    if not text:
        return "Object"
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"\s+", " ", words.replace("_", " ").strip())


def _dist3(a: Dict | Tuple[float, float, float], b: Dict | Tuple[float, float, float]) -> float:
    ax, ay, az = (float(a["x"]), float(a["y"]), float(a["z"])) if isinstance(a, dict) else a
    bx, by, bz = (float(b["x"]), float(b["y"]), float(b["z"])) if isinstance(b, dict) else b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _bbox_map(event) -> Dict[str, Tuple[float, float, float, float]]:
    det = event.instance_detections2D or {}
    return {
        object_id: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for object_id, box in det.items()
    }


def _bbox_quality(
    bbox: Optional[Tuple[float, float, float, float]],
    *,
    width: int,
    height: int,
) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width < 1.0 or box_height < 1.0:
        return None
    return {
        "width": box_width,
        "height": box_height,
        "area_ratio": (box_width * box_height) / float(width * height),
        "center_x_ratio": ((x1 + x2) * 0.5) / float(width),
        "center_y_ratio": ((y1 + y2) * 0.5) / float(height),
        "edge_margin_ratio": min(
            x1 / float(width),
            y1 / float(height),
            (width - x2) / float(width),
            (height - y2) / float(height),
        ),
    }


def _good_visible_object(
    event,
    obj: Optional[Dict],
    *,
    width: int,
    height: int,
    relaxed: bool = False,
    anchor: bool = False,
) -> bool:
    if obj is None or not bool(obj.get("visible", False)) or obj.get("position") is None:
        return False
    object_type = obj.get("objectType", "Unknown")
    if object_type in EXCLUDED_OBJECT_TYPES:
        return False
    bbox = _bbox_map(event).get(obj["objectId"])
    quality = _bbox_quality(bbox, width=width, height=height)
    if quality is None:
        return False
    if float(obj.get("distance", 999.0)) > MAX_TARGET_DISTANCE:
        return False
    min_area = 0.018 if anchor else (0.006 if relaxed else MIN_BBOX_AREA_RATIO)
    min_width = 54.0 if anchor else (26.0 if relaxed else 38.0)
    min_height = 44.0 if anchor else (20.0 if relaxed else 28.0)
    return (
        min_area <= quality["area_ratio"] <= MAX_BBOX_AREA_RATIO
        and quality["width"] >= min_width
        and quality["height"] >= min_height
        and 0.08 <= quality["center_x_ratio"] <= 0.92
        and 0.08 <= quality["center_y_ratio"] <= 0.92
        and quality["edge_margin_ratio"] >= (0.005 if relaxed else 0.02)
    )


def _screen_center_distance(q1: Optional[Dict[str, float]], q2: Optional[Dict[str, float]]) -> float:
    if q1 is None or q2 is None:
        return 0.0
    return math.sqrt(
        (q1["center_x_ratio"] - q2["center_x_ratio"]) ** 2
        + (q1["center_y_ratio"] - q2["center_y_ratio"]) ** 2
    )


def _aabb_size(obj: Optional[Dict]) -> Dict[str, float]:
    if obj is None:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    aabb = obj.get("axisAlignedBoundingBox") or {}
    size = aabb.get("size") or {}
    return {
        "x": float(size.get("x", 0.0)),
        "y": float(size.get("y", 0.0)),
        "z": float(size.get("z", 0.0)),
    }


def _aabb_bottom_y(obj: Optional[Dict]) -> Optional[float]:
    if obj is None:
        return None
    center = ((obj.get("axisAlignedBoundingBox") or {}).get("center") or {})
    size = ((obj.get("axisAlignedBoundingBox") or {}).get("size") or {})
    if not center or not size:
        return None
    return float(center.get("y", 0.0)) - float(size.get("y", 0.0)) * 0.5


def _aabb_bounds(obj: Optional[Dict]) -> Optional[Tuple[float, float, float, float, float, float]]:
    if obj is None:
        return None
    aabb = obj.get("axisAlignedBoundingBox") or {}
    center = aabb.get("center") or {}
    size = aabb.get("size") or {}
    if not center or not size:
        return None
    cx = float(center.get("x", 0.0))
    cy = float(center.get("y", 0.0))
    cz = float(center.get("z", 0.0))
    sx = float(size.get("x", 0.0)) * 0.5
    sy = float(size.get("y", 0.0)) * 0.5
    sz = float(size.get("z", 0.0)) * 0.5
    return (cx - sx, cy - sy, cz - sz, cx + sx, cy + sy, cz + sz)


def _aabb_intersection_ratio(obj_a: Optional[Dict], obj_b: Optional[Dict]) -> float:
    bounds_a = _aabb_bounds(obj_a)
    bounds_b = _aabb_bounds(obj_b)
    if bounds_a is None or bounds_b is None:
        return 0.0
    ax1, ay1, az1, ax2, ay2, az2 = bounds_a
    bx1, by1, bz1, bx2, by2, bz2 = bounds_b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    if ix <= 0.0 or iy <= 0.0 or iz <= 0.0:
        return 0.0
    inter = ix * iy * iz
    vol_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1) * (az2 - az1))
    vol_b = max(1e-9, (bx2 - bx1) * (by2 - by1) * (bz2 - bz1))
    return inter / min(vol_a, vol_b)


def _rotation_y(obj: Optional[Dict]) -> Optional[float]:
    if obj is None:
        return None
    rotation = obj.get("rotation") or {}
    if not rotation:
        return None
    return float(rotation.get("y", 0.0))


def _normalize_angle(angle: float) -> float:
    return angle % 360.0


def _angular_diff(a: float, b: float) -> float:
    diff = (_normalize_angle(a) - _normalize_angle(b) + 180.0) % 360.0 - 180.0
    return abs(diff)


def _support_parent_id(obj: Optional[Dict]) -> Optional[str]:
    if obj is None:
        return None
    parents = obj.get("parentReceptacles") or []
    return parents[-1] if parents else None


def _object_type_from_id(object_id: Optional[str]) -> Optional[str]:
    if not object_id:
        return None
    return object_id.split("|", 1)[0]


def _support_parent_type(obj: Optional[Dict]) -> Optional[str]:
    return _object_type_from_id(_support_parent_id(obj))


def _world_to_local_xz(dx: float, dz: float, yaw_deg: float) -> Tuple[float, float]:
    theta = math.radians(yaw_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    local_x = cos_t * dx - sin_t * dz
    local_z = sin_t * dx + cos_t * dz
    return local_x, local_z


def _local_to_world_xz(local_x: float, local_z: float, yaw_deg: float) -> Tuple[float, float]:
    theta = math.radians(yaw_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = cos_t * local_x + sin_t * local_z
    dz = -sin_t * local_x + cos_t * local_z
    return dx, dz


def _horizontal_direction_from_local(local_x: float, local_z: float) -> str:
    azim_deg = math.degrees(math.atan2(local_x, local_z))
    bins = [
        (-22.5, 22.5, "正前方"),
        (22.5, 75.0, "右前方"),
        (75.0, 105.0, "正右方"),
        (105.0, 157.5, "右后方"),
        (157.5, 180.1, "正后方"),
        (-180.1, -157.5, "正后方"),
        (-157.5, -105.0, "左后方"),
        (-105.0, -75.0, "正左方"),
        (-75.0, -22.5, "左前方"),
    ]
    for lo, hi, name in bins:
        if lo <= azim_deg < hi:
            return name
    return "正前方"


def _location_word(local_x: float, local_z: float) -> str:
    ang = math.degrees(math.atan2(local_x, max(local_z, 1e-6)))
    if ang < -15.0:
        return "左侧"
    if ang > 15.0:
        return "右侧"
    return "中间"


def _euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m


def _paired_events_have_overlap(events: Sequence[object]) -> bool:
    return all(
        len(_visible_object_ids(first).intersection(_visible_object_ids(second))) >= MIN_SHARED_OBJECTS
        for first, second in zip(events, events[1:])
    )


def _chain_has_differences(events: Sequence[object]) -> bool:
    return all(_frame_diff_score(first.frame, second.frame) >= MIN_FRAME_DIFF for first, second in zip(events, events[1:]))


def _reset_scene(controller, scene: str) -> None:
    try:
        controller.reset(scene=scene)
    except TypeError:
        controller.reset(scene)


def _teleport_to_state(controller, state: StateRecord):
    return controller.step(
        action="Teleport",
        position=state.position,
        rotation={"x": 0, "y": state.yaw, "z": 0},
        horizon=state.horizon,
        standing=True,
    )


def _teleport_to_pose(controller, pose: PoseState):
    return controller.step(
        action="Teleport",
        position={"x": pose.x, "y": pose.y, "z": pose.z},
        rotation={"x": 0, "y": pose.yaw, "z": 0},
        horizon=pose.pitch,
        standing=True,
    )


def _execute_action(controller, step_id: int, action: ActionSpec) -> Tuple[object, StepRecord]:
    before = _pose_from_event(controller.last_event)
    event = controller.step(action=action.action_api, **action.action_args)
    success = bool(event.metadata.get("lastActionSuccess", False))
    if not success and action.action_api.startswith("Move"):
        original = float(action.action_args["moveMagnitude"])
        for fallback in (0.75, 0.5, 0.25, 0.15, 0.1):
            if fallback >= original:
                continue
            event = controller.step(action=action.action_api, moveMagnitude=fallback)
            success = bool(event.metadata.get("lastActionSuccess", False))
            if success:
                action = ActionSpec(
                    action_api=action.action_api,
                    action_args={"moveMagnitude": fallback},
                    action_text=action.action_text,
                )
                break
    after = _pose_from_event(event)
    if not success:
        raise RuntimeError(f"action_failed:{action.action_api}:{action.action_args}")
    return event, StepRecord(
        step_id=step_id,
        action_api=action.action_api,
        action_args=action.action_args,
        success=success,
        before_pose=asdict(before),
        after_pose=asdict(after),
    )


class ControllerPool:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._cache: Dict[str, object] = {}

    def get(self, scene: str):
        controller = self._cache.get(scene)
        if controller is None:
            controller = create_controller(
                scene,
                width=self.args.width,
                height=self.args.height,
                grid_size=self.args.grid_size,
                rotate_step_degrees=30,
                field_of_view=self.args.field_of_view,
                visibility_distance=self.args.visibility_distance,
                use_cloud_rendering=self.args.use_cloud_rendering,
                render_instance_segmentation=True,
            )
            self._cache[scene] = controller
        return controller

    def close(self) -> None:
        for controller in self._cache.values():
            try:
                controller.stop()
            except Exception:
                pass
        self._cache.clear()


class BankTaskExecutor:
    def __init__(self, args: argparse.Namespace, class_id: int):
        self.args = args
        self.class_id = class_id
        self.specs = {
            subcat: spec_for(class_id, subcat)
            for subcat in {key[1] for key in spec_for.__globals__["TASK_EXECUTION_SPECS"] if key[0] == class_id}
        }
        self.states = self._load_states(args.state_bank)
        self.proposals = self._load_proposals(args.proposals)
        self.output_root = args.output_root
        self.image_root = self.output_root / "images"
        self.meta_root = self.output_root / "meta"
        self.rows: List[dict] = []
        self.records: List[dict] = []
        self.subcat_success = Counter()
        self.pool = ControllerPool(args)

    def _load_states(self, path: Path) -> Dict[str, StateRecord]:
        mapping: Dict[str, StateRecord] = {}
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                state = _state_from_dict(json.loads(line))
                mapping[state.state_id] = state
        return mapping

    def _load_proposals(self, path: Path) -> List[TaskProposal]:
        proposals: List[TaskProposal] = []
        subcats = None if self.args.subcats is None else {part.strip() for part in self.args.subcats.split(",") if part.strip()}
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                proposal = _proposal_from_dict(json.loads(line))
                if proposal.class_id != self.class_id:
                    continue
                if subcats and proposal.subcat not in subcats:
                    continue
                proposals.append(proposal)
        grouped: Dict[str, List[TaskProposal]] = defaultdict(list)
        for proposal in proposals:
            grouped[proposal.subcat].append(proposal)
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
            if not added:
                break
            round_idx += 1
        if self.args.max_proposals is not None:
            ordered = ordered[: self.args.max_proposals]
        return ordered

    def _prepare_output(self) -> None:
        if self.args.reset_output and self.output_root.exists():
            shutil.rmtree(self.output_root)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.meta_root.mkdir(parents=True, exist_ok=True)

    def _sample_dir(self, proposal: TaskProposal) -> Path:
        sample_dir = self.image_root / proposal.proposal_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    def run(self) -> int:
        self._prepare_output()
        try:
            for proposal in self.proposals:
                target = self.specs[proposal.subcat].target_success_count
                if self.args.target_per_subcat is not None:
                    target = self.args.target_per_subcat
                if target is not None and self.subcat_success[proposal.subcat] >= target:
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
                        {"proposal_id": proposal.proposal_id, "subcat": proposal.subcat, "status": "missing_state"}
                    )
                    continue
                controller = self.pool.get(state.scene)
                try:
                    _reset_scene(controller, state.scene)
                    row = self._execute_one(controller, state, proposal)
                    self.rows.append(row)
                    self.records.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "subcat": proposal.subcat,
                            "status": "success",
                        }
                    )
                    self.subcat_success[proposal.subcat] += 1
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
            self.pool.close()
        self._write_meta()
        return 0

    def _execute_one(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        if self.class_id == 2:
            return self._execute_class02(controller, state, proposal)
        if self.class_id == 3:
            return self._execute_class03(controller, state, proposal)
        if self.class_id == 4:
            return self._execute_class04(controller, state, proposal)
        if self.class_id == 5:
            return self._execute_class05(controller, state, proposal)
        if self.class_id == 6:
            return self._execute_class06(controller, state, proposal)
        if self.class_id == 7:
            return self._execute_class07(controller, state, proposal)
        if self.class_id == 8:
            return self._execute_class08(controller, state, proposal)
        if self.class_id == 9:
            return self._execute_class09(controller, state, proposal)
        if self.class_id == 10:
            return self._execute_class10(controller, state, proposal)
        raise RuntimeError(f"unsupported_class:{self.class_id}")

    def _row_base(self, proposal: TaskProposal, state: StateRecord) -> dict:
        return {
            "sample_id": proposal.proposal_id,
            "scene": state.scene,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _write_meta(self) -> None:
        qa_jsonl_path = self.meta_root / "qa_data.jsonl"
        qa_json_path = self.meta_root / "qa_data.json"
        stats_path = self.meta_root / "stats.json"
        records_path = self.meta_root / "records.json"

        with qa_jsonl_path.open("w", encoding="utf-8") as fout:
            for row in self.rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        qa_json_path.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8")
        records_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

        stats = {
            "class_id": self.class_id,
            "task_type": sorted({row["task_type"] for row in self.rows}),
            "requested_proposals": len(self.proposals),
            "generated_samples": len(self.rows),
            "success_by_subcat": dict(self.subcat_success),
            "jsonl_path": str(qa_jsonl_path),
            "json_path": str(qa_json_path),
            "records_path": str(records_path),
            "image_root": str(self.image_root),
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    def _execute_class02(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        target_id = str(proposal.payload["target_object_id"])
        anchor_id = str(proposal.payload["reference_object_id"])
        start_pose = PoseState(
            x=float(state.position["x"]),
            y=float(state.position["y"]),
            z=float(state.position["z"]),
            yaw=float(state.yaw),
            pitch=float(state.horizon),
        )
        rng = random.Random(proposal.proposal_id)
        directions = ["Left", "Right"]
        hinted_direction = proposal.payload.get("scan_direction")
        if hinted_direction in {"Left", "Right"}:
            directions = [str(hinted_direction), "Right" if hinted_direction == "Left" else "Left"]
        rng.shuffle(directions)
        if hinted_direction in {"Left", "Right"}:
            directions.sort(key=lambda item: item != hinted_direction)
        best = None
        best_score = None

        for direction in directions:
            sign = 1.0 if direction == "Right" else -1.0
            for deg1 in (45, 60, 75, 90):
                for deg2 in (30, 45, 60, 75, 90):
                    pose_a = PoseState(
                        x=start_pose.x,
                        y=start_pose.y,
                        z=start_pose.z,
                        yaw=(start_pose.yaw - sign * deg1) % 360.0,
                        pitch=start_pose.pitch,
                    )
                    evt_a = _teleport_to_pose(controller, pose_a)
                    if not bool(evt_a.metadata.get("lastActionSuccess", False)):
                        continue
                    anchor_a = _find_object(evt_a, anchor_id)
                    target_a = _find_object(evt_a, target_id)
                    if not _good_visible_object(evt_a, anchor_a, width=self.args.width, height=self.args.height, anchor=True):
                        continue
                    if target_a is not None and bool(target_a.get("visible", False)):
                        continue

                    action1 = ActionSpec(
                        action_api=f"Rotate{direction}",
                        action_args={"degrees": deg1},
                        action_text=f"向{'右' if direction == 'Right' else '左'}旋转{deg1}度",
                    )
                    evt_b, step1 = _execute_action(controller, 1, action1)
                    action2 = ActionSpec(
                        action_api=f"Rotate{direction}",
                        action_args={"degrees": deg2},
                        action_text=f"向{'右' if direction == 'Right' else '左'}旋转{deg2}度",
                    )
                    evt_c, step2 = _execute_action(controller, 2, action2)
                    target_c = _find_object(evt_c, target_id)
                    anchor_c = _find_object(evt_c, anchor_id)
                    if not _good_visible_object(evt_c, target_c, width=self.args.width, height=self.args.height):
                        continue
                    if anchor_c is not None and bool(anchor_c.get("visible", False)):
                        continue
                    events = [evt_a, evt_b, evt_c]
                    if not _paired_events_have_overlap(events):
                        continue
                    score = _frame_diff_score(evt_a.frame, evt_c.frame)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = (evt_a, evt_b, evt_c, step1, step2)

        if best is None:
            raise RuntimeError("class02_no_valid_sequence")
        evt_a, evt_b, evt_c, step1, step2 = best
        anchor = _find_object(evt_a, anchor_id)
        target = _find_object(evt_c, target_id)
        if anchor is None or target is None:
            raise RuntimeError("class02_missing_anchor_or_target")
        anchor_pos = anchor["position"]
        target_pos = target["position"]
        local_x, local_z = _world_to_local_xz(
            float(target_pos["x"]) - float(anchor_pos["x"]),
            float(target_pos["z"]) - float(anchor_pos["z"]),
            _pose_from_event(evt_a).yaw,
        )
        answer = _horizontal_direction_from_local(local_x, local_z)
        sample_dir = self._sample_dir(proposal)
        frame_paths = [
            sample_dir / "frame_000_A.png",
            sample_dir / "frame_001_B.png",
            sample_dir / "frame_002_C.png",
        ]
        for path, event in zip(frame_paths, [evt_a, evt_b, evt_c]):
            _save_frame(event.frame, path)
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "multi_image_overlap_localization",
                "input": {
                    "frame_paths": [str(path) for path in frame_paths],
                    "frame_A": str(frame_paths[0]),
                    "frame_B": str(frame_paths[1]),
                    "frame_C": str(frame_paths[2]),
                },
                "question": (
                    f"结合图A、图B、图C的重叠区域，并以图A的视角朝向为参考，"
                    f"请判断图C中的【{_display_name(target.get('objectType', 'object'))}】"
                    f"相对于图A中的【{_display_name(anchor.get('objectType', 'object'))}】"
                    f"在图A视角下更接近哪个水平方位？"
                ),
                "answer": answer,
                "gt": {
                    "api_actions": [step1.action_api, step2.action_api],
                    "steps": [asdict(step1), asdict(step2)],
                    "start_state": {"text": _pose_from_event(evt_a).to_text(), "raw": asdict(_pose_from_event(evt_a))},
                    "end_state": {"text": _pose_from_event(evt_c).to_text(), "raw": asdict(_pose_from_event(evt_c))},
                    "objects": {
                        "target_object_from_C": {
                            "id": target["objectId"],
                            "type": target.get("objectType", "Unknown"),
                            "display_name": _display_name(target.get("objectType", "object")),
                            "position": target["position"],
                            "distance": float(target.get("distance", 999.0)),
                            "bbox_quality": _bbox_quality(
                                _bbox_map(evt_c).get(target["objectId"]),
                                width=self.args.width,
                                height=self.args.height,
                            ),
                        },
                        "anchor_object_from_A": {
                            "id": anchor["objectId"],
                            "type": anchor.get("objectType", "Unknown"),
                            "display_name": _display_name(anchor.get("objectType", "object")),
                            "position": anchor["position"],
                            "distance": float(anchor.get("distance", 999.0)),
                            "bbox_quality": _bbox_quality(
                                _bbox_map(evt_a).get(anchor["objectId"]),
                                width=self.args.width,
                                height=self.args.height,
                            ),
                        },
                    },
                    "visibility_check": {
                        "A_frame": {
                            "target_visible": target["objectId"] in _visible_object_ids(evt_a),
                            "anchor_visible": anchor["objectId"] in _visible_object_ids(evt_a),
                        },
                        "B_frame": {
                            "target_visible": target["objectId"] in _visible_object_ids(evt_b),
                            "anchor_visible": anchor["objectId"] in _visible_object_ids(evt_b),
                        },
                        "C_frame": {
                            "target_visible": target["objectId"] in _visible_object_ids(evt_c),
                            "anchor_visible": anchor["objectId"] in _visible_object_ids(evt_c),
                        },
                    },
                    "relation": {
                        "world_vector": {
                            "dx": float(target_pos["x"]) - float(anchor_pos["x"]),
                            "dy": float(target_pos["y"]) - float(anchor_pos["y"]),
                            "dz": float(target_pos["z"]) - float(anchor_pos["z"]),
                        },
                        "local_vector": {"x": local_x, "z": local_z},
                        "horizontal": answer,
                    },
                },
            }
        )
        return base

    def _execute_class03(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        evt_a = _teleport_to_state(controller, state)
        if not bool(evt_a.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class03_teleport_failed")
        near_id = str(proposal.payload["near_object_id"])
        far_id = str(proposal.payload["far_object_id"])
        obj_near = _find_object(evt_a, near_id)
        obj_far = _find_object(evt_a, far_id)
        if not _good_visible_object(evt_a, obj_near, width=self.args.width, height=self.args.height):
            raise RuntimeError("class03_near_not_visible")
        if not _good_visible_object(evt_a, obj_far, width=self.args.width, height=self.args.height):
            raise RuntimeError("class03_far_not_visible")

        pose_a = _pose_from_event(evt_a)
        cam_xyz = (pose_a.x, pose_a.y, pose_a.z)
        dist_near = _dist3(cam_xyz, obj_near["position"])
        dist_far = _dist3(cam_xyz, obj_far["position"])
        if dist_far - dist_near < 0.5:
            raise RuntimeError("class03_depth_gap_too_small")

        best = None
        best_score = None
        for action_api in ("MoveAhead", "MoveLeft", "MoveRight", "MoveBack"):
            for magnitude in (1.0, 0.75, 0.5, 0.25):
                _teleport_to_state(controller, state)
                evt_b = controller.step(action=action_api, moveMagnitude=magnitude)
                if not bool(evt_b.metadata.get("lastActionSuccess", False)):
                    continue
                if near_id not in _visible_object_ids(evt_b) or far_id not in _visible_object_ids(evt_b):
                    continue
                score = _frame_diff_score(evt_a.frame, evt_b.frame)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (action_api, magnitude, evt_b)

        if best is None:
            raise RuntimeError("class03_no_valid_motion")
        action_api, magnitude, evt_b = best
        step = StepRecord(
            step_id=1,
            action_api=action_api,
            action_args={"moveMagnitude": magnitude},
            success=True,
            before_pose=asdict(pose_a),
            after_pose=asdict(_pose_from_event(evt_b)),
        )
        sample_dir = self._sample_dir(proposal)
        frame_a = sample_dir / "frame_000_A.png"
        frame_b = sample_dir / "frame_001_B.png"
        _save_frame(evt_a.frame, frame_a)
        _save_frame(evt_b.frame, frame_b)
        nearer_label = "A" if dist_near < dist_far else "B"
        near_name = _pretty_name(obj_near.get("objectType", "object"))
        far_name = _pretty_name(obj_far.get("objectType", "object"))
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "parallax_depth_inference",
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b)],
                    "frame_A": str(frame_a),
                    "frame_B": str(frame_b),
                },
                "question": f"在这两张图像中，【{near_name}】和【{far_name}】哪个距离相机更近？",
                "answer": f"【{near_name if nearer_label == 'A' else far_name}】距离更近。",
                "gt": {
                    "api_actions": [action_api],
                    "steps": [asdict(step)],
                    "start_state": {"text": pose_a.to_text(), "raw": asdict(pose_a)},
                    "end_state": {"text": _pose_from_event(evt_b).to_text(), "raw": asdict(_pose_from_event(evt_b))},
                    "camera_position_A": {"x": cam_xyz[0], "y": cam_xyz[1], "z": cam_xyz[2]},
                    "objects": {
                        "object_a": {
                            "id": near_id,
                            "type": obj_near.get("objectType", "Unknown"),
                            "name": near_name,
                            "position": obj_near["position"],
                            "distance_to_camera_A": dist_near,
                        },
                        "object_b": {
                            "id": far_id,
                            "type": obj_far.get("objectType", "Unknown"),
                            "name": far_name,
                            "position": obj_far["position"],
                            "distance_to_camera_A": dist_far,
                        },
                    },
                    "depth_gap": abs(dist_far - dist_near),
                    "nearer_object": {
                        "label": nearer_label,
                        "id": near_id if nearer_label == "A" else far_id,
                        "name": near_name if nearer_label == "A" else far_name,
                    },
                    "visibility_check": {
                        "A_frame": {"A": True, "B": True},
                        "B_frame": {
                            "A": near_id in _visible_object_ids(evt_b),
                            "B": far_id in _visible_object_ids(evt_b),
                        },
                    },
                },
            }
        )
        return base

    def _build_route_actions(self, start_pose: PoseState, target_position: Dict) -> Tuple[List[ActionSpec], str]:
        dx = float(target_position["x"]) - start_pose.x
        dz = float(target_position["z"]) - start_pose.z
        local_x, local_z = _world_to_local_xz(dx, dz, start_pose.yaw)
        azim_deg = math.degrees(math.atan2(local_x, local_z))
        abs_angle = abs(azim_deg)
        turn_direction = "Right" if local_x >= 0 else "Left"
        target_distance = _dist3((start_pose.x, start_pose.y, start_pose.z), target_position)
        long_stride = 0.75 if target_distance >= 2.8 else 0.5
        medium_stride = 0.75 if target_distance >= 3.2 else 0.5
        short_stride = 0.5
        if abs_angle >= 18:
            turn_deg = min(90, max(30, int(round(abs_angle / 15.0)) * 15))
            return (
                [
                    ActionSpec(
                        action_api=f"Rotate{turn_direction}",
                        action_args={"degrees": float(turn_deg)},
                        action_text=f"向{'右' if turn_direction == 'Right' else '左'}旋转{turn_deg}度",
                    ),
                    ActionSpec("MoveAhead", {"moveMagnitude": long_stride}, f"向前移动{long_stride:.2f}米"),
                    ActionSpec("MoveAhead", {"moveMagnitude": medium_stride}, f"向前移动{medium_stride:.2f}米"),
                ],
                "turn_then_advance",
            )
        return (
            [
                ActionSpec("MoveAhead", {"moveMagnitude": long_stride}, f"向前移动{long_stride:.2f}米"),
                ActionSpec("MoveAhead", {"moveMagnitude": medium_stride}, f"向前移动{medium_stride:.2f}米"),
                ActionSpec("MoveAhead", {"moveMagnitude": short_stride}, f"向前移动{short_stride:.2f}米"),
            ],
            "direct_advance",
        )

    def _distance_progression(self, events: Sequence[object], target_position: Dict) -> List[float]:
        distances = []
        for event in events:
            pose = _pose_from_event(event)
            distances.append(_dist3((pose.x, pose.y, pose.z), target_position))
        return distances

    def _progression_is_logical(self, distances: Sequence[float]) -> bool:
        if len(distances) < 2 or distances[-1] > distances[0] - 0.2:
            return False
        return all(cur <= prev + 0.15 for prev, cur in zip(distances, distances[1:]))

    def _collect_route_events(
        self,
        controller,
        state: StateRecord,
        actions: Sequence[ActionSpec],
    ) -> Tuple[List[object], List[StepRecord]]:
        evt0 = _teleport_to_state(controller, state)
        if not bool(evt0.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("route_teleport_failed")
        events = [evt0]
        steps: List[StepRecord] = []
        for step_id, action in enumerate(actions, start=1):
            event, step = _execute_action(controller, step_id, action)
            events.append(event)
            steps.append(step)
        return events, steps

    def _shuffle_labels(
        self,
        proposal: TaskProposal,
        time_events: Sequence[object],
        labels: Sequence[str],
    ) -> Tuple[Dict[str, str], Dict[str, Path], str]:
        rng = random.Random(proposal.proposal_id)
        indices = list(range(len(time_events)))
        rng.shuffle(indices)
        sample_dir = self._sample_dir(proposal)
        label_to_time: Dict[str, str] = {}
        label_to_path: Dict[str, Path] = {}
        for label, index in zip(labels, indices):
            label_to_time[label] = f"T{index + 1}"
            path = sample_dir / f"frame_{label}.png"
            _save_frame(time_events[index].frame, path)
            label_to_path[label] = path
        time_to_label = {time: label for label, time in label_to_time.items()}
        answer_seq = " -> ".join(time_to_label[f"T{i+1}"] for i in range(len(time_events)))
        return label_to_time, label_to_path, answer_seq

    def _execute_class04(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        evt0 = _teleport_to_state(controller, state)
        if not bool(evt0.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class04_teleport_failed")
        pose0 = _pose_from_event(evt0)

        route_candidates: List[Tuple[List[ActionSpec], str, str, Dict]] = []
        if proposal.subcat == "start_visible":
            target = _find_object(evt0, str(proposal.payload["target_object_id"]))
            if target is None or not _good_visible_object(evt0, target, width=self.args.width, height=self.args.height):
                raise RuntimeError("class04_target_not_visible")
            actions, route_type = self._build_route_actions(pose0, target["position"])
            route_candidates.append((actions, route_type, target["objectId"], target))
        else:
            anchor = _find_object(evt0, str(proposal.payload["anchor_object_id"]))
            if anchor is not None and anchor.get("position") is not None:
                actions, route_type = self._build_route_actions(pose0, anchor["position"])
                route_candidates.append((actions, f"anchor_guided_{route_type}", "", {}))
            for direction, turn_deg in (("Left", 60), ("Right", 60), ("Left", 90), ("Right", 90), ("Left", 45), ("Right", 45)):
                route_candidates.append(
                    (
                        [
                            ActionSpec(f"Rotate{direction}", {"degrees": float(turn_deg)}, f"向{'左' if direction == 'Left' else '右'}旋转{turn_deg}度"),
                            ActionSpec("MoveAhead", {"moveMagnitude": 0.75}, "向前移动0.75米"),
                            ActionSpec("MoveAhead", {"moveMagnitude": 0.5}, "向前移动0.50米"),
                        ],
                        "search_nonvisible_target",
                        "",
                        {},
                    )
                )

        chosen = None
        for actions, route_type, preset_target_id, preset_target in route_candidates:
            events, steps = self._collect_route_events(controller, state, actions[:3])
            if len(events) != 4:
                continue
            target = None
            target_id = preset_target_id
            if proposal.subcat == "start_visible":
                target = _find_object(events[-1], target_id)
            else:
                start_visible = _visible_object_ids(events[0])
                final_visible = _visible_objects(events[-1])
                final_visible.sort(key=lambda obj: float(obj.get("distance", 999.0)))
                for obj in final_visible:
                    if obj["objectId"] in start_visible:
                        continue
                    if not _good_visible_object(events[-1], obj, width=self.args.width, height=self.args.height):
                        continue
                    target = obj
                    target_id = obj["objectId"]
                    break
            if target is None:
                continue
            if proposal.subcat == "start_not_visible" and target_id in _visible_object_ids(events[0]):
                continue
            distances = self._distance_progression(events, target["position"])
            if not self._progression_is_logical(distances):
                continue
            chosen = (events, steps, route_type, target, distances)
            break

        if chosen is None:
            raise RuntimeError("class04_no_valid_route")
        events, steps, route_type, target, distances = chosen
        label_to_time, label_to_path, answer_seq = self._shuffle_labels(proposal, events, ["A", "B", "C", "D"])
        frame_paths = [str(label_to_path[label]) for label in sorted(label_to_path)]
        target_name = _pretty_name(target.get("objectType", "object"))
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "movement_sequence_sorting",
                "input": {
                    "frame_paths": frame_paths,
                    "labeled_frames": {label: str(path) for label, path in sorted(label_to_path.items())},
                },
                "question": (
                    f"智能体正朝着【{target_name}】所在区域移动，目标不一定会一直出现在视野中。"
                    f"请根据空间布局和相机位姿变化，给出这4张图片（A, B, C, D）正确的时序发生顺序。"
                ),
                "answer": f"正确的顺序是：{', '.join(answer_seq.split(' -> '))}",
                "gt": {
                    "target": {
                        "id": target["objectId"],
                        "type": target.get("objectType", "Unknown"),
                        "name": target_name,
                        "position": target["position"],
                    },
                    "route_type": route_type,
                    "api_actions": [step.action_api for step in steps],
                    "steps": [asdict(step) for step in steps],
                    "start_state": {"text": _pose_from_event(events[0]).to_text(), "raw": asdict(_pose_from_event(events[0]))},
                    "end_state": {"text": _pose_from_event(events[-1]).to_text(), "raw": asdict(_pose_from_event(events[-1]))},
                    "time_order": [f"T{i+1}" for i in range(len(events))],
                    "label_to_time": label_to_time,
                    "correct_label_sequence": answer_seq,
                    "target_distance_progression": distances,
                    "target_visibility_by_time": [target["objectId"] in _visible_object_ids(event) for event in events],
                    "start_visibility_mode": proposal.subcat,
                },
            }
        )
        return base

    def _execute_class05(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        evt_a = _teleport_to_state(controller, state)
        if not bool(evt_a.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class05_teleport_failed")
        pose_a = _pose_from_event(evt_a)
        if proposal.subcat == "rotation":
            action_candidates = [
                ("RotateLeft", "向左旋转"),
                ("RotateRight", "向右旋转"),
            ]
            values = [90.0, 60.0, 45.0, 30.0]
            larger = float(proposal.payload.get("degrees", 90.0))
            larger = min(values, key=lambda item: abs(item - larger))
            smaller = next((value for value in values if value < larger), 30.0)
            arg_name = "degrees"
            family = "rotation"
            unit = "degrees"
        else:
            action_candidates = [
                ("MoveAhead", "向前移动"),
                ("MoveBack", "向后移动"),
                ("MoveLeft", "向左移动"),
                ("MoveRight", "向右移动"),
            ]
            values = [1.0, 0.75, 0.5, 0.25]
            larger = max(0.5, float(proposal.payload.get("move_magnitude", 0.75)))
            larger = min(values, key=lambda item: abs(item - larger))
            smaller = next((value for value in values if value < larger), 0.25)
            arg_name = "moveMagnitude"
            family = "translation"
            unit = "meters"

        chosen = None
        best_score = None
        for action_api, text in action_candidates:
            outcomes = []
            valid = True
            for tag, planned in (("B", larger), ("C", smaller)):
                _teleport_to_state(controller, state)
                evt = controller.step(action=action_api, **{arg_name: planned})
                if not bool(evt.metadata.get("lastActionSuccess", False)):
                    valid = False
                    break
                end_pose = _pose_from_event(evt)
                if family == "rotation":
                    actual_value = _angular_diff(end_pose.yaw, pose_a.yaw)
                else:
                    actual_value = _dist3((pose_a.x, pose_a.y, pose_a.z), (end_pose.x, end_pose.y, end_pose.z))
                outcomes.append((tag, planned, actual_value, evt, end_pose))
            if not valid or len(outcomes) != 2:
                continue
            score = (
                _frame_diff_score(evt_a.frame, outcomes[0][3].frame)
                + _frame_diff_score(evt_a.frame, outcomes[1][3].frame)
                + _frame_diff_score(outcomes[0][3].frame, outcomes[1][3].frame)
            )
            if best_score is None or score > best_score:
                best_score = score
                chosen = (action_api, text, outcomes)

        if chosen is None:
            raise RuntimeError("class05_no_valid_action")
        action_api, text, outcomes = chosen
        sample_dir = self._sample_dir(proposal)
        frame_a = sample_dir / "frame_A.png"
        frame_b = sample_dir / "frame_B.png"
        frame_c = sample_dir / "frame_C.png"
        _save_frame(evt_a.frame, frame_a)
        _save_frame(outcomes[0][3].frame, frame_b)
        _save_frame(outcomes[1][3].frame, frame_c)

        def frame_gt(item):
            tag, planned, actual_value, event, end_pose = item
            if family == "rotation":
                action_text = f"{text}{int(round(planned))}度"
            else:
                action_text = f"{text}{planned:.2f}米"
            return {
                "tag": tag,
                "family": family,
                "action_api": action_api,
                "action_text": action_text,
                "planned_value": planned,
                "actual_value": actual_value,
                "unit": unit,
                "success": True,
                "start_pose": asdict(pose_a),
                "end_pose": asdict(end_pose),
            }

        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "movement_degree_comparison",
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b), str(frame_c)],
                    "labeled_frames": {"A": str(frame_a), "B": str(frame_b), "C": str(frame_c)},
                    "reference_frame": str(frame_a),
                },
                "question": (
                    "图A是起始视角，图B和图C都表示智能体从图A出发，各执行了一次同类型但不同幅度的单动作后的结果。"
                    "这个动作类型可能是前后左右移动，也可能是左右转动。请判断图B和图C里，哪一张对应的动作幅度更大？"
                ),
                "answer": "【图B】对应的动作幅度更大。",
                "gt": {
                    "start_pose": {"text": pose_a.to_text(), "raw": asdict(pose_a)},
                    "comparison_type": family,
                    "action_api": action_api,
                    "action_description": text,
                    "same_action_type_constraint": True,
                    "larger_image": "B",
                    "smaller_image": "C",
                    "frame_A_role": "shared_start",
                    "frame_B": frame_gt(outcomes[0]),
                    "frame_C": frame_gt(outcomes[1]),
                },
            }
        )
        return base

    def _object_state_summary(self, obj: Optional[Dict]) -> Dict:
        if obj is None:
            return {"exists": False}
        rotation = obj.get("rotation") or {}
        return {
            "exists": True,
            "visible": bool(obj.get("visible", False)),
            "is_open": bool(obj.get("isOpen", False)),
            "is_toggled": bool(obj.get("isToggled", False)),
            "rotation_y": float(rotation.get("y", 0.0)),
        }

    def _execute_class06(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        evt_a = _teleport_to_state(controller, state)
        if not bool(evt_a.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class06_teleport_failed")
        target_id = str(proposal.payload["target_object_id"])
        target = _find_object(evt_a, target_id)
        if not _good_visible_object(evt_a, target, width=self.args.width, height=self.args.height):
            raise RuntimeError("class06_target_not_visible")
        object_type = target.get("objectType", "Unknown")

        if proposal.subcat == "open_close":
            if not bool(target.get("openable", False)):
                raise RuntimeError("class06_not_openable")
            is_open = bool(target.get("isOpen", False))
            action_api = "CloseObject" if is_open else "OpenObject"
            action_args = {"objectId": target_id, "forceAction": True}
            question = "场景中哪个物体的开合状态发生了变化？"
        elif proposal.subcat == "toggle":
            if not bool(target.get("toggleable", False)):
                raise RuntimeError("class06_not_toggleable")
            is_toggled = bool(target.get("isToggled", False))
            action_api = "ToggleObjectOff" if is_toggled else "ToggleObjectOn"
            action_args = {"objectId": target_id, "forceAction": True}
            question = "场景中哪个物体的开关状态发生了变化？"
        elif proposal.subcat == "rotate":
            if object_type in ROTATION_EXCLUDED_TYPES or not (target.get("pickupable", False) or target.get("moveable", False)):
                raise RuntimeError("class06_not_rotatable")
            current_rot = target.get("rotation") or {}
            current_y = float(current_rot.get("y", 0.0))
            target_y = (_normalize_angle(current_y + 90.0)) % 360.0
            action_api = "TeleportObject"
            action_args = {
                "objectId": target_id,
                "position": target["position"],
                "rotation": {"x": 0.0, "y": target_y, "z": 0.0},
                "forceAction": True,
            }
            question = "场景中哪个物体的朝向发生了变化？"
        else:
            if not (target.get("pickupable", False) or target.get("moveable", False)):
                raise RuntimeError("class06_not_removable")
            action_api = "DisableObject"
            action_args = {"objectId": target_id}
            question = "场景中哪个物体被移走了？"

        evt_after = controller.step(action=action_api, **action_args)
        if not bool(evt_after.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class06_action_failed")
        controller.step(action="AdvancePhysicsStep", simSeconds=0.25)
        evt_b = controller.last_event
        after_target = _find_object(evt_b, target_id)
        before_state = self._object_state_summary(target)
        after_state = self._object_state_summary(after_target)

        if proposal.subcat == "open_close":
            if after_target is None or not _good_visible_object(evt_b, after_target, width=self.args.width, height=self.args.height, relaxed=True):
                raise RuntimeError("class06_open_close_after_invisible")
            if before_state["is_open"] == after_state["is_open"]:
                raise RuntimeError("class06_open_close_no_change")
            change_text = "由打开变成了关闭" if before_state["is_open"] else "由关闭变成了打开"
        elif proposal.subcat == "toggle":
            if after_target is None or not _good_visible_object(evt_b, after_target, width=self.args.width, height=self.args.height, relaxed=True):
                raise RuntimeError("class06_toggle_after_invisible")
            if before_state["is_toggled"] == after_state["is_toggled"]:
                raise RuntimeError("class06_toggle_no_change")
            change_text = "由开启变成了关闭" if before_state["is_toggled"] else "由关闭变成了开启"
        elif proposal.subcat == "rotate":
            if after_target is None or not _good_visible_object(evt_b, after_target, width=self.args.width, height=self.args.height, relaxed=True):
                raise RuntimeError("class06_rotate_after_invisible")
            delta = _angular_diff(before_state["rotation_y"], after_state["rotation_y"])
            if delta < 40.0:
                raise RuntimeError("class06_rotate_delta_too_small")
            change_text = f"朝向发生了变化，旋转了约{int(round(delta))}度"
        else:
            if after_target is not None and bool(after_target.get("visible", False)):
                raise RuntimeError("class06_remove_still_visible")
            change_text = "被移走了"

        sample_dir = self._sample_dir(proposal)
        frame_a = sample_dir / "frame_000_A.png"
        frame_b = sample_dir / "frame_001_B.png"
        _save_frame(evt_a.frame, frame_a)
        _save_frame(evt_b.frame, frame_b)
        target_name = _pretty_name(object_type)
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "object_state_attribute_changes",
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b)],
                    "frame_A": str(frame_a),
                    "frame_B": str(frame_b),
                },
                "question": question,
                "answer": f"【{target_name}】{change_text}。",
                "gt": {
                    "target": {"id": target_id, "type": object_type, "name": target_name},
                    "change_category": proposal.subcat,
                    "change_text": change_text,
                    "action_api": action_api,
                    "action_params": action_args,
                    "target_state_before": before_state,
                    "target_state_after": after_state,
                    "start_state": {"text": _pose_from_event(evt_a).to_text(), "raw": asdict(_pose_from_event(evt_a))},
                    "end_state": {"text": _pose_from_event(evt_b).to_text(), "raw": asdict(_pose_from_event(evt_b))},
                },
            }
        )
        return base

    def _object_snapshot(self, event, obj: Optional[Dict]) -> Dict:
        if obj is None:
            return {"exists": False}
        return {
            "exists": True,
            "id": obj["objectId"],
            "type": obj.get("objectType", "Unknown"),
            "name": _pretty_name(obj.get("objectType", "object")),
            "visible": bool(obj.get("visible", False)),
            "distance": float(obj.get("distance", 999.0)),
            "position": obj.get("position"),
            "rotation": obj.get("rotation"),
            "parent_receptacles": obj.get("parentReceptacles") or [],
            "bbox_quality": _bbox_quality(
                _bbox_map(event).get(obj["objectId"]),
                width=self.args.width,
                height=self.args.height,
            ),
            "aabb_size": _aabb_size(obj),
            "aabb_bottom_y": _aabb_bottom_y(obj),
        }

    def _qualified_name(self, primary: Dict, counterpart: Dict) -> str:
        if primary.get("name") != counterpart.get("name"):
            return primary.get("name", "object")
        q_a = primary.get("bbox_quality") or {}
        q_b = counterpart.get("bbox_quality") or {}
        cx_a = q_a.get("center_x_ratio")
        cx_b = q_b.get("center_x_ratio")
        if cx_a is not None and cx_b is not None:
            return f"更靠左的{primary['name']}" if cx_a < cx_b else f"更靠右的{primary['name']}"
        return primary.get("name", "object")

    def _execute_class07(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        evt_a = _teleport_to_state(controller, state)
        if not bool(evt_a.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class07_teleport_failed")
        object_a_id = str(proposal.payload["first_object_id"])
        object_b_id = str(proposal.payload["second_object_id"])
        obj_a = _find_object(evt_a, object_a_id)
        obj_b = _find_object(evt_a, object_b_id)
        if not _good_visible_object(evt_a, obj_a, width=self.args.width, height=self.args.height):
            raise RuntimeError("class07_object_a_not_visible")
        if not _good_visible_object(evt_a, obj_b, width=self.args.width, height=self.args.height):
            raise RuntimeError("class07_object_b_not_visible")
        support_id_a = _support_parent_id(obj_a)
        support_id_b = _support_parent_id(obj_b)
        support_type = _support_parent_type(obj_a)
        if support_id_a is None or support_id_a != support_id_b:
            raise RuntimeError("class07_support_mismatch")
        if support_type in DISALLOWED_SUPPORT_TYPES:
            raise RuntimeError("class07_disallowed_support")
        pair_distance = _dist3(obj_a["position"], obj_b["position"])
        if not (0.12 <= pair_distance <= 1.65):
            raise RuntimeError("class07_pair_distance_invalid")

        pos_a_before = dict(obj_a["position"])
        pos_b_before = dict(obj_b["position"])
        rot_a_before = dict(obj_a["rotation"])
        rot_b_before = dict(obj_b["rotation"])
        lift_height = min(0.48, max(0.22, 0.10 + max(_aabb_size(obj_a).values(), default=0.0) * 0.72))
        pair_dx = float(pos_b_before["x"]) - float(pos_a_before["x"])
        pair_dz = float(pos_b_before["z"]) - float(pos_a_before["z"])
        pair_norm = math.hypot(pair_dx, pair_dz)
        if pair_norm > 1e-6:
            perp_x = -pair_dz / pair_norm
            perp_z = pair_dx / pair_norm
        else:
            perp_x, perp_z = 1.0, 0.0
        side_offset = min(0.22, max(0.12, pair_distance * 0.35))
        temp_positions = [
            {
                "x": float(pos_a_before["x"]) + perp_x * side_offset,
                "y": float(pos_a_before["y"]) + lift_height,
                "z": float(pos_a_before["z"]) + perp_z * side_offset,
            },
            {
                "x": float(pos_a_before["x"]) - perp_x * side_offset,
                "y": float(pos_a_before["y"]) + lift_height,
                "z": float(pos_a_before["z"]) - perp_z * side_offset,
            },
            {
                "x": float(pos_a_before["x"]),
                "y": float(pos_a_before["y"]) + lift_height,
                "z": float(pos_a_before["z"]),
            },
        ]
        last_error = "class07_teleport_object_failed"
        best_swap = None
        for temp_pos_a in temp_positions:
            evt_start = _teleport_to_state(controller, state)
            if not bool(evt_start.metadata.get("lastActionSuccess", False)):
                last_error = "class07_teleport_failed"
                continue
            obj_a_start = _find_object(evt_start, object_a_id)
            obj_b_start = _find_object(evt_start, object_b_id)
            if not _good_visible_object(evt_start, obj_a_start, width=self.args.width, height=self.args.height):
                last_error = "class07_object_a_not_visible"
                continue
            if not _good_visible_object(evt_start, obj_b_start, width=self.args.width, height=self.args.height):
                last_error = "class07_object_b_not_visible"
                continue
            action_sequence = [
                {
                    "api": "TeleportObject",
                    "objectId": object_a_id,
                    "position": temp_pos_a,
                    "rotation": rot_a_before,
                    "forceAction": True,
                    "purpose": "temporary_lift_and_clear_slot",
                },
                {
                    "api": "TeleportObject",
                    "objectId": object_b_id,
                    "position": pos_a_before,
                    "rotation": rot_b_before,
                    "forceAction": True,
                    "purpose": "move_object_b_to_object_a_position",
                },
                {
                    "api": "TeleportObject",
                    "objectId": object_a_id,
                    "position": pos_b_before,
                    "rotation": rot_a_before,
                    "forceAction": True,
                    "purpose": "move_object_a_to_object_b_position",
                },
                {"api": "AdvancePhysicsStep", "simSeconds": 0.50, "purpose": "settle_objects"},
            ]
            failed = False
            for kwargs in action_sequence[:-1]:
                evt = controller.step(action="TeleportObject", **{k: v for k, v in kwargs.items() if k not in {"api", "purpose"}})
                if not bool(evt.metadata.get("lastActionSuccess", False)):
                    last_error = "class07_teleport_object_failed"
                    failed = True
                    break
            if failed:
                continue
            controller.step(action="AdvancePhysicsStep", simSeconds=0.50)
            evt_b = controller.last_event
            obj_a_after = _find_object(evt_b, object_a_id)
            obj_b_after = _find_object(evt_b, object_b_id)
            if not _good_visible_object(evt_b, obj_a_after, width=self.args.width, height=self.args.height, relaxed=True):
                last_error = "class07_object_a_after_invisible"
                continue
            if not _good_visible_object(evt_b, obj_b_after, width=self.args.width, height=self.args.height, relaxed=True):
                last_error = "class07_object_b_after_invisible"
                continue
            err_a = _dist3(obj_a_after["position"], pos_b_before)
            err_b = _dist3(obj_b_after["position"], pos_a_before)
            if err_a > 0.16 or err_b > 0.16:
                last_error = "class07_position_error_large"
                continue
            best_swap = (
                evt_start,
                evt_b,
                obj_a_start,
                obj_b_start,
                obj_a_after,
                obj_b_after,
                err_a,
                err_b,
                action_sequence,
            )
            break
        if best_swap is None:
            raise RuntimeError(last_error)
        evt_a, evt_b, obj_a, obj_b, obj_a_after, obj_b_after, err_a, err_b, action_sequence = best_swap

        before_a = self._object_snapshot(evt_a, obj_a)
        before_b = self._object_snapshot(evt_a, obj_b)
        after_a = self._object_snapshot(evt_b, obj_a_after)
        after_b = self._object_snapshot(evt_b, obj_b_after)
        answer_name_a = self._qualified_name(before_a, before_b)
        answer_name_b = self._qualified_name(before_b, before_a)

        sample_dir = self._sample_dir(proposal)
        frame_a = sample_dir / "frame_000_A.png"
        frame_b = sample_dir / "frame_001_B.png"
        _save_frame(evt_a.frame, frame_a)
        _save_frame(evt_b.frame, frame_b)
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "object_position_swapping",
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b)],
                    "frame_A": str(frame_a),
                    "frame_B": str(frame_b),
                },
                "question": "图A是原始场景，图B是交换两个物体位置后的场景。场景中哪两个物体交换了位置？",
                "answer": f"【{answer_name_a}】和【{answer_name_b}】交换了位置。",
                "gt": {
                    "swap_pair": [
                        {"before": before_a, "after": after_a, "answer_name": answer_name_a},
                        {"before": before_b, "after": after_b, "answer_name": answer_name_b},
                    ],
                    "support_parent": {"id": support_id_a, "type": support_type},
                    "swap_metrics": {
                        "position_error_a_to_target_b": err_a,
                        "position_error_b_to_target_a": err_b,
                        "screen_center_shift_a": _screen_center_distance(before_a.get("bbox_quality"), after_a.get("bbox_quality")),
                        "screen_center_shift_b": _screen_center_distance(before_b.get("bbox_quality"), after_b.get("bbox_quality")),
                    },
                    "action_sequence": action_sequence,
                    "start_state": {"text": _pose_from_event(evt_a).to_text(), "raw": asdict(_pose_from_event(evt_a))},
                    "end_state": {"text": _pose_from_event(evt_b).to_text(), "raw": asdict(_pose_from_event(evt_b))},
                },
            }
        )
        return base

    def _attempt_object_move(
        self,
        controller,
        state: StateRecord,
        mover_id: str,
        local_offset: Tuple[float, float],
        *,
        require_screen_shift: bool = True,
    ) -> Optional[Dict]:
        evt_a = _teleport_to_state(controller, state)
        if not bool(evt_a.metadata.get("lastActionSuccess", False)):
            return None
        mover_before = _find_object(evt_a, mover_id)
        if not _good_visible_object(evt_a, mover_before, width=self.args.width, height=self.args.height):
            return None
        if not (mover_before.get("pickupable", False) or mover_before.get("moveable", False)):
            return None
        support_type = _support_parent_type(mover_before)
        if support_type in DISALLOWED_SUPPORT_TYPES:
            return None
        pose = _pose_from_event(evt_a)
        world_dx, world_dz = _local_to_world_xz(local_offset[0], local_offset[1], pose.yaw)
        pos_before = dict(mover_before["position"])
        rot_before = dict(mover_before["rotation"])
        target_pos = {
            "x": float(pos_before["x"]) + world_dx,
            "y": float(pos_before["y"]),
            "z": float(pos_before["z"]) + world_dz,
        }
        move_distance = _dist3(pos_before, target_pos)
        if not (0.16 <= move_distance <= 0.48):
            return None
        lift_height = min(0.50, max(0.22, 0.12 + max(_aabb_size(mover_before).values(), default=0.0) * 0.75))
        temp_pos = {
            "x": float(pos_before["x"]),
            "y": float(pos_before["y"]) + lift_height,
            "z": float(pos_before["z"]),
        }
        step1 = controller.step(
            action="TeleportObject",
            objectId=mover_id,
            position=temp_pos,
            rotation=rot_before,
            forceAction=True,
        )
        if not bool(step1.metadata.get("lastActionSuccess", False)):
            return None
        step2 = controller.step(
            action="TeleportObject",
            objectId=mover_id,
            position=target_pos,
            rotation=rot_before,
            forceAction=True,
        )
        if not bool(step2.metadata.get("lastActionSuccess", False)):
            return None
        controller.step(action="AdvancePhysicsStep", simSeconds=0.50)
        evt_b = controller.last_event
        mover_after = _find_object(evt_b, mover_id)
        if not _good_visible_object(evt_b, mover_after, width=self.args.width, height=self.args.height, relaxed=True):
            return None
        if _dist3(mover_after["position"], target_pos) > 0.08:
            return None
        q_before = _bbox_quality(_bbox_map(evt_a).get(mover_id), width=self.args.width, height=self.args.height)
        q_after = _bbox_quality(_bbox_map(evt_b).get(mover_id), width=self.args.width, height=self.args.height)
        if require_screen_shift and _screen_center_distance(q_before, q_after) < 0.03:
            return None
        return {
            "event_a": evt_a,
            "event_b": evt_b,
            "mover_before": mover_before,
            "mover_after": mover_after,
            "support_parent": {"id": _support_parent_id(mover_before), "type": support_type},
            "action_sequence": [
                {
                    "api": "TeleportObject",
                    "objectId": mover_id,
                    "position": temp_pos,
                    "rotation": rot_before,
                    "forceAction": True,
                    "purpose": "temporary_lift",
                },
                {
                    "api": "TeleportObject",
                    "objectId": mover_id,
                    "position": target_pos,
                    "rotation": rot_before,
                    "forceAction": True,
                    "purpose": "move_object",
                },
                {"api": "AdvancePhysicsStep", "simSeconds": 0.50, "purpose": "settle_object"},
            ],
            "move_text": f"移动约{move_distance:.2f}米",
            "local_offset": {"x": local_offset[0], "z": local_offset[1]},
        }

    def _class08_good_target_view(self, event, obj: Optional[Dict], *, relaxed: bool = False) -> bool:
        if obj is None or not bool(obj.get("visible", False)) or obj.get("position") is None:
            return False
        if obj.get("objectType", "Unknown") in EXCLUDED_OBJECT_TYPES:
            return False
        q = _bbox_quality(_bbox_map(event).get(obj["objectId"]), width=self.args.width, height=self.args.height)
        if q is None:
            return False
        if float(obj.get("distance", 999.0)) > 4.2:
            return False
        if relaxed:
            return (
                q["area_ratio"] >= 0.008
                and q["width"] >= 34.0
                and q["height"] >= 26.0
                and 0.10 <= q["center_x_ratio"] <= 0.90
                and 0.10 <= q["center_y_ratio"] <= 0.90
                and q["edge_margin_ratio"] >= 0.01
            )
        return (
            0.012 <= q["area_ratio"] <= 0.20
            and q["width"] >= 40.0
            and q["height"] >= 32.0
            and 0.14 <= q["center_x_ratio"] <= 0.86
            and 0.12 <= q["center_y_ratio"] <= 0.88
            and q["edge_margin_ratio"] >= 0.015
        )

    def _class08_safe_mover(self, obj: Dict) -> bool:
        if obj.get("objectType", "Unknown") in EXCLUDED_OBJECT_TYPES:
            return False
        if obj.get("position") is None:
            return False
        if float(obj.get("distance", 999.0)) > 4.2:
            return False
        if bool(obj.get("pickupable", False)):
            pass
        elif bool(obj.get("moveable", False)):
            size = _aabb_size(obj)
            if max(size["x"], size["y"], size["z"]) > 0.55 or size["y"] > 0.50:
                return False
        else:
            return False
        if _support_parent_type(obj) in DISALLOWED_SUPPORT_TYPES:
            return False
        return True

    def _class08_candidate_movers(self, event, preferred_object_id: Optional[str]) -> List[Dict]:
        movers = []
        for obj in _visible_objects(event):
            if not self._class08_safe_mover(obj):
                continue
            if not self._class08_good_target_view(event, obj):
                continue
            movers.append(obj)
        movers.sort(
            key=lambda obj: (
                0 if str(obj.get("objectId")) == str(preferred_object_id or "") else 1,
                0 if _support_parent_type(obj) in PREFERRED_SUPPORT_TYPES else 1,
                float(obj.get("distance", 999.0)),
            )
        )
        return movers

    def _class08_visible_type_counts(self, event) -> Counter:
        return Counter(obj.get("objectType", "Unknown") for obj in _visible_objects(event))

    def _class08_candidate_targets(self, event, exclude_ids: Optional[set] = None) -> List[Dict]:
        exclude_ids = exclude_ids or set()
        counts = self._class08_visible_type_counts(event)
        objs = []
        for obj in _visible_objects(event):
            if obj.get("objectId") in exclude_ids:
                continue
            if obj.get("objectType", "Unknown") in EXCLUDED_OBJECT_TYPES:
                continue
            if counts.get(obj.get("objectType", "Unknown"), 0) != 1:
                continue
            if not self._class08_good_target_view(event, obj, relaxed=True):
                continue
            objs.append(obj)
        objs.sort(key=lambda obj: float(obj.get("distance", 999.0)))
        return objs

    def _class08_restore_view(self, controller, scene_name: str, pose: PoseState):
        _reset_scene(controller, scene_name)
        evt = _teleport_to_pose(controller, pose)
        if not bool(evt.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class08_restore_view_failed")
        return evt

    def _class08_move_text(self, local_dx: float, local_dz: float) -> str:
        parts = []
        if local_dz <= -0.12:
            parts.append("向观察者靠近")
        elif local_dz >= 0.12:
            parts.append("远离观察者")
        if local_dx <= -0.12:
            parts.append("向左")
        elif local_dx >= 0.12:
            parts.append("向右")
        if not parts:
            parts.append("轻微移动")
        magnitude = math.sqrt(local_dx ** 2 + local_dz ** 2)
        return f"{'并'.join(parts)}约{magnitude:.2f}米"

    def _class08_search_best_view(
        self,
        controller,
        scene_name: str,
        seed_pose: PoseState,
        preferred_object_id: Optional[str],
        rng: random.Random,
    ) -> Optional[Tuple[PoseState, object]]:
        candidates: List[PoseState] = [seed_pose]
        _reset_scene(controller, scene_name)
        evt = controller.step(action="GetReachablePositions")
        reachable = evt.metadata.get("actionReturn", [])
        if not reachable:
            try:
                evt_seed = self._class08_restore_view(controller, scene_name, seed_pose)
            except RuntimeError:
                return None
            return seed_pose, evt_seed
        for _ in range(26):
            pos = rng.choice(reachable)
            candidates.append(
                PoseState(
                    x=float(pos["x"]),
                    y=float(pos["y"]),
                    z=float(pos["z"]),
                    yaw=float(rng.choice(CLASS08_VIEW_SEARCH_YAWS)),
                    pitch=float(rng.choice(CLASS08_VIEW_SEARCH_HORIZONS)),
                )
            )
        best_pose = None
        best_event = None
        best_score = None
        for pose in candidates:
            evt_try = _teleport_to_pose(controller, pose)
            if not bool(evt_try.metadata.get("lastActionSuccess", False)):
                continue
            movers = self._class08_candidate_movers(evt_try, preferred_object_id)
            targets = self._class08_candidate_targets(evt_try)
            preferred_bonus = 0
            if preferred_object_id is not None:
                preferred_obj = _find_object(evt_try, str(preferred_object_id))
                if self._class08_good_target_view(evt_try, preferred_obj):
                    preferred_bonus = 2
            score = (preferred_bonus, len(movers), len(targets))
            if best_score is None or score > best_score:
                best_score = score
                best_pose = pose
                best_event = evt_try
            if score[1] >= 3 and score[2] >= 5:
                break
        if best_pose is None or best_event is None:
            return None
        return best_pose, best_event

    def _class08_attempt_move_from_pose(
        self,
        controller,
        scene_name: str,
        start_pose: PoseState,
        mover_id: str,
        local_offset: Tuple[float, float],
        *,
        require_screen_shift: bool,
    ) -> Optional[Dict]:
        base_evt = self._class08_restore_view(controller, scene_name, start_pose)
        mover_base = _find_object(base_evt, mover_id)
        if not self._class08_good_target_view(base_evt, mover_base):
            return None
        scales = (1.00, 0.88, 0.76, 0.64, 0.52)
        for scale in scales:
            evt_a = self._class08_restore_view(controller, scene_name, start_pose)
            mover_before = _find_object(evt_a, mover_id)
            if not self._class08_good_target_view(evt_a, mover_before):
                continue
            support_parent_id = _support_parent_id(mover_before)
            pos_before = dict(mover_before["position"])
            rot_before = dict(mover_before["rotation"])
            scaled_offset = (local_offset[0] * scale, local_offset[1] * scale)
            world_dx, world_dz = _local_to_world_xz(scaled_offset[0], scaled_offset[1], start_pose.yaw)
            target_pos = {
                "x": float(pos_before["x"]) + world_dx,
                "y": float(pos_before["y"]),
                "z": float(pos_before["z"]) + world_dz,
            }
            move_distance = _dist3(pos_before, target_pos)
            if not (0.16 <= move_distance <= 0.48):
                continue
            temp_pos = {
                "x": float(pos_before["x"]),
                "y": float(pos_before["y"]) + min(0.50, max(0.22, 0.12 + max(_aabb_size(mover_before).values(), default=0.0) * 0.75)),
                "z": float(pos_before["z"]),
            }
            step1 = controller.step(
                action="TeleportObject",
                objectId=mover_before["objectId"],
                position=temp_pos,
                rotation=rot_before,
                forceAction=True,
            )
            if not bool(step1.metadata.get("lastActionSuccess", False)):
                continue
            step2 = controller.step(
                action="TeleportObject",
                objectId=mover_before["objectId"],
                position=target_pos,
                rotation=rot_before,
                forceAction=True,
            )
            if not bool(step2.metadata.get("lastActionSuccess", False)):
                continue
            controller.step(action="AdvancePhysicsStep", simSeconds=0.50)
            evt_b = controller.last_event
            mover_after = _find_object(evt_b, mover_id)
            if not self._class08_good_target_view(evt_b, mover_after, relaxed=True):
                continue
            if _dist3(mover_after["position"], target_pos) > 0.08:
                continue
            if support_parent_id:
                parents_after = mover_after.get("parentReceptacles") or []
                if support_parent_id not in parents_after:
                    continue
            if float(mover_after["position"]["y"]) > float(pos_before["y"]) + 0.02:
                continue
            bottom_before = _aabb_bottom_y(mover_before)
            bottom_after = _aabb_bottom_y(mover_after)
            if bottom_before is not None and bottom_after is not None and abs(bottom_after - bottom_before) > 0.02:
                continue
            overlap_bad = False
            for other in evt_b.metadata.get("objects", []):
                other_id = str(other.get("objectId", ""))
                if other_id == mover_id:
                    continue
                if other.get("position") is None:
                    continue
                if other_id == support_parent_id or _support_parent_id(other) == mover_id:
                    continue
                if _aabb_intersection_ratio(mover_after, other) > 0.08:
                    overlap_bad = True
                    break
            if overlap_bad:
                continue
            q_before = _bbox_quality(_bbox_map(evt_a).get(mover_id), width=self.args.width, height=self.args.height)
            q_after = _bbox_quality(_bbox_map(evt_b).get(mover_id), width=self.args.width, height=self.args.height)
            if require_screen_shift and _screen_center_distance(q_before, q_after) < 0.035:
                continue
            return {
                "event_a": evt_a,
                "event_b": evt_b,
                "mover_before": mover_before,
                "mover_after": mover_after,
                "support_parent": {"id": support_parent_id, "type": _support_parent_type(mover_before)},
                "action_sequence": [
                    {
                        "api": "TeleportObject",
                        "objectId": mover_before["objectId"],
                        "position": temp_pos,
                        "rotation": rot_before,
                        "forceAction": True,
                        "purpose": "temporary_lift",
                    },
                    {
                        "api": "TeleportObject",
                        "objectId": mover_before["objectId"],
                        "position": target_pos,
                        "rotation": rot_before,
                        "forceAction": True,
                        "purpose": "move_object",
                    },
                    {"api": "AdvancePhysicsStep", "simSeconds": 0.50, "purpose": "settle_object"},
                ],
                "move_text": self._class08_move_text(scaled_offset[0], scaled_offset[1]),
                "local_offset": {"x": scaled_offset[0], "z": scaled_offset[1]},
            }
        return None

    def _class08_evaluate_distance_change(self, event_a, event_b, mover_id: str) -> Optional[Dict]:
        mover_before = _find_object(event_a, mover_id)
        mover_after = _find_object(event_b, mover_id)
        if mover_before is None or mover_after is None:
            return None
        dist_before = float(mover_before.get("distance", 999.0))
        dist_after = float(mover_after.get("distance", 999.0))
        delta = dist_after - dist_before
        if abs(delta) < 0.15:
            return None
        return {
            "relation": "更近" if delta < 0 else "更远",
            "distance_before": dist_before,
            "distance_after": dist_after,
            "distance_delta": delta,
        }

    def _class08_target_stable_after_motion(
        self,
        target_before: Dict,
        target_after: Optional[Dict],
        mover_after: Dict,
    ) -> bool:
        if target_after is None or target_after.get("position") is None:
            return False
        support_before = _support_parent_id(target_before)
        support_after = _support_parent_id(target_after)
        if support_before and support_before != support_after:
            return False
        if _dist3(target_before["position"], target_after["position"]) > 0.035:
            return False
        bottom_before = _aabb_bottom_y(target_before)
        bottom_after = _aabb_bottom_y(target_after)
        if bottom_before is not None and bottom_after is not None and abs(bottom_after - bottom_before) > 0.018:
            return False
        rot_before = _rotation_y(target_before)
        rot_after = _rotation_y(target_after)
        if rot_before is not None and rot_after is not None and _angular_diff(rot_before, rot_after) > 8.0:
            return False
        if _aabb_intersection_ratio(mover_after, target_after) > 0.03:
            return False
        return True

    def _class08_best_occlusion_target(self, event_a, event_b, mover_id: str) -> Optional[Dict]:
        mover_before = _find_object(event_a, mover_id)
        mover_after = _find_object(event_b, mover_id)
        if mover_before is None or mover_after is None:
            return None
        bbox_before_map = _bbox_map(event_a)
        bbox_after_map = _bbox_map(event_b)
        mover_bbox_before = bbox_before_map.get(mover_id)
        mover_bbox_after = bbox_after_map.get(mover_id)
        best = None
        for target_before in self._class08_candidate_targets(event_a, exclude_ids={mover_id}):
            target_after = _find_object(event_b, target_before["objectId"])
            if not self._class08_target_stable_after_motion(target_before, target_after, mover_after):
                continue
            q_before = _bbox_quality(bbox_before_map.get(target_before["objectId"]), width=self.args.width, height=self.args.height)
            q_after = _bbox_quality(bbox_after_map.get(target_before["objectId"]), width=self.args.width, height=self.args.height) if target_after else None
            if q_before is None:
                continue
            area_before = q_before["area_ratio"]
            area_after = q_after["area_ratio"] if q_after else 0.0
            overlap_before = self._bbox_overlap_ratio(bbox_before_map.get(target_before["objectId"]), mover_bbox_before)
            overlap_after = self._bbox_overlap_ratio(bbox_after_map.get(target_before["objectId"]), mover_bbox_after)
            visible_after = bool(target_after and target_after.get("visible", False) and q_after is not None)
            harder_strength = (2.0 if not visible_after else 0.0) + max(0.0, overlap_after - overlap_before) + max(
                0.0,
                (area_before - area_after) / max(area_before, 1e-6),
            )
            easier_strength = max(0.0, overlap_before - overlap_after) + max(
                0.0,
                (area_after - area_before) / max(area_before, 1e-6),
            )
            change_type = None
            strength = 0.0
            if harder_strength >= 0.45 and (not visible_after or (overlap_after > overlap_before + 0.10 and area_after < area_before * 0.72)):
                change_type = "harder"
                strength = harder_strength
            elif easier_strength >= 0.36 and overlap_before > overlap_after + 0.10 and area_after > area_before * 1.20:
                change_type = "easier"
                strength = easier_strength
            if change_type is None:
                continue
            candidate = {
                "target_before": target_before,
                "target_after": target_after,
                "change_type": change_type,
                "strength": strength,
                "visible_after": visible_after,
                "area_before": area_before,
                "area_after": area_after,
                "overlap_before": overlap_before,
                "overlap_after": overlap_after,
            }
            if best is None or candidate["strength"] > best["strength"]:
                best = candidate
        return best

    def _class08_build_distance_sample(
        self,
        controller,
        scene_name: str,
        start_pose: PoseState,
        preferred_object_id: Optional[str],
        rng: random.Random,
    ) -> Optional[Dict]:
        evt_a = self._class08_restore_view(controller, scene_name, start_pose)
        movers = self._class08_candidate_movers(evt_a, preferred_object_id)
        if not movers:
            return None
        offsets = list(CLASS08_LOCAL_MOVE_OFFSETS)
        rng.shuffle(offsets)
        for mover in movers[:CLASS08_MAX_CANDIDATE_ATTEMPTS]:
            prioritized = sorted(
                offsets,
                key=lambda off: (
                    0 if abs(off[1]) >= abs(off[0]) else 1,
                    abs(abs(off[1]) - 0.36),
                    abs(off[0]),
                ),
            )
            for offset in prioritized:
                attempt = self._class08_attempt_move_from_pose(
                    controller,
                    scene_name,
                    start_pose,
                    str(mover["objectId"]),
                    offset,
                    require_screen_shift=True,
                )
                if attempt is None:
                    continue
                change = self._class08_evaluate_distance_change(
                    attempt["event_a"],
                    attempt["event_b"],
                    str(mover["objectId"]),
                )
                if change is None:
                    continue
                return {
                    "event_a": attempt["event_a"],
                    "event_b": attempt["event_b"],
                    "question": "图A到图B中，发生移动的那个物体相对观察者是更近了还是更远了？",
                    "answer": change["relation"],
                    "gt": {
                        "question_family": "distance_change",
                        "mover": {
                            "before": self._object_snapshot(attempt["event_a"], attempt["mover_before"]),
                            "after": self._object_snapshot(attempt["event_b"], attempt["mover_after"]),
                        },
                        "action_sequence": attempt["action_sequence"],
                        "move_text": attempt["move_text"],
                        "local_offset": attempt["local_offset"],
                        "support_parent": attempt["support_parent"],
                        **change,
                        "start_pose": asdict(start_pose),
                    },
                }
        return None

    def _class08_build_occlusion_sample(
        self,
        controller,
        scene_name: str,
        start_pose: PoseState,
        preferred_object_id: Optional[str],
        rng: random.Random,
    ) -> Optional[Dict]:
        evt_a = self._class08_restore_view(controller, scene_name, start_pose)
        movers = self._class08_candidate_movers(evt_a, preferred_object_id)
        if not movers:
            return None
        offsets = list(CLASS08_LOCAL_MOVE_OFFSETS)
        rng.shuffle(offsets)
        prioritized = sorted(
            offsets,
            key=lambda off: (
                0 if abs(off[0]) >= abs(off[1]) else 1,
                abs(abs(off[0]) - 0.30),
                abs(off[1]),
            ),
        )
        for mover in movers[:CLASS08_MAX_CANDIDATE_ATTEMPTS]:
            for offset in prioritized:
                attempt = self._class08_attempt_move_from_pose(
                    controller,
                    scene_name,
                    start_pose,
                    str(mover["objectId"]),
                    offset,
                    require_screen_shift=True,
                )
                if attempt is None:
                    continue
                change = self._class08_best_occlusion_target(
                    attempt["event_a"],
                    attempt["event_b"],
                    str(mover["objectId"]),
                )
                if change is None:
                    continue
                target_name = _pretty_name(change["target_before"].get("objectType", "object"))
                answer = "更不容易看见了" if change["change_type"] == "harder" else "更容易看见了"
                return {
                    "event_a": attempt["event_a"],
                    "event_b": attempt["event_b"],
                    "question": (
                        "图A到图B中，一个物体发生了移动。"
                        f"它的移动让【{target_name}】变得更容易看见还是更不容易看见？"
                    ),
                    "answer": answer,
                    "gt": {
                        "question_family": "occlusion_change",
                        "mover": {
                            "before": self._object_snapshot(attempt["event_a"], attempt["mover_before"]),
                            "after": self._object_snapshot(attempt["event_b"], attempt["mover_after"]),
                        },
                        "affected_object": {
                            "before": self._object_snapshot(attempt["event_a"], change["target_before"]),
                            "after": self._object_snapshot(attempt["event_b"], change["target_after"]),
                        },
                        "action_sequence": attempt["action_sequence"],
                        "move_text": attempt["move_text"],
                        "local_offset": attempt["local_offset"],
                        "support_parent": attempt["support_parent"],
                        "change_type": change["change_type"],
                        "visible_after": change["visible_after"],
                        "target_area_before": change["area_before"],
                        "target_area_after": change["area_after"],
                        "overlap_before": change["overlap_before"],
                        "overlap_after": change["overlap_after"],
                        "start_pose": asdict(start_pose),
                    },
                }
        return None

    def _execute_class08(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        preferred_mover_id = str(proposal.payload["target_object_id"])
        rng = random.Random(proposal.proposal_id)
        seed_pose = PoseState(
            x=float(state.position["x"]),
            y=float(state.position["y"]),
            z=float(state.position["z"]),
            yaw=float(state.yaw),
            pitch=float(state.horizon),
        )
        searched = self._class08_search_best_view(controller, state.scene, seed_pose, preferred_mover_id, rng)
        if searched is None:
            raise RuntimeError("class08_no_candidate_view")
        start_pose, _ = searched
        if proposal.subcat == "distance_change":
            chosen = self._class08_build_distance_sample(
                controller,
                state.scene,
                start_pose,
                preferred_mover_id,
                rng,
            )
        else:
            chosen = self._class08_build_occlusion_sample(
                controller,
                state.scene,
                start_pose,
                preferred_mover_id,
                rng,
            )
        if chosen is None:
            raise RuntimeError("class08_no_valid_motion")
        sample_dir = self._sample_dir(proposal)
        frame_a = sample_dir / "frame_A.png"
        frame_b = sample_dir / "frame_B.png"
        _save_frame(chosen["event_a"].frame, frame_a)
        _save_frame(chosen["event_b"].frame, frame_b)
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "dynamic_movement_occlusion",
                "input": {
                    "frame_paths": [str(frame_a), str(frame_b)],
                    "frame_A": str(frame_a),
                    "frame_B": str(frame_b),
                },
                "question": chosen["question"],
                "answer": chosen["answer"],
                "gt": chosen["gt"],
            }
        )
        return base

    def _bbox_overlap_ratio(
        self,
        bbox_a: Optional[Tuple[float, float, float, float]],
        bbox_b: Optional[Tuple[float, float, float, float]],
    ) -> float:
        if bbox_a is None or bbox_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        return inter / area

    def _execute_class09(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        evt = _teleport_to_state(controller, state)
        if not bool(evt.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class09_teleport_failed")
        anchor = _find_object(evt, str(proposal.payload["anchor_object_id"]))
        target = _find_object(evt, str(proposal.payload["target_object_id"]))
        if not _is_visible_object(anchor):
            raise RuntimeError("class09_anchor_not_visible")
        if not _is_visible_object(target):
            raise RuntimeError("class09_target_not_visible")
        pa = anchor["position"]
        pb = target["position"]
        ra = anchor.get("rotation", {"x": 0, "y": 0, "z": 0})
        pos_a = np.array([float(pa["x"]), float(pa["y"]), float(pa["z"])], dtype=float)
        pos_b = np.array([float(pb["x"]), float(pb["y"]), float(pb["z"])], dtype=float)
        rotation = _euler_xyz_to_matrix(float(ra.get("x", 0.0)), float(ra.get("y", 0.0)), float(ra.get("z", 0.0)))
        centered = pos_b - pos_a
        local = rotation.T @ centered
        direction_word = _horizontal_direction_from_local(float(local[0]), float(local[2]))
        sample_dir = self._sample_dir(proposal)
        frame_a = sample_dir / "frame_A.png"
        _save_frame(evt.frame, frame_a)
        anchor_name = _pretty_name(anchor.get("objectType", "object"))
        target_name = _pretty_name(target.get("objectType", "object"))
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "imagined_perspective_taking",
                "input": {"frame_paths": [str(frame_a)], "frame_A": str(frame_a)},
                "question": (
                    f"想象你现在位于图中的 [{anchor_name}] 位置，"
                    f"并且把该物体当前朝向的正前方定义为你的正前方。请问 [{target_name}] 在你的什么方位？"
                ),
                "answer": direction_word,
                "gt": {
                    "preferred_direction": direction_word,
                    "object_A": {
                        "id": anchor["objectId"],
                        "name": anchor_name,
                        "position": pa,
                        "rotation_euler": {
                            "x": float(ra.get("x", 0.0)),
                            "y": float(ra.get("y", 0.0)),
                            "z": float(ra.get("z", 0.0)),
                        },
                    },
                    "object_B": {"id": target["objectId"], "name": target_name, "position": pb},
                    "rotation_matrix_A": rotation.tolist(),
                    "pos_B_centered": centered.tolist(),
                    "pos_B_local": local.tolist(),
                    "axis_convention": "AI2-THOR局部坐标：Z为forward，X为right，Y为up",
                    "direction_word": direction_word,
                },
            }
        )
        return base

    def _candidate_sequences_for_class10(self, subcat: str) -> List[List[Tuple[str, Dict[str, float], str]]]:
        moves = [
            ("MoveAhead", {"moveMagnitude": 0.6}, "向前走0.60米"),
            ("MoveAhead", {"moveMagnitude": 1.0}, "向前走1.00米"),
            ("MoveBack", {"moveMagnitude": 0.6}, "向后退0.60米"),
            ("MoveLeft", {"moveMagnitude": 0.8}, "向左平移0.80米"),
            ("MoveRight", {"moveMagnitude": 0.8}, "向右平移0.80米"),
        ]
        turns = [
            ("RotateLeft", {"degrees": 45.0}, "向左转45度"),
            ("RotateRight", {"degrees": 45.0}, "向右转45度"),
            ("RotateLeft", {"degrees": 60.0}, "向左转60度"),
            ("RotateRight", {"degrees": 60.0}, "向右转60度"),
        ]
        if subcat == "trend_disappear":
            return [[step] for step in moves[:4] + turns[:2]]
        sequences = []
        for move in moves:
            for turn in turns:
                sequences.append([move, turn])
                sequences.append([turn, move])
        sequences.extend(
            [
                [moves[3], moves[0], turns[0]],
                [moves[4], moves[0], turns[1]],
                [moves[0], moves[3], turns[0]],
                [moves[0], moves[4], turns[1]],
            ]
        )
        if subcat == "two_frame_direction":
            sequences.extend([[step] for step in moves[:4] + turns[:4]])
        return sequences

    def _reset_and_apply_sequence(
        self,
        controller,
        state: StateRecord,
        sequence: Sequence[Tuple[str, Dict[str, float], str]],
    ) -> Tuple[object, List[object]]:
        evt_a = _teleport_to_state(controller, state)
        if not bool(evt_a.metadata.get("lastActionSuccess", False)):
            raise RuntimeError("class10_teleport_failed")
        events = []
        for action_api, action_args, _ in sequence:
            evt = controller.step(action=action_api, **action_args)
            if not bool(evt.metadata.get("lastActionSuccess", False)):
                raise RuntimeError(f"class10_action_failed:{action_api}")
            events.append(evt)
        return evt_a, events

    def _direction_to_target(self, pose: PoseState, obj: Dict) -> Tuple[str, float]:
        dx = float(obj["position"]["x"]) - pose.x
        dz = float(obj["position"]["z"]) - pose.z
        local_x, local_z = _world_to_local_xz(dx, dz, pose.yaw)
        return _horizontal_direction_from_local(local_x, local_z), math.degrees(math.atan2(local_x, local_z))

    def _execute_class10(self, controller, state: StateRecord, proposal: TaskProposal) -> dict:
        target_id = str(proposal.payload["target_object_id"])
        sequences = self._candidate_sequences_for_class10(proposal.subcat)
        chosen = None

        for sequence in sequences:
            try:
                evt_a, events = self._reset_and_apply_sequence(controller, state, sequence)
            except Exception:
                continue
            target_a = _find_object(evt_a, target_id)
            if not _good_visible_object(evt_a, target_a, width=self.args.width, height=self.args.height):
                continue
            target_end = _find_object(events[-1], target_id)
            if target_end is None or target_end.get("position") is None:
                continue
            start_pose = _pose_from_event(evt_a)
            end_pose = _pose_from_event(events[-1])
            seq_text = "，然后".join(text for _, _, text in sequence)
            action_sequence = [{"api": action_api, **action_args, "text": text} for action_api, action_args, text in sequence]

            if proposal.subcat == "single_visibility":
                final_visible = bool(target_end.get("visible", False))
                loc = None
                if final_visible:
                    dx = float(target_end["position"]["x"]) - end_pose.x
                    dz = float(target_end["position"]["z"]) - end_pose.z
                    local_x, local_z = _world_to_local_xz(dx, dz, end_pose.yaw)
                    loc = _location_word(local_x, local_z)
                chosen = {
                    "frame_events": [("frame_A", evt_a)],
                    "question": (
                        f"如果我从当前视角依次{seq_text}，那么【{_pretty_name(target_a.get('objectType', 'object'))}】还能看见吗？"
                        "如果还能看见，它会更靠左侧、中间还是右侧？"
                    ),
                    "answer": f"能看见，它会在{loc}。" if final_visible else "不能看见，它会移出视野。",
                    "gt": {
                        "question_family": "single_visibility",
                        "target": {"id": target_id, "name": _pretty_name(target_a.get("objectType", "object"))},
                        "action_sequence": action_sequence,
                        "action_text": seq_text,
                        "start_pose": asdict(start_pose),
                        "result_visibility": final_visible,
                        "result_location": loc,
                    },
                }
                break
            if proposal.subcat == "single_direction":
                direction_word, _ = self._direction_to_target(end_pose, target_end)
                chosen = {
                    "frame_events": [("frame_A", evt_a)],
                    "question": f"如果我从当前视角依次{seq_text}，那么【{_pretty_name(target_a.get('objectType', 'object'))}】相对我会在什么方位？",
                    "answer": direction_word,
                    "gt": {
                        "question_family": "single_direction",
                        "preferred_direction": direction_word,
                        "target": {"id": target_id, "name": _pretty_name(target_a.get("objectType", "object"))},
                        "action_sequence": action_sequence,
                        "action_text": seq_text,
                        "start_pose": asdict(start_pose),
                        "direction_word": direction_word,
                    },
                }
                break
            if proposal.subcat == "two_frame_direction":
                direction_word, _ = self._direction_to_target(end_pose, target_end)
                chosen = {
                    "frame_events": [("frame_A", evt_a), ("frame_B", events[-1])],
                    "question": (
                        "图A和图B对应连续移动前后两个视角。"
                        f"如果我移动到图B的位置，那么图A里的【{_pretty_name(target_a.get('objectType', 'object'))}】相对我会在什么方位？"
                    ),
                    "answer": direction_word,
                    "gt": {
                        "question_family": "two_frame_direction",
                        "preferred_direction": direction_word,
                        "target": {"id": target_id, "name": _pretty_name(target_a.get("objectType", "object"))},
                        "action_sequence": action_sequence,
                        "action_text": seq_text,
                        "start_pose": asdict(start_pose),
                        "direction_word": direction_word,
                    },
                }
                break
            if proposal.subcat == "trend_disappear":
                evt_b = events[-1]
                target_b = _find_object(evt_b, target_id)
                if target_b is None or not bool(target_b.get("visible", False)):
                    continue
                try:
                    for action_api, action_args, _ in sequence:
                        evt_c = controller.step(action=action_api, **action_args)
                        if not bool(evt_c.metadata.get("lastActionSuccess", False)):
                            raise RuntimeError("trend_second_pass_fail")
                except Exception:
                    continue
                target_c = _find_object(evt_c, target_id)
                if target_c is None or bool(target_c.get("visible", False)):
                    continue
                comparison = None
                for obj in _visible_objects(evt_b):
                    if obj["objectId"] == target_id:
                        continue
                    obj_c = _find_object(evt_c, obj["objectId"])
                    if obj_c is not None and bool(obj_c.get("visible", False)):
                        comparison = obj
                        break
                if comparison is None:
                    continue
                chosen = {
                    "frame_events": [("frame_A", evt_a), ("frame_B", evt_b)],
                    "question": (
                        "图A到图B体现了同一种运动趋势。"
                        f"如果我继续按照这个趋势再移动一步，视野中的【{_pretty_name(target_b.get('objectType', 'object'))}】"
                        f"和【{_pretty_name(comparison.get('objectType', 'object'))}】谁会先移出视野？"
                    ),
                    "answer": f"【{_pretty_name(target_b.get('objectType', 'object'))}】会先移出视野。",
                    "gt": {
                        "question_family": "trend_disappear",
                        "answer_object": {"id": target_id, "name": _pretty_name(target_b.get("objectType", "object"))},
                        "comparison_object": {
                            "id": comparison["objectId"],
                            "name": _pretty_name(comparison.get("objectType", "object")),
                        },
                        "action_sequence": action_sequence,
                        "action_text": seq_text,
                        "start_pose": asdict(start_pose),
                    },
                }
                break

        if chosen is None:
            raise RuntimeError("class10_no_valid_sequence")
        sample_dir = self._sample_dir(proposal)
        frame_paths = []
        input_obj = {"frame_paths": frame_paths}
        for label, event in chosen["frame_events"]:
            path = sample_dir / f"{label}.png"
            _save_frame(event.frame, path)
            input_obj[label] = str(path)
            frame_paths.append(str(path))
        base = self._row_base(proposal, state)
        base.update(
            {
                "task_type": "imagined_movement_consequence",
                "input": input_obj,
                "question": chosen["question"],
                "answer": chosen["answer"],
                "gt": chosen["gt"],
            }
        )
        return base


def parse_args(default_class_id: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Execute class-{default_class_id:02d} task proposals into final QA samples.")
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


def main_for_class(default_class_id: int) -> int:
    args = parse_args(default_class_id)
    executor = BankTaskExecutor(args, default_class_id)
    return executor.run()
