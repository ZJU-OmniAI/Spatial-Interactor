#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSSD 第8类：动态移动与遮挡 (Dynamic Movement / Occlusion)

逻辑尽量对齐 /path/to/workspace/AI2THOR/8/generate_dynamic_movement_occlusion_data.py，
仅把底层执行替换为 Habitat/HSSD 的 scene_instance.json 改写。

说明：
1. 保留 AI2THOR 中 distance_change / occlusion_change 两个题族和同样的调度顺序。
2. 视角固定为正常人高度 + 平视，并额外过滤两侧墙面过多、目标不显眼的起始视角。
3. HSSD 不提供 AI2THOR 的 2D instance bbox，这里改用“投影中心 + 深度前后关系 + 显著性变化”
   近似判断遮挡变强/变弱，但题面、输出结构和采样流程保持一致。
"""

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner


START_YAWS = list(range(0, 360, 15))
START_HORIZONS = [0]
FAMILY_SCHEDULE = [
    "distance_change",
    "occlusion_change",
    "distance_change",
    "occlusion_change",
    "distance_change",
    "occlusion_change",
]

MAX_TARGET_DISTANCE = 4.2
MIN_MOVE_DISTANCE = 0.16
MAX_MOVE_DISTANCE = 0.48
MIN_DISTANCE_CHANGE = 0.15
MIN_CENTER_SHIFT = 0.035
MIN_FRAME_DIFF = 0.45
MAX_CANDIDATE_ATTEMPTS = 14
MIN_TARGET_PROMINENCE = 2.6
PSEUDO_OCCLUSION_RADIUS = 0.24

LOCAL_MOVE_OFFSETS = [
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
]

EXCLUDED_OBJECT_NAMES = {
    "accent chest",
    "balcony rail",
    "bar counter",
    "bath tub",
    "bathtub",
    "bed",
    "bench",
    "bin",
    "blind",
    "cabinet",
    "car",
    "chair",
    "computer",
    "coat rack",
    "corner unit",
    "counter",
    "counter top",
    "curtain",
    "desk",
    "dishwasher",
    "door",
    "drawer",
    "fence",
    "floor",
    "floor unit",
    "fridge",
    "garage door",
    "garland",
    "grill",
    "hedge",
    "kitchen",
    "kitchen island",
    "lamp",
    "light",
    "mat",
    "mirror",
    "ottoman",
    "pendant light",
    "piano",
    "plant",
    "planter",
    "rack",
    "rail",
    "range hood",
    "refrigerator",
    "rug",
    "shelf",
    "shower curtain",
    "sideboard",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "storage open",
    "storage unit",
    "table",
    "toilet",
    "tv",
    "tv stand",
    "umbrella stand",
    "wall",
    "wardrobe",
    "washer dryer",
    "washing machine",
    "window",
    "wreath",
}

BAD_NAME_SUBSTRINGS = (
    " part",
    "bay ",
    "for head",
    "numbers clock",
    "time clock",
)


def _frame_diff_score(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def world_to_local_xz(dx: float, dz: float, yaw_deg: float) -> Tuple[float, float]:
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    lx = c * dx - s * dz
    lz = s * dx + c * dz
    return lx, lz


def local_to_world_xz(lx: float, lz: float, yaw_deg: float) -> Tuple[float, float]:
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    dx = c * lx + s * lz
    dz = -s * lx + c * lz
    return dx, dz


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


class HSSDDynamicMovementOcclusionGenerator:
    def __init__(
        self,
        output_root: str,
        dataset_root: str,
        image_width: int = 640,
        image_height: int = 480,
        hfov: int = 105,
        sensor_height: float = 1.6,
        agent_radius: float = 0.1,
        seed: int = 42,
        max_episode_retry: int = 60,
        scene_dataset_config: Optional[str] = None,
    ):
        self.output_root = output_root
        self.image_dir = os.path.join(output_root, "images")
        self.meta_dir = os.path.join(output_root, "meta")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        self.dataset_root = os.path.abspath(dataset_root)
        self.image_width = image_width
        self.image_height = image_height
        self.hfov = hfov
        self.sensor_height = sensor_height
        self.agent_radius = agent_radius
        self.max_episode_retry = max_episode_retry

        self.rng = random.Random(seed)
        self.ctx = HabitatDatasetContext(
            dataset="hssd",
            dataset_root=self.dataset_root,
            image_width=image_width,
            image_height=image_height,
            seed=seed,
        )
        if scene_dataset_config:
            config_path = self._resolve_scene_dataset_config(scene_dataset_config)
            self.ctx.scene_dataset_config = config_path
            self.ctx.scene_paths = self._discover_scene_paths(config_path)
            if not self.ctx.scene_paths:
                raise RuntimeError(f"未找到任何 HSSD 场景文件: {config_path}")

        self.runner = HabitatSceneRunner(
            ctx=self.ctx,
            image_width=image_width,
            image_height=image_height,
            hfov=hfov,
            sensor_height=sensor_height,
            agent_radius=agent_radius,
            seed=seed,
        )

    @staticmethod
    def _resolve_scene_dataset_config(explicit_path: str) -> str:
        config_path = os.path.abspath(explicit_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"scene_dataset_config 不存在: {config_path}")
        return config_path

    @staticmethod
    def _discover_scene_paths(scene_dataset_config: str) -> List[str]:
        config_dir = os.path.dirname(scene_dataset_config)
        with open(scene_dataset_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        found: List[str] = []
        for rel_dir in config.get("scene_instances", {}).get("paths", {}).get(".json", []):
            abs_dir = os.path.join(config_dir, rel_dir)
            if not os.path.isdir(abs_dir):
                continue
            found.extend(str(path) for path in sorted(Path(abs_dir).glob("*.scene_instance.json")))
        return found

    @staticmethod
    def _to_pose_state(pose) -> PoseState:
        return PoseState(
            x=float(pose.x),
            y=float(pose.y),
            z=float(pose.z),
            yaw=float(pose.yaw),
            pitch=float(pose.pitch),
        )

    def _body_position_from_pose(self, pose: PoseState) -> np.ndarray:
        return np.array([pose.x, pose.y - self.sensor_height, pose.z], dtype=np.float32)

    def _reset_output_dirs(self) -> None:
        for path in (self.image_dir, self.meta_dir):
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

    def _prepare_sample_dir(self, sample_idx: int) -> str:
        sample_dir = os.path.join(self.image_dir, f"sample_{sample_idx:05d}")
        if os.path.isdir(sample_dir):
            shutil.rmtree(sample_dir)
        os.makedirs(sample_dir, exist_ok=True)
        return sample_dir

    def _random_scene(self) -> str:
        return self.rng.choice(self.ctx.scene_paths)

    @staticmethod
    def _save_frame(np_rgb_frame, save_path: str) -> None:
        Image.fromarray(np_rgb_frame).save(save_path)

    @staticmethod
    def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _position_to_dict(position: Sequence[float]) -> Dict[str, float]:
        return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}

    @staticmethod
    def _screen_center_distance(a, b) -> float:
        ax, ay = float(a.pixel_x), float(a.pixel_y)
        bx, by = float(b.pixel_x), float(b.pixel_y)
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    @staticmethod
    def _screen_center_distance_ratio(a, b, image_width: int, image_height: int) -> float:
        return HSSDDynamicMovementOcclusionGenerator._screen_center_distance(a, b) / float(
            max(image_width, image_height)
        )

    def _is_meaningful_name(self, name: str) -> bool:
        normalized = (name or "").strip().lower()
        if not normalized or len(normalized) < 2:
            return False
        if normalized in EXCLUDED_OBJECT_NAMES:
            return False
        if normalized == "object":
            return False
        if any(token in normalized for token in BAD_NAME_SUBSTRINGS):
            return False
        if normalized.replace(" ", "").isdigit():
            return False
        alpha_count = sum(ch.isalpha() for ch in normalized)
        if alpha_count < max(2, len(normalized) // 5):
            return False
        return True

    def _good_target_view(self, obj, relaxed: bool = False) -> bool:
        x_ratio = obj.pixel_x / float(self.image_width)
        y_ratio = obj.pixel_y / float(self.image_height)
        prominence = self.runner.target_prominence_score(obj)
        if relaxed:
            return (
                0.14 <= x_ratio <= 0.86
                and 0.12 <= y_ratio <= 0.88
                and obj.distance <= MAX_TARGET_DISTANCE + 0.4
                and obj.local_z > 0.25
                and prominence >= 1.9
            )
        return (
            0.20 <= x_ratio <= 0.80
            and 0.16 <= y_ratio <= 0.84
            and obj.distance <= MAX_TARGET_DISTANCE
            and obj.local_z > 0.35
            and prominence >= MIN_TARGET_PROMINENCE
        )

    def _visible_object_map_by_index(self, visible_objects: Sequence) -> Dict[int, object]:
        return {int(obj.index): obj for obj in visible_objects}

    def _candidate_movers(self, scene_objects: Sequence[Dict], obs: Dict) -> List:
        visible = self.runner.visible_objects(scene_objects, obs["depth"])
        candidates = []
        for obj in visible:
            scene_obj = scene_objects[obj.index]
            if not self._is_meaningful_name(obj.name):
                continue
            if not self._good_target_view(obj):
                continue
            if float(obj.distance) > MAX_TARGET_DISTANCE:
                continue
            y = float(scene_obj["position"][1])
            if y < 0.05 or y > 2.30:
                continue
            if scene_obj.get("motion_type", "STATIC") == "KINEMATIC":
                continue
            candidates.append(obj)
        candidates.sort(
            key=lambda obj: (
                -self.runner.target_prominence_score(obj),
                float(obj.distance),
            )
        )
        return candidates

    def _candidate_targets(self, scene_objects: Sequence[Dict], obs: Dict, exclude_indexes: Optional[set] = None) -> List:
        exclude_indexes = exclude_indexes or set()
        visible = self.runner.unique_visible_objects(scene_objects, obs["depth"])
        candidates = []
        for obj in visible:
            if int(obj.index) in exclude_indexes:
                continue
            if not self._is_meaningful_name(obj.name):
                continue
            if not self._good_target_view(obj, relaxed=True):
                continue
            candidates.append(obj)
        candidates.sort(
            key=lambda obj: (
                -self.runner.target_prominence_score(obj),
                float(obj.distance),
            )
        )
        return candidates

    def _teleport_best_view(self, scene_objects: Sequence[Dict]) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best = None
        best_score = -1e9
        for _ in range(48):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            pitch = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=pitch)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue

            movers = self._candidate_movers(scene_objects, obs)
            targets = self._candidate_targets(scene_objects, obs)
            score = self.runner.view_quality_score(scene_objects, obs, require_unique=True)
            score += 4.5 * len(movers)
            score += 2.0 * len(targets)
            if movers:
                score += 1.5 * self.runner.target_prominence_score(movers[0])
            if targets:
                score += 0.8 * self.runner.target_prominence_score(targets[0])

            candidate = {"obs": obs, "movers": movers, "targets": targets}
            if score > best_score:
                best_score = score
                best = candidate
            if len(movers) >= 2 and len(targets) >= 4 and score >= 22.0:
                return candidate

        if best is None or not best["movers"]:
            raise RuntimeError("未找到包含合适可移动物体的视角")
        return best

    @staticmethod
    def _move_text(local_dx: float, local_dz: float) -> str:
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

    def _object_snapshot(self, visible_obj, scene_obj: Optional[Dict]) -> Dict:
        if scene_obj is None:
            return {"exists": False}
        payload = {
            "exists": True,
            "index": int(scene_obj["index"]),
            "template_name": scene_obj["template_name"],
            "name": scene_obj["name"],
            "position": self._position_to_dict(scene_obj["position"]),
            "rotation": [float(v) for v in scene_obj["rotation"]],
            "motion_type": scene_obj.get("motion_type", "STATIC"),
        }
        if visible_obj is not None:
            payload.update(
                {
                    "visible": True,
                    "distance": float(visible_obj.distance),
                    "pixel": {"x": float(visible_obj.pixel_x), "y": float(visible_obj.pixel_y)},
                    "local_position": {"x": float(visible_obj.local_x), "z": float(visible_obj.local_z)},
                    "prominence": float(self.runner.target_prominence_score(visible_obj)),
                }
            )
        else:
            payload["visible"] = False
        return payload

    def _make_scene_copy_move_object(self, scene_data: Dict, target_index: int, offset_x: float, offset_z: float) -> Dict:
        new_scene = copy.deepcopy(scene_data)
        target = new_scene["object_instances"][target_index]
        target["translation"][0] = float(target["translation"][0]) + float(offset_x)
        target["translation"][2] = float(target["translation"][2]) + float(offset_z)
        return new_scene

    def _distance_offset_priority(self, mover) -> List[Tuple[float, float]]:
        ranked = []
        current_dist = math.hypot(float(mover.local_x), float(mover.local_z))
        for local_dx, local_dz in LOCAL_MOVE_OFFSETS:
            next_dist = math.hypot(float(mover.local_x) + float(local_dx), float(mover.local_z) + float(local_dz))
            predicted_delta = next_dist - current_dist
            ranked.append(
                (
                    -abs(predicted_delta),
                    0 if abs(local_dz) >= abs(local_dx) else 1,
                    abs(abs(local_dz) - 0.30),
                    abs(local_dx),
                    (local_dx, local_dz),
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        return [item[-1] for item in ranked]

    def _nearest_neighbor_distance(self, position: Sequence[float], scene_objects: Sequence[Dict], exclude_index: int) -> float:
        best = 1e9
        for obj in scene_objects:
            if int(obj["index"]) == int(exclude_index):
                continue
            d = self._dist3(position, obj["position"])
            if d < best:
                best = d
        return best

    def _attempt_move(
        self,
        sample_idx: int,
        scene_path: str,
        scene_data: Dict,
        scene_objects: Sequence[Dict],
        start_obs: Dict,
        mover_before,
        local_offset: Tuple[float, float],
    ) -> Optional[Dict]:
        start_pose = start_obs["pose"]
        world_dx, world_dz = local_to_world_xz(local_offset[0], local_offset[1], start_pose.yaw)
        scene_obj_before = scene_objects[mover_before.index]
        target_position = [
            float(scene_obj_before["position"][0]) + float(world_dx),
            float(scene_obj_before["position"][1]),
            float(scene_obj_before["position"][2]) + float(world_dz),
        ]
        move_distance = self._dist3(target_position, scene_obj_before["position"])
        if not (MIN_MOVE_DISTANCE <= move_distance <= MAX_MOVE_DISTANCE):
            return None

        nearest_after = self._nearest_neighbor_distance(target_position, scene_objects, mover_before.index)
        if nearest_after < 0.02:
            return None

        modified_scene = self._make_scene_copy_move_object(scene_data, mover_before.index, world_dx, world_dz)
        modified_scene_path = self.ctx.save_temp_scene(modified_scene, scene_path, task_id=8, sample_idx=sample_idx)
        modified_scene_objects = self.ctx.scene_objects(modified_scene)

        self.runner.build(modified_scene_path, navmesh_key_path=scene_path)
        after_obs = self.runner.set_agent_state(
            position=self._body_position_from_pose(start_pose),
            yaw=start_pose.yaw,
            pitch=start_pose.pitch,
        )
        after_obs["pose"] = self._to_pose_state(after_obs["pose"])

        after_visible_all = self._visible_object_map_by_index(
            self.runner.visible_objects(modified_scene_objects, after_obs["depth"])
        )
        mover_after = after_visible_all.get(int(mover_before.index))
        if mover_after is None:
            return None

        shift = self._screen_center_distance_ratio(mover_before, mover_after, self.image_width, self.image_height)
        if shift < MIN_CENTER_SHIFT:
            return None

        frame_diff = _frame_diff_score(start_obs["rgb"], after_obs["rgb"])
        if frame_diff < MIN_FRAME_DIFF:
            return None

        return {
            "event_a": start_obs,
            "event_b": after_obs,
            "mover_before": mover_before,
            "mover_after": mover_after,
            "scene_obj_before": scene_obj_before,
            "scene_obj_after": modified_scene_objects[mover_before.index],
            "scene_objects_after": modified_scene_objects,
            "modified_scene_path": modified_scene_path,
            "target_position": self._position_to_dict(target_position),
            "move_distance": float(move_distance),
            "local_offset": {"x": float(local_offset[0]), "z": float(local_offset[1])},
            "move_text": self._move_text(local_offset[0], local_offset[1]),
            "action_sequence": [
                {
                    "api": "ModifySceneInstanceTranslation",
                    "object_index": int(mover_before.index),
                    "world_offset": {"x": float(world_dx), "z": float(world_dz)},
                    "target_position": self._position_to_dict(target_position),
                    "purpose": "move_object",
                }
            ],
            "screen_center_shift": float(shift),
            "frame_diff": float(frame_diff),
        }

    def _evaluate_distance_change(self, attempt: Dict) -> Optional[Dict]:
        mover_before = attempt["mover_before"]
        mover_after = attempt["mover_after"]
        dist_before = float(mover_before.distance)
        dist_after = float(mover_after.distance)
        delta = dist_after - dist_before
        if abs(delta) < MIN_DISTANCE_CHANGE:
            return None
        relation = "更近" if delta < 0 else "更远"
        return {
            "relation": relation,
            "distance_before": dist_before,
            "distance_after": dist_after,
            "distance_delta": delta,
        }

    def _pseudo_occlusion_score(self, mover, target) -> float:
        if mover is None or target is None:
            return 0.0
        sep = self._screen_center_distance_ratio(mover, target, self.image_width, self.image_height)
        closeness = max(0.0, 1.0 - sep / PSEUDO_OCCLUSION_RADIUS)
        if mover.distance + 0.10 < target.distance:
            frontness = 1.0
        elif mover.distance < target.distance + 0.10:
            frontness = 0.45
        else:
            frontness = 0.12
        return float(closeness * frontness)

    def _best_occlusion_target(self, scene_objects: Sequence[Dict], attempt: Dict) -> Optional[Dict]:
        before_visible_unique = self._candidate_targets(scene_objects, attempt["event_a"], exclude_indexes={int(attempt["mover_before"].index)})
        after_visible_all = self._visible_object_map_by_index(
            self.runner.visible_objects(attempt["scene_objects_after"], attempt["event_b"]["depth"])
        )

        best = None
        for target_before in before_visible_unique:
            if int(target_before.index) == int(attempt["mover_before"].index):
                continue
            target_after = after_visible_all.get(int(target_before.index))
            prominence_before = float(self.runner.target_prominence_score(target_before))
            prominence_after = float(self.runner.target_prominence_score(target_after)) if target_after is not None else 0.0
            occlusion_before = self._pseudo_occlusion_score(attempt["mover_before"], target_before)
            occlusion_after = self._pseudo_occlusion_score(attempt["mover_after"], target_after)
            visible_after = target_after is not None and self._good_target_view(target_after, relaxed=True)
            if target_after is not None and not visible_after:
                target_after = None

            harder_strength = 0.0
            easier_strength = 0.0
            if not visible_after:
                harder_strength = 1.25 + max(0.0, occlusion_after - occlusion_before)
            else:
                harder_strength = (
                    max(0.0, occlusion_after - occlusion_before) * 1.8
                    + max(0.0, prominence_before - prominence_after) * 0.22
                )
                easier_strength = (
                    max(0.0, occlusion_before - occlusion_after) * 1.8
                    + max(0.0, prominence_after - prominence_before) * 0.22
                )

            change_type = None
            strength = 0.0
            if harder_strength >= 0.42 and (
                not visible_after
                or (occlusion_after > occlusion_before + 0.12 and prominence_after < prominence_before - 0.6)
            ):
                change_type = "harder"
                strength = harder_strength
            elif easier_strength >= 0.36 and (
                occlusion_before > occlusion_after + 0.12 and prominence_after > prominence_before + 0.5
            ):
                change_type = "easier"
                strength = easier_strength

            if change_type is None:
                continue

            candidate = {
                "target_before": target_before,
                "target_after": target_after,
                "change_type": change_type,
                "strength": float(strength),
                "visible_after": bool(visible_after),
                "prominence_before": prominence_before,
                "prominence_after": prominence_after,
                "occlusion_before": occlusion_before,
                "occlusion_after": occlusion_after,
            }
            if best is None or candidate["strength"] > best["strength"]:
                best = candidate
        return best

    def _build_distance_sample(
        self,
        sample_idx: int,
        scene_path: str,
        scene_data: Dict,
        scene_objects: Sequence[Dict],
        start: Dict,
    ) -> Optional[Dict]:
        movers = start["movers"]
        if not movers:
            return None

        for mover in movers[: min(MAX_CANDIDATE_ATTEMPTS, 2)]:
            prioritized = self._distance_offset_priority(mover)[:2]
            for offset in prioritized:
                attempt = self._attempt_move(sample_idx, scene_path, scene_data, scene_objects, start["obs"], mover, offset)
                if attempt is None:
                    continue
                change = self._evaluate_distance_change(attempt)
                if change is None:
                    continue
                return {
                    "frame_obs": [("frame_A", attempt["event_a"]), ("frame_B", attempt["event_b"])],
                    "question_family": "distance_change",
                    "question": "图A到图B中，发生移动的那个物体相对观察者是更近了还是更远了？",
                    "answer": change["relation"],
                    "gt": {
                        "mover": {
                            "before": self._object_snapshot(attempt["mover_before"], attempt["scene_obj_before"]),
                            "after": self._object_snapshot(attempt["mover_after"], attempt["scene_obj_after"]),
                        },
                        "action_sequence": attempt["action_sequence"],
                        "move_text": attempt["move_text"],
                        "local_offset": attempt["local_offset"],
                        "target_position": attempt["target_position"],
                        "screen_center_shift": attempt["screen_center_shift"],
                        "frame_diff": attempt["frame_diff"],
                        "modified_scene_path": attempt["modified_scene_path"],
                        **change,
                    },
                }
        return None

    def _build_occlusion_sample(
        self,
        sample_idx: int,
        scene_path: str,
        scene_data: Dict,
        scene_objects: Sequence[Dict],
        start: Dict,
    ) -> Optional[Dict]:
        movers = start["movers"]
        if not movers:
            return None

        offsets = list(LOCAL_MOVE_OFFSETS)
        self.rng.shuffle(offsets)
        prioritized = sorted(
            offsets,
            key=lambda off: (
                0 if abs(off[0]) >= abs(off[1]) else 1,
                abs(abs(off[0]) - 0.30),
                abs(off[1]),
            ),
        )
        for mover in movers[:MAX_CANDIDATE_ATTEMPTS]:
            for offset in prioritized:
                attempt = self._attempt_move(sample_idx, scene_path, scene_data, scene_objects, start["obs"], mover, offset)
                if attempt is None:
                    continue
                change = self._best_occlusion_target(scene_objects, attempt)
                if change is None:
                    continue
                target_name = change["target_before"].name
                answer = "更不容易看见了" if change["change_type"] == "harder" else "更容易看见了"
                target_after_scene = (
                    attempt["scene_objects_after"][change["target_before"].index]
                    if change["target_before"] is not None
                    else None
                )
                return {
                    "frame_obs": [("frame_A", attempt["event_a"]), ("frame_B", attempt["event_b"])],
                    "question_family": "occlusion_change",
                    "question": (
                        "图A到图B中，一个物体发生了移动。"
                        f"它的移动让【{target_name}】变得更容易看见还是更不容易看见？"
                    ),
                    "answer": answer,
                    "gt": {
                        "mover": {
                            "before": self._object_snapshot(attempt["mover_before"], attempt["scene_obj_before"]),
                            "after": self._object_snapshot(attempt["mover_after"], attempt["scene_obj_after"]),
                        },
                        "affected_object": {
                            "before": self._object_snapshot(
                                change["target_before"],
                                scene_objects[change["target_before"].index],
                            ),
                            "after": self._object_snapshot(change["target_after"], target_after_scene),
                        },
                        "action_sequence": attempt["action_sequence"],
                        "move_text": attempt["move_text"],
                        "local_offset": attempt["local_offset"],
                        "target_position": attempt["target_position"],
                        "screen_center_shift": attempt["screen_center_shift"],
                        "frame_diff": attempt["frame_diff"],
                        "modified_scene_path": attempt["modified_scene_path"],
                        "change_type": change["change_type"],
                        "visible_after": change["visible_after"],
                        "prominence_before": change["prominence_before"],
                        "prominence_after": change["prominence_after"],
                        "occlusion_before": change["occlusion_before"],
                        "occlusion_after": change["occlusion_after"],
                    },
                }
        return None

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)

        self.runner.build(scene_path, navmesh_key_path=scene_path)
        start = self._teleport_best_view(scene_objects)
        family = FAMILY_SCHEDULE[sample_idx % len(FAMILY_SCHEDULE)]

        if family == "distance_change":
            result = self._build_distance_sample(sample_idx, scene_path, scene_data, scene_objects, start)
            if result is None:
                result = self._build_occlusion_sample(sample_idx, scene_path, scene_data, scene_objects, start)
        else:
            result = self._build_occlusion_sample(sample_idx, scene_path, scene_data, scene_objects, start)
            if result is None:
                result = self._build_distance_sample(sample_idx, scene_path, scene_data, scene_objects, start)
        if result is None:
            raise RuntimeError("当前视角未找到合适的动态移动样本")

        sample_dir = self._prepare_sample_dir(sample_idx)
        frame_paths = []
        input_obj = {"frame_paths": frame_paths}
        for label, obs in result["frame_obs"]:
            path = os.path.join(sample_dir, f"{label}.png")
            self._save_frame(obs["rgb"], path)
            input_obj[label] = path
            frame_paths.append(path)

        start_pose = start["obs"]["pose"]
        end_pose = result["frame_obs"][-1][1]["pose"]
        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "dynamic_movement_occlusion",
            "scene": os.path.basename(scene_path),
            "input": input_obj,
            "question": result["question"],
            "answer": result["answer"],
            "gt": {
                "question_family": result["question_family"],
                "start_state": {"text": start_pose.to_text(), "raw": asdict(start_pose)},
                "end_state": {"text": end_pose.to_text(), "raw": asdict(end_pose)},
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
                **result["gt"],
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_data(self, num_samples: int) -> Tuple[str, str, int]:
        self._reset_output_dirs()

        rows: List[Dict] = []
        success = 0
        for i in range(num_samples):
            generated = False
            for retry in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    row = self._run_one_sample(i, scene_path=scene_path)
                    rows.append(row)
                    success += 1
                    generated = True
                    print(f"[OK] 已生成样本 {success}/{num_samples}: {row['sample_id']}")
                    break
                except Exception as exc:
                    print(f"[WARN] 样本 {i} 第{retry}次失败(scene={os.path.basename(scene_path)}): {exc}")
                    traceback.print_exc()
                finally:
                    self.runner.shutdown()
            if not generated:
                print(f"[SKIP] 样本 {i} 达到最大重试次数，已跳过")

        jsonl_path = os.path.join(self.meta_dir, "qa_data.jsonl")
        json_path = os.path.join(self.meta_dir, "qa_data.json")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        stats = {
            "requested_samples": num_samples,
            "generated_samples": success,
            "jsonl_path": jsonl_path,
            "json_path": json_path,
            "image_root": self.image_dir,
            "config": {
                "field_of_view": self.hfov,
                "sensor_height": self.sensor_height,
                "family_schedule": FAMILY_SCHEDULE,
                "max_target_distance": MAX_TARGET_DISTANCE,
                "min_move_distance": MIN_MOVE_DISTANCE,
                "max_move_distance": MAX_MOVE_DISTANCE,
                "min_distance_change": MIN_DISTANCE_CHANGE,
                "min_center_shift": MIN_CENTER_SHIFT,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
            "question_distribution": dict(Counter(row["gt"]["question_family"] for row in rows)),
            "scene_distribution": dict(Counter(row["scene"] for row in rows)),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return jsonl_path, json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSSD 第8类：动态移动与遮挡 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/HSSD/8")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scene_dataset_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_retry", type=int, default=60)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=105)
    parser.add_argument("--sensor_height", type=float, default=1.6)
    parser.add_argument("--agent_radius", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = HSSDDynamicMovementOcclusionGenerator(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        image_width=args.image_width,
        image_height=args.image_height,
        hfov=args.hfov,
        sensor_height=args.sensor_height,
        agent_radius=args.agent_radius,
        seed=args.seed,
        max_episode_retry=args.max_episode_retry,
        scene_dataset_config=args.scene_dataset_config,
    )
    jsonl_path, json_path, success = gen.generate_data(args.num_samples)
    print("\n===== 生成完成 =====")
    print(f"成功生成: {success}/{args.num_samples}")
    print(f"JSONL: {jsonl_path}")
    print(f"JSON : {json_path}")


if __name__ == "__main__":
    main()
