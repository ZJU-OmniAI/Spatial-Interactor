#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReplicaCAD 第10类：想象进行运动 (Imagined Movement Consequence)

逻辑尽量对齐 /path/to/workspace/AI2THOR/10/generate_imagined_movement_consequence_data.py，
并参考 /path/to/workspace/HSSD/10/generate_imagined_movement_consequence_data.py，
仅把底层执行替换为 Habitat/ReplicaCAD。
"""

import argparse
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

from habitat_qa_generators import (
    HabitatDatasetContext,
    HabitatSceneRunner,
    _horizontal_location,
    _make_move_action,
    _make_rotate_action,
)


START_YAWS = list(range(0, 360, 15))
START_HORIZONS = [0]
MOVE_MAGNITUDES = [0.6, 0.8, 1.0]
ROTATE_DEGREES = [30, 45, 60, 75, 90]
FAMILY_SCHEDULE = [
    "single_visibility",
    "two_frame_direction",
    "single_direction",
    "trend_disappear",
    "two_frame_direction",
    "single_direction",
    "single_visibility",
    "trend_disappear",
    "two_frame_direction",
    "single_direction",
]
DIRECTION_SCHEDULE = [
    "左前方",
    "右前方",
    "左后方",
    "右后方",
    "正左方",
    "正右方",
    "正前方",
    "正后方",
    "左前方",
    "右后方",
]

EXCLUDED_OBJECT_NAMES = {
    "cabinet",
    "counter",
    "counter top",
    "countertop",
    "curtain",
    "door",
    "drawer",
    "floor",
    "kitchen counter",
    "kitchen cupboard",
    "lamp",
    "light",
    "mirror",
    "plant",
    "rack",
    "rug",
    "wall",
    "window",
}

BAD_NAME_SUBSTRINGS = (
    " part",
    "camera",
    "ceiling",
    " component",
    "handle",
    "knob",
    " module",
    " panel",
    "shade",
    "switch",
)


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


def world_to_local_xz(dx: float, dz: float, yaw_deg: float) -> Tuple[float, float]:
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    lx = c * dx - s * dz
    lz = s * dx + c * dz
    return lx, lz


def direction_from_local(x: float, z: float) -> str:
    ang = math.degrees(math.atan2(x, z))
    bins = [
        (-22.5, 22.5, "正前方"),
        (22.5, 67.5, "右前方"),
        (67.5, 112.5, "正右方"),
        (112.5, 165.0, "右后方"),
        (165.0, 180.1, "正后方"),
        (-180.1, -165.0, "正后方"),
        (-165.0, -112.5, "左后方"),
        (-112.5, -67.5, "正左方"),
        (-75.0, -22.5, "左前方"),
    ]
    for lo, hi, name in bins:
        if lo <= ang < hi:
            return name
    return "正前方"


def direction_center(direction: str) -> float:
    mapping = {
        "正前方": 0.0,
        "右前方": 45.0,
        "正右方": 90.0,
        "右后方": 135.0,
        "正后方": 180.0,
        "左后方": -135.0,
        "正左方": -90.0,
        "左前方": -45.0,
    }
    return mapping[direction]


def angle_distance(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


class ReplicaCADImaginedMovementConsequenceGenerator:
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
        max_episode_retry: int = 40,
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
            dataset="replicacad",
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
                raise RuntimeError(f"未找到任何 ReplicaCAD 场景文件: {config_path}")

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
            for path in sorted(Path(abs_dir).glob("*.scene_instance.json")):
                if path.name == "empty_stage.scene_instance.json":
                    continue
                found.append(str(path))
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
        if any(ch.isdigit() for ch in normalized):
            return False
        if normalized.replace(" ", "").isdigit():
            return False
        alpha_count = sum(ch.isalpha() for ch in normalized)
        if alpha_count < max(2, len(normalized) // 5):
            return False
        return True

    def _good_target_view(self, obj) -> bool:
        x_ratio = obj.pixel_x / float(self.image_width)
        y_ratio = obj.pixel_y / float(self.image_height)
        prominence = self.runner.target_prominence_score(obj)
        return (
            0.14 <= x_ratio <= 0.86
            and 0.12 <= y_ratio <= 0.88
            and obj.distance <= 5.5
            and obj.local_z > 0.15
            and prominence >= 1.7
        )

    def _good_visible_objects(self, scene_objects: Sequence[Dict], obs: Dict) -> List:
        visible = self.runner.unique_visible_objects(scene_objects, obs["depth"])
        objs = []
        for obj in visible:
            if not self._is_meaningful_name(obj.name):
                continue
            if not self._good_target_view(obj):
                continue
            objs.append(obj)
        objs.sort(key=lambda obj: (float(obj.distance), -self.runner.target_prominence_score(obj)))
        return objs

    def _teleport_best_view(self, scene_objects: Sequence[Dict]) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")
        best = None
        best_score = -1e9
        for _ in range(30):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            pitch = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(body_pos, yaw, pitch)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            visible = self._good_visible_objects(scene_objects, obs)
            score = self.runner.view_quality_score(scene_objects, obs, require_unique=True)
            score += 2.8 * len(visible)
            if visible:
                score += self.runner.target_prominence_score(visible[0])
            if score > best_score:
                best_score = score
                best = {"obs": obs, "visible": visible}
            if len(visible) >= 4 and score >= 12.0:
                return {"obs": obs, "visible": visible}
        if best is None or not best["visible"]:
            raise RuntimeError("未找到包含足够目标物体的视角")
        return best

    def _move_step(self):
        return self.rng.choice(
            [
                _make_move_action("MoveAhead", self.rng.choice(MOVE_MAGNITUDES)),
                _make_move_action("MoveBack", self.rng.choice(MOVE_MAGNITUDES)),
                _make_move_action("MoveLeft", self.rng.choice(MOVE_MAGNITUDES)),
                _make_move_action("MoveRight", self.rng.choice(MOVE_MAGNITUDES)),
            ]
        )

    def _turn_step(self):
        return self.rng.choice(
            [
                _make_rotate_action("RotateLeft", self.rng.choice(ROTATE_DEGREES)),
                _make_rotate_action("RotateRight", self.rng.choice(ROTATE_DEGREES)),
            ]
        )

    def _sample_sequence(self, family: str):
        templates = {
            "single_visibility": [["move", "turn"], ["turn", "move"], ["move", "move", "turn"]],
            "single_direction": [["move", "turn"], ["turn", "move"], ["move", "move", "turn"]],
            "two_frame_direction": [["move"], ["turn"], ["move", "turn"]],
            "trend_disappear": [["move"], ["turn"]],
        }
        template = self.rng.choice(templates[family])
        steps = []
        for kind in template:
            steps.append(self._move_step() if kind == "move" else self._turn_step())
        return steps

    def _apply_sequence(self, steps) -> Optional[List[Dict]]:
        events = []
        for action in steps:
            obs = self.runner.execute_action(action)
            obs["pose"] = self._to_pose_state(obs["pose"])
            events.append(obs)
        return events

    @staticmethod
    def _sequence_text(steps) -> str:
        return "，然后".join(action.action_text for action in steps)

    def _direction_to_object(self, pose: PoseState, obj) -> Tuple[str, float]:
        dx = float(obj.position[0]) - pose.x
        dz = float(obj.position[2]) - pose.z
        lx, lz = world_to_local_xz(dx, dz, pose.yaw)
        return direction_from_local(lx, lz), math.degrees(math.atan2(lx, lz))

    def _location_word(self, pose: PoseState, obj) -> str:
        dx = float(obj.position[0]) - pose.x
        dz = float(obj.position[2]) - pose.z
        lx, lz = world_to_local_xz(dx, dz, pose.yaw)
        ang = math.degrees(math.atan2(lx, max(lz, 1e-6)))
        if ang < -15:
            return "左侧"
        if ang > 15:
            return "右侧"
        return "中间"

    @staticmethod
    def _visible_map(visible_objects: Sequence) -> Dict[int, object]:
        return {int(obj.index): obj for obj in visible_objects}

    def _pick_direction_target(
        self,
        scene_objects: Sequence[Dict],
        start_obs: Dict,
        end_obs: Dict,
        preferred_direction: str,
        prefer_hidden_end: bool = False,
    ):
        preferred_center = direction_center(preferred_direction)
        final_visible_map = self._visible_map(self.runner.unique_visible_objects(scene_objects, end_obs["depth"]))
        candidates = []
        for obj in self._good_visible_objects(scene_objects, start_obs):
            final_obj = final_visible_map.get(int(obj.index))
            if final_obj is None:
                continue
            dir_word, ang = self._direction_to_object(end_obs["pose"], final_obj)
            score = (
                0 if dir_word == preferred_direction else 1,
                0 if prefer_hidden_end and final_obj.pixel_x < 0.0 else 1,
                angle_distance(ang, preferred_center),
                float(obj.distance),
            )
            candidates.append((obj, dir_word, score))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[2])
        return candidates[0][0], candidates[0][1]

    def _pick_visibility_target(self, scene_objects: Sequence[Dict], start_obs: Dict, end_obs: Dict):
        final_visible_map = self._visible_map(self.runner.unique_visible_objects(scene_objects, end_obs["depth"]))
        candidates = []
        for obj in self._good_visible_objects(scene_objects, start_obs):
            final_obj = final_visible_map.get(int(obj.index))
            if final_obj is None:
                score = (0, 0, float(obj.distance))
                candidates.append((obj, False, None, score))
                continue
            loc = self._location_word(end_obs["pose"], final_obj)
            score = (1, 0 if loc in {"左侧", "右侧"} else 1, float(obj.distance))
            candidates.append((obj, True, loc, score))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[3])
        return candidates[0][0], candidates[0][1], candidates[0][2]

    def _pick_disappear_pair(self, scene_objects: Sequence[Dict], obs_b: Dict, obs_c: Dict):
        objs_b = self._good_visible_objects(scene_objects, obs_b)
        if len(objs_b) < 2:
            return None
        vis_c_map = self._visible_map(self.runner.unique_visible_objects(scene_objects, obs_c["depth"]))
        candidates = []
        for i in range(len(objs_b)):
            for j in range(i + 1, len(objs_b)):
                obj1, obj2 = objs_b[i], objs_b[j]
                vis1 = int(obj1.index) in vis_c_map
                vis2 = int(obj2.index) in vis_c_map
                if vis1 == vis2:
                    continue
                answer_obj = obj1 if not vis1 else obj2
                keep_obj = obj2 if answer_obj is obj1 else obj1
                candidates.append((answer_obj, keep_obj, float(answer_obj.distance) + float(keep_obj.distance)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[2])
        return candidates[0][0], candidates[0][1]

    def _target_gt(self, obj) -> Dict:
        return {
            "index": int(obj.index),
            "name": obj.name,
            "position": {"x": float(obj.position[0]), "y": float(obj.position[1]), "z": float(obj.position[2])},
            "distance": float(obj.distance),
            "pixel": {"x": float(obj.pixel_x), "y": float(obj.pixel_y)},
        }

    def _reset_pose(self, pose: PoseState) -> None:
        self.runner.set_agent_state(self._body_position_from_pose(pose), pose.yaw, pose.pitch)

    def _build_single_visibility(self, scene_objects: Sequence[Dict], start_obs: Dict, idx: int) -> Optional[Dict]:
        start_pose = start_obs["pose"]
        steps = self._sample_sequence("single_visibility")
        events = self._apply_sequence(steps)
        if not events:
            self._reset_pose(start_pose)
            return None
        end_obs = events[-1]
        picked = self._pick_visibility_target(scene_objects, start_obs, end_obs)
        if picked is None:
            self._reset_pose(start_pose)
            return None
        target, final_visible, loc = picked
        seq_text = self._sequence_text(steps)
        question = (
            f"如果我从当前视角依次{seq_text}，那么【{target.name}】还能看见吗？"
            "如果还能看见，它会更靠左侧、中间还是右侧？"
        )
        if final_visible:
            answer = f"能看见，它会在{loc}。"
        else:
            answer = "不能看见，它会移出视野。"
        self._reset_pose(start_pose)
        return {
            "frame_obs": [("frame_A", start_obs)],
            "question_family": "single_visibility",
            "question": question,
            "answer": answer,
            "gt": {
                "target": self._target_gt(target),
                "action_sequence": [
                    {"api": action.action_api, **action.action_params, "text": action.action_text} for action in steps
                ],
                "action_text": seq_text,
                "start_pose": asdict(start_pose),
                "result_visibility": bool(final_visible),
                "result_location": loc,
            },
        }

    def _build_single_direction(self, scene_objects: Sequence[Dict], start_obs: Dict, idx: int) -> Optional[Dict]:
        start_pose = start_obs["pose"]
        steps = self._sample_sequence("single_direction")
        events = self._apply_sequence(steps)
        if not events:
            self._reset_pose(start_pose)
            return None
        end_obs = events[-1]
        preferred_direction = DIRECTION_SCHEDULE[idx % len(DIRECTION_SCHEDULE)]
        picked = self._pick_direction_target(scene_objects, start_obs, end_obs, preferred_direction)
        if picked is None:
            self._reset_pose(start_pose)
            return None
        target, dir_word = picked
        seq_text = self._sequence_text(steps)
        question = f"如果我从当前视角依次{seq_text}，那么【{target.name}】相对我会在什么方位？"
        self._reset_pose(start_pose)
        return {
            "frame_obs": [("frame_A", start_obs)],
            "question_family": "single_direction",
            "question": question,
            "answer": dir_word,
            "gt": {
                "preferred_direction": preferred_direction,
                "target": self._target_gt(target),
                "action_sequence": [
                    {"api": action.action_api, **action.action_params, "text": action.action_text} for action in steps
                ],
                "action_text": seq_text,
                "start_pose": asdict(start_pose),
                "direction_word": dir_word,
            },
        }

    def _build_two_frame_direction(self, scene_objects: Sequence[Dict], start_obs: Dict, idx: int) -> Optional[Dict]:
        start_pose = start_obs["pose"]
        steps = self._sample_sequence("two_frame_direction")
        events = self._apply_sequence(steps)
        if not events:
            self._reset_pose(start_pose)
            return None
        obs_b = events[-1]
        preferred_direction = DIRECTION_SCHEDULE[(idx + 3) % len(DIRECTION_SCHEDULE)]
        picked = self._pick_direction_target(scene_objects, start_obs, obs_b, preferred_direction)
        if picked is None:
            self._reset_pose(start_pose)
            return None
        target, dir_word = picked
        question = (
            f"图A和图B对应连续移动前后两个视角。"
            f"如果我移动到图B的位置，那么图A里的【{target.name}】相对我会在什么方位？"
        )
        self._reset_pose(start_pose)
        return {
            "frame_obs": [("frame_A", start_obs), ("frame_B", obs_b)],
            "question_family": "two_frame_direction",
            "question": question,
            "answer": dir_word,
            "gt": {
                "preferred_direction": preferred_direction,
                "target": self._target_gt(target),
                "action_sequence": [
                    {"api": action.action_api, **action.action_params, "text": action.action_text} for action in steps
                ],
                "action_text": self._sequence_text(steps),
                "start_pose": asdict(start_pose),
                "direction_word": dir_word,
            },
        }

    def _build_trend_disappear(self, scene_objects: Sequence[Dict], start_obs: Dict, idx: int) -> Optional[Dict]:
        start_pose = start_obs["pose"]
        steps = self._sample_sequence("trend_disappear")
        events_b = self._apply_sequence(steps)
        if not events_b:
            self._reset_pose(start_pose)
            return None
        obs_b = events_b[-1]
        events_c = self._apply_sequence(steps)
        if not events_c:
            self._reset_pose(start_pose)
            return None
        obs_c = events_c[-1]
        picked = self._pick_disappear_pair(scene_objects, obs_b, obs_c)
        if picked is None:
            self._reset_pose(start_pose)
            return None
        answer_obj, keep_obj = picked
        question = (
            "图A到图B体现了同一种运动趋势。"
            f"如果我继续按照这个趋势再移动一步，视野中的【{answer_obj.name}】和【{keep_obj.name}】谁会先移出视野？"
        )
        answer = f"【{answer_obj.name}】会先移出视野。"
        self._reset_pose(start_pose)
        return {
            "frame_obs": [("frame_A", start_obs), ("frame_B", obs_b)],
            "question_family": "trend_disappear",
            "question": question,
            "answer": answer,
            "gt": {
                "answer_object": self._target_gt(answer_obj),
                "comparison_object": self._target_gt(keep_obj),
                "action_sequence": [
                    {"api": action.action_api, **action.action_params, "text": action.action_text} for action in steps
                ],
                "action_text": self._sequence_text(steps),
                "start_pose": asdict(start_pose),
            },
        }

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)
        self.runner.build(scene_path, navmesh_key_path=scene_path)
        start = self._teleport_best_view(scene_objects)
        start_obs = start["obs"]
        family = FAMILY_SCHEDULE[sample_idx % len(FAMILY_SCHEDULE)]
        if family == "single_visibility":
            result = self._build_single_visibility(scene_objects, start_obs, sample_idx)
        elif family == "single_direction":
            result = self._build_single_direction(scene_objects, start_obs, sample_idx)
        elif family == "two_frame_direction":
            result = self._build_two_frame_direction(scene_objects, start_obs, sample_idx)
        else:
            result = self._build_trend_disappear(scene_objects, start_obs, sample_idx)
        if result is None:
            raise RuntimeError("当前视角未找到合适的想象运动样本")

        sample_dir = self._prepare_sample_dir(sample_idx)
        frame_paths = []
        input_obj = {"frame_paths": frame_paths}
        for label, obs in result["frame_obs"]:
            path = os.path.join(sample_dir, f"{label}.png")
            self._save_frame(obs["rgb"], path)
            input_obj[label] = path
            frame_paths.append(path)

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "imagined_movement_consequence",
            "scene": os.path.basename(scene_path),
            "input": input_obj,
            "question": result["question"],
            "answer": result["answer"],
            "gt": {
                "question_family": result["question_family"],
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
                **result["gt"],
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_data(self, num_samples: int):
        self._reset_output_dirs()
        rows: List[Dict] = []
        success = 0
        for i in range(num_samples):
            generated = False
            for retry in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    row = self._run_one_sample(i, scene_path)
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
                "move_magnitudes": MOVE_MAGNITUDES,
                "rotate_degrees": ROTATE_DEGREES,
                "direction_schedule": DIRECTION_SCHEDULE,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
            "question_distribution": dict(Counter(row["gt"]["question_family"] for row in rows)),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return jsonl_path, json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReplicaCAD 第10类：想象进行运动 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/ReplicaCAD/10")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scene_dataset_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_retry", type=int, default=40)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=105)
    parser.add_argument("--sensor_height", type=float, default=1.6)
    parser.add_argument("--agent_radius", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = ReplicaCADImaginedMovementConsequenceGenerator(
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
