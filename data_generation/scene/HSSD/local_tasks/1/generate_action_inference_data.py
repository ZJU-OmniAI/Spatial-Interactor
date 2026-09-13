#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSSD 第1类：动作逆推 (Action Inference) 数据生成器。

逻辑对齐 /path/to/workspace/AI2THOR/1/generate_action_inference_data.py，
仅将底层接口替换为 Habitat/HSSD。
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from habitat_sim.agent import ActionSpec as HabitatActionSpec
from habitat_sim.agent.controls import ActuationSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner


MOVE_MAGNITUDES = [0.4, 0.6, 0.8, 1.0]
ROTATE_DEGREES = [30, 45, 60, 75, 90]
LOOK_DEGREES = [30, 45, 60]
START_YAWS = list(range(0, 360, 15))
START_HORIZON_CANDIDATES = [-30, 0, 30, 60]

MIN_SHARED_OBJECTS = 1
MIN_CHAIN_FRAME_DIFF = 6.0

VARIANT_TWO_IMAGE_SINGLE = "two_image_single_action"
VARIANT_TWO_IMAGE_DOUBLE = "two_image_double_action"
VARIANT_MULTI_IMAGE_CHAIN = "multi_image_single_action_chain"
VARIANT_CYCLE = [
    VARIANT_TWO_IMAGE_SINGLE,
    VARIANT_TWO_IMAGE_DOUBLE,
    VARIANT_MULTI_IMAGE_CHAIN,
]

MOVE_APIS = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]
ROTATE_APIS = ["RotateLeft", "RotateRight"]
LOOK_APIS = ["LookUp", "LookDown"]

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
    action_key: str
    action_api: str
    action_params: Dict[str, float]
    action_natural: str
    family: str

    @property
    def action_text(self) -> str:
        return self.action_natural


@dataclass
class StepRecord:
    step_id: int
    action_api: str
    action_params: Dict[str, float]
    action_natural: str
    success: bool
    before_pose: Dict
    after_pose: Dict


def _frame_diff_score(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


class HSSDActionInferenceRunner(HabitatSceneRunner):
    def _build_action_space(self) -> Dict[str, HabitatActionSpec]:
        action_space: Dict[str, HabitatActionSpec] = {}
        for mag in MOVE_MAGNITUDES:
            for action_api, key_prefix in MOVE_KEY_PREFIX.items():
                key = f"{key_prefix}_{mag:.2f}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(mag)),
                )
        for deg in ROTATE_DEGREES:
            for action_api, key_prefix in ROTATE_KEY_PREFIX.items():
                key = f"{key_prefix}_{int(deg)}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(deg)),
                )
        for deg in LOOK_DEGREES:
            for action_api, key_prefix in LOOK_KEY_PREFIX.items():
                key = f"{key_prefix}_{int(deg)}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(deg)),
                )
        return action_space


class HSSDActionInferenceDataGenerator:
    def __init__(
        self,
        output_root: str,
        dataset_root: str,
        image_width: int = 640,
        image_height: int = 480,
        hfov: int = 90,
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

        self.move_magnitudes = list(MOVE_MAGNITUDES)
        self.rotate_degrees = list(ROTATE_DEGREES)
        self.look_degrees = list(LOOK_DEGREES)
        self.variant_cycle = list(VARIANT_CYCLE)

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

        self.runner = HSSDActionInferenceRunner(
            ctx=self.ctx,
            image_width=image_width,
            image_height=image_height,
            hfov=hfov,
            sensor_height=sensor_height,
            agent_radius=agent_radius,
            seed=seed,
        )

        self.current_scene_path: Optional[str] = None
        self.current_scene_objects: List[Dict] = []

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

    @staticmethod
    def _save_frame(np_rgb_frame, save_path: str) -> None:
        Image.fromarray(np_rgb_frame).save(save_path)

    @staticmethod
    def _angle_delta(v1: float, v2: float) -> float:
        return abs((v2 - v1 + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _xy_distance(p1: PoseState, p2: PoseState) -> float:
        return math.hypot(p2.x - p1.x, p2.z - p1.z)

    def _visible_object_ids(self, obs: Dict) -> set:
        return {obj.index for obj in self.runner.visible_objects(self.current_scene_objects, obs["depth"])}

    def _view_richness_score(self, obs: Dict) -> float:
        return self.runner.view_quality_score(self.current_scene_objects, obs, require_unique=True)

    def _pair_has_enough_overlap(self, first_obs: Dict, second_obs: Dict) -> bool:
        first_ids = self._visible_object_ids(first_obs)
        second_ids = self._visible_object_ids(second_obs)
        if not first_ids or not second_ids:
            return False
        return len(first_ids.intersection(second_ids)) >= MIN_SHARED_OBJECTS

    def _saved_views_have_enough_overlap(self, observations: Sequence[Dict]) -> bool:
        if len(observations) < 2:
            return False
        for first_obs, second_obs in zip(observations, observations[1:]):
            if not self._pair_has_enough_overlap(first_obs, second_obs):
                return False
        return True

    def _random_scene(self) -> str:
        return self.rng.choice(self.ctx.scene_paths)

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

    def _make_move_action(self, action_api: str, magnitude: float) -> ActionSpec:
        prefix = {
            "MoveAhead": "向前移动",
            "MoveBack": "向后移动",
            "MoveLeft": "向左平移",
            "MoveRight": "向右平移",
        }[action_api]
        return ActionSpec(
            action_key={
                "MoveAhead": f"move_forward_{magnitude:.2f}",
                "MoveBack": f"move_backward_{magnitude:.2f}",
                "MoveLeft": f"move_left_{magnitude:.2f}",
                "MoveRight": f"move_right_{magnitude:.2f}",
            }[action_api],
            action_api=action_api,
            action_params={"moveMagnitude": float(magnitude)},
            action_natural=f"{prefix}{magnitude:.2f}米",
            family="move",
        )

    def _make_rotate_action(self, action_api: str, degrees: int) -> ActionSpec:
        prefix = {
            "RotateLeft": "向左旋转",
            "RotateRight": "向右旋转",
        }[action_api]
        return ActionSpec(
            action_key={
                "RotateLeft": f"turn_left_{int(degrees)}",
                "RotateRight": f"turn_right_{int(degrees)}",
            }[action_api],
            action_api=action_api,
            action_params={"degrees": float(degrees)},
            action_natural=f"{prefix}{int(degrees)}度",
            family="rotate",
        )

    def _make_look_action(self, action_api: str, degrees: int) -> ActionSpec:
        prefix = {
            "LookUp": "向上抬头",
            "LookDown": "向下低头",
        }[action_api]
        return ActionSpec(
            action_key={
                "LookUp": f"look_up_{int(degrees)}",
                "LookDown": f"look_down_{int(degrees)}",
            }[action_api],
            action_api=action_api,
            action_params={"degrees": float(degrees)},
            action_natural=f"{prefix}{int(degrees)}度",
            family="look",
        )

    def _sample_move_action(self, magnitudes: Optional[Sequence[float]] = None) -> ActionSpec:
        action_api = self.rng.choice(MOVE_APIS)
        magnitude = self.rng.choice(list(magnitudes) if magnitudes else self.move_magnitudes)
        return self._make_move_action(action_api, float(magnitude))

    def _sample_rotate_action(self, degrees_choices: Optional[Sequence[int]] = None) -> ActionSpec:
        action_api = self.rng.choice(ROTATE_APIS)
        degrees = self.rng.choice(list(degrees_choices) if degrees_choices else self.rotate_degrees)
        return self._make_rotate_action(action_api, int(degrees))

    def _sample_look_action(self, degrees_choices: Optional[Sequence[int]] = None) -> ActionSpec:
        action_api = self.rng.choice(LOOK_APIS)
        degrees = self.rng.choice(list(degrees_choices) if degrees_choices else self.look_degrees)
        return self._make_look_action(action_api, int(degrees))

    def _sample_action_spec(
        self,
        families: Optional[Sequence[str]] = None,
        move_magnitudes: Optional[Sequence[float]] = None,
        rotate_degrees: Optional[Sequence[int]] = None,
        look_degrees: Optional[Sequence[int]] = None,
    ) -> ActionSpec:
        allowed_families = list(families) if families else ["move", "rotate", "look"]
        family = self.rng.choice(allowed_families)
        if family == "move":
            return self._sample_move_action(magnitudes=move_magnitudes)
        if family == "rotate":
            return self._sample_rotate_action(degrees_choices=rotate_degrees)
        return self._sample_look_action(degrees_choices=look_degrees)

    @staticmethod
    def _is_exact_duplicate(first: ActionSpec, second: ActionSpec) -> bool:
        return first.action_api == second.action_api and first.action_params == second.action_params

    @staticmethod
    def _is_direct_inverse(first: ActionSpec, second: ActionSpec) -> bool:
        inverse_pairs = {
            ("MoveAhead", "MoveBack"),
            ("MoveBack", "MoveAhead"),
            ("MoveLeft", "MoveRight"),
            ("MoveRight", "MoveLeft"),
            ("RotateLeft", "RotateRight"),
            ("RotateRight", "RotateLeft"),
            ("LookUp", "LookDown"),
            ("LookDown", "LookUp"),
        }
        if (first.action_api, second.action_api) not in inverse_pairs:
            return False
        if first.family == "move":
            return float(first.action_params["moveMagnitude"]) == float(second.action_params["moveMagnitude"])
        return float(first.action_params["degrees"]) == float(second.action_params["degrees"])

    @staticmethod
    def _is_ninety_degree_angle_action(action: ActionSpec) -> bool:
        return action.family in {"rotate", "look"} and float(action.action_params.get("degrees", 0.0)) == 90.0

    def _has_forbidden_consecutive_ninety(self, actions: Sequence[ActionSpec]) -> bool:
        for first, second in zip(actions, actions[1:]):
            if self._is_ninety_degree_angle_action(first) and self._is_ninety_degree_angle_action(second):
                return True
        return False

    def _sample_two_image_single_actions(self) -> List[ActionSpec]:
        actions = [
            self._sample_action_spec(
                move_magnitudes=self.move_magnitudes,
                rotate_degrees=self.rotate_degrees,
                look_degrees=self.look_degrees,
            )
        ]
        if self._has_forbidden_consecutive_ninety(actions):
            raise RuntimeError("两图单动作采样命中了非法的连续90度动作")
        return actions

    def _sample_two_image_double_actions(self) -> List[ActionSpec]:
        large_move_magnitudes = [0.8, 1.0]
        for _ in range(80):
            first = self._sample_action_spec(
                move_magnitudes=large_move_magnitudes,
                rotate_degrees=self.rotate_degrees,
                look_degrees=self.look_degrees,
            )
            second_families = ["move", "rotate", "look"]
            if first.family == "move":
                second_families = ["rotate", "look", "move"]
            elif first.family == "rotate":
                second_families = ["move", "look", "rotate"]
            elif first.family == "look":
                second_families = ["move", "rotate", "look"]

            move_present = first.family == "move"
            second_rotate_choices = [15, 30, 45] if move_present else self.rotate_degrees
            second_look_choices = [15, 30, 45] if move_present else self.look_degrees
            second = self._sample_action_spec(
                families=second_families,
                move_magnitudes=large_move_magnitudes,
                rotate_degrees=second_rotate_choices,
                look_degrees=second_look_choices,
            )
            if second.family == "move":
                move_present = True
            if move_present and first.family in {"rotate", "look"} and float(first.action_params.get("degrees", 0.0)) > 45:
                continue
            if self._is_exact_duplicate(first, second):
                continue
            if self._is_direct_inverse(first, second):
                continue
            if first.family == "move" and second.family == "move":
                continue
            actions = [first, second]
            if self._has_forbidden_consecutive_ninety(actions):
                continue
            return actions
        raise RuntimeError("无法采样出有效的两图双动作组合")

    def _sample_multi_image_chain_actions(self, min_steps: int, max_steps: int) -> List[ActionSpec]:
        step_count = self.rng.randint(min_steps, max_steps)
        actions: List[ActionSpec] = []
        while len(actions) < step_count:
            candidate = self._sample_action_spec()
            if actions and self._is_exact_duplicate(actions[-1], candidate):
                continue
            if actions and self._is_direct_inverse(actions[-1], candidate):
                continue
            if actions and self._is_ninety_degree_angle_action(actions[-1]) and self._is_ninety_degree_angle_action(candidate):
                continue
            actions.append(candidate)
        return actions

    def _sample_actions_for_variant(
        self,
        variant: str,
        min_steps: int,
        max_steps: int,
    ) -> List[ActionSpec]:
        if variant == VARIANT_TWO_IMAGE_SINGLE:
            return self._sample_two_image_single_actions()
        if variant == VARIANT_TWO_IMAGE_DOUBLE:
            return self._sample_two_image_double_actions()
        if variant == VARIANT_MULTI_IMAGE_CHAIN:
            return self._sample_multi_image_chain_actions(min_steps=min_steps, max_steps=max_steps)
        raise ValueError(f"未知样本类型: {variant}")

    def _choose_start_horizon(self, actions: Sequence[ActionSpec]) -> int:
        cumulative_pitch_delta = 0.0
        min_prefix_delta = 0.0
        max_prefix_delta = 0.0

        for action in actions:
            if action.action_api == "LookUp":
                cumulative_pitch_delta -= float(action.action_params["degrees"])
            elif action.action_api == "LookDown":
                cumulative_pitch_delta += float(action.action_params["degrees"])
            min_prefix_delta = min(min_prefix_delta, cumulative_pitch_delta)
            max_prefix_delta = max(max_prefix_delta, cumulative_pitch_delta)

        valid_candidates = [
            horizon
            for horizon in START_HORIZON_CANDIDATES
            if -30.0 <= horizon + min_prefix_delta and horizon + max_prefix_delta <= 60.0
        ]
        if valid_candidates:
            return self.rng.choice(valid_candidates)
        return 0

    def _reset_agent_randomly(self, actions: Sequence[ActionSpec], require_rich_view: bool = False) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        horizon = self._choose_start_horizon(actions)
        candidate_trials = 24 if require_rich_view else 6
        best_obs = None
        best_score = -1.0

        for _ in range(candidate_trials):
            position = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            obs = self.runner.set_agent_state(position=position, yaw=yaw, pitch=horizon)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.is_dark_view(obs):
                continue
            if self.runner.has_excessive_side_walls(obs):
                continue
            score = self._view_richness_score(obs)
            if score > best_score:
                best_score = score
                best_obs = obs
            if not require_rich_view and score >= 8.0:
                return obs
            if require_rich_view and score >= 12.0:
                return obs

        if best_obs is None:
            raise RuntimeError("随机初始化智能体失败")
        return best_obs

    @staticmethod
    def _chain_frames_have_enough_differences(observations: Sequence[Dict]) -> bool:
        if len(observations) < 2:
            return False
        for first_obs, second_obs in zip(observations, observations[1:]):
            if _frame_diff_score(first_obs["rgb"], second_obs["rgb"]) < MIN_CHAIN_FRAME_DIFF:
                return False
        return True

    def _execute_action(self, step_id: int, action: ActionSpec) -> Tuple[Dict, StepRecord]:
        before_pose = self._to_pose_state(self.runner.extract_pose())

        if action.family == "look":
            original_deg = float(action.action_params["degrees"])
            desired_pitch = before_pose.pitch - original_deg if action.action_api == "LookUp" else before_pose.pitch + original_deg
            if desired_pitch < -30.0 or desired_pitch > 60.0:
                for fallback_deg in (45, 30, 15):
                    if fallback_deg >= original_deg:
                        continue
                    candidate_pitch = before_pose.pitch - fallback_deg if action.action_api == "LookUp" else before_pose.pitch + fallback_deg
                    if -30.0 <= candidate_pitch <= 60.0:
                        action = self._make_look_action(action.action_api, fallback_deg)
                        break

        obs = self.runner.execute_action(action)
        obs["pose"] = self._to_pose_state(obs["pose"])
        success = True

        if action.family == "move":
            distance = self._xy_distance(before_pose, obs["pose"])
            if distance < max(0.08, float(action.action_params["moveMagnitude"]) * 0.35):
                original_mag = float(action.action_params["moveMagnitude"])
                success = False
                for fallback_mag in (0.8, 0.6, 0.4, 0.2):
                    if fallback_mag >= original_mag:
                        continue
                    candidate = self._make_move_action(action.action_api, fallback_mag)
                    candidate_obs = self.runner.execute_action(candidate)
                    candidate_obs["pose"] = self._to_pose_state(candidate_obs["pose"])
                    distance = self._xy_distance(before_pose, candidate_obs["pose"])
                    if distance >= max(0.08, fallback_mag * 0.35):
                        action = candidate
                        obs = candidate_obs
                        success = True
                        break
                if not success:
                    raise RuntimeError(f"位移动作变化过小: {action.action_api} {action.action_params}")
        elif action.family == "rotate":
            yaw_delta = self._angle_delta(before_pose.yaw, obs["pose"].yaw)
            if yaw_delta < max(15.0, float(action.action_params["degrees"]) * 0.6):
                raise RuntimeError(f"旋转动作变化过小: {action.action_api} {action.action_params}")
        elif action.family == "look":
            pitch_delta = abs(obs["pose"].pitch - before_pose.pitch)
            if pitch_delta < max(15.0, float(action.action_params["degrees"]) * 0.6):
                raise RuntimeError(f"俯仰动作变化过小: {action.action_api} {action.action_params}")

        return obs, StepRecord(
            step_id=step_id,
            action_api=action.action_api,
            action_params=action.action_params,
            action_natural=action.action_natural,
            success=success,
            before_pose=asdict(before_pose),
            after_pose=asdict(obs["pose"]),
        )

    def _has_meaningful_total_change(self, start_pose: PoseState, end_pose: PoseState) -> bool:
        return (
            self._xy_distance(start_pose, end_pose) >= 0.08
            or self._angle_delta(start_pose.yaw, end_pose.yaw) >= 15.0
            or abs(end_pose.pitch - start_pose.pitch) >= 15.0
        )

    def _format_answer(self, variant: str, actions: Sequence[ActionSpec]) -> str:
        texts = [action.action_natural for action in actions]
        if variant == VARIANT_TWO_IMAGE_SINGLE:
            return texts[0]
        if variant == VARIANT_TWO_IMAGE_DOUBLE:
            return f"先{texts[0]}，再{texts[1]}"
        return "；".join(f"第{idx + 1}步{text}" for idx, text in enumerate(texts))

    def _sample_question(self, variant: str) -> str:
        candidates = {
            VARIANT_TWO_IMAGE_SINGLE: [
                "给定首尾两张图，中间只执行了一个动作。这个动作是什么？",
                "观察图A和图B，相机在中间只做了一次操作，请回答这个单动作。",
            ],
            VARIANT_TWO_IMAGE_DOUBLE: [
                "给定首尾两张图，中间连续执行了两个动作。请按顺序回答这两个动作。",
                "观察图A和图B，相机在两帧之间做了两步连续动作，请依次写出它们。",
            ],
            VARIANT_MULTI_IMAGE_CHAIN: [
                "观察整段图像序列。每相邻两帧之间只执行了一个动作，请按顺序回答所有动作。",
                "下面是一串连续视角变化图像，每一帧到下一帧都只有一个动作。请从前到后写出动作序列。",
            ],
        }
        return self.rng.choice(candidates[variant])

    def _run_one_episode(
        self,
        sample_idx: int,
        variant: str,
        min_steps: int,
        max_steps: int,
        scene_path: str,
    ) -> Dict:
        actions = self._sample_actions_for_variant(
            variant=variant,
            min_steps=min_steps,
            max_steps=max_steps,
        )

        scene_data = self.ctx.load_scene(scene_path)
        self.current_scene_objects = self.ctx.scene_objects(scene_data)
        self.current_scene_path = scene_path
        self.runner.build(scene_path, navmesh_key_path=scene_path)

        start_obs = self._reset_agent_randomly(
            actions,
            require_rich_view=True,
        )

        sample_img_dir = self._prepare_sample_dir(sample_idx)
        start_pose = start_obs["pose"]

        frame_paths: List[str] = []
        saved_observations: List[Dict] = []
        start_frame_path = os.path.join(sample_img_dir, "frame_000_start.png")
        self._save_frame(start_obs["rgb"], start_frame_path)
        frame_paths.append(start_frame_path)
        saved_observations.append(start_obs)

        step_records: List[StepRecord] = []
        save_all_intermediate = variant == VARIANT_MULTI_IMAGE_CHAIN

        current_obs = start_obs
        for step_id, action in enumerate(actions, start=1):
            current_obs, step_record = self._execute_action(step_id=step_id, action=action)
            step_records.append(step_record)

            should_save = save_all_intermediate or step_id == len(actions)
            if should_save:
                frame_path = os.path.join(sample_img_dir, f"frame_{step_id:03d}.png")
                self._save_frame(current_obs["rgb"], frame_path)
                frame_paths.append(frame_path)
                saved_observations.append(current_obs)

        end_pose = current_obs["pose"]
        if not self._has_meaningful_total_change(start_pose, end_pose):
            raise RuntimeError("首尾视角变化过小")
        if variant == VARIANT_MULTI_IMAGE_CHAIN and not self._chain_frames_have_enough_differences(saved_observations):
            raise RuntimeError("多图动作链相邻帧变化不明显")
        if variant != VARIANT_MULTI_IMAGE_CHAIN and not self._saved_views_have_enough_overlap(saved_observations):
            raise RuntimeError("问答帧之间的可见内容重叠不足")

        answer = self._format_answer(variant=variant, actions=actions)
        question = self._sample_question(variant)

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "action_inference",
            "sample_variant": variant,
            "scene": os.path.basename(scene_path),
            "input": {
                "frame_paths": frame_paths,
                "start_frame": frame_paths[0],
                "end_frame": frame_paths[-1],
                "num_frames": len(frame_paths),
            },
            "question": question,
            "answer": answer,
            "gt": {
                "variant": variant,
                "action_count": len(actions),
                "api_actions": [action.action_api for action in actions],
                "action_texts": [action.action_natural for action in actions],
                "steps": [asdict(step) for step in step_records],
                "start_state": {
                    "text": start_pose.to_text(),
                    "raw": asdict(start_pose),
                },
                "end_state": {
                    "text": end_pose.to_text(),
                    "raw": asdict(end_pose),
                },
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_action_inference_data(
        self,
        num_samples: int,
        min_steps: int = 2,
        max_steps: int = 4,
    ) -> Tuple[str, str, int]:
        if min_steps < 2:
            min_steps = 2
        if max_steps < min_steps:
            max_steps = min_steps

        self._reset_output_dirs()

        all_samples: List[Dict] = []
        success_count = 0
        variant_counts = {variant: 0 for variant in self.variant_cycle}

        for sample_idx in range(num_samples):
            variant = self.variant_cycle[sample_idx % len(self.variant_cycle)]
            generated = False

            for attempt_idx in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    sample = self._run_one_episode(
                        sample_idx=sample_idx,
                        variant=variant,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        scene_path=scene_path,
                    )
                    all_samples.append(sample)
                    success_count += 1
                    variant_counts[variant] += 1
                    generated = True
                    print(
                        f"[OK] 已生成样本 {sample_idx + 1}/{num_samples}: "
                        f"{sample['sample_id']} ({variant})"
                    )
                    break
                except Exception as exc:
                    print(
                        f"[WARN] 样本 {sample_idx} 第{attempt_idx}次失败 "
                        f"(variant={variant}, scene={os.path.basename(scene_path)}): {exc}"
                    )
                    traceback.print_exc()
                finally:
                    self.runner.shutdown()

            if not generated:
                print(f"[SKIP] 样本 {sample_idx} 多次重试失败，已跳过")

        qa_jsonl_path = os.path.join(self.meta_dir, "qa_data.jsonl")
        qa_json_path = os.path.join(self.meta_dir, "qa_data.json")
        stats_path = os.path.join(self.meta_dir, "stats.json")

        with open(qa_jsonl_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with open(qa_json_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        stats = {
            "requested_samples": num_samples,
            "generated_samples": success_count,
            "image_root": self.image_dir,
            "jsonl_path": qa_jsonl_path,
            "json_path": qa_json_path,
            "variant_counts": variant_counts,
            "config": {
                "dataset_root": self.dataset_root,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
                "multi_image_min_steps": min_steps,
                "multi_image_max_steps": max_steps,
                "move_magnitudes": self.move_magnitudes,
                "rotate_degrees": self.rotate_degrees,
                "look_degrees": self.look_degrees,
                "min_shared_objects": MIN_SHARED_OBJECTS,
                "variant_cycle": self.variant_cycle,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "hfov": self.hfov,
                "sensor_height": self.sensor_height,
                "agent_radius": self.agent_radius,
            },
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return qa_jsonl_path, qa_json_path, success_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSSD 第1类：动作逆推 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/HSSD/1", help="输出目录")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data", help="Habitat 数据根目录")
    parser.add_argument("--scene_dataset_config", type=str, default=None, help="显式指定 scene_dataset_config")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数")
    parser.add_argument("--min_steps", type=int, default=2, help="多图链样本的最少动作步数")
    parser.add_argument("--max_steps", type=int, default=4, help="多图链样本的最多动作步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_episode_retry", type=int, default=40, help="单样本最大重试次数")
    parser.add_argument("--image_width", type=int, default=640, help="图像宽度")
    parser.add_argument("--image_height", type=int, default=480, help="图像高度")
    parser.add_argument("--hfov", type=int, default=90, help="水平视场角")
    parser.add_argument("--sensor_height", type=float, default=1.6, help="相机高度")
    parser.add_argument("--agent_radius", type=float, default=0.1, help="导航体半径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = HSSDActionInferenceDataGenerator(
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

    qa_jsonl_path, qa_json_path, success_count = generator.generate_action_inference_data(
        num_samples=args.num_samples,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )

    print("\n===== 生成完成 =====")
    print(f"成功生成: {success_count}/{args.num_samples}")
    print(f"JSONL: {qa_jsonl_path}")
    print(f"JSON : {qa_json_path}")


if __name__ == "__main__":
    main()
