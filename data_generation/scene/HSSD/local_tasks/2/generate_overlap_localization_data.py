#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSSD 第2类：多图重叠定位 (Multi-image Overlap Localization) 数据生成器。

逻辑尽量对齐 /path/to/workspace/AI2THOR/2/generate_overlap_localization_data.py，
仅把底层接口替换为 Habitat/HSSD。
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


MOVE_MAGNITUDES = [0.2, 0.4, 0.6, 0.8, 1.0]
ROTATE_DEGREES = [15, 30, 45, 60, 75, 90]
START_YAWS = list(range(0, 360, 15))
START_HORIZONS = [0]

AMBIGUOUS_NAME_PATTERNS = {
    "floor",
    "wall",
    "ceiling",
    "cabinet",
    "drawer",
    "shelf",
    "shelving",
    "counter",
    "countertop",
    "window",
    "door",
    "mirror",
    "room",
    "sink",
    "faucet",
    "stove knob",
    "burner",
    "blind",
    "curtain",
}

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


class HSSDOverlapLocalizationRunner(HabitatSceneRunner):
    def _build_action_space(self) -> Dict[str, HabitatActionSpec]:
        action_space: Dict[str, HabitatActionSpec] = {}
        for mag in MOVE_MAGNITUDES:
            for _, key_prefix in MOVE_KEY_PREFIX.items():
                key = f"{key_prefix}_{mag:.2f}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(mag)),
                )
        for deg in ROTATE_DEGREES:
            for _, key_prefix in ROTATE_KEY_PREFIX.items():
                key = f"{key_prefix}_{int(deg)}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(deg)),
                )
        return action_space


class HSSDOverlapLocalizationDataGenerator:
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
        min_obj_pair_dist: float = 1.0,
        max_obj_pair_dist: float = 3.0,
        overlap_min_shared_objs: int = 1,
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
        self.min_obj_pair_dist = min_obj_pair_dist
        self.max_obj_pair_dist = max_obj_pair_dist
        self.overlap_min_shared_objs = overlap_min_shared_objs

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

        self.runner = HSSDOverlapLocalizationRunner(
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

    def _all_visible_objects(self, obs: Dict) -> List[object]:
        return self.runner.visible_objects(self.current_scene_objects, obs["depth"])

    def _visible_object_ids(self, obs: Dict) -> List[int]:
        return [obj.index for obj in self._all_visible_objects(obs)]

    def _view_richness_score(self, obs: Dict) -> float:
        return self.runner.view_quality_score(self.current_scene_objects, obs, require_unique=True)

    def _teleport_random_pose(self) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best_obs = None
        best_score = -1.0
        for _ in range(28):
            position = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            horizon = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=position, yaw=yaw, pitch=horizon)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            score = self._view_richness_score(obs)
            if score > best_score:
                best_obs = obs
                best_score = score
            if score >= 14.0:
                return obs

        if best_obs is None:
            raise RuntimeError("随机传送失败")
        return best_obs

    def _sample_rotate_action(
        self,
        direction: Optional[str] = None,
        degree_choices: Optional[Sequence[int]] = None,
    ) -> ActionSpec:
        side = direction if direction else self.rng.choice(["Left", "Right"])
        degrees = self.rng.choice(list(degree_choices) if degree_choices else ROTATE_DEGREES)
        action_api = f"Rotate{side}"
        action_natural = f"向{'左' if side == 'Left' else '右'}旋转{int(degrees)}度"
        return ActionSpec(
            action_key=f"{ROTATE_KEY_PREFIX[action_api]}_{int(degrees)}",
            action_api=action_api,
            action_params={"degrees": float(degrees)},
            action_natural=action_natural,
        )

    def _sample_action_sequence(self) -> List[ActionSpec]:
        direction = self.rng.choice(["Left", "Right"])
        first_degrees = self.rng.choice([30, 45, 60])
        second_degrees = self.rng.choice([15, 30, 45])
        return [
            self._sample_rotate_action(direction=direction, degree_choices=[first_degrees]),
            self._sample_rotate_action(direction=direction, degree_choices=[second_degrees]),
        ]

    @staticmethod
    def _rotate_world_to_agent_local(dx: float, dz: float, agent_yaw_deg: float) -> Tuple[float, float]:
        theta = math.radians(agent_yaw_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        local_x = cos_t * dx - sin_t * dz
        local_z = sin_t * dx + cos_t * dz
        return local_x, local_z

    @staticmethod
    def _horizontal_direction_from_local(local_x: float, local_z: float) -> str:
        azim_deg = math.degrees(math.atan2(local_x, local_z))
        bins = [
            (-22.5, 22.5, "正前方"),
            (22.5, 75.0, "右前方"),
            (75.0, 105.0, "正右方"),
            (105.0, 157.5, "右后方"),
            (157.5, 180.0, "正后方"),
            (-180.0, -157.5, "正后方"),
            (-157.5, -105.0, "左后方"),
            (-105.0, -75.0, "正左方"),
            (-75.0, -22.5, "左前方"),
        ]
        for lo, hi, name in bins:
            if lo <= azim_deg < hi:
                return name
        return "正前方"

    def _vector_to_answer_text(
        self,
        target_pos: Tuple[float, float, float],
        anchor_pos: Tuple[float, float, float],
        ref_agent_yaw: float,
    ) -> Tuple[str, Dict]:
        dx = target_pos[0] - anchor_pos[0]
        dy = target_pos[1] - anchor_pos[1]
        dz = target_pos[2] - anchor_pos[2]
        local_x, local_z = self._rotate_world_to_agent_local(dx, dz, ref_agent_yaw)
        horiz = self._horizontal_direction_from_local(local_x, local_z)
        return horiz, {
            "world_vector": {"dx": dx, "dy": dy, "dz": dz},
            "local_vector": {"x": local_x, "z": local_z},
            "horizontal": horiz,
        }

    def _is_good_name(self, name: str) -> bool:
        normalized = " ".join((name or "").lower().split())
        if not normalized or len(normalized) < 2:
            return False
        for pattern in AMBIGUOUS_NAME_PATTERNS:
            if pattern in normalized:
                return False
        return True

    def _point_quality(self, obj) -> Dict[str, float]:
        cx_ratio = float(obj.pixel_x) / float(self.image_width)
        cy_ratio = float(obj.pixel_y) / float(self.image_height)
        edge_margin = min(cx_ratio, cy_ratio, 1.0 - cx_ratio, 1.0 - cy_ratio)
        center_dist = math.hypot(cx_ratio - 0.5, cy_ratio - 0.5)
        return {
            "center_x_ratio": cx_ratio,
            "center_y_ratio": cy_ratio,
            "edge_margin_ratio": edge_margin,
            "center_distance_ratio": center_dist,
        }

    def _visible_name_counts(self, visible_objects: Sequence[object]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in visible_objects:
            counts[obj.name] = counts.get(obj.name, 0) + 1
        return counts

    def _candidate_objects(
        self,
        obs: Dict,
        opposite_visible_ids: Sequence[int],
        middle_visible_ids: Sequence[int],
        require_middle_hidden: bool,
    ) -> List[Dict]:
        visible_objects = self._all_visible_objects(obs)
        name_counts = self._visible_name_counts(visible_objects)
        opposite_set = set(opposite_visible_ids)
        middle_set = set(middle_visible_ids)
        candidates: List[Dict] = []

        for obj in visible_objects:
            if obj.index in opposite_set:
                continue
            if require_middle_hidden and obj.index in middle_set:
                continue
            if not self._is_good_name(obj.name):
                continue
            if name_counts.get(obj.name, 0) != 1:
                continue
            candidates.append(
                {
                    "index": obj.index,
                    "name": obj.name,
                    "template_name": obj.template_name,
                    "position": {"x": obj.position[0], "y": obj.position[1], "z": obj.position[2]},
                    "distance": float(obj.distance),
                    "pixel": {"x": float(obj.pixel_x), "y": float(obj.pixel_y)},
                    "point_quality": self._point_quality(obj),
                }
            )
        candidates.sort(
            key=lambda item: (
                item["point_quality"]["center_distance_ratio"],
                item["distance"],
            )
        )
        return candidates

    @staticmethod
    def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _select_object_pair(self, obs_a: Dict, obs_b: Dict, obs_c: Dict) -> Optional[Dict]:
        ids_a = set(self._visible_object_ids(obs_a))
        ids_b = set(self._visible_object_ids(obs_b))
        ids_c = set(self._visible_object_ids(obs_c))

        for middle_mode in ("both_hidden", "target_hidden", "ignore_b"):
            anchor_candidates = self._candidate_objects(
                obs_a,
                opposite_visible_ids=ids_c,
                middle_visible_ids=ids_b,
                require_middle_hidden=(middle_mode == "both_hidden"),
            )
            target_candidates = self._candidate_objects(
                obs_c,
                opposite_visible_ids=ids_a,
                middle_visible_ids=ids_b,
                require_middle_hidden=(middle_mode != "ignore_b"),
            )

            best_pair = None
            best_score = None
            for target in target_candidates:
                target_pos = (
                    target["position"]["x"],
                    target["position"]["y"],
                    target["position"]["z"],
                )
                for anchor in anchor_candidates:
                    if target["name"] == anchor["name"]:
                        continue
                    anchor_pos = (
                        anchor["position"]["x"],
                        anchor["position"]["y"],
                        anchor["position"]["z"],
                    )
                    pair_distance = self._dist3(target_pos, anchor_pos)
                    if not (self.min_obj_pair_dist <= pair_distance <= self.max_obj_pair_dist):
                        continue

                    quality = target["point_quality"]
                    anchor_in_b = anchor["index"] in ids_b
                    target_in_b = target["index"] in ids_b
                    target_is_centered = (
                        0.20 <= quality["center_x_ratio"] <= 0.80
                        and 0.15 <= quality["center_y_ratio"] <= 0.85
                        and quality["edge_margin_ratio"] >= 0.04
                    )
                    score = (
                        0 if not target_in_b else 1,
                        0 if target_is_centered else 1,
                        0 if middle_mode == "both_hidden" and not anchor_in_b else 1,
                        0 if not anchor_in_b else 1,
                        abs(pair_distance - 1.8),
                        quality["center_distance_ratio"],
                        -target["distance"],
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = {
                            "target_object": target,
                            "anchor_object": anchor,
                            "pair_distance": pair_distance,
                            "middle_mode": middle_mode,
                            "visibility_check": {
                                "A_frame": {
                                    "target_visible": target["index"] in ids_a,
                                    "anchor_visible": anchor["index"] in ids_a,
                                },
                                "B_frame": {
                                    "target_visible": target["index"] in ids_b,
                                    "anchor_visible": anchor["index"] in ids_b,
                                },
                                "C_frame": {
                                    "target_visible": target["index"] in ids_c,
                                    "anchor_visible": anchor["index"] in ids_c,
                                },
                            },
                        }
            if best_pair is not None:
                return best_pair
        return None

    def _run_actions_collect_observations(self, actions: Sequence[ActionSpec]) -> Tuple[List[Dict], List[StepRecord]]:
        observations = [{"rgb": self.runner.get_observations()["rgb"], "depth": self.runner.get_observations()["depth"]}]
        # Replace with current full observation once to avoid repeated sim calls in later logic.
        observations[0] = self.runner.get_observations()
        observations[0]["pose"] = self._to_pose_state(observations[0]["pose"])
        steps: List[StepRecord] = []

        for step_id, action in enumerate(actions, start=1):
            before_pose = self._to_pose_state(self.runner.extract_pose())
            obs = self.runner.execute_action(action)
            obs["pose"] = self._to_pose_state(obs["pose"])
            after_pose = obs["pose"]
            steps.append(
                StepRecord(
                    step_id=step_id,
                    action_api=action.action_api,
                    action_params=action.action_params,
                    action_natural=action.action_natural,
                    success=True,
                    before_pose=asdict(before_pose),
                    after_pose=asdict(after_pose),
                )
            )
            observations.append(obs)

        return observations, steps

    def _check_overlap(self, observations: Sequence[Dict]) -> bool:
        for first_obs, second_obs in zip(observations, observations[1:]):
            s1 = set(self._visible_object_ids(first_obs))
            s2 = set(self._visible_object_ids(second_obs))
            if len(s1.intersection(s2)) < self.overlap_min_shared_objs:
                return False
        return True

    def _build_question(self, target_name: str, anchor_name: str) -> str:
        return (
            f"结合图A、图B、图C的重叠区域，并以图A的视角朝向为参考，"
            f"请判断图C中的【{target_name}】相对于图A中的【{anchor_name}】在图A视角下更接近哪个水平方位？"
        )

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        self.current_scene_objects = self.ctx.scene_objects(scene_data)
        self.current_scene_path = scene_path
        self.runner.build(scene_path, navmesh_key_path=scene_path)

        start_obs = self._teleport_random_pose()
        start_pose = start_obs["pose"]

        actions = None
        steps = None
        observations = None
        pair = None
        for _ in range(10):
            self.runner.set_agent_state(
                position=np.array([start_pose.x, start_pose.y - self.sensor_height, start_pose.z], dtype=np.float32),
                yaw=int(round(start_pose.yaw)) % 360,
                pitch=int(round(start_pose.pitch)),
            )
            candidate_actions = self._sample_action_sequence()
            candidate_observations, candidate_steps = self._run_actions_collect_observations(candidate_actions)
            if len(candidate_observations) != 3:
                continue
            if not self._check_overlap(candidate_observations):
                continue
            obs_a, obs_b, obs_c = candidate_observations
            candidate_pair = self._select_object_pair(obs_a, obs_b, obs_c)
            if candidate_pair is None:
                continue
            actions = candidate_actions
            steps = candidate_steps
            observations = candidate_observations
            pair = candidate_pair
            break

        if actions is None or steps is None or observations is None or pair is None:
            raise RuntimeError("未找到满足互不可见要求的物体对")

        obs_a, obs_b, obs_c = observations
        visibility_check = pair["visibility_check"]
        if visibility_check["A_frame"]["target_visible"]:
            raise RuntimeError("图C目标物体在图A中仍可见")
        if not visibility_check["A_frame"]["anchor_visible"]:
            raise RuntimeError("图A锚点物体在图A中不可见")
        if not visibility_check["C_frame"]["target_visible"]:
            raise RuntimeError("图C目标物体在图C中不可见")
        if visibility_check["C_frame"]["anchor_visible"]:
            raise RuntimeError("图A锚点物体在图C中仍可见")

        pose_a = obs_a["pose"]
        pose_b = obs_b["pose"]
        pose_c = obs_c["pose"]
        target_pos = (
            pair["target_object"]["position"]["x"],
            pair["target_object"]["position"]["y"],
            pair["target_object"]["position"]["z"],
        )
        anchor_pos = (
            pair["anchor_object"]["position"]["x"],
            pair["anchor_object"]["position"]["y"],
            pair["anchor_object"]["position"]["z"],
        )
        answer_text, relation_detail = self._vector_to_answer_text(
            target_pos=target_pos,
            anchor_pos=anchor_pos,
            ref_agent_yaw=pose_a.yaw,
        )

        sample_img_dir = self._prepare_sample_dir(sample_idx)
        frame_names = ["frame_000_A.png", "frame_001_B.png", "frame_002_C.png"]
        frame_paths: List[str] = []
        for frame_name, obs in zip(frame_names, observations):
            frame_path = os.path.join(sample_img_dir, frame_name)
            self._save_frame(obs["rgb"], frame_path)
            frame_paths.append(frame_path)

        question = self._build_question(
            target_name=pair["target_object"]["name"],
            anchor_name=pair["anchor_object"]["name"],
        )

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "multi_image_overlap_localization",
            "scene": os.path.basename(scene_path),
            "input": {
                "frame_paths": frame_paths,
                "frame_A": frame_paths[0],
                "frame_B": frame_paths[1],
                "frame_C": frame_paths[2],
            },
            "question": question,
            "answer": answer_text,
            "gt": {
                "api_actions": [action.action_api for action in actions],
                "steps": [asdict(step) for step in steps],
                "start_state": {"text": pose_a.to_text(), "raw": asdict(pose_a)},
                "mid_state": {"text": pose_b.to_text(), "raw": asdict(pose_b)},
                "end_state": {"text": pose_c.to_text(), "raw": asdict(pose_c)},
                "objects": {
                    "target_object_from_C": pair["target_object"],
                    "anchor_object_from_A": pair["anchor_object"],
                    "pair_distance": pair["pair_distance"],
                },
                "visibility_check": visibility_check,
                "relation": relation_detail,
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_data(self, num_samples: int) -> Tuple[str, str, int]:
        self._reset_output_dirs()
        all_samples: List[Dict] = []
        success = 0

        for sample_idx in range(num_samples):
            generated = False
            for attempt_idx in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    row = self._run_one_sample(sample_idx, scene_path=scene_path)
                    all_samples.append(row)
                    success += 1
                    generated = True
                    print(f"[OK] 已生成样本 {sample_idx + 1}/{num_samples}: {row['sample_id']}")
                    break
                except Exception as exc:
                    print(
                        f"[WARN] 样本 {sample_idx} 第{attempt_idx}次失败 "
                        f"(scene={os.path.basename(scene_path)}): {exc}"
                    )
                    traceback.print_exc()
                finally:
                    self.runner.shutdown()
            if not generated:
                print(f"[SKIP] 样本 {sample_idx} 多次重试失败，已跳过")

        qa_jsonl_path = os.path.join(self.meta_dir, "qa_data.jsonl")
        qa_json_path = os.path.join(self.meta_dir, "qa_data.json")

        with open(qa_jsonl_path, "w", encoding="utf-8") as f:
            for row in all_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(qa_json_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        stats = {
            "requested_samples": num_samples,
            "generated_samples": success,
            "image_root": self.image_dir,
            "jsonl_path": qa_jsonl_path,
            "json_path": qa_json_path,
            "config": {
                "min_obj_pair_dist": self.min_obj_pair_dist,
                "max_obj_pair_dist": self.max_obj_pair_dist,
                "overlap_min_shared_objs": self.overlap_min_shared_objs,
                "strict_cross_visibility": True,
                "prefer_middle_frame_hidden": True,
                "action_mode": "rotation_only",
                "answer_mode": "horizontal_only",
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return qa_jsonl_path, qa_json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSSD 第2类：多图重叠定位 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/HSSD/2", help="输出根目录")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data", help="Habitat 数据根目录")
    parser.add_argument("--scene_dataset_config", type=str, default=None, help="显式指定 scene_dataset_config")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_episode_retry", type=int, default=40, help="单样本最大重试次数")
    parser.add_argument("--min_obj_pair_dist", type=float, default=1.0, help="目标物体对最小距离")
    parser.add_argument("--max_obj_pair_dist", type=float, default=3.0, help="目标物体对最大距离")
    parser.add_argument("--overlap_min_shared_objs", type=int, default=1, help="相邻帧最少共享可见物体数")
    parser.add_argument("--image_width", type=int, default=640, help="图像宽度")
    parser.add_argument("--image_height", type=int, default=480, help="图像高度")
    parser.add_argument("--hfov", type=int, default=90, help="水平视场角")
    parser.add_argument("--sensor_height", type=float, default=1.6, help="相机高度")
    parser.add_argument("--agent_radius", type=float, default=0.1, help="导航体半径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = HSSDOverlapLocalizationDataGenerator(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        image_width=args.image_width,
        image_height=args.image_height,
        hfov=args.hfov,
        sensor_height=args.sensor_height,
        agent_radius=args.agent_radius,
        seed=args.seed,
        max_episode_retry=args.max_episode_retry,
        min_obj_pair_dist=args.min_obj_pair_dist,
        max_obj_pair_dist=args.max_obj_pair_dist,
        overlap_min_shared_objs=args.overlap_min_shared_objs,
        scene_dataset_config=args.scene_dataset_config,
    )

    qa_jsonl_path, qa_json_path, success_count = generator.generate_data(args.num_samples)
    print("\n===== 生成完成 =====")
    print(f"成功生成: {success_count}/{args.num_samples}")
    print(f"JSONL: {qa_jsonl_path}")
    print(f"JSON : {qa_json_path}")


if __name__ == "__main__":
    main()
