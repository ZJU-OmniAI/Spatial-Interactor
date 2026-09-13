#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSSD 第4类：运动过程排序 (Movement Sequence Sorting) 数据生成器。

逻辑对齐 /path/to/workspace/AI2THOR/4/generate_movement_sequence_sorting_data.py，
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


ROTATE_DEGREES = [15, 30, 45, 60, 75, 90]
MOVE_MAGNITUDES = [0.5, 0.75, 1.0]
FALLBACK_MOVE_MAGNITUDES = [0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
LABELS = ["A", "B", "C", "D", "E", "F"]
START_YAWS = list(range(0, 360, 15))
START_HORIZONS = [0]

EXCLUDED_TARGET_NAMES = {
    "blind",
    "burner",
    "cabinet",
    "ceiling",
    "counter",
    "curtain",
    "door",
    "drawer",
    "floor",
    "mirror",
    "room",
    "shelf",
    "sink",
    "wall",
    "window",
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
    action_args: Dict[str, float]
    action_text: str

    @property
    def action_params(self) -> Dict[str, float]:
        return self.action_args


@dataclass
class StepRecord:
    step_id: int
    action_api: str
    action_args: Dict
    success: bool
    before_pose: Dict
    after_pose: Dict


class HSSDMovementSequenceRunner(HabitatSceneRunner):
    def _build_action_space(self) -> Dict[str, HabitatActionSpec]:
        action_space: Dict[str, HabitatActionSpec] = {}
        for mag in FALLBACK_MOVE_MAGNITUDES:
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


class HSSDMovementSequenceSortingGenerator:
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
        max_episode_retry: int = 50,
        sequence_len: int = 4,
        min_target_dist: float = 1.9,
        max_target_dist: float = 6.0,
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
        self.sequence_len = sequence_len
        self.min_target_dist = min_target_dist
        self.max_target_dist = max_target_dist

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

        self.runner = HSSDMovementSequenceRunner(
            ctx=self.ctx,
            image_width=image_width,
            image_height=image_height,
            hfov=hfov,
            sensor_height=sensor_height,
            agent_radius=agent_radius,
            seed=seed,
        )
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
    def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _rotate_world_to_agent_local(dx: float, dz: float, agent_yaw_deg: float) -> Tuple[float, float]:
        theta = math.radians(agent_yaw_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        local_x = cos_t * dx - sin_t * dz
        local_z = sin_t * dx + cos_t * dz
        return local_x, local_z

    def _body_position_from_pose(self, pose: PoseState) -> np.ndarray:
        return np.array([pose.x, pose.y - self.sensor_height, pose.z], dtype=np.float32)

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

    def _visible_name_counts(self, obs: Dict) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self._all_visible_objects(obs):
            counts[obj.name] = counts.get(obj.name, 0) + 1
        return counts

    def _target_local_vector(self, pose: PoseState, target_pos: Tuple[float, float, float]) -> Tuple[float, float]:
        dx = target_pos[0] - pose.x
        dz = target_pos[2] - pose.z
        return self._rotate_world_to_agent_local(dx, dz, pose.yaw)

    def _candidate_targets(self, obs: Dict) -> List[object]:
        pose = obs["pose"]
        name_counts = self._visible_name_counts(obs)
        candidates: List[Tuple[float, object]] = []

        for obj in self._all_visible_objects(obs):
            if obj.name in EXCLUDED_TARGET_NAMES:
                continue
            if name_counts.get(obj.name, 0) != 1:
                continue
            distance = float(obj.distance)
            if not (self.min_target_dist <= distance <= self.max_target_dist):
                continue
            local_x, local_z = self._target_local_vector(pose, obj.position)
            if local_z <= 0.4:
                continue
            abs_angle = abs(math.degrees(math.atan2(local_x, local_z)))
            prominence = self.runner.target_prominence_score(obj)
            score = prominence - 0.15 * distance - 0.01 * min(abs_angle, 90.0) - 0.05 * abs(local_x)
            candidates.append((score, obj))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [obj for _, obj in candidates]

    def _teleport_random_pose(self) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best_candidate = None
        best_score = -1.0
        for _ in range(30):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            horizon = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=horizon)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            candidates = self._candidate_targets(obs)
            score = 0.0
            if candidates:
                top_candidates = candidates[: min(3, len(candidates))]
                avg_top_dist = sum(float(obj.distance) for obj in top_candidates) / len(top_candidates)
                ahead_bonus = 0.0
                try:
                    ahead_obs = self.runner.execute_action(
                        ActionSpec(
                            action_key="move_forward_0.50",
                            action_api="MoveAhead",
                            action_args={"moveMagnitude": 0.5},
                            action_text="向前移动0.50米",
                        )
                    )
                    ahead_pose = self._to_pose_state(ahead_obs["pose"])
                    moved = math.hypot(ahead_pose.x - obs["pose"].x, ahead_pose.z - obs["pose"].z)
                    ahead_bonus = 2.0 if moved >= 0.15 else 0.0
                except Exception:
                    ahead_bonus = 0.0
                obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=horizon)
                obs["pose"] = self._to_pose_state(obs["pose"])
                score = len(candidates) * 10.0 + avg_top_dist + ahead_bonus
            view_score = self.runner.view_quality_score(self.current_scene_objects, obs, require_unique=True)
            score += view_score
            if score > best_score:
                best_score = score
                best_candidate = (body_pos, yaw, horizon)

        if best_candidate is None:
            raise RuntimeError("随机传送失败")

        body_pos, yaw, horizon = best_candidate
        obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=horizon)
        obs["pose"] = self._to_pose_state(obs["pose"])
        return obs

    def _choose_targets(self, obs: Dict) -> List[object]:
        candidates = self._candidate_targets(obs)
        if not candidates:
            return []
        return candidates[: min(8, len(candidates))]

    def _make_rotate_action(self, direction: str, degrees: int) -> ActionSpec:
        action_api = f"Rotate{direction}"
        action_text = f"向{'左' if direction == 'Left' else '右'}旋转{degrees}度"
        return ActionSpec(
            action_key=f"{ROTATE_KEY_PREFIX[action_api]}_{degrees}",
            action_api=action_api,
            action_args={"degrees": float(degrees)},
            action_text=action_text,
        )

    @staticmethod
    def _direction_from_local_x(local_x: float) -> str:
        return "Right" if local_x > 0 else "Left"

    def _build_route_actions(self, start_pose: PoseState, target_pos: Tuple[float, float, float]) -> Tuple[List[ActionSpec], str]:
        local_x, local_z = self._target_local_vector(start_pose, target_pos)
        azim_deg = math.degrees(math.atan2(local_x, local_z))
        abs_angle = abs(azim_deg)
        target_distance = self._dist3((start_pose.x, start_pose.y, start_pose.z), target_pos)
        turn_direction = self._direction_from_local_x(
            local_x if abs(local_x) > 1e-6 else (1.0 if self.rng.random() > 0.5 else -1.0)
        )
        long_stride = 0.75 if target_distance >= 2.8 else 0.5
        medium_stride = 0.75 if target_distance >= 3.2 else 0.5
        short_stride = 0.5

        if abs_angle >= 18:
            turn_deg = min(90, max(30, int(round(abs_angle / 15.0)) * 15))
            actions = [
                self._make_rotate_action(turn_direction, turn_deg),
                ActionSpec("move_forward_%.2f" % long_stride, "MoveAhead", {"moveMagnitude": long_stride}, f"向前移动{long_stride:.2f}米"),
                ActionSpec("move_forward_%.2f" % medium_stride, "MoveAhead", {"moveMagnitude": medium_stride}, f"向前移动{medium_stride:.2f}米"),
            ]
            return actions, "turn_then_advance"

        actions = [
            ActionSpec("move_forward_%.2f" % long_stride, "MoveAhead", {"moveMagnitude": long_stride}, f"向前移动{long_stride:.2f}米"),
            ActionSpec("move_forward_%.2f" % medium_stride, "MoveAhead", {"moveMagnitude": medium_stride}, f"向前移动{medium_stride:.2f}米"),
            ActionSpec("move_forward_%.2f" % short_stride, "MoveAhead", {"moveMagnitude": short_stride}, f"向前移动{short_stride:.2f}米"),
        ]
        return actions, "direct_advance"

    def _execute_action(self, step_id: int, action: ActionSpec) -> Tuple[Dict, StepRecord]:
        before = self._to_pose_state(self.runner.extract_pose())
        obs = self.runner.execute_action(action)
        obs["pose"] = self._to_pose_state(obs["pose"])
        success = True
        final_action = action

        if action.action_api == "MoveAhead":
            moved = math.hypot(obs["pose"].x - before.x, obs["pose"].z - before.z)
            if moved < max(0.08, float(action.action_args["moveMagnitude"]) * 0.35):
                success = False
                original = float(action.action_args["moveMagnitude"])
                for fallback_mag in (0.75, 0.5, 0.25, 0.15, 0.1):
                    if fallback_mag >= original:
                        continue
                    candidate = ActionSpec(
                        action_key=f"move_forward_{fallback_mag:.2f}",
                        action_api="MoveAhead",
                        action_args={"moveMagnitude": fallback_mag},
                        action_text=f"向前移动{fallback_mag:.2f}米",
                    )
                    obs = self.runner.execute_action(candidate)
                    obs["pose"] = self._to_pose_state(obs["pose"])
                    moved = math.hypot(obs["pose"].x - before.x, obs["pose"].z - before.z)
                    if moved >= max(0.08, fallback_mag * 0.35):
                        final_action = candidate
                        success = True
                        break
        if not success:
            raise RuntimeError(f"动作失败: {action.action_api} {action.action_args}")

        return obs, StepRecord(
            step_id=step_id,
            action_api=final_action.action_api,
            action_args=final_action.action_args,
            success=success,
            before_pose=asdict(before),
            after_pose=asdict(obs["pose"]),
        )

    def _distance_progression_to_target(self, observations: Sequence[Dict], target_pos: Tuple[float, float, float]) -> List[float]:
        distances: List[float] = []
        for obs in observations:
            pose = obs["pose"]
            distances.append(self._dist3((pose.x, pose.y, pose.z), target_pos))
        return distances

    @staticmethod
    def _progression_is_logical(distances: Sequence[float]) -> bool:
        if len(distances) < 2:
            return False
        if distances[-1] > distances[0] - 0.3:
            return False
        for prev, cur in zip(distances, distances[1:]):
            if cur > prev + 0.08:
                return False
        return True

    def _collect_time_sequence(self, target) -> Tuple[List[Dict], List[StepRecord], str, List[float]]:
        start_obs = self.runner.get_observations()
        start_obs["pose"] = self._to_pose_state(start_obs["pose"])
        start_pose = start_obs["pose"]
        target_pos = tuple(float(v) for v in target.position)
        actions, route_type = self._build_route_actions(start_pose, target_pos)

        observations = [start_obs]
        steps: List[StepRecord] = []

        for step_id, action in enumerate(actions[: self.sequence_len - 1], start=1):
            nxt_obs, step_record = self._execute_action(step_id=step_id, action=action)
            steps.append(step_record)
            observations.append(nxt_obs)

        if len(observations) != self.sequence_len:
            raise RuntimeError("采集序列长度不足")

        distances = self._distance_progression_to_target(observations, target_pos)
        if not self._progression_is_logical(distances):
            raise RuntimeError("轨迹不够连贯或没有明显朝目标区域推进")

        return observations, steps, route_type, distances

    def _shuffle_and_save(self, sample_idx: int, observations: List[Dict]) -> Tuple[Dict[str, str], Dict[str, str], str]:
        n = len(observations)
        labels = LABELS[:n]
        indices = list(range(n))
        self.rng.shuffle(indices)

        sample_dir = self._prepare_sample_dir(sample_idx)
        label_to_time: Dict[str, str] = {}
        label_to_path: Dict[str, str] = {}

        for label, t_idx in zip(labels, indices):
            t_name = f"T{t_idx + 1}"
            out_path = os.path.join(sample_dir, f"frame_{label}.png")
            self._save_frame(observations[t_idx]["rgb"], out_path)
            label_to_time[label] = t_name
            label_to_path[label] = out_path

        time_to_label = {t: l for l, t in label_to_time.items()}
        ordered_labels = [time_to_label[f"T{i + 1}"] for i in range(n)]
        answer_seq = " -> ".join(ordered_labels)
        return label_to_time, label_to_path, answer_seq

    def _visible_id_set(self, obs: Dict) -> set:
        return {obj.index for obj in self._all_visible_objects(obs)}

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        self.current_scene_objects = self.ctx.scene_objects(scene_data)
        self.runner.build(scene_path, navmesh_key_path=scene_path)

        base_obs = self._teleport_random_pose()
        base_pose = base_obs["pose"]
        candidate_targets = self._choose_targets(base_obs)
        if not candidate_targets:
            raise RuntimeError("未找到合适的远距离明确目标物体")

        target = None
        observations = None
        steps = None
        route_type = None
        target_distance_progression = None

        for candidate_target in candidate_targets:
            try:
                self.runner.set_agent_state(
                    position=self._body_position_from_pose(base_pose),
                    yaw=int(round(base_pose.yaw)) % 360,
                    pitch=int(round(base_pose.pitch)),
                )
                observations, steps, route_type, target_distance_progression = self._collect_time_sequence(candidate_target)
                target = candidate_target
                break
            except Exception:
                self.runner.set_agent_state(
                    position=self._body_position_from_pose(base_pose),
                    yaw=int(round(base_pose.yaw)) % 360,
                    pitch=int(round(base_pose.pitch)),
                )
                continue

        if target is None or observations is None or steps is None or route_type is None or target_distance_progression is None:
            raise RuntimeError("当前视角下所有候选目标都未形成稳定轨迹")

        target_id = target.index
        target_name = target.name
        target_pos = tuple(float(v) for v in target.position)
        start_pose = observations[0]["pose"]
        end_pose = observations[-1]["pose"]

        label_to_time, label_to_path, answer_seq = self._shuffle_and_save(sample_idx, observations)
        labels_sorted = sorted(label_to_path.keys())
        frame_paths = [label_to_path[label] for label in labels_sorted]

        question = (
            f"智能体正朝着【{target_name}】所在区域移动，目标不一定会一直出现在视野中。"
            f"请根据空间布局和相机位姿变化，给出这{len(labels_sorted)}张图片"
            f"（{', '.join(labels_sorted)}）正确的时序发生顺序。"
        )
        answer = f"正确的顺序是：{', '.join(answer_seq.split(' -> '))}"

        target_visibility = [target_id in self._visible_id_set(obs) for obs in observations]

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "movement_sequence_sorting",
            "scene": os.path.basename(scene_path),
            "input": {
                "frame_paths": frame_paths,
                "labeled_frames": {label: label_to_path[label] for label in labels_sorted},
            },
            "question": question,
            "answer": answer,
            "gt": {
                "target": {
                    "id": target_id,
                    "type": target.template_name,
                    "name": target_name,
                    "position": {"x": target_pos[0], "y": target_pos[1], "z": target_pos[2]},
                },
                "route_type": route_type,
                "api_actions": [step.action_api for step in steps],
                "steps": [asdict(step) for step in steps],
                "start_state": {"text": start_pose.to_text(), "raw": asdict(start_pose)},
                "end_state": {"text": end_pose.to_text(), "raw": asdict(end_pose)},
                "time_order": [f"T{i + 1}" for i in range(len(observations))],
                "label_to_time": label_to_time,
                "correct_label_sequence": answer_seq,
                "target_distance_progression": target_distance_progression,
                "target_visibility_by_time": target_visibility,
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_data(self, num_samples: int) -> Tuple[str, str, int]:
        self._reset_output_dirs()
        all_samples: List[Dict] = []
        success_count = 0

        for sample_idx in range(num_samples):
            generated = False
            for retry in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    sample = self._run_one_sample(sample_idx, scene_path=scene_path)
                    all_samples.append(sample)
                    success_count += 1
                    generated = True
                    print(f"[OK] 已生成样本 {sample_idx + 1}/{num_samples}: {sample['sample_id']}")
                    break
                except Exception as exc:
                    print(f"[WARN] 样本 {sample_idx} 第{retry}次失败(scene={os.path.basename(scene_path)}): {exc}")
                    traceback.print_exc()
                finally:
                    self.runner.shutdown()
            if not generated:
                print(f"[SKIP] 样本 {sample_idx} 达到最大重试次数，已跳过")

        qa_jsonl_path = os.path.join(self.meta_dir, "qa_data.jsonl")
        qa_json_path = os.path.join(self.meta_dir, "qa_data.json")

        with open(qa_jsonl_path, "w", encoding="utf-8") as f:
            for row in all_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(qa_json_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        stats = {
            "requested_samples": num_samples,
            "generated_samples": success_count,
            "image_root": self.image_dir,
            "jsonl_path": qa_jsonl_path,
            "json_path": qa_json_path,
            "config": {
                "sequence_len": self.sequence_len,
                "move_magnitudes": MOVE_MAGNITUDES,
                "rotate_degrees": ROTATE_DEGREES,
                "min_target_dist": self.min_target_dist,
                "max_target_dist": self.max_target_dist,
                "max_episode_retry": self.max_episode_retry,
                "excluded_target_names": sorted(EXCLUDED_TARGET_NAMES),
                "require_unique_object_name_in_view": True,
                "allow_target_leave_view": True,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return qa_jsonl_path, qa_json_path, success_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSSD 第4类：运动过程排序 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/HSSD/4", help="输出根目录")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data", help="Habitat 数据根目录")
    parser.add_argument("--scene_dataset_config", type=str, default=None, help="显式指定 scene_dataset_config")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sequence_len", type=int, default=4, help="时间序列长度")
    parser.add_argument("--max_episode_retry", type=int, default=50, help="单样本最大重试次数")
    parser.add_argument("--min_target_dist", type=float, default=1.9, help="目标最小距离")
    parser.add_argument("--max_target_dist", type=float, default=6.0, help="目标最大距离")
    parser.add_argument("--image_width", type=int, default=640, help="图像宽度")
    parser.add_argument("--image_height", type=int, default=480, help="图像高度")
    parser.add_argument("--hfov", type=int, default=90, help="水平视场角")
    parser.add_argument("--sensor_height", type=float, default=1.6, help="相机高度")
    parser.add_argument("--agent_radius", type=float, default=0.1, help="导航体半径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gen = HSSDMovementSequenceSortingGenerator(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        image_width=args.image_width,
        image_height=args.image_height,
        hfov=args.hfov,
        sensor_height=args.sensor_height,
        agent_radius=args.agent_radius,
        seed=args.seed,
        sequence_len=args.sequence_len,
        max_episode_retry=args.max_episode_retry,
        min_target_dist=args.min_target_dist,
        max_target_dist=args.max_target_dist,
        scene_dataset_config=args.scene_dataset_config,
    )

    qa_jsonl_path, qa_json_path, success_count = gen.generate_data(args.num_samples)

    print("\n===== 生成完成 =====")
    print(f"成功生成: {success_count}/{args.num_samples}")
    print(f"JSONL: {qa_jsonl_path}")
    print(f"JSON : {qa_json_path}")


if __name__ == "__main__":
    main()
