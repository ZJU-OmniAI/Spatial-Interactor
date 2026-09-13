#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReplicaCAD 第5类：运动程度比较 (Movement Degree Comparison)

逻辑对齐 /path/to/workspace/AI2THOR/5/generate_movement_degree_comparison_data.py，
仅把底层接口替换为 Habitat/ReplicaCAD。
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
from habitat_sim.agent import ActionSpec as HabitatActionSpec
from habitat_sim.agent.controls import ActuationSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner


GRID_SIZE = 0.25
MOVE_MAGNITUDES = [0.25, 0.5, 0.75, 1.0]
FALLBACK_MOVE_MAGNITUDES = [0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
ROTATE_DEGREES = [30, 60, 90]
START_YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
# 第5类固定平视，避免出现抬头/低头视角。
START_HORIZONS = [0]


@dataclass(frozen=True)
class ActionTemplate:
    family: str
    action_api: str
    arg_name: str
    values: Tuple[float, ...]
    description: str
    unit: str


ACTION_TEMPLATES: Tuple[ActionTemplate, ...] = (
    ActionTemplate("translation", "MoveAhead", "moveMagnitude", tuple(MOVE_MAGNITUDES), "向前移动", "meters"),
    ActionTemplate("translation", "MoveBack", "moveMagnitude", tuple(MOVE_MAGNITUDES), "向后移动", "meters"),
    ActionTemplate("translation", "MoveLeft", "moveMagnitude", tuple(MOVE_MAGNITUDES), "向左移动", "meters"),
    ActionTemplate("translation", "MoveRight", "moveMagnitude", tuple(MOVE_MAGNITUDES), "向右移动", "meters"),
    ActionTemplate("rotation", "RotateLeft", "degrees", tuple(ROTATE_DEGREES), "向左旋转", "degrees"),
    ActionTemplate("rotation", "RotateRight", "degrees", tuple(ROTATE_DEGREES), "向右旋转", "degrees"),
)

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
class ActionOutcome:
    tag: str
    family: str
    action_api: str
    action_text: str
    planned_value: float
    actual_value: float
    unit: str
    success: bool
    start_pose: Dict
    end_pose: Dict


@dataclass
class ActionSpec:
    action_key: str
    action_api: str
    action_params: Dict[str, float]
    action_text: str


class ReplicaCADMovementDegreeRunner(HabitatSceneRunner):
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


class ReplicaCADMovementDegreeComparisonGenerator:
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
        max_episode_retry: int = 80,
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

        self.runner = ReplicaCADMovementDegreeRunner(
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
    def _dist(a: PoseState, b: PoseState) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    @staticmethod
    def _angular_diff_deg(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _teleport_random_start(self) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")
        scene_data = self.ctx.load_scene(self.runner.current_scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)
        best_obs = None
        best_score = -1e9
        for _ in range(24):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            horizon = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=horizon)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            score = self.runner.view_quality_score(scene_objects, obs, require_unique=True)
            if score > best_score:
                best_score = score
                best_obs = obs
            if score >= 12.0:
                return obs
        if best_obs is None:
            raise RuntimeError("随机起点失败")
        return best_obs

    def _reset_to_start(self, start_pose: PoseState) -> Dict:
        obs = self.runner.set_agent_state(
            position=self._body_position_from_pose(start_pose),
            yaw=int(round(start_pose.yaw)) % 360,
            pitch=int(round(start_pose.pitch)),
        )
        obs["pose"] = self._to_pose_state(obs["pose"])
        return obs

    def _ordered_templates_for_sample(self, sample_idx: int) -> List[ActionTemplate]:
        offset = sample_idx % len(ACTION_TEMPLATES)
        preferred = ACTION_TEMPLATES[offset]
        remaining = list(ACTION_TEMPLATES[:offset]) + list(ACTION_TEMPLATES[offset + 1 :])
        self.rng.shuffle(remaining)
        return [preferred] + remaining

    def _sample_value_pair(self, values: Sequence[float]) -> Tuple[float, float]:
        a, b = self.rng.sample(list(values), 2)
        small, large = sorted((float(a), float(b)))
        return small, large

    def _action_text(self, template: ActionTemplate, value: float) -> str:
        if template.unit == "degrees":
            return f"{template.description}{int(value)}度"
        return f"{template.description}{value:.2f}米"

    def _make_action_spec(self, template: ActionTemplate, value: float) -> ActionSpec:
        if template.family == "rotation":
            key = f"{ROTATE_KEY_PREFIX[template.action_api]}_{int(value)}"
        else:
            key = f"{MOVE_KEY_PREFIX[template.action_api]}_{value:.2f}"
        return ActionSpec(
            action_key=key,
            action_api=template.action_api,
            action_params={template.arg_name: float(value)},
            action_text=self._action_text(template, value),
        )

    def _measure_action_amount(self, template: ActionTemplate, start_pose: PoseState, end_pose: PoseState) -> float:
        if template.family == "rotation":
            return self._angular_diff_deg(end_pose.yaw, start_pose.yaw)
        return self._dist(start_pose, end_pose)

    def _valid_amount(self, template: ActionTemplate, planned: float, actual: float) -> bool:
        if template.family == "rotation":
            return actual >= planned - 1.0
        return actual >= planned * 0.8

    def _execute_single_action(self, tag: str, template: ActionTemplate, value: float) -> Tuple[ActionOutcome, Dict]:
        start_pose = self._to_pose_state(self.runner.extract_pose())
        action = self._make_action_spec(template, value)
        obs = self.runner.execute_action(action)
        obs["pose"] = self._to_pose_state(obs["pose"])
        end_pose = obs["pose"]
        actual_value = self._measure_action_amount(template, start_pose, end_pose)

        if not self._valid_amount(template, value, actual_value):
            if template.family == "translation":
                original = float(value)
                for fallback in (0.75, 0.5, 0.25, 0.15, 0.1):
                    if fallback >= original:
                        continue
                    action = self._make_action_spec(template, fallback)
                    obs = self.runner.execute_action(action)
                    obs["pose"] = self._to_pose_state(obs["pose"])
                    end_pose = obs["pose"]
                    actual_value = self._measure_action_amount(template, start_pose, end_pose)
                    if self._valid_amount(template, fallback, actual_value):
                        value = fallback
                        break
                else:
                    raise RuntimeError(
                        f"{tag}动作失败或实际幅度不足: {template.action_api} planned={value} actual={actual_value:.3f}"
                    )
            else:
                raise RuntimeError(
                    f"{tag}动作失败或实际幅度不足: {template.action_api} planned={value} actual={actual_value:.3f}"
                )

        return ActionOutcome(
            tag=tag,
            family=template.family,
            action_api=template.action_api,
            action_text=self._action_text(template, value),
            planned_value=value,
            actual_value=actual_value,
            unit=template.unit,
            success=True,
            start_pose=asdict(start_pose),
            end_pose=asdict(end_pose),
        ), obs

    def _run_branch(self, tag: str, start_pose: PoseState, template: ActionTemplate, value: float) -> Tuple[ActionOutcome, Dict]:
        self._reset_to_start(start_pose)
        return self._execute_single_action(tag=tag, template=template, value=value)

    @staticmethod
    def _record_is_farther(record_a: ActionOutcome, record_b: ActionOutcome) -> bool:
        return record_a.actual_value > record_b.actual_value

    def _validate_comparison_gap(self, template: ActionTemplate, records: Sequence[ActionOutcome]) -> bool:
        actual_values = sorted(record.actual_value for record in records)
        gap = actual_values[-1] - actual_values[0]
        if template.family == "rotation":
            return gap >= 15.0
        return gap >= 0.18

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        self.runner.build(scene_path, navmesh_key_path=scene_path)
        start_obs = self._teleport_random_start()
        start_event_frame = start_obs["rgb"]
        start_pose = start_obs["pose"]

        template = None
        record_b = None
        record_c = None

        for candidate in self._ordered_templates_for_sample(sample_idx):
            try:
                small_value, large_value = self._sample_value_pair(candidate.values)
                if self.rng.random() < 0.5:
                    value_b, value_c = small_value, large_value
                else:
                    value_b, value_c = large_value, small_value

                record_b, _ = self._run_branch("B", start_pose, candidate, value_b)
                record_c, _ = self._run_branch("C", start_pose, candidate, value_c)
                if not self._validate_comparison_gap(candidate, [record_b, record_c]):
                    raise RuntimeError("两次动作实际幅度差异不足")
                template = candidate
                break
            except Exception:
                record_b = None
                record_c = None
                template = None
                self._reset_to_start(start_pose)

        if template is None or record_b is None or record_c is None:
            raise RuntimeError("未找到合适的同类型动作幅度比较样本")

        sample_dir = self._prepare_sample_dir(sample_idx)
        frame_a_path = os.path.join(sample_dir, "frame_A.png")
        frame_b_path = os.path.join(sample_dir, "frame_B.png")
        frame_c_path = os.path.join(sample_dir, "frame_C.png")

        self._save_frame(start_event_frame, frame_a_path)

        self._reset_to_start(start_pose)
        _, obs_b = self._execute_single_action(tag="B", template=template, value=record_b.planned_value)
        self._save_frame(obs_b["rgb"], frame_b_path)

        self._reset_to_start(start_pose)
        _, obs_c = self._execute_single_action(tag="C", template=template, value=record_c.planned_value)
        self._save_frame(obs_c["rgb"], frame_c_path)

        larger_image = "B" if self._record_is_farther(record_b, record_c) else "C"
        smaller_image = "C" if larger_image == "B" else "B"

        question = (
            "图A是起始视角，图B和图C都表示智能体从图A出发，"
            "各执行了一次同类型但不同幅度的单动作后的结果。"
            "这个动作类型可能是前后左右移动，也可能是左右转动。"
            "请判断图B和图C里，哪一张对应的动作幅度更大？"
        )
        answer = f"【图{larger_image}】对应的动作幅度更大。"

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "movement_degree_comparison",
            "scene": os.path.basename(scene_path),
            "input": {
                "frame_paths": [frame_a_path, frame_b_path, frame_c_path],
                "labeled_frames": {
                    "A": frame_a_path,
                    "B": frame_b_path,
                    "C": frame_c_path,
                },
                "reference_frame": frame_a_path,
            },
            "question": question,
            "answer": answer,
            "gt": {
                "start_pose": {
                    "text": start_pose.to_text(),
                    "raw": asdict(start_pose),
                },
                "comparison_type": template.family,
                "action_api": template.action_api,
                "action_description": template.description,
                "same_action_type_constraint": True,
                "larger_image": larger_image,
                "smaller_image": smaller_image,
                "frame_A_role": "shared_start",
                "frame_B": asdict(record_b),
                "frame_C": asdict(record_c),
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_data(self, num_samples: int) -> Tuple[str, str, int]:
        self._reset_output_dirs()

        rows: List[Dict] = []
        success = 0

        for sample_idx in range(num_samples):
            generated = False
            for retry in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    row = self._run_one_sample(sample_idx, scene_path=scene_path)
                    rows.append(row)
                    success += 1
                    generated = True
                    print(f"[OK] 已生成样本 {sample_idx + 1}/{num_samples}: {row['sample_id']}")
                    break
                except Exception as exc:
                    print(f"[WARN] 样本 {sample_idx} 第{retry}次失败(scene={os.path.basename(scene_path)}): {exc}")
                    traceback.print_exc()
                finally:
                    self.runner.shutdown()

            if not generated:
                print(f"[SKIP] 样本 {sample_idx} 达到最大重试次数，已跳过")

        jsonl_path = os.path.join(self.meta_dir, "qa_data.jsonl")
        json_path = os.path.join(self.meta_dir, "qa_data.json")

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        action_counter = Counter(row["gt"]["action_api"] for row in rows)
        family_counter = Counter(row["gt"]["comparison_type"] for row in rows)
        stats = {
            "requested_samples": num_samples,
            "generated_samples": success,
            "image_root": self.image_dir,
            "jsonl_path": jsonl_path,
            "json_path": json_path,
            "config": {
                "max_episode_retry": self.max_episode_retry,
                "grid_size": GRID_SIZE,
                "move_magnitudes": MOVE_MAGNITUDES,
                "rotate_degrees": ROTATE_DEGREES,
                "fixed_num_frames": 3,
                "same_action_type_constraint": True,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
            "action_distribution": dict(action_counter),
            "family_distribution": dict(family_counter),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return jsonl_path, json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReplicaCAD 第5类：运动程度比较 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/ReplicaCAD/5")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scene_dataset_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_retry", type=int, default=80)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=90)
    parser.add_argument("--sensor_height", type=float, default=1.6)
    parser.add_argument("--agent_radius", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = ReplicaCADMovementDegreeComparisonGenerator(
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
