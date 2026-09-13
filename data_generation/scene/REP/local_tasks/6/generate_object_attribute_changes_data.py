#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReplicaCAD 第6类：改变物体属性 (Object State/Attribute Changes)

逻辑尽量对齐 /path/to/workspace/AI2THOR/6/generate_object_attribute_changes_data.py，
仅把底层执行替换为 Habitat/ReplicaCAD。

说明：
1. ReplicaCAD scene_instance.json 不包含 AI2THOR 那种 open/toggle 运行时状态字段，
   因此这里稳定支持 rotate/remove 两类可直接通过场景 JSON 改写验证的属性变化。
2. 视角固定为正常人高度与平视，避免抬头/低头视角。
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
import quaternion
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner


START_YAWS = list(range(0, 360, 15))
START_HORIZONS = [0]
MAX_TARGET_DISTANCE = 3.2
MIN_FRAME_DIFF = 4.0
ROTATE_DEGREES = [45, 60, 90, 120]
CATEGORY_SCHEDULE = ["rotate", "open_close", "toggle", "remove", "rotate"]

EXCLUDED_TARGET_NAMES = {
    "bathtub",
    "bed",
    "bench",
    "cabinet",
    "chair",
    "counter",
    "counter top",
    "curtain",
    "desk",
    "door",
    "drawer",
    "floor",
    "fridge",
    "light",
    "mirror",
    "piano",
    "rack",
    "shelf",
    "sink",
    "sofa",
    "stool",
    "table",
    "toilet",
    "tv stand",
    "wall",
    "window",
}

ROTATION_EXCLUDED_NAMES = {
    "basket",
    "bowl",
    "book",
    "cup",
    "dish",
    "frame",
    "mug",
    "plate",
    "rug",
    "tv",
    "vase",
}

BAD_NAME_SUBSTRINGS = (
    " part",
    " component",
    " module",
    " panel",
    " numbers clock",
    " time clock",
)


def _frame_diff_score(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def _quat_from_wxyz(values: Sequence[float]) -> quaternion.quaternion:
    return quaternion.quaternion(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def _quat_to_wxyz(q: quaternion.quaternion) -> List[float]:
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def _yaw_quaternion(degrees: float) -> quaternion.quaternion:
    radians = math.radians(float(degrees))
    return quaternion.quaternion(math.cos(radians / 2.0), 0.0, math.sin(radians / 2.0), 0.0)


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
class OperationSpec:
    category: str
    action_api: str
    action_kwargs: Dict
    question: str


class ReplicaCADObjectAttributeChangeGenerator:
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
        max_episode_retry: int = 50,
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
    def _position_to_dict(position: Sequence[float]) -> Dict[str, float]:
        return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}

    @staticmethod
    def _is_meaningful_name(name: str) -> bool:
        normalized = (name or "").strip().lower()
        if not normalized or len(normalized) < 2:
            return False
        if normalized == "object" or normalized.replace(" ", "").isdigit():
            return False
        if any(token in normalized for token in BAD_NAME_SUBSTRINGS):
            return False
        if any(ch.isdigit() for ch in normalized):
            return False
        alpha_count = sum(ch.isalpha() for ch in normalized)
        return alpha_count >= max(2, len(normalized.replace(" ", "")) // 5)

    def _good_target_view(self, obj, relaxed: bool = False) -> bool:
        x_ratio = obj.pixel_x / float(self.image_width)
        y_ratio = obj.pixel_y / float(self.image_height)
        prominence = self.runner.target_prominence_score(obj)
        if relaxed:
            return (
                0.20 <= x_ratio <= 0.80
                and 0.16 <= y_ratio <= 0.84
                and 0.60 <= obj.distance <= 2.40
                and obj.local_z > 0.45
                and prominence >= 2.0
            )
        return (
            0.26 <= x_ratio <= 0.74
            and 0.20 <= y_ratio <= 0.80
            and 0.80 <= obj.distance <= 2.00
            and obj.local_z > 0.60
            and prominence >= 2.4
        )

    def _candidate_objects(self, scene_objects: Sequence[Dict], obs: Dict) -> List:
        visible = self.runner.unique_visible_objects(scene_objects, obs["depth"])
        strict_candidates = []
        relaxed_candidates = []
        for obj in visible:
            if obj.name in EXCLUDED_TARGET_NAMES:
                continue
            if not self._is_meaningful_name(obj.name):
                continue
            if self._good_target_view(obj):
                strict_candidates.append(obj)
            elif self._good_target_view(obj, relaxed=True):
                relaxed_candidates.append(obj)
        candidates = strict_candidates if strict_candidates else relaxed_candidates
        candidates.sort(key=lambda obj: self.runner.target_prominence_score(obj), reverse=True)
        return candidates

    def _teleport_random_start(self, scene_objects: Sequence[Dict]) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best = None
        best_score = -1
        for _ in range(40):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            pitch = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=pitch)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            candidates = self._candidate_objects(scene_objects, obs)
            score = self.runner.view_quality_score(scene_objects, obs, require_unique=True)
            if candidates:
                score += self.runner.target_prominence_score(candidates[0])
            if score > best_score:
                best_score = score
                best = {"obs": obs, "candidates": candidates}
            if candidates and score >= 20.0:
                return best

        if best is None:
            raise RuntimeError("随机起点失败")
        return best

    def _find_matching_visible(self, visible_after: Sequence, target) -> Optional:
        matches = [
            obj
            for obj in visible_after
            if obj.name == target.name and obj.template_name == target.template_name
        ]
        if not matches:
            return None
        matches.sort(
            key=lambda obj: (
                (obj.position[0] - target.position[0]) ** 2
                + (obj.position[1] - target.position[1]) ** 2
                + (obj.position[2] - target.position[2]) ** 2
            )
        )
        return matches[0]

    def _rotation_operation(self, target) -> Optional[OperationSpec]:
        if target.name in ROTATION_EXCLUDED_NAMES:
            return None
        degrees = self.rng.choice(ROTATE_DEGREES)
        return OperationSpec(
            category="rotate",
            action_api="ModifyScene/RotateObject",
            action_kwargs={"target_index": int(target.index), "degrees": float(degrees)},
            question="场景中哪个物体的朝向发生了变化？",
        )

    def _possible_ops(self, target) -> List[OperationSpec]:
        return [
            OperationSpec(
                category="remove",
                action_api="ModifyScene/RemoveObject",
                action_kwargs={"target_index": int(target.index)},
                question="场景中哪个物体被移走了？",
            )
        ]

    def _ordered_categories(self, sample_idx: int, available: List[str]) -> List[str]:
        preferred = CATEGORY_SCHEDULE[sample_idx % len(CATEGORY_SCHEDULE)]
        ordered = [preferred] + [cat for cat in CATEGORY_SCHEDULE if cat != preferred]
        deduped = []
        for cat in ordered:
            if cat in available and cat not in deduped:
                deduped.append(cat)
        for cat in available:
            if cat not in deduped:
                deduped.append(cat)
        return deduped

    def _make_scene_copy_with_rotation(self, scene_data: Dict, target_index: int, degrees: float) -> Dict:
        new_scene = copy.deepcopy(scene_data)
        target = new_scene["object_instances"][target_index]
        current_q = _quat_from_wxyz(target.get("rotation", [1.0, 0.0, 0.0, 0.0]))
        target["rotation"] = _quat_to_wxyz(_yaw_quaternion(degrees) * current_q)
        return new_scene

    def _make_scene_copy_without_object(self, scene_data: Dict, target_index: int) -> Dict:
        new_scene = copy.deepcopy(scene_data)
        del new_scene["object_instances"][target_index]
        return new_scene

    def _object_state_summary(self, scene_obj: Optional[Dict], visible_obj) -> Dict:
        if scene_obj is None:
            return {"exists": False, "visible": False}
        return {
            "exists": True,
            "visible": visible_obj is not None,
            "position": self._position_to_dict(scene_obj["position"]),
            "rotation": [float(v) for v in scene_obj["rotation"]],
        }

    def _describe_change(
        self,
        target,
        op: OperationSpec,
        before_scene_objects: Sequence[Dict],
        after_scene_objects: Sequence[Dict],
        before_obs: Dict,
        after_obs: Dict,
    ) -> Tuple[bool, str, Dict, Dict]:
        before_visible_target = target
        after_visible = self.runner.unique_visible_objects(after_scene_objects, after_obs["depth"])
        diff_score = _frame_diff_score(before_obs["rgb"], after_obs["rgb"])

        before_scene_obj = before_scene_objects[target.index]
        if op.category == "rotate":
            after_scene_obj = after_scene_objects[target.index]
            after_visible_target = self._find_matching_visible(after_visible, target)
            before_state = self._object_state_summary(before_scene_obj, before_visible_target)
            after_state = self._object_state_summary(after_scene_obj, after_visible_target)
            if after_visible_target is None or after_visible_target.local_z <= 0.15:
                return False, "", before_state, after_state
            if diff_score < 1.5:
                return False, "", before_state, after_state
            degrees = int(round(float(op.action_kwargs["degrees"])))
            text = f"朝向发生了变化，旋转了约{degrees}度"
            return True, text, before_state, after_state

        if op.category == "remove":
            before_state = self._object_state_summary(before_scene_obj, before_visible_target)
            after_state = {"exists": False, "visible": False}
            return True, "被移走了", before_state, after_state

        return False, "", {}, {}

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)

        self.runner.build(scene_path, navmesh_key_path=scene_path)
        start = self._teleport_random_start(scene_objects)
        start_obs = start["obs"]
        candidates = start["candidates"]
        if not candidates:
            raise RuntimeError("当前视角没有合适的可修改目标物体")

        ops_by_category: Dict[str, List[Tuple[object, OperationSpec]]] = {}
        for target in candidates:
            for op in self._possible_ops(target):
                ops_by_category.setdefault(op.category, []).append((target, op))

        if not ops_by_category:
            raise RuntimeError("当前视角没有可执行的属性变化操作")

        ordered_categories = self._ordered_categories(sample_idx, list(ops_by_category.keys()))
        chosen_target = None
        chosen_op = None
        change_text = None
        before_target_state = None
        after_target_state = None
        chosen_after_obs = None
        modified_scene_path = None
        best_diff_score = -1.0

        for category in ordered_categories:
            candidates_in_category = list(ops_by_category.get(category, []))
            self.rng.shuffle(candidates_in_category)
            if category == "rotate":
                candidates_in_category = candidates_in_category[:4]
            if category == "remove":
                candidates_in_category = candidates_in_category[:10]
            for target, op in candidates_in_category:
                if op.category == "rotate":
                    modified_scene = self._make_scene_copy_with_rotation(
                        scene_data, op.action_kwargs["target_index"], op.action_kwargs["degrees"]
                    )
                elif op.category == "remove":
                    modified_scene = self._make_scene_copy_without_object(scene_data, op.action_kwargs["target_index"])
                else:
                    continue

                current_modified_scene_path = self.ctx.save_temp_scene(
                    modified_scene, scene_path, task_id=6, sample_idx=sample_idx
                )
                after_scene_objects = self.ctx.scene_objects(modified_scene)

                self.runner.build(current_modified_scene_path, navmesh_key_path=scene_path)
                pose = start_obs["pose"]
                current_after_obs = self.runner.set_agent_state(
                    position=self._body_position_from_pose(pose),
                    yaw=pose.yaw,
                    pitch=pose.pitch,
                )
                current_after_obs["pose"] = self._to_pose_state(current_after_obs["pose"])

                valid, text, before_state, after_state = self._describe_change(
                    target=target,
                    op=op,
                    before_scene_objects=scene_objects,
                    after_scene_objects=after_scene_objects,
                    before_obs=start_obs,
                    after_obs=current_after_obs,
                )
                if not valid:
                    continue

                current_diff_score = _frame_diff_score(start_obs["rgb"], current_after_obs["rgb"])
                if current_diff_score <= best_diff_score:
                    continue

                chosen_target = target
                chosen_op = op
                change_text = text
                before_target_state = before_state
                after_target_state = after_state
                best_diff_score = current_diff_score
                modified_scene_path = current_modified_scene_path
                chosen_after_obs = {
                    "rgb": np.array(current_after_obs["rgb"], copy=True),
                    "depth": np.array(current_after_obs["depth"], copy=True),
                    "pose": current_after_obs["pose"],
                    "sensor_state": current_after_obs["sensor_state"],
                }
            if chosen_op is not None:
                break

        if chosen_target is None or chosen_op is None or chosen_after_obs is None or change_text is None:
            raise RuntimeError("未能找到有效的属性变化样本")

        sample_dir = self._prepare_sample_dir(sample_idx)
        frame_a = os.path.join(sample_dir, "frame_000_A.png")
        frame_b = os.path.join(sample_dir, "frame_001_B.png")
        self._save_frame(start_obs["rgb"], frame_a)
        self._save_frame(chosen_after_obs["rgb"], frame_b)

        answer = f"【{chosen_target.name}】{change_text}。"
        start_pose = start_obs["pose"]
        end_pose = chosen_after_obs["pose"]

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "object_state_attribute_changes",
            "scene": os.path.basename(scene_path),
            "input": {
                "frame_paths": [frame_a, frame_b],
                "frame_A": frame_a,
                "frame_B": frame_b,
            },
            "question": chosen_op.question,
            "answer": answer,
            "gt": {
                "target": {
                    "index": int(chosen_target.index),
                    "template_name": chosen_target.template_name,
                    "name": chosen_target.name,
                    "position": self._position_to_dict(chosen_target.position),
                },
                "change_category": chosen_op.category,
                "change_text": change_text,
                "action_api": chosen_op.action_api,
                "action_params": chosen_op.action_kwargs,
                "target_state_before": before_target_state,
                "target_state_after": after_target_state,
                "start_state": {"text": start_pose.to_text(), "raw": asdict(start_pose)},
                "end_state": {"text": end_pose.to_text(), "raw": asdict(end_pose)},
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "modified_scene_path": modified_scene_path,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def generate_data(self, num_samples: int) -> Tuple[str, str, int]:
        self._reset_output_dirs()

        all_samples: List[Dict] = []
        success = 0

        for i in range(num_samples):
            generated = False
            for retry in range(1, self.max_episode_retry + 1):
                scene_path = self._random_scene()
                try:
                    row = self._run_one_sample(i, scene_path=scene_path)
                    all_samples.append(row)
                    success += 1
                    generated = True
                    print(f"[OK] 已生成样本 {i + 1}/{num_samples}: {row['sample_id']}")
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
            for row in all_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        category_counter = Counter(row["gt"]["change_category"] for row in all_samples)
        action_counter = Counter(row["gt"]["action_api"] for row in all_samples)
        stats = {
            "requested_samples": num_samples,
            "generated_samples": success,
            "image_root": self.image_dir,
            "jsonl_path": jsonl_path,
            "json_path": json_path,
            "config": {
                "max_episode_retry": self.max_episode_retry,
                "max_target_distance": MAX_TARGET_DISTANCE,
                "field_of_view": self.hfov,
                "sensor_height": self.sensor_height,
                "fixed_horizons": START_HORIZONS,
                "category_schedule": CATEGORY_SCHEDULE,
                "implemented_change_categories": ["remove"],
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
            "category_distribution": dict(category_counter),
            "action_distribution": dict(action_counter),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return jsonl_path, json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReplicaCAD 第6类：改变物体属性 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/ReplicaCAD/6")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scene_dataset_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_retry", type=int, default=50)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=105)
    parser.add_argument("--sensor_height", type=float, default=1.6)
    parser.add_argument("--agent_radius", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = ReplicaCADObjectAttributeChangeGenerator(
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
