#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSSD 第3类：视差推断深度 (Parallax Depth Inference) 数据生成器。

逻辑对齐 /path/to/workspace/AI2THOR/3/generate_parallax_depth_data.py，
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from habitat_sim.agent import ActionSpec as HabitatActionSpec
from habitat_sim.agent.controls import ActuationSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner


MOVE_MAGNITUDES = [0.25, 0.5]
ROTATE_DEGREES = 30
START_YAWS = list(range(0, 360, 30))
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
class StepRecord:
    step_id: int
    action_api: str
    action_args: Dict
    success: bool
    before_pose: Dict
    after_pose: Dict


@dataclass
class ActionSpec:
    action_key: str
    action_api: str
    action_params: Dict[str, float]
    action_natural: str

    @property
    def action_text(self) -> str:
        return self.action_natural


class HSSDParallaxRunner(HabitatSceneRunner):
    def _build_action_space(self) -> Dict[str, HabitatActionSpec]:
        action_space: Dict[str, HabitatActionSpec] = {}
        for mag in MOVE_MAGNITUDES:
            for _, key_prefix in MOVE_KEY_PREFIX.items():
                key = f"{key_prefix}_{mag:.2f}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(mag)),
                )
        for deg in [ROTATE_DEGREES]:
            for _, key_prefix in ROTATE_KEY_PREFIX.items():
                key = f"{key_prefix}_{int(deg)}"
                action_space[key] = HabitatActionSpec(
                    name=key_prefix,
                    actuation=ActuationSpec(amount=float(deg)),
                )
        return action_space


class HSSDParallaxDepthDataGenerator:
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
        max_episode_retry: int = 30,
        min_depth_gap: float = 0.5,
        min_obj_dist: float = 0.5,
        max_obj_dist: float = 6.0,
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
        self.min_depth_gap = min_depth_gap
        self.min_obj_dist = min_obj_dist
        self.max_obj_dist = max_obj_dist

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

        self.runner = HSSDParallaxRunner(
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
    def _euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

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

    def _is_good_name(self, name: str) -> bool:
        normalized = " ".join((name or "").lower().split())
        if not normalized or len(normalized) < 2:
            return False
        for pattern in AMBIGUOUS_NAME_PATTERNS:
            if pattern in normalized:
                return False
        return True

    def _all_visible_objects(self, obs: Dict) -> List[object]:
        return self.runner.visible_objects(self.current_scene_objects, obs["depth"])

    def _visible_name_counts(self, obs: Dict) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self._all_visible_objects(obs):
            counts[obj.name] = counts.get(obj.name, 0) + 1
        return counts

    def _candidate_visible_objects(self, obs: Dict) -> List[object]:
        counts = self._visible_name_counts(obs)
        candidates: List[object] = []
        for obj in self._all_visible_objects(obs):
            if not self._is_good_name(obj.name):
                continue
            if counts.get(obj.name, 0) != 1:
                continue
            candidates.append(obj)
        candidates.sort(key=lambda obj: self.runner.target_prominence_score(obj), reverse=True)
        return candidates

    def _teleport_random_pose(self) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best_obs = None
        best_score = -1.0
        for _ in range(20):
            position = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            horizon = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=position, yaw=yaw, pitch=horizon)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            score = self.runner.view_quality_score(self.current_scene_objects, obs, require_unique=True)
            if score > best_score:
                best_score = score
                best_obs = obs
            if score >= 12.0:
                return obs

        if best_obs is None:
            raise RuntimeError("随机传送失败")
        return best_obs

    def _make_move_ahead_action(self, magnitude: float) -> ActionSpec:
        return ActionSpec(
            action_key=f"move_forward_{magnitude:.2f}",
            action_api="MoveAhead",
            action_params={"moveMagnitude": float(magnitude)},
            action_natural=f"向前移动{magnitude:.2f}米",
        )

    def _select_depth_pair(self, obs_a: Dict, cam_xyz: Tuple[float, float, float]) -> Optional[Dict]:
        visibles = self._candidate_visible_objects(obs_a)
        if len(visibles) < 2:
            return None

        candidates = []
        for i in range(len(visibles)):
            for j in range(i + 1, len(visibles)):
                oa = visibles[i]
                ob = visibles[j]
                pa = tuple(float(v) for v in oa.position)
                pb = tuple(float(v) for v in ob.position)
                da = self._euclidean_distance(cam_xyz, pa)
                db = self._euclidean_distance(cam_xyz, pb)
                if not (self.min_obj_dist <= da <= self.max_obj_dist):
                    continue
                if not (self.min_obj_dist <= db <= self.max_obj_dist):
                    continue
                gap = abs(da - db)
                if gap < self.min_depth_gap:
                    continue
                if oa.name == ob.name:
                    continue

                near_obj = oa if da < db else ob
                far_obj = ob if da < db else oa
                candidates.append(
                    {
                        "object_a": oa,
                        "object_b": ob,
                        "dist_a": da,
                        "dist_b": db,
                        "depth_gap": gap,
                        "near_object_id": near_obj.index,
                        "far_object_id": far_obj.index,
                    }
                )

        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                item["depth_gap"],
                min(
                    self.runner.target_prominence_score(item["object_a"]),
                    self.runner.target_prominence_score(item["object_b"]),
                ),
            ),
            reverse=True,
        )
        top_k = candidates[: min(8, len(candidates))]
        return self.rng.choice(top_k)

    def _visible_id_set(self, obs: Dict) -> set:
        return {obj.index for obj in self._all_visible_objects(obs)}

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        self.current_scene_objects = self.ctx.scene_objects(scene_data)
        self.runner.build(scene_path, navmesh_key_path=scene_path)

        obs_a = self._teleport_random_pose()
        pose_a = obs_a["pose"]
        cam_xyz = (pose_a.x, pose_a.y, pose_a.z)

        pair = self._select_depth_pair(obs_a, cam_xyz)
        if pair is None:
            raise RuntimeError("未找到满足深度差要求的可见物体对")

        obj_a = pair["object_a"]
        obj_b = pair["object_b"]
        obj_a_id = obj_a.index
        obj_b_id = obj_b.index

        move_mag = self.rng.choice(MOVE_MAGNITUDES)
        action = self._make_move_ahead_action(move_mag)
        before_pose = self._to_pose_state(self.runner.extract_pose())
        obs_b = self.runner.execute_action(action)
        obs_b["pose"] = self._to_pose_state(obs_b["pose"])
        after_pose = obs_b["pose"]

        moved = math.hypot(after_pose.x - before_pose.x, after_pose.z - before_pose.z)
        if moved < max(0.08, move_mag * 0.35):
            raise RuntimeError("MoveAhead 变化过小")

        step_record = StepRecord(
            step_id=1,
            action_api="MoveAhead",
            action_args={"moveMagnitude": move_mag},
            success=True,
            before_pose=asdict(before_pose),
            after_pose=asdict(after_pose),
        )

        vis_b = self._visible_id_set(obs_b)
        if obj_a_id not in vis_b or obj_b_id not in vis_b:
            raise RuntimeError("平移后目标物体不再同时可见")

        pa = tuple(float(v) for v in obj_a.position)
        pb = tuple(float(v) for v in obj_b.position)
        dist_a = self._euclidean_distance(cam_xyz, pa)
        dist_b = self._euclidean_distance(cam_xyz, pb)
        if abs(dist_a - dist_b) < self.min_depth_gap:
            raise RuntimeError("深度差不足")

        nearer = "A" if dist_a < dist_b else "B"
        name_a = obj_a.name
        name_b = obj_b.name
        question = f"在这两张图像中，【{name_a}】和【{name_b}】哪个距离相机更近？"
        answer = f"【{name_a if nearer == 'A' else name_b}】距离更近。"

        sample_img_dir = self._prepare_sample_dir(sample_idx)
        frame_a_path = os.path.join(sample_img_dir, "frame_000_A.png")
        frame_b_path = os.path.join(sample_img_dir, "frame_001_B.png")
        self._save_frame(obs_a["rgb"], frame_a_path)
        self._save_frame(obs_b["rgb"], frame_b_path)

        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "parallax_depth_inference",
            "scene": os.path.basename(scene_path),
            "input": {
                "frame_paths": [frame_a_path, frame_b_path],
                "frame_A": frame_a_path,
                "frame_B": frame_b_path,
            },
            "question": question,
            "answer": answer,
            "gt": {
                "api_actions": ["MoveAhead"],
                "steps": [asdict(step_record)],
                "start_state": {"text": pose_a.to_text(), "raw": asdict(pose_a)},
                "end_state": {"text": after_pose.to_text(), "raw": asdict(after_pose)},
                "camera_position_A": {"x": cam_xyz[0], "y": cam_xyz[1], "z": cam_xyz[2]},
                "objects": {
                    "object_a": {
                        "id": obj_a_id,
                        "type": obj_a.template_name,
                        "name": name_a,
                        "position": {"x": pa[0], "y": pa[1], "z": pa[2]},
                        "distance_to_camera_A": dist_a,
                    },
                    "object_b": {
                        "id": obj_b_id,
                        "type": obj_b.template_name,
                        "name": name_b,
                        "position": {"x": pb[0], "y": pb[1], "z": pb[2]},
                        "distance_to_camera_A": dist_b,
                    },
                },
                "depth_gap": abs(dist_a - dist_b),
                "nearer_object": {
                    "label": nearer,
                    "id": obj_a_id if nearer == "A" else obj_b_id,
                    "name": name_a if nearer == "A" else name_b,
                },
                "visibility_check": {
                    "A_frame": {"A": True, "B": True},
                    "B_frame": {
                        "A": obj_a_id in vis_b,
                        "B": obj_b_id in vis_b,
                    },
                },
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
                "move_magnitudes": MOVE_MAGNITUDES,
                "min_depth_gap": self.min_depth_gap,
                "min_obj_dist": self.min_obj_dist,
                "max_obj_dist": self.max_obj_dist,
                "max_episode_retry": self.max_episode_retry,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
                "require_unique_object_name_in_view": True,
            },
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return qa_jsonl_path, qa_json_path, success_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSSD 第3类：视差推断深度 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/HSSD/3", help="输出根目录")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data", help="Habitat 数据根目录")
    parser.add_argument("--scene_dataset_config", type=str, default=None, help="显式指定 scene_dataset_config")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min_depth_gap", type=float, default=0.5, help="最小深度差阈值")
    parser.add_argument("--max_episode_retry", type=int, default=30, help="单样本最大重试次数")
    parser.add_argument("--image_width", type=int, default=640, help="图像宽度")
    parser.add_argument("--image_height", type=int, default=480, help="图像高度")
    parser.add_argument("--hfov", type=int, default=90, help="水平视场角")
    parser.add_argument("--sensor_height", type=float, default=1.6, help="相机高度")
    parser.add_argument("--agent_radius", type=float, default=0.1, help="导航体半径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = HSSDParallaxDepthDataGenerator(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        image_width=args.image_width,
        image_height=args.image_height,
        hfov=args.hfov,
        sensor_height=args.sensor_height,
        agent_radius=args.agent_radius,
        seed=args.seed,
        max_episode_retry=args.max_episode_retry,
        min_depth_gap=args.min_depth_gap,
        scene_dataset_config=args.scene_dataset_config,
    )
    qa_jsonl_path, qa_json_path, success_count = generator.generate_data(args.num_samples)
    print("\n===== 生成完成 =====")
    print(f"成功生成: {success_count}/{args.num_samples}")
    print(f"JSONL: {qa_jsonl_path}")
    print(f"JSON : {qa_json_path}")


if __name__ == "__main__":
    main()
