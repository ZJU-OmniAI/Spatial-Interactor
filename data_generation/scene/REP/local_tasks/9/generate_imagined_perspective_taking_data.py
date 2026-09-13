#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReplicaCAD 第9类：想象代入视角 (Imagined Perspective Taking)

逻辑尽量对齐 /path/to/workspace/AI2THOR/9/generate_imagined_perspective_taking_data.py，
并参考 /path/to/workspace/HSSD/9/generate_imagined_perspective_taking_data.py，
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
from typing import Dict, List, Optional, Sequence

import numpy as np
import quaternion
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner


START_YAWS = list(range(0, 360, 15))
START_HORIZONS = [0]
DIRECTION_SCHEDULE = [
    "右前方",
    "左前方",
    "右后方",
    "左后方",
    "正前方",
    "正后方",
    "正右方",
    "正左方",
    "右前方",
    "左后方",
]

PREFERRED_ANCHOR_NAMES = {
    "bed",
    "chair",
    "desk",
    "fridge",
    "microwave",
    "shelf",
    "sink",
    "sofa",
    "stool",
    "table",
    "toilet",
    "tv",
    "tv stand",
    "tvstand",
}

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
    "handle",
    "knob",
    "shade",
    "switch",
)

MIN_PAIR_DISTANCE = 0.35
MAX_PAIR_DISTANCE = 5.0
MIN_TARGET_DISTANCE = 0.35
MAX_TARGET_DISTANCE = 4.6


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


def _quat_from_wxyz(values: Sequence[float]) -> quaternion.quaternion:
    return quaternion.quaternion(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def direction_from_local(x: float, z: float) -> str:
    ang = math.degrees(math.atan2(x, z))
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


class ReplicaCADImaginedPerspectiveGenerator:
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
        if normalized.replace(" ", "").isdigit():
            return False
        alpha_count = sum(ch.isalpha() for ch in normalized)
        if alpha_count < max(2, len(normalized) // 5):
            return False
        return True

    def _good_visible_object(self, obj, anchor: bool = False) -> bool:
        x_ratio = obj.pixel_x / float(self.image_width)
        y_ratio = obj.pixel_y / float(self.image_height)
        prominence = self.runner.target_prominence_score(obj)
        if anchor:
            return (
                0.18 <= x_ratio <= 0.82
                and 0.16 <= y_ratio <= 0.84
                and MIN_TARGET_DISTANCE <= obj.distance <= 3.8
                and obj.local_z > 0.25
                and prominence >= 2.8
            )
        return (
            0.15 <= x_ratio <= 0.85
            and 0.12 <= y_ratio <= 0.88
            and MIN_TARGET_DISTANCE <= obj.distance <= MAX_TARGET_DISTANCE
            and obj.local_z > 0.15
            and prominence >= 1.8
        )

    def _safe_object(self, obj) -> bool:
        return self._is_meaningful_name(obj.name)

    def _candidate_pairs(self, scene_objects: Sequence[Dict], obs: Dict) -> List[Dict]:
        visible = self.runner.unique_visible_objects(scene_objects, obs["depth"])
        scene_object_map = {int(obj["index"]): obj for obj in scene_objects}

        anchors = []
        for obj in visible:
            if not self._safe_object(obj):
                continue
            if not self._good_visible_object(obj, anchor=True):
                continue
            anchors.append(obj)

        if not anchors:
            return []

        pairs: List[Dict] = []
        for obj_a in anchors:
            scene_a = scene_object_map[obj_a.index]
            q_a = _quat_from_wxyz(scene_a["rotation"])
            rot_inv = quaternion.as_rotation_matrix(q_a.conjugate())
            pos_a = np.array(obj_a.position, dtype=np.float32)
            for obj_b in visible:
                if obj_b.index == obj_a.index or obj_b.name == obj_a.name:
                    continue
                if not self._safe_object(obj_b):
                    continue
                if not self._good_visible_object(obj_b, anchor=False):
                    continue

                pos_b = np.array(obj_b.position, dtype=np.float32)
                centered = pos_b - pos_a
                dist = float(np.linalg.norm(centered))
                if dist < MIN_PAIR_DISTANCE or dist > MAX_PAIR_DISTANCE:
                    continue

                local = rot_inv @ centered
                dir_word = direction_from_local(float(local[0]), float(local[2]))
                ang = math.degrees(math.atan2(float(local[0]), float(local[2])))
                pairs.append(
                    {
                        "object_A": obj_a,
                        "object_B": obj_b,
                        "rotation_quat_A": scene_a["rotation"],
                        "centered": centered,
                        "local": local,
                        "direction": dir_word,
                        "angle": ang,
                        "distance": dist,
                        "anchor_preferred": obj_a.name in PREFERRED_ANCHOR_NAMES,
                        "anchor_score": self.runner.target_prominence_score(obj_a),
                        "target_score": self.runner.target_prominence_score(obj_b),
                    }
                )
        return pairs

    def _pick_pair(self, scene_objects: Sequence[Dict], obs: Dict, preferred_direction: str) -> Optional[Dict]:
        pairs = self._candidate_pairs(scene_objects, obs)
        if not pairs:
            return None

        preferred_center = direction_center(preferred_direction)
        pool = [pair for pair in pairs if pair["direction"] == preferred_direction] or pairs
        pool.sort(
            key=lambda pair: (
                0 if pair["direction"] == preferred_direction else 1,
                0 if pair["anchor_preferred"] else 1,
                angle_distance(pair["angle"], preferred_center),
                abs(pair["distance"] - 1.7),
                -(pair["anchor_score"] + pair["target_score"]),
            )
        )
        return pool[0]

    def _teleport_best_view(self, scene_objects: Sequence[Dict], preferred_direction: str) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best = None
        best_score = -1e9
        for _ in range(36):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            pitch = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=pitch)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue

            pairs = self._candidate_pairs(scene_objects, obs)
            exact_count = sum(1 for pair in pairs if pair["direction"] == preferred_direction)
            preferred_anchor_count = sum(1 for pair in pairs if pair["anchor_preferred"])
            score = self.runner.view_quality_score(scene_objects, obs, require_unique=True)
            score += 5.0 * exact_count + 2.5 * preferred_anchor_count + 1.5 * len(pairs)
            if pairs:
                score += 0.8 * (pairs[0]["anchor_score"] + pairs[0]["target_score"])
            candidate = {"obs": obs, "pairs": pairs}
            if score > best_score:
                best_score = score
                best = candidate
            if exact_count > 0 and preferred_anchor_count > 0 and score >= 18.0:
                return candidate

        if best is None or not best["pairs"]:
            raise RuntimeError("未找到包含可用代入视角物体对的视角")
        return best

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        preferred_direction = DIRECTION_SCHEDULE[sample_idx % len(DIRECTION_SCHEDULE)]
        scene_data = self.ctx.load_scene(scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)

        self.runner.build(scene_path, navmesh_key_path=scene_path)
        start = self._teleport_best_view(scene_objects, preferred_direction)
        pair = self._pick_pair(scene_objects, start["obs"], preferred_direction)
        if pair is None:
            raise RuntimeError("当前视角没有合适的代入视角样本")

        obj_a = pair["object_A"]
        obj_b = pair["object_B"]
        sample_dir = self._prepare_sample_dir(sample_idx)
        frame_a = os.path.join(sample_dir, "frame_A.png")
        self._save_frame(start["obs"]["rgb"], frame_a)

        start_pose = start["obs"]["pose"]
        question = (
            f"想象你现在位于图中的 [{obj_a.name}] 位置，"
            f"并且把该物体当前朝向的正前方定义为你的正前方。"
            f"请问 [{obj_b.name}] 在你的什么方位？"
        )
        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "imagined_perspective_taking",
            "scene": os.path.basename(scene_path),
            "input": {"frame_paths": [frame_a], "frame_A": frame_a},
            "question": question,
            "answer": pair["direction"],
            "gt": {
                "preferred_direction": preferred_direction,
                "object_A": {
                    "index": int(obj_a.index),
                    "name": obj_a.name,
                    "position": {"x": float(obj_a.position[0]), "y": float(obj_a.position[1]), "z": float(obj_a.position[2])},
                    "rotation_quaternion_wxyz": [float(v) for v in pair["rotation_quat_A"]],
                    "pixel": {"x": float(obj_a.pixel_x), "y": float(obj_a.pixel_y)},
                },
                "object_B": {
                    "index": int(obj_b.index),
                    "name": obj_b.name,
                    "position": {"x": float(obj_b.position[0]), "y": float(obj_b.position[1]), "z": float(obj_b.position[2])},
                    "pixel": {"x": float(obj_b.pixel_x), "y": float(obj_b.pixel_y)},
                },
                "pos_B_centered": [float(v) for v in pair["centered"]],
                "pos_B_local": [float(v) for v in pair["local"]],
                "axis_convention": "Habitat/ReplicaCAD局部坐标：Z为forward，X为right，Y为up",
                "direction_word": pair["direction"],
                "start_state": {"text": start_pose.to_text(), "raw": asdict(start_pose)},
                "scene_path": os.path.abspath(scene_path),
                "scene_dataset_config": self.ctx.scene_dataset_config,
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
                "direction_schedule": DIRECTION_SCHEDULE,
                "min_pair_distance": MIN_PAIR_DISTANCE,
                "max_pair_distance": MAX_PAIR_DISTANCE,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
            "direction_distribution": dict(Counter(row["gt"]["direction_word"] for row in rows)),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return jsonl_path, json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReplicaCAD 第9类：想象代入视角 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/ReplicaCAD/9")
    parser.add_argument("--dataset_root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scene_dataset_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
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
    gen = ReplicaCADImaginedPerspectiveGenerator(
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
