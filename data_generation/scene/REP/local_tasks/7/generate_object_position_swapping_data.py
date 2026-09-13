#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReplicaCAD 第7类：交换两个物体位置 (Object Position Swapping)

逻辑对齐 /path/to/workspace/AI2THOR/7/generate_object_position_swapping_data.py，
并参考 /path/to/workspace/HSSD/7/generate_object_position_swapping_data.py，
仅把底层执行替换为 Habitat/ReplicaCAD。
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
MAX_TARGET_DISTANCE = 3.2
MIN_PAIR_WORLD_DISTANCE = 0.18
MAX_PAIR_WORLD_DISTANCE = 1.20
MAX_PAIR_Y_GAP = 0.12
MIN_PAIR_SCREEN_CENTER_DISTANCE = 0.06
MIN_SCREEN_SHIFT = 0.08
MAX_PAIR_ATTEMPTS_PER_VIEW = 20

EXCLUDED_OBJECT_NAMES = {
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


class ReplicaCADObjectPositionSwappingGenerator:
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

    @staticmethod
    def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _dist2_xz(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[2]) - float(b[2])) ** 2)

    @staticmethod
    def _position_to_dict(position: Sequence[float]) -> Dict[str, float]:
        return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}

    @staticmethod
    def _screen_center_distance(a, b) -> float:
        ax, ay = float(a.pixel_x), float(a.pixel_y)
        bx, by = float(b.pixel_x), float(b.pixel_y)
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

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
                0.18 <= x_ratio <= 0.82
                and 0.14 <= y_ratio <= 0.86
                and 0.60 <= obj.distance <= MAX_TARGET_DISTANCE
                and obj.local_z > 0.35
                and prominence >= 1.9
            )
        return (
            0.24 <= x_ratio <= 0.76
            and 0.18 <= y_ratio <= 0.82
            and 0.75 <= obj.distance <= MAX_TARGET_DISTANCE
            and obj.local_z > 0.45
            and prominence >= 2.3
        )

    def _candidate_objects(self, scene_objects: Sequence[Dict], obs: Dict) -> List:
        visible = self.runner.unique_visible_objects(scene_objects, obs["depth"])
        candidates = []
        for obj in visible:
            if obj.name in EXCLUDED_OBJECT_NAMES:
                continue
            if not self._is_meaningful_name(obj.name):
                continue
            if not self._good_target_view(obj):
                continue
            candidates.append(obj)
        candidates.sort(key=lambda obj: self.runner.target_prominence_score(obj), reverse=True)
        return candidates

    def _teleport_best_view(self, scene_objects: Sequence[Dict]) -> Dict:
        if not self.runner.sim.pathfinder.is_loaded:
            raise RuntimeError("pathfinder 未加载")

        best = None
        best_score = -1e9
        for _ in range(40):
            body_pos = np.array(self.runner.sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            yaw = self.rng.choice(START_YAWS)
            pitch = self.rng.choice(START_HORIZONS)
            obs = self.runner.set_agent_state(position=body_pos, yaw=yaw, pitch=pitch)
            obs["pose"] = self._to_pose_state(obs["pose"])
            if self.runner.has_excessive_side_walls(obs):
                continue
            candidates = self._candidate_objects(scene_objects, obs)
            pair_count = len(self._pair_candidates(candidates))
            score = self.runner.view_quality_score(scene_objects, obs, require_unique=True)
            if candidates:
                score += self.runner.target_prominence_score(candidates[0])
            score += 4.0 * pair_count
            if score > best_score:
                best_score = score
                best = {"obs": obs, "candidates": candidates}
            if pair_count >= 1 and candidates and score >= 18.0:
                return best

        if best is None:
            raise RuntimeError("未找到包含可交换物体对的合适视角")
        return best

    def _pair_candidates(self, candidates: Sequence) -> List[Tuple[object, object, Tuple]]:
        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                if a.name == b.name:
                    continue
                if abs(float(a.position[1]) - float(b.position[1])) > MAX_PAIR_Y_GAP:
                    continue
                pair_world_distance = self._dist2_xz(a.position, b.position)
                if not (MIN_PAIR_WORLD_DISTANCE <= pair_world_distance <= MAX_PAIR_WORLD_DISTANCE):
                    continue
                screen_dist = self._screen_center_distance(a, b) / float(max(self.image_width, self.image_height))
                if screen_dist < MIN_PAIR_SCREEN_CENTER_DISTANCE:
                    continue
                score = (
                    abs(pair_world_distance - 0.45),
                    -(self.runner.target_prominence_score(a) + self.runner.target_prominence_score(b)),
                    float(a.distance) + float(b.distance),
                )
                pairs.append((a, b, score))
        pairs.sort(key=lambda item: item[2])
        return pairs

    @staticmethod
    def _visible_object_map(visible_objects: Sequence) -> Dict[str, object]:
        return {obj.name: obj for obj in visible_objects}

    def _make_scene_copy_swap_positions(self, scene_data: Dict, idx_a: int, idx_b: int) -> Dict:
        new_scene = copy.deepcopy(scene_data)
        obj_a = new_scene["object_instances"][idx_a]
        obj_b = new_scene["object_instances"][idx_b]
        obj_a["translation"], obj_b["translation"] = obj_b["translation"], obj_a["translation"]
        return new_scene

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
        }
        if visible_obj is not None:
            payload.update(
                {
                    "visible": True,
                    "distance": float(visible_obj.distance),
                    "pixel": {"x": float(visible_obj.pixel_x), "y": float(visible_obj.pixel_y)},
                }
            )
        else:
            payload["visible"] = False
        return payload

    def _run_one_sample(self, sample_idx: int, scene_path: str) -> Dict:
        scene_data = self.ctx.load_scene(scene_path)
        scene_objects = self.ctx.scene_objects(scene_data)

        self.runner.build(scene_path, navmesh_key_path=scene_path)
        start = self._teleport_best_view(scene_objects)
        start_obs = start["obs"]
        candidates = start["candidates"]
        pairs = self._pair_candidates(candidates)
        if not pairs:
            raise RuntimeError("当前视角没有满足约束的交换物体对")

        chosen = None
        chosen_after_obs = None
        modified_scene_path = None
        scene_objects_after = None
        for obj_a, obj_b, _ in pairs[:MAX_PAIR_ATTEMPTS_PER_VIEW]:
            modified_scene = self._make_scene_copy_swap_positions(scene_data, obj_a.index, obj_b.index)
            current_modified_scene_path = self.ctx.save_temp_scene(
                modified_scene, scene_path, task_id=7, sample_idx=sample_idx
            )
            current_scene_objects_after = self.ctx.scene_objects(modified_scene)

            self.runner.build(current_modified_scene_path, navmesh_key_path=scene_path)
            pose = start_obs["pose"]
            current_after_obs = self.runner.set_agent_state(
                position=self._body_position_from_pose(pose),
                yaw=pose.yaw,
                pitch=pose.pitch,
            )
            current_after_obs["pose"] = self._to_pose_state(current_after_obs["pose"])

            after_visible = self._visible_object_map(
                self.runner.unique_visible_objects(current_scene_objects_after, current_after_obs["depth"])
            )
            a_after = after_visible.get(obj_a.name)
            b_after = after_visible.get(obj_b.name)
            if a_after is None or b_after is None:
                continue
            if not self._good_target_view(a_after, relaxed=True) or not self._good_target_view(b_after, relaxed=True):
                continue

            shift_a = self._screen_center_distance(obj_a, a_after) / float(max(self.image_width, self.image_height))
            shift_b = self._screen_center_distance(obj_b, b_after) / float(max(self.image_width, self.image_height))
            frame_diff = _frame_diff_score(start_obs["rgb"], current_after_obs["rgb"])
            if shift_a < MIN_SCREEN_SHIFT or shift_b < MIN_SCREEN_SHIFT:
                continue
            if frame_diff < 0.8:
                continue

            chosen = {
                "obj_a_before": self._object_snapshot(obj_a, scene_objects[obj_a.index]),
                "obj_b_before": self._object_snapshot(obj_b, scene_objects[obj_b.index]),
                "obj_a_after": self._object_snapshot(a_after, current_scene_objects_after[obj_a.index]),
                "obj_b_after": self._object_snapshot(b_after, current_scene_objects_after[obj_b.index]),
                "screen_center_shift_a": shift_a,
                "screen_center_shift_b": shift_b,
                "frame_diff": frame_diff,
            }
            chosen_after_obs = current_after_obs
            modified_scene_path = current_modified_scene_path
            scene_objects_after = current_scene_objects_after
            break

        if chosen is None or chosen_after_obs is None or modified_scene_path is None or scene_objects_after is None:
            raise RuntimeError("尝试多个候选物体对后，仍未得到稳定交换样本")

        sample_dir = self._prepare_sample_dir(sample_idx)
        frame_a = os.path.join(sample_dir, "frame_000_A.png")
        frame_b = os.path.join(sample_dir, "frame_001_B.png")
        self._save_frame(start_obs["rgb"], frame_a)
        self._save_frame(chosen_after_obs["rgb"], frame_b)

        start_pose = start_obs["pose"]
        end_pose = chosen_after_obs["pose"]
        return {
            "sample_id": f"sample_{sample_idx:05d}",
            "task_type": "object_position_swapping",
            "scene": os.path.basename(scene_path),
            "input": {"frame_paths": [frame_a, frame_b], "frame_A": frame_a, "frame_B": frame_b},
            "question": "图A是原始场景，图B是交换两个物体位置后的场景。场景中哪两个物体交换了位置？",
            "answer": f"【{chosen['obj_a_before']['name']}】和【{chosen['obj_b_before']['name']}】交换了位置。",
            "gt": {
                "swap_pair": [
                    {
                        "before": chosen["obj_a_before"],
                        "after": chosen["obj_a_after"],
                        "answer_name": chosen["obj_a_before"]["name"],
                    },
                    {
                        "before": chosen["obj_b_before"],
                        "after": chosen["obj_b_after"],
                        "answer_name": chosen["obj_b_before"]["name"],
                    },
                ],
                "swap_metrics": {
                    "screen_center_shift_a": chosen["screen_center_shift_a"],
                    "screen_center_shift_b": chosen["screen_center_shift_b"],
                    "frame_diff": chosen["frame_diff"],
                },
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
                "max_target_distance": MAX_TARGET_DISTANCE,
                "min_pair_world_distance": MIN_PAIR_WORLD_DISTANCE,
                "max_pair_world_distance": MAX_PAIR_WORLD_DISTANCE,
                "scene_dataset_config": self.ctx.scene_dataset_config,
                "scene_count": len(self.ctx.scene_paths),
            },
            "scene_distribution": dict(Counter(row["scene"] for row in rows)),
        }
        with open(os.path.join(self.meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.runner.shutdown()
        return jsonl_path, json_path, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReplicaCAD 第7类：交换两个物体位置 数据生成器")
    parser.add_argument("--output_root", type=str, default="/path/to/workspace/ReplicaCAD/7")
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
    gen = ReplicaCADObjectPositionSwappingGenerator(
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
