from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, "/path/to/workspace")

from habitat_qa_generators import HabitatDatasetContext, HabitatSceneRunner

from scene_catalog import SCENE_DATASET_CONFIG, scene_path


DATASET_ROOT = "/path/to/workspace/habitat_data"


def create_context(
    *,
    dataset_root: str = DATASET_ROOT,
    width: int,
    height: int,
    seed: int = 42,
) -> HabitatDatasetContext:
    ctx = HabitatDatasetContext(
        dataset="replicacad",
        dataset_root=dataset_root,
        image_width=width,
        image_height=height,
        seed=seed,
    )
    ctx.scene_dataset_config = str(SCENE_DATASET_CONFIG)
    return ctx


def create_runner(
    *,
    width: int,
    height: int,
    field_of_view: int = 90,
    sensor_height: float = 1.6,
    agent_radius: float = 0.1,
    seed: int = 42,
    dataset_root: str = DATASET_ROOT,
    enable_color_sensor: bool = True,
    enable_depth_sensor: bool = True,
) -> HabitatSceneRunner:
    ctx = create_context(dataset_root=dataset_root, width=width, height=height, seed=seed)
    return HabitatSceneRunner(
        ctx=ctx,
        image_width=width,
        image_height=height,
        hfov=field_of_view,
        sensor_height=sensor_height,
        agent_radius=agent_radius,
        seed=seed,
        enable_color_sensor=enable_color_sensor,
        enable_depth_sensor=enable_depth_sensor,
    )


def load_scene(
    runner: HabitatSceneRunner,
    scene_name: str,
) -> Tuple[str, Dict, List[Dict]]:
    scene_file = scene_path(scene_name)
    scene_data = runner.ctx.load_scene(scene_file)
    scene_objects = runner.ctx.scene_objects(scene_data)
    runner.build(scene_file, navmesh_key_path=scene_file)
    return scene_file, scene_data, scene_objects
