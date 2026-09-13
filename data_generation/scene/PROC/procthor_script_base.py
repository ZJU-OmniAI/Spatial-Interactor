#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import json
import os
import random
import re
import sys
from typing import Callable, Dict, Optional, Sequence, Tuple

import ai2thor.fifo_server
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from scene_catalog import all_scene_names, house_spec_path, scene_name as normalize_scene_name


AI2THOR_ROOT = "/path/to/workspace/AI2THOR"
PROCTHOR_ROOT = "/path/to/workspace/ProcTHOR"
HOUSE_LABEL_PREFIX = "ProcTHOR_house_"


class TolerantFifoServer(ai2thor.fifo_server.FifoServer):
    """Discard stale startup frames that can precede ChangeResolution on some hosts."""

    def receive(self, timeout=None):
        while True:
            metadata, files = self._recv_message(
                timeout=self.timeout if timeout is None else timeout
            )
            if metadata is None:
                raise ValueError("no metadata received from recv_message")
            sequence_id = metadata.get("sequenceId")
            if sequence_id != self.sequence_id:
                if isinstance(sequence_id, int) and sequence_id < self.sequence_id:
                    continue
                raise ValueError(
                    f"Sequence id mismatch: {sequence_id} vs {self.sequence_id}"
                )
            return self.create_event(metadata, files)


def load_ai2thor_script(script_id: str, filename: str):
    module_name = f"_ai2thor_{script_id}_{filename.replace('.py', '')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    file_path = os.path.join(AI2THOR_ROOT, script_id, filename)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AI2THOR script: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def remap_default_output_root(args, script_id: str, ai2thor_default: str):
    if getattr(args, "output_root", None) == ai2thor_default:
        args.output_root = os.path.join(PROCTHOR_ROOT, script_id)
    return args


def random_house_label(rng: random.Random) -> str:
    scenes = all_scene_names()
    if not scenes:
        raise RuntimeError("persistent Proc manifest is empty")
    return str(rng.choice(scenes))


def extract_house_seed(scene_name: str) -> int:
    normalized = scene_name or ""
    match = re.search(r"(\d+)$", normalized)
    if not match:
        raise ValueError(f"Invalid ProcTHOR scene label: {scene_name}")
    return int(match.group(1))


def _inject_scene_name(event, scene_name: str) -> None:
    if event is not None and isinstance(getattr(event, "metadata", None), dict):
        event.metadata["sceneName"] = scene_name


def _is_black_frame(event) -> bool:
    frame = getattr(event, "frame", None)
    if frame is None:
        return False
    if getattr(frame, "size", 0) == 0:
        return False
    try:
        return int(frame.max()) == 0
    except Exception:
        return False


def patch_controller_scene_name(controller: Controller, scene_name: str, house_data: Dict) -> Controller:
    original_step = controller.step
    original_reset = controller.reset

    def _matches_current_scene(requested_scene) -> bool:
        if requested_scene is None:
            return True
        if not isinstance(requested_scene, str):
            return False
        try:
            return normalize_scene_name(requested_scene) == scene_name
        except Exception:
            return str(requested_scene) == scene_name

    def _refresh_black_frame(event, *, allow_pass_refresh: bool):
        if not allow_pass_refresh or not _is_black_frame(event):
            return event
        try:
            refresh_event = original_step(action="Pass")
        except Exception:
            return event
        return refresh_event if not _is_black_frame(refresh_event) else event

    def step_with_scene(*args, **kwargs):
        action_name = kwargs.get("action")
        if action_name is None and args and isinstance(args[0], dict):
            action_name = args[0].get("action")
        event = original_step(*args, **kwargs)
        event = _refresh_black_frame(event, allow_pass_refresh=action_name != "Pass")
        _inject_scene_name(event, scene_name)
        return event

    def reset_with_scene(*args, **kwargs):
        patched_args = list(args)
        patched_kwargs = dict(kwargs)
        if "scene" in patched_kwargs:
            requested_scene = patched_kwargs.get("scene")
            if _matches_current_scene(requested_scene):
                patched_kwargs["scene"] = house_data
        elif patched_args:
            requested_scene = patched_args[0]
            if _matches_current_scene(requested_scene):
                patched_args[0] = house_data
        else:
            patched_kwargs["scene"] = house_data
        event = original_reset(*patched_args, **patched_kwargs)
        event = _refresh_black_frame(event, allow_pass_refresh=True)
        _inject_scene_name(event, scene_name)
        return event

    controller.step = step_with_scene
    controller.reset = reset_with_scene
    _inject_scene_name(getattr(controller, "last_event", None), scene_name)
    controller._procthor_scene_name = scene_name
    controller._procthor_house_data = house_data
    return controller


def teleport_to_default_agent(controller: Controller, house_data: Dict) -> bool:
    metadata = house_data.get("metadata", {})
    agent_meta = metadata.get("agent", {}) if isinstance(metadata, dict) else {}
    position = agent_meta.get("position")
    if not isinstance(position, dict):
        return False
    rotation = agent_meta.get("rotation", {"x": 0, "y": 0, "z": 0})
    if not isinstance(rotation, dict):
        rotation = {"x": 0, "y": 0, "z": 0}
    event = controller.step(
        action="Teleport",
        position=position,
        rotation=rotation,
        horizon=int(agent_meta.get("horizon", 0)),
        standing=bool(agent_meta.get("standing", True)),
    )
    return bool(event.metadata.get("lastActionSuccess", False))


def build_procthor_controller(
    scene_name: str,
    *,
    width: int,
    height: int,
    grid_size: float,
    rotate_step_degrees: float,
    field_of_view: float,
    use_cloud_rendering: bool,
    visibility_distance: Optional[float] = None,
    render_instance_segmentation: bool = False,
    snap_to_grid: bool = False,
    quality: str = "Low",
    gpu_device: Optional[int] = None,
    server_timeout: float = 120.0,
    server_start_timeout: float = 120.0,
    split: str = "train",
) -> Tuple[Controller, int, Dict]:
    ai2thor.fifo_server.FifoServer = TolerantFifoServer
    kwargs = {
        "scene": "Procedural",
        "branch": "nanna",
        "quality": quality,
        "width": width,
        "height": height,
        "gridSize": grid_size,
        "rotateStepDegrees": rotate_step_degrees,
        "snapToGrid": snap_to_grid,
        "fieldOfView": field_of_view,
        "server_timeout": server_timeout,
        "server_start_timeout": server_start_timeout,
    }
    if visibility_distance is not None:
        kwargs["visibilityDistance"] = visibility_distance
    if render_instance_segmentation:
        kwargs["renderInstanceSegmentation"] = True
    if gpu_device is not None:
        kwargs["gpu_device"] = gpu_device
    if use_cloud_rendering:
        kwargs["platform"] = CloudRendering

    controller = Controller(**kwargs)
    scene_name = scene_name or random_house_label(random.Random())
    scene_name = normalize_scene_name(scene_name)
    house_seed = extract_house_seed(scene_name)
    spec_path = house_spec_path(scene_name)
    house_data = json.loads(spec_path.read_text(encoding="utf-8"))
    metadata = house_data.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("sourceSceneId", scene_name)
        metadata.setdefault("procthor_split", split)
    controller = patch_controller_scene_name(controller, scene_name, house_data)
    controller.reset(scene=house_data)
    teleport_to_default_agent(controller, house_data)
    _inject_scene_name(getattr(controller, "last_event", None), scene_name)
    return controller, house_seed, house_data


class ProcTHORSceneAdapterMixin:
    def _random_procthor_scene(self) -> str:
        return random_house_label(self.rng)

    def _build_procthor_controller(self, scene_name: str, **kwargs) -> Controller:
        controller, house_seed, house_data = build_procthor_controller(scene_name, **kwargs)
        self.current_house_seed = house_seed
        self.current_house_data = house_data
        return controller


def pick_reachable_view(
    controller: Controller,
    rng: random.Random,
    *,
    yaws: Sequence[int],
    horizons: Sequence[int],
    max_trials: int,
    score_fn: Callable,
    accept_fn: Callable,
) -> bool:
    event = controller.step(action="GetReachablePositions")
    reachable = event.metadata.get("actionReturn", [])
    if not reachable:
        return False

    successful_candidates = []
    for _ in range(max_trials):
        pos = rng.choice(reachable)
        yaw = rng.choice(list(yaws))
        horizon = rng.choice(list(horizons))
        tp = controller.step(
            action="Teleport",
            position=pos,
            rotation={"x": 0, "y": yaw, "z": 0},
            horizon=horizon,
            standing=True,
        )
        if not bool(tp.metadata.get("lastActionSuccess", False)):
            continue
        score = score_fn(tp)
        successful_candidates.append((score, pos, yaw, horizon))
        if accept_fn(score):
            return True

    successful_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, pos, yaw, horizon in successful_candidates:
        for _ in range(2):
            tp = controller.step(
                action="Teleport",
                position=pos,
                rotation={"x": 0, "y": yaw, "z": 0},
                horizon=horizon,
                standing=True,
            )
            if bool(tp.metadata.get("lastActionSuccess", False)):
                return True
    return False
