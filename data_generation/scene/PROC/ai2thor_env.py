from __future__ import annotations

from typing import Dict

from ai2thor.controller import Controller

from procthor_script_base import build_procthor_controller


def create_controller(
    scene: str,
    width: int,
    height: int,
    *,
    grid_size: float = 0.25,
    rotate_step_degrees: int = 30,
    field_of_view: int = 90,
    visibility_distance: float = 1.5,
    use_cloud_rendering: bool = False,
    render_instance_segmentation: bool = True,
) -> Controller:
    controller, _, _ = build_procthor_controller(
        scene_name=scene,
        width=width,
        height=height,
        grid_size=grid_size,
        rotate_step_degrees=rotate_step_degrees,
        field_of_view=field_of_view,
        use_cloud_rendering=use_cloud_rendering,
        visibility_distance=visibility_distance,
        render_instance_segmentation=render_instance_segmentation,
    )
    return controller
