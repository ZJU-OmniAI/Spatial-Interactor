from __future__ import annotations

from typing import Dict, Optional

from ai2thor.controller import Controller


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
    kwargs: Dict[str, object] = {
        "scene": scene,
        "width": width,
        "height": height,
        "gridSize": grid_size,
        "snapToGrid": False,
        "rotateStepDegrees": rotate_step_degrees,
        "fieldOfView": field_of_view,
        "visibilityDistance": visibility_distance,
        "renderInstanceSegmentation": render_instance_segmentation,
    }
    if use_cloud_rendering:
        kwargs["platform"] = "CloudRendering"
    return Controller(**kwargs)

