from __future__ import annotations

import math
from typing import Dict, List, Sequence

from habitat_qa_generators import VisibleObject

from schema import StateRecord, VisibleObjectRecord


DARK_BRIGHTNESS_THRESHOLD = 32.0
MIN_GOOD_BBOX_AREA = 0.010
MIN_PROMINENT_BBOX_AREA = 0.018
MIN_VISIBLE_OBJECTS = 3
MAX_NEAR_OBJECTS_SHARE = 0.55

NON_SALIENT_OBJECT_NAMES = {
    "bathtub",
    "cabinet",
    "counter",
    "counter top",
    "curtain",
    "door",
    "drawer",
    "floor",
    "light",
    "mirror",
    "rack",
    "shelf",
    "sink",
    "toilet",
    "tv stand",
    "wall",
    "window",
}

IMMOVABLE_OBJECT_NAMES = {
    "bathtub",
    "bed",
    "bench",
    "cabinet",
    "car",
    "chair",
    "counter",
    "counter top",
    "curtain",
    "desk",
    "door",
    "drawer",
    "floor",
    "fridge",
    "home gym",
    "light",
    "mirror",
    "piano",
    "post box",
    "rack",
    "shelf",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "toilet",
    "tv",
    "tv stand",
    "wall",
    "washing machine",
    "window",
}


def meaningful_object_name(name: str) -> bool:
    clean = str(name or "").strip().lower()
    if len(clean) < 2:
        return False
    if clean == "object":
        return False
    if any(ch.isdigit() for ch in clean):
        return False
    alpha_count = sum(ch.isalpha() for ch in clean)
    return alpha_count >= max(2, len(clean.replace(" ", "")) // 5)


def salient_object_name(name: str) -> bool:
    clean = str(name or "").strip().lower()
    return meaningful_object_name(clean) and clean not in NON_SALIENT_OBJECT_NAMES


def _pseudo_bbox_area_ratio(obj: VisibleObject, image_width: int, image_height: int) -> float:
    x_ratio = float(obj.pixel_x) / float(image_width)
    y_ratio = float(obj.pixel_y) / float(image_height)
    center_bonus = max(0.55, 1.25 - 0.8 * abs(x_ratio - 0.5) - 0.5 * abs(y_ratio - 0.5))
    distance = max(float(obj.distance), 0.55)
    return max(0.004, min(0.22, (0.12 / (distance * distance)) * center_bonus))


def _rotation_y_degrees(rotation: Sequence[float] | None) -> float | None:
    if not rotation or len(rotation) != 4:
        return None
    w, x, y, z = [float(v) for v in rotation]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def visible_object_records(
    scene_objects: Sequence[Dict],
    visible_objects: Sequence[VisibleObject],
    *,
    image_width: int,
    image_height: int,
) -> List[VisibleObjectRecord]:
    by_index = {int(obj["index"]): obj for obj in scene_objects}
    records: List[VisibleObjectRecord] = []
    for obj in visible_objects:
        scene_obj = by_index.get(int(obj.index), {})
        motion_type = str(scene_obj.get("motion_type", "STATIC")).upper()
        name = str(obj.name)
        heuristic_moveable = meaningful_object_name(name) and name not in IMMOVABLE_OBJECT_NAMES
        bbox_area_ratio = _pseudo_bbox_area_ratio(obj, image_width=image_width, image_height=image_height)
        records.append(
            VisibleObjectRecord(
                object_id=str(obj.index),
                object_type=str(scene_obj.get("template_name", obj.template_name)),
                name=name,
                distance=float(obj.distance),
                pickupable=(motion_type == "DYNAMIC" or heuristic_moveable) and obj.distance <= 1.8,
                moveable=(motion_type != "STATIC") or heuristic_moveable,
                openable=False,
                toggleable=False,
                receptacle=False,
                rotation_y=_rotation_y_degrees(scene_obj.get("rotation")),
                bbox_area_ratio=bbox_area_ratio,
                center_x_ratio=float(obj.pixel_x) / float(image_width),
                center_y_ratio=float(obj.pixel_y) / float(image_height),
                edge_margin_ratio=min(
                    float(obj.pixel_x) / float(image_width),
                    float(obj.pixel_y) / float(image_height),
                    1.0 - float(obj.pixel_x) / float(image_width),
                    1.0 - float(obj.pixel_y) / float(image_height),
                ),
            )
        )
    records.sort(
        key=lambda item: (
            item.bbox_area_ratio if item.bbox_area_ratio is not None else -1.0,
            -item.distance,
        ),
        reverse=True,
    )
    return records


def state_metrics(brightness: float, visible_objects: List[VisibleObjectRecord]) -> Dict[str, float | int | bool]:
    far_count = sum(1 for obj in visible_objects if obj.distance >= 1.2)
    near_count = sum(1 for obj in visible_objects if obj.distance <= 0.75)
    meaningful_count = sum(1 for obj in visible_objects if meaningful_object_name(obj.name))
    good_bbox_count = sum(
        1 for obj in visible_objects if obj.bbox_area_ratio is not None and obj.bbox_area_ratio >= MIN_GOOD_BBOX_AREA
    )
    prominent_count = sum(
        1
        for obj in visible_objects
        if obj.bbox_area_ratio is not None and obj.bbox_area_ratio >= MIN_PROMINENT_BBOX_AREA
    )
    salient_prominent_count = sum(
        1
        for obj in visible_objects
        if salient_object_name(obj.name)
        and obj.bbox_area_ratio is not None
        and obj.bbox_area_ratio >= MIN_PROMINENT_BBOX_AREA
    )
    salient_center_count = sum(
        1
        for obj in visible_objects
        if salient_object_name(obj.name)
        and obj.bbox_area_ratio is not None
        and obj.bbox_area_ratio >= MIN_GOOD_BBOX_AREA
        and obj.center_x_ratio is not None
        and obj.center_y_ratio is not None
        and 0.18 <= obj.center_x_ratio <= 0.82
        and 0.18 <= obj.center_y_ratio <= 0.82
    )
    actionable_count = sum(1 for obj in visible_objects if obj.pickupable or obj.moveable)
    is_dark = brightness < DARK_BRIGHTNESS_THRESHOLD
    is_cramped = (
        len(visible_objects) < MIN_VISIBLE_OBJECTS
        or (len(visible_objects) > 0 and (near_count / float(len(visible_objects))) > MAX_NEAR_OBJECTS_SHARE)
        or far_count == 0
    )
    return {
        "brightness": round(float(brightness), 3),
        "visible_count": len(visible_objects),
        "far_count": far_count,
        "near_count": near_count,
        "meaningful_count": meaningful_count,
        "good_bbox_count": good_bbox_count,
        "prominent_count": prominent_count,
        "salient_prominent_count": salient_prominent_count,
        "salient_center_count": salient_center_count,
        "actionable_count": actionable_count,
        "is_dark": is_dark,
        "is_cramped": is_cramped,
        "is_good_state": (
            (not is_dark)
            and (not is_cramped)
            and good_bbox_count >= 1
            and meaningful_count >= 2
            and salient_center_count >= 1
        ),
    }


def state_suitability(visible_objects: List[VisibleObjectRecord], metrics: Dict[str, object]) -> Dict[str, object]:
    good_named = [obj for obj in visible_objects if meaningful_object_name(obj.name)]
    salient_named = [obj for obj in visible_objects if salient_object_name(obj.name)]
    good_bbox = [obj for obj in visible_objects if (obj.bbox_area_ratio or 0.0) >= MIN_GOOD_BBOX_AREA]
    swappable = [obj for obj in visible_objects if obj.pickupable or obj.moveable]
    actionable = [obj for obj in visible_objects if obj.pickupable or obj.moveable]

    depth_pair_exists = False
    ordered = sorted(good_bbox, key=lambda item: item.distance)
    for first in ordered:
        for second in ordered:
            if second.distance - first.distance >= 0.8:
                depth_pair_exists = True
                break
        if depth_pair_exists:
            break

    return {
        "class01": bool(metrics["is_good_state"]) and len(salient_named) >= 1,
        "class02": len(good_bbox) >= 2,
        "class03": depth_pair_exists,
        "class04": len(good_bbox) >= 1,
        "class05": bool(metrics["is_good_state"]),
        "class06": len(actionable) >= 1,
        "class07": len(swappable) >= 2,
        "class08": len(swappable) >= 1 or len(good_bbox) >= 1,
        "class09": len(good_named) >= 2,
        "class10": len(good_named) >= 1 and bool(metrics["is_good_state"]),
    }


def build_state_record(
    *,
    scene: str,
    scene_id: int,
    state_index: int,
    position: Sequence[float],
    yaw: int,
    horizon: int,
    scene_objects: Sequence[Dict],
    visible_objects: Sequence[VisibleObject],
    brightness: float,
    image_width: int,
    image_height: int,
) -> StateRecord:
    objects = visible_object_records(
        scene_objects,
        visible_objects,
        image_width=image_width,
        image_height=image_height,
    )
    metrics = state_metrics(brightness=brightness, visible_objects=objects)
    suitability = state_suitability(objects, metrics)
    return StateRecord(
        scene=scene,
        scene_id=int(scene_id),
        state_id=f"{scene}__state_{state_index:06d}",
        position={
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        yaw=int(yaw),
        horizon=int(horizon),
        metrics=metrics,
        visible_objects=objects,
        suitability=suitability,
    )
