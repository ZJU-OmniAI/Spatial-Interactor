from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from schema import StateRecord, VisibleObjectRecord


DARK_BRIGHTNESS_THRESHOLD = 42.0
MIN_GOOD_BBOX_AREA = 0.010
MIN_PROMINENT_BBOX_AREA = 0.018
MIN_VISIBLE_OBJECTS = 3
MAX_NEAR_OBJECTS_SHARE = 0.55

BAD_NAME_SUBSTRINGS = {
    "object",
    "item",
    "mesh",
    "prop",
    "instance",
}

NON_SALIENT_OBJECT_NAMES = {
    "ceiling",
    "counter",
    "counter top",
    "countertop",
    "door",
    "floor",
    "room",
    "wall",
    "window",
}


def camel_to_words(text: str) -> str:
    if not text:
        return "object"
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = words.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", words)


def meaningful_object_name(name: str) -> bool:
    clean = camel_to_words(name)
    if any(ch.isdigit() for ch in clean):
        return False
    if len(clean) < 3:
        return False
    return not any(part in clean for part in BAD_NAME_SUBSTRINGS)


def salient_object_name(name: str) -> bool:
    clean = camel_to_words(name)
    if not meaningful_object_name(clean):
        return False
    return clean not in NON_SALIENT_OBJECT_NAMES


def bbox_map(event) -> Dict[str, Tuple[float, float, float, float]]:
    det = event.instance_detections2D or {}
    return {
        object_id: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for object_id, box in det.items()
    }


def bbox_quality(
    bbox: Optional[Tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < 1.0 or height < 1.0:
        return None
    area_ratio = (width * height) / float(image_width * image_height)
    return {
        "area_ratio": area_ratio,
        "center_x_ratio": ((x1 + x2) * 0.5) / float(image_width),
        "center_y_ratio": ((y1 + y2) * 0.5) / float(image_height),
        "edge_margin_ratio": min(
            x1 / float(image_width),
            y1 / float(image_height),
            (image_width - x2) / float(image_width),
            (image_height - y2) / float(image_height),
        ),
    }


def frame_brightness(frame: np.ndarray) -> float:
    return float(np.mean(frame.astype(np.float32)))


def visible_object_records(event, image_width: int, image_height: int) -> List[VisibleObjectRecord]:
    boxes = bbox_map(event)
    records: List[VisibleObjectRecord] = []
    for obj in event.metadata.get("objects", []):
        if not obj.get("visible", False):
            continue
        object_type = str(obj.get("objectType", "Unknown"))
        bbox_info = bbox_quality(boxes.get(obj.get("objectId")), image_width=image_width, image_height=image_height)
        records.append(
            VisibleObjectRecord(
                object_id=str(obj.get("objectId")),
                object_type=object_type,
                name=camel_to_words(object_type),
                distance=float(obj.get("distance", 999.0)),
                pickupable=bool(obj.get("pickupable", False)),
                moveable=bool(obj.get("moveable", False)),
                openable=bool(obj.get("openable", False)),
                toggleable=bool(obj.get("toggleable", False)),
                receptacle=bool(obj.get("receptacle", False)),
                rotation_y=float((obj.get("rotation") or {}).get("y")) if obj.get("rotation") else None,
                bbox_area_ratio=None if bbox_info is None else float(bbox_info["area_ratio"]),
                center_x_ratio=None if bbox_info is None else float(bbox_info["center_x_ratio"]),
                center_y_ratio=None if bbox_info is None else float(bbox_info["center_y_ratio"]),
                edge_margin_ratio=None if bbox_info is None else float(bbox_info["edge_margin_ratio"]),
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


def state_metrics(frame: np.ndarray, visible_objects: List[VisibleObjectRecord]) -> Dict[str, float | int | bool]:
    brightness = frame_brightness(frame)
    far_count = sum(1 for obj in visible_objects if obj.distance >= 1.2)
    near_count = sum(1 for obj in visible_objects if obj.distance <= 0.65)
    meaningful_count = sum(1 for obj in visible_objects if meaningful_object_name(obj.name))
    good_bbox_count = sum(
        1
        for obj in visible_objects
        if obj.bbox_area_ratio is not None and obj.bbox_area_ratio >= MIN_GOOD_BBOX_AREA
    )
    prominent_count = sum(
        1
        for obj in visible_objects
        if obj.bbox_area_ratio is not None and obj.bbox_area_ratio >= MIN_PROMINENT_BBOX_AREA
    )
    salient_prominent_count = sum(
        1
        for obj in visible_objects
        if salient_object_name(obj.name) and obj.bbox_area_ratio is not None and obj.bbox_area_ratio >= MIN_PROMINENT_BBOX_AREA
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
    actionable_count = sum(
        1 for obj in visible_objects if obj.pickupable or obj.moveable or obj.openable or obj.toggleable
    )
    is_dark = brightness < DARK_BRIGHTNESS_THRESHOLD
    is_cramped = (
        len(visible_objects) < MIN_VISIBLE_OBJECTS
        or (len(visible_objects) > 0 and (near_count / float(len(visible_objects))) > MAX_NEAR_OBJECTS_SHARE)
        or far_count == 0
    )
    return {
        "brightness": round(brightness, 3),
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
            and good_bbox_count >= 2
            and meaningful_count >= 2
            and salient_prominent_count >= 1
            and salient_center_count >= 1
        ),
    }


def state_suitability(visible_objects: List[VisibleObjectRecord], metrics: Dict[str, object]) -> Dict[str, object]:
    good_named = [obj for obj in visible_objects if meaningful_object_name(obj.name)]
    salient_named = [obj for obj in visible_objects if salient_object_name(obj.name)]
    good_bbox = [obj for obj in visible_objects if (obj.bbox_area_ratio or 0.0) >= MIN_GOOD_BBOX_AREA]
    swappable = [obj for obj in visible_objects if obj.pickupable or obj.moveable]
    actionable = [
        obj for obj in visible_objects if obj.pickupable or obj.moveable or obj.openable or obj.toggleable
    ]
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
    scene: str,
    scene_id: int,
    state_index: int,
    position: Dict[str, float],
    yaw: int,
    horizon: int,
    event,
    image_width: int,
    image_height: int,
) -> StateRecord:
    objects = visible_object_records(event, image_width=image_width, image_height=image_height)
    metrics = state_metrics(event.frame, objects)
    suitability = state_suitability(objects, metrics)
    return StateRecord(
        scene=scene,
        scene_id=scene_id,
        state_id=f"{scene}__state_{state_index:06d}",
        position={
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        },
        yaw=int(yaw),
        horizon=int(horizon),
        metrics=metrics,
        visible_objects=objects,
        suitability=suitability,
    )
