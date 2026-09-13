from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal, VisibleObjectRecord
from .base import BaseEnumerator


DISALLOWED_MOTION_TYPES = {
    "ArmChair",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Cabinet",
    "Chair",
    "CoffeeTable",
    "CounterTop",
    "Curtains",
    "Desk",
    "DiningTable",
    "Door",
    "Drawer",
    "Dresser",
    "Floor",
    "Fridge",
    "GarbageCan",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Safe",
    "Shelf",
    "ShelvingUnit",
    "SideTable",
    "Sink",
    "SinkBasin",
    "Sofa",
    "StoveBurner",
    "TVStand",
    "Television",
    "Toilet",
    "Wall",
    "Window",
}

ALLOWED_NON_PICKUPABLE_MOTION_TYPES = {
    "DeskLamp",
    "Statue",
    "Vase",
}


def _motion_candidate(obj: VisibleObjectRecord) -> bool:
    area = float(obj.bbox_area_ratio or 0.0)
    cx = obj.center_x_ratio
    cy = obj.center_y_ratio
    edge = float(obj.edge_margin_ratio or 0.0)
    return (
        (obj.pickupable or obj.object_type in ALLOWED_NON_PICKUPABLE_MOTION_TYPES)
        and obj.object_type not in DISALLOWED_MOTION_TYPES
        and 0.008 <= area <= 0.12
        and 0.7 <= obj.distance <= 3.2
        and cx is not None
        and cy is not None
        and 0.12 <= cx <= 0.88
        and 0.12 <= cy <= 0.86
        and edge >= 0.015
    )


def _neighbor_density(target: VisibleObjectRecord, visible_objects: List[VisibleObjectRecord]) -> float:
    cx = float(target.center_x_ratio or 0.5)
    cy = float(target.center_y_ratio or 0.5)
    score = 0.0
    for other in visible_objects:
        if other.object_id == target.object_id:
            continue
        if other.center_x_ratio is None or other.center_y_ratio is None:
            continue
        dx = abs(float(other.center_x_ratio) - cx)
        dy = abs(float(other.center_y_ratio) - cy)
        if dx <= 0.18 and dy <= 0.18:
            score += 1.0
    return score


class Class08Enumerator(BaseEnumerator):
    class_id = 8
    subcategories = ("distance_change", "occlusion_change")

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class08")):
            return []
        ranked = sorted(
            [obj for obj in state.visible_objects if _motion_candidate(obj)],
            key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance),
            reverse=True,
        )
        if not ranked:
            return []
        distance_props = [
            self._proposal(
                state,
                subcat="distance_change",
                score=(
                    float(target.bbox_area_ratio or 0.0)
                    + max(0.0, 2.6 - target.distance)
                    + 0.25 * max(0.0, 0.10 - abs(float(target.center_x_ratio or 0.5) - 0.5))
                ),
                payload={"target_object_id": target.object_id, "target_name": target.name},
            )
            for target in ranked
        ]
        occlusion_props = [
            self._proposal(
                state,
                subcat="occlusion_change",
                score=(
                    float(target.bbox_area_ratio or 0.0)
                    + 0.35 * _neighbor_density(target, ranked)
                    + 0.15 * max(0.0, 2.6 - target.distance)
                ),
                payload={"target_object_id": target.object_id, "target_name": target.name},
            )
            for target in ranked
        ]
        return self._topk(distance_props, limit=12) + self._topk(occlusion_props, limit=18)
