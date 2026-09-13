from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal, VisibleObjectRecord
from .base import BaseEnumerator


DISALLOWED_SWAP_TYPES = {
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


def _swap_visible(obj: VisibleObjectRecord) -> bool:
    area = float(obj.bbox_area_ratio or 0.0)
    cx = obj.center_x_ratio
    cy = obj.center_y_ratio
    edge = float(obj.edge_margin_ratio or 0.0)
    actionable = bool(obj.pickupable or obj.moveable)
    if not actionable:
        return False
    if obj.moveable and not obj.pickupable and area > 0.11:
        return False
    return (
        actionable
        and obj.object_type not in DISALLOWED_SWAP_TYPES
        and 0.008 <= area <= 0.14
        and obj.distance <= 3.2
        and cx is not None
        and cy is not None
        and 0.10 <= cx <= 0.90
        and 0.12 <= cy <= 0.86
        and edge >= 0.015
    )


class Class07Enumerator(BaseEnumerator):
    class_id = 7
    subcategories = ("default",)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        movable = [obj for obj in state.visible_objects if _swap_visible(obj)]
        if len(movable) < 2 or not bool(state.suitability.get("class07")):
            return []
        ranked = sorted(movable, key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance), reverse=True)
        proposals: List[TaskProposal] = []
        for idx, first in enumerate(ranked):
            for jdx, second in enumerate(ranked):
                if idx >= jdx:
                    continue
                if first.object_type == second.object_type:
                    continue
                distance_gap = abs(first.distance - second.distance)
                cx_gap = abs(float(first.center_x_ratio or 0.5) - float(second.center_x_ratio or 0.5))
                cy_gap = abs(float(first.center_y_ratio or 0.5) - float(second.center_y_ratio or 0.5))
                if distance_gap > 0.55:
                    continue
                if cy_gap > 0.22:
                    continue
                if not (0.06 <= cx_gap <= 0.60):
                    continue
                score = (
                    float(first.bbox_area_ratio or 0.0)
                    + float(second.bbox_area_ratio or 0.0)
                    + 0.30 * max(0.0, 0.55 - distance_gap)
                    + 0.18 * max(0.0, 0.22 - cy_gap)
                    + 0.15 * cx_gap
                    + 0.06 * int(bool(first.pickupable))
                    + 0.06 * int(bool(second.pickupable))
                )
                proposals.append(
                    self._proposal(
                        state,
                        subcat="default",
                        score=score,
                        payload={
                            "first_object_id": first.object_id,
                            "second_object_id": second.object_id,
                        },
                    )
                )
        return self._topk(proposals, limit=16)
