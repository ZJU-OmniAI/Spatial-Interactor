from __future__ import annotations

from typing import Iterable
from typing import List

from schema import StateRecord, TaskProposal, VisibleObjectRecord
from .base import BaseEnumerator

EXCLUDED_RELATION_TYPES = {
    "Bathtub",
    "BathtubBasin",
    "Blinds",
    "Burner",
    "Cabinet",
    "Ceiling",
    "Counter",
    "CounterTop",
    "Countertop",
    "Curtains",
    "DeskLamp",
    "Door",
    "Doorway",
    "Drawer",
    "Faucet",
    "Floor",
    "LightSwitch",
    "Mirror",
    "Room",
    "Shelf",
    "ShelvingUnit",
    "ShowerCurtain",
    "ShowerDoor",
    "ShowerGlass",
    "SinkBasin",
    "StandardDoor",
    "StoveBurner",
    "StoveKnob",
    "Wall",
    "Window",
}


def _strong_visible(obj: VisibleObjectRecord) -> bool:
    area = float(obj.bbox_area_ratio or 0.0)
    cx = obj.center_x_ratio
    cy = obj.center_y_ratio
    edge = float(obj.edge_margin_ratio or 0.0)
    return (
        obj.object_type not in EXCLUDED_RELATION_TYPES
        and
        0.010 <= area <= 0.22
        and obj.distance <= 4.2
        and cx is not None
        and cy is not None
        and 0.10 <= cx <= 0.90
        and 0.10 <= cy <= 0.90
        and edge >= 0.02
    )


class Class02Enumerator(BaseEnumerator):
    class_id = 2
    subcategories = ("default",)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class02")):
            return []
        objs = [
            obj
            for obj in state.visible_objects
            if _strong_visible(obj)
        ]
        if len(objs) < 2:
            return []
        proposals: List[TaskProposal] = []
        ranked = sorted(objs, key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance), reverse=True)
        for first_idx, first in enumerate(ranked):
            for second_idx, second in enumerate(ranked):
                if first_idx == second_idx:
                    continue
                first_cx = float(first.center_x_ratio or 0.5)
                second_cx = float(second.center_x_ratio or 0.5)
                first_cy = float(first.center_y_ratio or 0.5)
                second_cy = float(second.center_y_ratio or 0.5)
                horizontal_gap = abs(second_cx - first_cx)
                vertical_gap = abs(second_cy - first_cy)
                if horizontal_gap < 0.18:
                    continue
                if vertical_gap > 0.40:
                    continue
                if first_cx == second_cx:
                    continue
                scan_direction = "Right" if second_cx > first_cx else "Left"
                opposite_sides = (
                    (first_cx <= 0.46 and second_cx >= 0.54)
                    or (first_cx >= 0.54 and second_cx <= 0.46)
                )
                score = (
                    2.5 * horizontal_gap
                    + float(first.bbox_area_ratio or 0.0)
                    + float(second.bbox_area_ratio or 0.0)
                    + (0.35 if opposite_sides else 0.0)
                    - 0.20 * abs(first.distance - second.distance)
                )
                proposals.append(
                    self._proposal(
                        state,
                        subcat="default",
                        score=score,
                        payload={
                            "reference_object_id": first.object_id,
                            "reference_name": first.name,
                            "target_object_id": second.object_id,
                            "target_name": second.name,
                            "scan_direction": scan_direction,
                        },
                    )
                )
        return self._topk(proposals, limit=10)
