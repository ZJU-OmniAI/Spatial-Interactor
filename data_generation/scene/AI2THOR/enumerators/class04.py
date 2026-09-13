from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal, VisibleObjectRecord
from .base import BaseEnumerator

EXCLUDED_ROUTE_TYPES = {
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


def _route_target_visible(obj: VisibleObjectRecord) -> bool:
    area = float(obj.bbox_area_ratio or 0.0)
    cx = obj.center_x_ratio
    cy = obj.center_y_ratio
    edge = float(obj.edge_margin_ratio or 0.0)
    return (
        obj.object_type not in EXCLUDED_ROUTE_TYPES
        and
        0.012 <= area <= 0.22
        and 1.2 <= obj.distance <= 4.2
        and cx is not None
        and cy is not None
        and 0.18 <= cx <= 0.82
        and 0.16 <= cy <= 0.84
        and edge >= 0.03
    )


class Class04Enumerator(BaseEnumerator):
    class_id = 4
    subcategories = ("start_visible", "start_not_visible")

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class04")):
            return []
        targets = [obj for obj in state.visible_objects if _route_target_visible(obj)]
        if not targets:
            return []
        ranked = sorted(targets, key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance), reverse=True)
        visible_props: List[TaskProposal] = []
        not_visible_props: List[TaskProposal] = []
        for idx, target in enumerate(ranked):
            cx = float(target.center_x_ratio or 0.5)
            center_bonus = max(0.0, 1.0 - abs(cx - 0.5) * 2.0)
            route_score = float(target.bbox_area_ratio or 0.0) + 0.18 * min(target.distance, 3.5) + 0.15 * center_bonus
            visible_props.append(
                self._proposal(
                    state,
                    subcat="start_visible",
                    score=route_score,
                    payload={"target_object_id": target.object_id, "target_name": target.name},
                )
            )
            not_visible_props.append(
                self._proposal(
                    state,
                    subcat="start_not_visible",
                    score=route_score * 0.85,
                    payload={
                        "anchor_object_id": target.object_id,
                        "anchor_name": target.name,
                        "target_strategy": "search_nonvisible_target_from_same_anchor",
                    },
                )
            )
        return self._topk(visible_props, limit=10) + self._topk(not_visible_props, limit=10)
