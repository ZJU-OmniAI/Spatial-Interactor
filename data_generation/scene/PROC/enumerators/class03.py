from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal, VisibleObjectRecord
from .base import BaseEnumerator

EXCLUDED_DEPTH_TYPES = {
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


def _depth_visible(obj: VisibleObjectRecord) -> bool:
    area = float(obj.bbox_area_ratio or 0.0)
    cx = obj.center_x_ratio
    cy = obj.center_y_ratio
    edge = float(obj.edge_margin_ratio or 0.0)
    return (
        obj.object_type not in EXCLUDED_DEPTH_TYPES
        and
        0.010 <= area <= 0.22
        and obj.distance <= 5.0
        and cx is not None
        and cy is not None
        and 0.08 <= cx <= 0.92
        and 0.08 <= cy <= 0.92
        and edge >= 0.02
    )


class Class03Enumerator(BaseEnumerator):
    class_id = 3
    subcategories = ("default",)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class03")):
            return []
        objs = sorted(
            [obj for obj in state.visible_objects if _depth_visible(obj)],
            key=lambda item: item.distance,
        )
        if len(objs) < 2:
            return []
        proposals: List[TaskProposal] = []
        for near in objs:
            for far in objs:
                if near.object_id == far.object_id:
                    continue
                gap = far.distance - near.distance
                if gap < 0.8:
                    continue
                if near.distance > 2.6 or far.distance > 5.0:
                    continue
                cx_gap = abs(float(near.center_x_ratio or 0.5) - float(far.center_x_ratio or 0.5))
                cy_gap = abs(float(near.center_y_ratio or 0.5) - float(far.center_y_ratio or 0.5))
                if cx_gap > 0.45 or cy_gap > 0.30:
                    continue
                proposals.append(
                    self._proposal(
                        state,
                        subcat="default",
                        score=(
                            1.6 * gap
                            + float(near.bbox_area_ratio or 0.0)
                            + float(far.bbox_area_ratio or 0.0)
                            - 0.8 * cx_gap
                            - 0.4 * cy_gap
                        ),
                        payload={
                            "near_object_id": near.object_id,
                            "near_name": near.name,
                            "far_object_id": far.object_id,
                            "far_name": far.name,
                        },
                        )
                )
        return self._topk(proposals, limit=10)
