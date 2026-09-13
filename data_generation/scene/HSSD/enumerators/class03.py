from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class03Enumerator(BaseEnumerator):
    class_id = 3
    subcategories = ("default",)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class03")):
            return []
        objs = sorted(
            [obj for obj in state.visible_objects if (obj.bbox_area_ratio or 0.0) >= 0.01],
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
                proposals.append(
                    self._proposal(
                        state,
                        subcat="default",
                        score=gap + float(near.bbox_area_ratio or 0.0) + float(far.bbox_area_ratio or 0.0),
                        payload={
                            "near_object_id": near.object_id,
                            "near_name": near.name,
                            "far_object_id": far.object_id,
                            "far_name": far.name,
                        },
                    )
                )
        return self._topk(proposals, limit=5)
