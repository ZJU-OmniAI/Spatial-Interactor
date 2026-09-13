from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class04Enumerator(BaseEnumerator):
    class_id = 4
    subcategories = ("start_visible", "start_not_visible")

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class04")):
            return []
        targets = [obj for obj in state.visible_objects if (obj.bbox_area_ratio or 0.0) >= 0.01]
        if not targets:
            return []
        ranked = sorted(targets, key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance), reverse=True)
        visible_props: List[TaskProposal] = []
        not_visible_props: List[TaskProposal] = []
        for idx, target in enumerate(ranked):
            visible_props.append(
                self._proposal(
                    state,
                    subcat="start_visible",
                    score=float(target.bbox_area_ratio or 0.0),
                    payload={"target_object_id": target.object_id, "target_name": target.name},
                )
            )
            not_visible_props.append(
                self._proposal(
                    state,
                    subcat="start_not_visible",
                    score=float(target.bbox_area_ratio or 0.0) * 0.8,
                    payload={
                        "anchor_object_id": target.object_id,
                        "anchor_name": target.name,
                        "target_strategy": "search_nonvisible_target_from_same_anchor",
                    },
                )
            )
        return self._topk(visible_props, limit=5) + self._topk(not_visible_props, limit=5)
