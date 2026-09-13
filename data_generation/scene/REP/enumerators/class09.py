from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class09Enumerator(BaseEnumerator):
    class_id = 9
    subcategories = ("default",)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class09")):
            return []
        if len(state.visible_objects) < 2:
            return []
        ranked = sorted(
            state.visible_objects,
            key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance),
            reverse=True,
        )
        proposals: List[TaskProposal] = []
        for first_idx, anchor in enumerate(ranked):
            for second_idx, target in enumerate(ranked):
                if first_idx == second_idx:
                    continue
                proposals.append(
                    self._proposal(
                        state,
                        subcat="default",
                        score=float(anchor.bbox_area_ratio or 0.0) + float(target.bbox_area_ratio or 0.0),
                        payload={
                            "anchor_object_id": anchor.object_id,
                            "anchor_name": anchor.name,
                            "target_object_id": target.object_id,
                            "target_name": target.name,
                        },
                    )
                )
        return self._topk(proposals, limit=5)
