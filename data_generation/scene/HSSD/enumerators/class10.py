from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class10Enumerator(BaseEnumerator):
    class_id = 10
    subcategories = ("single_visibility", "two_frame_direction", "single_direction", "trend_disappear")

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class10")):
            return []
        ranked = sorted(
            state.visible_objects,
            key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance),
            reverse=True,
        )
        if not ranked:
            return []
        results: List[TaskProposal] = []
        for subcat in self.subcategories:
            proposals = [
                self._proposal(
                    state,
                    subcat=subcat,
                    score=float(target.bbox_area_ratio or 0.0) + max(0.0, 2.0 - target.distance),
                    payload={"target_object_id": target.object_id},
                )
                for target in ranked
            ]
            results.extend(self._topk(proposals, limit=5))
        return results
