from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class02Enumerator(BaseEnumerator):
    class_id = 2
    subcategories = ("default",)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class02")):
            return []
        objs = [
            obj
            for obj in state.visible_objects
            if (obj.bbox_area_ratio or 0.0) >= 0.01
        ]
        if len(objs) < 2:
            return []
        proposals: List[TaskProposal] = []
        ranked = sorted(objs, key=lambda obj: ((obj.bbox_area_ratio or 0.0), -obj.distance), reverse=True)
        for first_idx, first in enumerate(ranked):
            for second_idx, second in enumerate(ranked):
                if first_idx == second_idx:
                    continue
                score = float(first.bbox_area_ratio or 0.0) + float(second.bbox_area_ratio or 0.0)
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
                        },
                    )
                )
        return self._topk(proposals, limit=5)
