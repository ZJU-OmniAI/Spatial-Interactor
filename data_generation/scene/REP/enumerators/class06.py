from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class06Enumerator(BaseEnumerator):
    class_id = 6
    subcategories = ("rotate", "open_close", "toggle", "remove")

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class06")):
            return []
        buckets = {subcat: [] for subcat in self.subcategories}
        for obj in state.visible_objects:
            score = float(obj.bbox_area_ratio or 0.0) + max(0.0, 2.0 - obj.distance)
            if obj.openable:
                buckets["open_close"].append(
                    self._proposal(state, subcat="open_close", score=2.0, payload={"target_object_id": obj.object_id})
                )
            if obj.toggleable:
                buckets["toggle"].append(
                    self._proposal(state, subcat="toggle", score=2.0, payload={"target_object_id": obj.object_id})
                )
            if obj.pickupable or obj.moveable:
                buckets["remove"].append(
                    self._proposal(state, subcat="remove", score=score, payload={"target_object_id": obj.object_id})
                )
                if obj.rotation_y is not None:
                    buckets["rotate"].append(
                        self._proposal(state, subcat="rotate", score=score - 0.1, payload={"target_object_id": obj.object_id})
                    )
        results: List[TaskProposal] = []
        for subcat in self.subcategories:
            results.extend(self._topk(buckets[subcat], limit=5))
        return results
