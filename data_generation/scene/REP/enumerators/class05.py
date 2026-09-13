from __future__ import annotations

from typing import List

from schema import StateRecord, TaskProposal
from .base import BaseEnumerator


class Class05Enumerator(BaseEnumerator):
    class_id = 5
    subcategories = ("translation", "rotation")

    TRANSLATIONS = (0.25, 0.35, 0.45, 0.55, 0.65)
    ROTATIONS = (30, 45, 60, 75, 90)

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class05")):
            return []
        translation_props = [
            self._proposal(
                state,
                subcat="translation",
                score=1.0 + 0.1 * idx,
                payload={"action_family": "translation", "move_magnitude": value},
            )
            for idx, value in enumerate(self.TRANSLATIONS)
        ]
        rotation_props = [
            self._proposal(
                state,
                subcat="rotation",
                score=1.0 + 0.1 * idx,
                payload={"action_family": "rotation", "degrees": value},
            )
            for idx, value in enumerate(self.ROTATIONS)
        ]
        return self._topk(translation_props, limit=5) + self._topk(rotation_props, limit=5)
