from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Dict, Iterable, List

from schema import StateRecord, TaskProposal


class BaseEnumerator(ABC):
    class_id: int
    subcategories: tuple[str, ...]

    @abstractmethod
    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        raise NotImplementedError

    def _proposal(
        self,
        state: StateRecord,
        *,
        subcat: str,
        score: float,
        payload: Dict[str, object],
    ) -> TaskProposal:
        key = f"{state.state_id}|{self.class_id}|{subcat}|{payload}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return TaskProposal(
            proposal_id=f"{state.state_id}__class{self.class_id:02d}__{subcat}__{digest}",
            scene=state.scene,
            state_id=state.state_id,
            class_id=self.class_id,
            subcat=subcat,
            score=score,
            payload=payload,
        )

    def _topk(self, items: List[TaskProposal], limit: int = 5) -> List[TaskProposal]:
        items.sort(key=lambda item: item.score, reverse=True)
        return items[:limit]
