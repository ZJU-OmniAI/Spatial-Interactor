from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class VisibleObjectRecord:
    object_id: str
    object_type: str
    name: str
    distance: float
    pickupable: bool
    moveable: bool
    openable: bool
    toggleable: bool
    receptacle: bool
    rotation_y: float | None
    bbox_area_ratio: float | None
    center_x_ratio: float | None
    center_y_ratio: float | None
    edge_margin_ratio: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateRecord:
    scene: str
    scene_id: int
    state_id: str
    position: Dict[str, float]
    yaw: int
    horizon: int
    metrics: Dict[str, Any]
    visible_objects: List[VisibleObjectRecord] = field(default_factory=list)
    suitability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["visible_objects"] = [obj.to_dict() for obj in self.visible_objects]
        return data


@dataclass
class TaskProposal:
    proposal_id: str
    scene: str
    state_id: str
    class_id: int
    subcat: str
    score: float
    payload: Dict[str, Any]
    source: str = "enumerated"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
