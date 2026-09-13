from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from schema import StateRecord, VisibleObjectRecord
from state_quality import salient_object_name


POSITION_BUCKET = 0.5
MAX_POSITION_DELTA = 0.35
MAX_YAW_DELTA = 30
MAX_CENTER_DELTA = 0.15
MAX_AREA_DELTA = 0.05
MIN_OBJECT_JACCARD = 0.6


@dataclass
class _StateSketch:
    state_id: str
    x: float
    z: float
    yaw: int
    horizon: int
    focus_key: str
    focus_name: str
    focus_center_x: float
    focus_center_y: float
    focus_area: float
    salient_object_ids: frozenset[str]


def _circular_yaw_delta(first: int, second: int) -> int:
    delta = abs(int(first) - int(second)) % 360
    return min(delta, 360 - delta)


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    if not first and not second:
        return 1.0
    union = first.union(second)
    if not union:
        return 0.0
    return len(first.intersection(second)) / float(len(union))


def _focus_rank(obj: VisibleObjectRecord) -> float:
    area = obj.bbox_area_ratio or 0.0
    center_x = obj.center_x_ratio if obj.center_x_ratio is not None else 0.5
    center_y = obj.center_y_ratio if obj.center_y_ratio is not None else 0.5
    center_bias = abs(center_x - 0.5) + abs(center_y - 0.5)
    actionable_bonus = 0.5 if (obj.pickupable or obj.moveable or obj.openable or obj.toggleable) else 0.0
    salient_bonus = 0.8 if salient_object_name(obj.name) else 0.0
    return area * 4.0 + salient_bonus + actionable_bonus - center_bias - max(0.0, obj.distance - 1.5) * 0.2


def _focus_object(visible_objects: Sequence[VisibleObjectRecord]) -> VisibleObjectRecord | None:
    ranked = [obj for obj in visible_objects if obj.bbox_area_ratio is not None]
    if not ranked:
        return None
    ranked.sort(key=_focus_rank, reverse=True)
    return ranked[0]


def _salient_object_ids(visible_objects: Sequence[VisibleObjectRecord]) -> frozenset[str]:
    selected = [
        obj.object_id
        for obj in visible_objects
        if salient_object_name(obj.name) and (obj.bbox_area_ratio or 0.0) >= 0.01
    ]
    if not selected:
        selected = [obj.object_id for obj in visible_objects[:4]]
    return frozenset(selected[:6])


def _build_sketch(record: StateRecord) -> _StateSketch | None:
    focus = _focus_object(record.visible_objects)
    if focus is None or focus.center_x_ratio is None or focus.center_y_ratio is None or focus.bbox_area_ratio is None:
        return None
    return _StateSketch(
        state_id=record.state_id,
        x=float(record.position["x"]),
        z=float(record.position["z"]),
        yaw=int(record.yaw),
        horizon=int(record.horizon),
        focus_key=str(focus.object_id or focus.name),
        focus_name=str(focus.name),
        focus_center_x=float(focus.center_x_ratio),
        focus_center_y=float(focus.center_y_ratio),
        focus_area=float(focus.bbox_area_ratio),
        salient_object_ids=_salient_object_ids(record.visible_objects),
    )


class SceneStateDeduper:
    def __init__(self) -> None:
        self._buckets: Dict[Tuple[str, int, int, int], List[_StateSketch]] = {}
        self.accepted = 0
        self.skipped = 0

    def _bucket_coords(self, x: float, z: float) -> Tuple[int, int]:
        return (int(round(x / POSITION_BUCKET)), int(round(z / POSITION_BUCKET)))

    def _bucket_keys(self, sketch: _StateSketch) -> List[Tuple[str, int, int, int]]:
        bx, bz = self._bucket_coords(sketch.x, sketch.z)
        keys: List[Tuple[str, int, int, int]] = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                keys.append((sketch.focus_key, sketch.horizon, bx + dx, bz + dz))
        return keys

    def _own_bucket_key(self, sketch: _StateSketch) -> Tuple[str, int, int, int]:
        bx, bz = self._bucket_coords(sketch.x, sketch.z)
        return (sketch.focus_key, sketch.horizon, bx, bz)

    def _is_near_duplicate(self, first: _StateSketch, second: _StateSketch) -> bool:
        if first.focus_key != second.focus_key:
            return False
        if abs(first.x - second.x) > MAX_POSITION_DELTA or abs(first.z - second.z) > MAX_POSITION_DELTA:
            return False
        if _circular_yaw_delta(first.yaw, second.yaw) > MAX_YAW_DELTA:
            return False
        if abs(first.focus_center_x - second.focus_center_x) > MAX_CENTER_DELTA:
            return False
        if abs(first.focus_center_y - second.focus_center_y) > MAX_CENTER_DELTA:
            return False
        if abs(first.focus_area - second.focus_area) > MAX_AREA_DELTA:
            return False
        if _jaccard(first.salient_object_ids, second.salient_object_ids) < MIN_OBJECT_JACCARD:
            return False
        return True

    def keep(self, record: StateRecord) -> bool:
        sketch = _build_sketch(record)
        if sketch is None:
            self.accepted += 1
            return True
        for key in self._bucket_keys(sketch):
            for existing in self._buckets.get(key, []):
                if self._is_near_duplicate(sketch, existing):
                    self.skipped += 1
                    return False
        self._buckets.setdefault(self._own_bucket_key(sketch), []).append(sketch)
        self.accepted += 1
        return True
