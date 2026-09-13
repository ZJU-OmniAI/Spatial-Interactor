from __future__ import annotations

from typing import Iterable, List


SCENE_IDS: List[int] = (
    list(range(1, 31))
    + list(range(201, 231))
    + list(range(301, 331))
    + list(range(401, 431))
)


def scene_name(scene_id: int) -> str:
    return f"FloorPlan{scene_id}"


def scene_id_from_name(scene: str) -> int:
    token = str(scene).strip()
    if token.startswith("FloorPlan"):
        token = token[len("FloorPlan") :]
    return int(token)


def room_type_for_scene(scene: str | int) -> str:
    scene_id = int(scene) if isinstance(scene, int) else scene_id_from_name(scene)
    if 1 <= scene_id <= 30:
        return "kitchen"
    if 201 <= scene_id <= 230:
        return "living_room"
    if 301 <= scene_id <= 330:
        return "bedroom"
    if 401 <= scene_id <= 430:
        return "bathroom"
    return "unknown"


def all_scene_names() -> List[str]:
    return [scene_name(scene_id) for scene_id in SCENE_IDS]


def normalize_scene_tokens(tokens: Iterable[str]) -> List[str]:
    values: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.startswith("FloorPlan"):
            values.append(token)
            continue
        values.append(scene_name(int(token)))
    return values
