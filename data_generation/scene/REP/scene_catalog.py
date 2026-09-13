from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


DATASET_ROOT = Path("/path/to/workspace/habitat_data/versioned_data/replica_cad_dataset")
SCENE_DATASET_CONFIG = DATASET_ROOT / "replicaCAD.scene_dataset_config.json"
SCENE_SUFFIX = ".scene_instance.json"


def _discover_scene_name_to_path() -> Dict[str, str]:
    with SCENE_DATASET_CONFIG.open("r", encoding="utf-8") as f:
        config = json.load(f)

    mapping: Dict[str, str] = {}
    for rel_dir in config.get("scene_instances", {}).get("paths", {}).get(".json", []):
        abs_dir = (SCENE_DATASET_CONFIG.parent / rel_dir).resolve()
        if not abs_dir.is_dir():
            continue
        for path in sorted(abs_dir.glob(f"*{SCENE_SUFFIX}")):
            stem = path.name[: -len(SCENE_SUFFIX)]
            mapping[stem] = str(path)
    return mapping


SCENE_NAME_TO_PATH = _discover_scene_name_to_path()
SCENE_NAMES = sorted(SCENE_NAME_TO_PATH)
SCENE_NAME_TO_ID = {name: idx for idx, name in enumerate(SCENE_NAMES, start=1)}


def all_scene_names() -> List[str]:
    return list(SCENE_NAMES)


def scene_id(scene_name: str) -> int:
    return SCENE_NAME_TO_ID[str(scene_name)]


def scene_path(scene_name: str) -> str:
    return SCENE_NAME_TO_PATH[str(scene_name)]


def normalize_scene_tokens(tokens: Iterable[str]) -> List[str]:
    values: List[str] = []
    for token in tokens:
        raw = token.strip()
        if not raw:
            continue
        if raw.endswith(SCENE_SUFFIX):
            raw = raw[: -len(SCENE_SUFFIX)]
        elif raw.endswith(".json"):
            raw = Path(raw).name.replace(SCENE_SUFFIX, "")
        if raw not in SCENE_NAME_TO_PATH:
            raise KeyError(f"未知 HSSD 场景: {token}")
        values.append(raw)
    return values


def room_type_for_scene(scene: str) -> str:
    return "unknown"
