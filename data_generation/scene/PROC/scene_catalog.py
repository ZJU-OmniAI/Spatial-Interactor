from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List


SCENE_ROOT = Path("/path/to/workspace/SCENE/PROC/sims_vsi_scenes")
SCENE_MANIFEST_PATH = Path("/path/to/workspace/SCENE/PROC/sims_vsi_scene_manifest_persistent.jsonl")


@lru_cache(maxsize=1)
def _manifest_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with SCENE_MANIFEST_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def _manifest_by_id() -> Dict[str, Dict[str, object]]:
    return {str(row["scene_id"]): row for row in _manifest_rows()}


def scene_name(scene_id: int | str) -> str:
    raw = str(scene_id).strip()
    if raw in _manifest_by_id():
        return raw
    match = re.search(r"(\d+)$", raw)
    if not match:
        raise ValueError(f"无法解析 Proc 场景标识: {scene_id}")
    normalized = match.group(1).zfill(6)
    if normalized not in _manifest_by_id():
        raise ValueError(f"Proc 场景不存在于 persistent manifest 中: {scene_id}")
    return normalized


def scene_id(scene: str) -> int:
    return int(scene_name(scene))


def scene_manifest(scene: str) -> Dict[str, object]:
    normalized = scene_name(scene)
    row = _manifest_by_id().get(normalized)
    if row is None:
        raise ValueError(f"Proc 场景不存在于 persistent manifest 中: {scene}")
    return row


def scene_dir(scene: str) -> Path:
    return Path(str(scene_manifest(scene)["scene_dir"]))


def house_spec_path(scene: str) -> Path:
    return Path(str(scene_manifest(scene)["house_spec"]))


def all_scene_names() -> List[str]:
    return [str(row["scene_id"]) for row in _manifest_rows()]


def normalize_scene_tokens(tokens: Iterable[str]) -> List[str]:
    values: List[str] = []
    seen = set()
    for token in tokens:
        raw = str(token).strip()
        if not raw:
            continue
        normalized = scene_name(raw)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def room_type_for_scene(scene: str) -> str:
    return "unknown"
