#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from habitat_proposal_executor import main_for_dataset
from scene_catalog import scene_path


def main_for_class(default_class_id: int) -> int:
    return main_for_dataset(
        dataset_name="hssd",
        resolve_scene_path=scene_path,
        default_class_id=default_class_id,
    )
