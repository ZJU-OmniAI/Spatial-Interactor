#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/path/to/workspace/SCENE")

from shared_qa_augmentation import main_with_default_root


if __name__ == "__main__":
    raise SystemExit(main_with_default_root(Path("/path/to/workspace/SCENEOUTPUT/HSSD/render_output")))
