#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def row_contact(row: dict) -> str | None:
    inp = row.get("input") or {}
    return inp.get("contact_sheet")


def row_frames(row: dict) -> list[str]:
    inp = row.get("input") or {}
    if inp.get("frame_paths"):
        return list(inp["frame_paths"])
    out = []
    for key in ("frame_A", "frame_B"):
        if inp.get(key):
            out.append(inp[key])
    return out


def load_pair(row: dict) -> np.ndarray | None:
    contact = row_contact(row)
    if contact:
        img = cv2.imread(contact)
        if img is not None:
            return cv2.resize(img, (960, 360))
    frames = []
    for path in row_frames(row)[:2]:
        img = cv2.imread(path)
        if img is None:
            return None
        frames.append(cv2.resize(img, (480, 360)))
    if len(frames) != 2:
        return None
    return np.concatenate(frames, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qa_json", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.qa_json.read_text(encoding="utf-8"))
    panels = []
    for i, row in enumerate(rows, 1):
        img = load_pair(row)
        if img is None:
            continue
        value = ""
        gt = row.get("gt") or {}
        approx = gt.get("approx_action") or {}
        if "value" in approx and "unit" in approx:
            value = f" {approx['value']}{approx['unit']}"
        text = f"{i}. {row.get('answer', '')}{value}"
        cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        panels.append(img)
    if not panels:
        raise SystemExit("no images")
    lines = []
    for idx in range(0, len(panels), 2):
        pair = panels[idx : idx + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        lines.append(np.concatenate(pair, axis=1))
    out = args.qa_json.parent / "overview_contact_sheet.png"
    cv2.imwrite(str(out), np.concatenate(lines, axis=0))
    print(out)


if __name__ == "__main__":
    main()
