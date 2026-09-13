#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_config = args.model_path / "tokenizer_config.json"
    if not source_config.is_file():
        raise FileNotFoundError(source_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        source = args.model_path / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)

    config_path = args.output_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["fix_mistral_regex"] = True
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
