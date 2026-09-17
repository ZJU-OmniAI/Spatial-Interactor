import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("datasets"), "requires requirements-data.txt")
class ImageReleaseTest(unittest.TestCase):
    def test_original_bytes_and_frame_order(self):
        import pyarrow.parquet as pq
        from datasets import load_dataset
        from PIL import Image

        script = Path(__file__).resolve().parents[1] / "data_preparation/export_image_subset.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, color in (("a.png", "red"), ("b.png", "blue")):
                Image.new("RGB", (12, 8), color).save(root / name)
            row = {
                "id": "example", "media_type": "image", "media_paths": ["b.png", "a.png"],
                "conversations": [{"from": "human", "value": "Compare."},
                                  {"from": "gpt", "value": "B"}],
                "metadata": {"source": "AI2-THOR", "scene": "test", "task_type": "action_inference",
                             "curriculum_level": "L2", "used_in_reported_sft": True},
            }
            source = root / "input.jsonl.gz"
            with gzip.open(source, "wt") as handle:
                handle.write(json.dumps(row) + "\n")
            output = root / "parquet"
            command = [sys.executable, str(script), "--input", str(source), "--media-root", str(root),
                       "--source", "AI2-THOR", "--output-dir", str(output)]
            subprocess.run(command, check=True, capture_output=True)
            path = next(output.glob("*.parquet"))
            record = pq.read_table(path).to_pylist()[0]
            self.assertEqual(record["images"][0]["bytes"], (root / "b.png").read_bytes())
            self.assertEqual(record["images"][1]["bytes"], (root / "a.png").read_bytes())
            self.assertEqual(record["conversations"], row["conversations"])
            decoded = next(iter(load_dataset("parquet", data_files=str(path), split="train", streaming=True)))
            self.assertEqual(decoded["images"][0].getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(decoded["images"][1].getpixel((0, 0)), (255, 0, 0))
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
