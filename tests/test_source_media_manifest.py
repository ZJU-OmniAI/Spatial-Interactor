import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class SourceManifestTest(unittest.TestCase):
    def test_no_local_media_path_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl.gz"
            row = {"id": "row-1", "media_type": "video",
                   "media_paths": ["video_clips/private.mp4"],
                   "metadata": {"source": "roomtour3d",
                                "gt_json": json.dumps({"video_id": "abc123", "path_id": "abc123_002", "start_frame": 8, "end_frame": 32})}}
            with gzip.open(source, "wt") as handle: handle.write(json.dumps(row) + "\n")
            output = root / "manifest.jsonl.gz"
            subprocess.run([sys.executable, str(ROOT / "data_preparation/build_source_media_manifest.py"), "--input", str(source), "--output", str(output)], check=True)
            with gzip.open(output, "rt") as handle: result = json.loads(handle.readline())
            self.assertEqual(result["source"], "RoomTour3D")
            self.assertEqual(result["source_locator"]["video_id"], "abc123")
            self.assertNotIn("video_clips", json.dumps(result))
            self.assertNotIn("/private", json.dumps(result))

if __name__ == "__main__": unittest.main()
