import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data_preparation/export_video_subset.py"


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class VideoReleaseTest(unittest.TestCase):
    def test_unique_clips_preserve_bytes_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "media/clips/example.mp4"
            clip.parent.mkdir(parents=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=640x360:rate=30", "-t", "1", "-c:v", "mpeg4", str(clip)],
                           check=True, capture_output=True)
            rows = [{"id": str(i), "metadata": {"source": "test"},
                     "media_type": "video", "media_paths": ["clips/example.mp4"]} for i in range(2)]
            rows.append({"metadata": {"source": "other"}, "media_type": "video",
                         "media_paths": ["unavailable.mp4"]})
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in rows))
            command = [sys.executable, str(SCRIPT), "--input", str(source), "--media-root",
                       str(root / "media"), "--source", "test", "--output-dir", str(root / "out")]
            subprocess.run(command, check=True, capture_output=True)
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual((summary["qa_rows"], summary["unique_videos"]), (2, 1))
            manifest = json.loads((root / "out/manifest.jsonl").read_text())
            with tarfile.open(root / "out/clips-00000.tar") as archive:
                self.assertEqual(archive.getnames(), ["clips/example.mp4"])
                self.assertEqual(archive.extractfile("clips/example.mp4").read(), clip.read_bytes())
            self.assertEqual(manifest["sha256"], hashlib.sha256(clip.read_bytes()).hexdigest())
            self.assertNotIn(directory, json.dumps(manifest))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_unsafe_paths_and_broken_video_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            media.mkdir()
            outside = root / "outside.mp4"
            outside.write_bytes(b"not a video")
            (media / "link.mp4").symlink_to(outside)
            (media / "broken.mp4").write_bytes(b"not a video")
            for i, path in enumerate(("../outside.mp4", str(outside), "link.mp4", "broken.mp4")):
                source = root / "input.jsonl"
                source.write_text(json.dumps({"metadata": {"source": "test"},
                                             "media_type": "video", "media_paths": [path]}) + "\n")
                output = root / f"out-{i}"
                result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(source),
                                         "--media-root", str(media), "--source", "test",
                                         "--output-dir", str(output)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
