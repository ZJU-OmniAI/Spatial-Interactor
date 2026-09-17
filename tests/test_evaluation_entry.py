import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("evaluation_entry", Path(__file__).parents[1] / "evaluation/run.py")
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


class EvaluationEntryTests(unittest.TestCase):
    def args(self, bench="vsi", *extra):
        return ENTRY.parser().parse_args([
            "--bench", bench, "--model", "org/model", "--family", "qwen25vl",
            "--toolkit", "/tmp/toolkit", "--output", "/tmp/evaluation", *extra,
        ])

    def test_vsi_fixed_budget(self):
        result = ENTRY.plan(self.args())
        self.assertEqual(result["config"]["data"]["VSI-Bench"]["nframe"], 32)
        self.assertIsNone(result["config"]["model"]["spatial-interactor"]["fps"])
        self.assertIn("exact_matching", result["command"])

    def test_image_tasks_do_not_receive_video_options(self):
        for bench in ("mindcube", "spbench", "mmsi", "viewspatial"):
            result = ENTRY.plan(self.args(bench))
            self.assertNotIn("nframe", result["config"]["model"]["spatial-interactor"])
            self.assertFalse(result["config"]["model"]["spatial-interactor"]["use_custom_prompt"])

    def test_lmms_keeps_task_budget(self):
        for bench in ("vsti", "sat-real", "sat-syn"):
            result = ENTRY.plan(self.args(bench))
            self.assertIn("--log_samples", result["command"])
            self.assertNotIn("--gen_kwargs", result["command"])
        self.assertIn("max_num_frames=32", ENTRY.plan(self.args("vsti"))["config"]["model_args"])

    def test_reject_invalid_options(self):
        for args in (self.args("vsi", "--frames", "0"),
                     self.args("sat-real", "--resume"),
                     self.args("sat-real", "--data-root", "/tmp/media"),
                     self.args("vsi", "--min-pixels", "0")):
            with self.assertRaises(ValueError):
                ENTRY.plan(args)

    def test_task_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task.yaml").write_text("task: example\n")
            state = ENTRY.source_state(root, root)
            self.assertEqual(len(state["task_files"]["task.yaml"]), 64)

    def test_launch_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            toolkit = root / "toolkit"
            toolkit.mkdir()
            (toolkit / "run.py").write_text(
                "import pathlib, sys\n"
                "p = pathlib.Path(sys.argv[sys.argv.index('--work-dir') + 1])\n"
                "p.mkdir(parents=True, exist_ok=True)\n"
                "(p / 'launched.txt').write_text('ok')\n"
            )
            output = root / "output"
            command = [sys.executable, str(Path(ENTRY.__file__)), "--bench", "vsi",
                       "--model", "org/model", "--family", "qwen25vl", "--toolkit", str(toolkit),
                       "--output", str(output)]
            dry_run = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, check=True)
            self.assertIn("command", json.loads(dry_run.stdout))
            self.assertFalse(output.exists())
            subprocess.run(command, capture_output=True, check=True)
            self.assertEqual((output / "results/launched.txt").read_text(), "ok")
            subprocess.run(command + ["--resume"], capture_output=True, check=True)
            changed = subprocess.run(command + ["--resume", "--frames", "16"], capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("configuration", changed.stderr)


if __name__ == "__main__":
    unittest.main()
