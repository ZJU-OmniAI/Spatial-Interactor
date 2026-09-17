from __future__ import annotations

import importlib.util
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def choice_row(row_id: str, video: str | None = None, task: str = "choice_task") -> dict:
    row = {
        "id": row_id,
        "conversations": [
            {"from": "human", "value": "<video>\nWhich option is correct?\nAnswer with only one letter."},
            {"from": "gpt", "value": "Answer: B"},
        ],
        "metadata": {"task_type": task},
    }
    if video:
        row["videos"] = [video]
    else:
        row["images"] = [f"{row_id}.jpg"]
    return row


class DataPipelineTest(unittest.TestCase):
    def run_script(self, relative: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / relative), *arguments],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_sft_mixture_normalizes_and_counts_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for source in ("l1", "l2", "vsi", "mindcube", "vsti"):
                path = root / f"{source}.jsonl"
                row = choice_row(source)
                if source == "mindcube":
                    row.pop("conversations")
                    row["messages"] = [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "B"},
                    ]
                write_jsonl(path, [row])
                paths[source] = path
            output = root / "sft"
            self.run_script(
                "data_preparation/build_sft_mixture.py",
                "--l1", str(paths["l1"]),
                "--l2", str(paths["l2"]),
                "--vsi", str(paths["vsi"]),
                "--mindcube", str(paths["mindcube"]),
                "--vsti", str(paths["vsti"]),
                "--output-dir", str(output),
                "--allow-count-mismatch",
            )
            rows = [json.loads(line) for line in (output / "spatial_interactor_sft.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all("conversations" in row and "messages" not in row for row in rows))
            summary = json.loads((output / "sft_mixture_summary.json").read_text())
            self.assertEqual(summary["public_rows"], 3)
            self.assertTrue(
                all(str(root) not in key for key in summary["input_sha256"])
            )

    def test_curriculum_assignment_uses_task_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "generated.jsonl"
            write_jsonl(
                source,
                [
                    choice_row("object_state_attribute_changes_1", task="object_state_attribute_changes"),
                    {
                        **choice_row("FloorPlan1__state_1__class05__translation__abc"),
                        "metadata": {},
                    },
                    choice_row("roomtour3d_simple_path_shape_1", "path.mp4", "roomtour3d_simple_path_shape"),
                ],
            )
            output = root / "curriculum"
            self.run_script(
                "data_preparation/assign_curriculum_levels.py",
                "--input", str(source),
                "--output-dir", str(output),
            )
            summary = json.loads((output / "curriculum_summary.json").read_text())
            self.assertEqual(summary["level_counts"], {"l1": 1, "l2": 1, "l3": 1})

    def test_multi_stage_manipulation_is_l1_but_excluded_from_sft(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated.jsonl"
            write_jsonl(
                generated,
                [choice_row("multi", task="long_horizon_manipulation_program")],
            )
            curriculum = root / "curriculum"
            self.run_script(
                "data_preparation/assign_curriculum_levels.py",
                "--input", str(generated),
                "--output-dir", str(curriculum),
            )
            self.assertEqual(
                len((curriculum / "lsi_l1.jsonl").read_text().splitlines()), 1
            )

            sources = {}
            for name in ("l2", "vsi", "mindcube", "vsti"):
                path = root / f"{name}.jsonl"
                write_jsonl(path, [choice_row(name)])
                sources[name] = path
            output = root / "sft"
            self.run_script(
                "data_preparation/build_sft_mixture.py",
                "--l1", str(curriculum / "lsi_l1.jsonl"),
                "--l2", str(sources["l2"]),
                "--vsi", str(sources["vsi"]),
                "--mindcube", str(sources["mindcube"]),
                "--vsti", str(sources["vsti"]),
                "--output-dir", str(output),
                "--allow-count-mismatch",
            )
            summary = json.loads((output / "sft_mixture_summary.json").read_text())
            self.assertEqual(summary["source_counts"].get("lsi_l1", 0), 0)
            self.assertEqual(
                summary["excluded_l1_task_counts"],
                {"long_horizon_manipulation_program": 1},
            )

    def test_release_dataset_is_portable_and_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            rows = [
                choice_row("l1", task="long_horizon_manipulation_program"),
                choice_row("l2", task="action_inference"),
                choice_row("l3", "video.mp4", "simple_path_shape"),
            ]
            rows[0]["metadata"]["source_file"] = "/private/build.json"
            rows[0]["metadata"]["gt"] = {"direction": "left"}
            write_jsonl(source, rows)
            output = root / "release"
            self.run_script(
                "data_preparation/prepare_release_dataset.py",
                "--input", str(source),
                "--output-dir", str(output),
                "--metadata", "full",
            )
            with gzip.open(output / "lsi_l1.jsonl.gz", "rt", encoding="utf-8") as handle:
                released = json.loads(handle.readline())
            self.assertEqual(released["metadata"]["curriculum_level"], "L1")
            self.assertFalse(released["metadata"]["used_in_reported_sft"])
            self.assertNotIn("source_file", released["metadata"])
            self.assertEqual(json.loads(released["metadata"]["gt_json"]), {"direction": "left"})
            self.assertEqual(released["media_type"], "image")
            self.assertTrue(released["media_paths"])
            self.assertNotIn("/private/build.json", json.dumps(released))

            minimal = root / "minimal"
            self.run_script(
                "data_preparation/prepare_release_dataset.py",
                "--input", str(output / "lsi_l1.jsonl.gz"),
                "--output-dir", str(minimal),
            )
            with gzip.open(minimal / "lsi_l1.jsonl.gz", "rt", encoding="utf-8") as handle:
                compact = json.loads(handle.readline())
            self.assertEqual(compact["conversations"], released["conversations"])
            self.assertEqual(compact["media_paths"], released["media_paths"])
            self.assertEqual(set(compact["metadata"]), {
                "curriculum_level", "task_type", "source", "scene", "used_in_reported_sft",
            })
            self.assertFalse(compact["metadata"]["used_in_reported_sft"])

            full_again = root / "full_again"
            self.run_script(
                "data_preparation/prepare_release_dataset.py",
                "--input", str(output / "lsi_l1.jsonl.gz"),
                "--output-dir", str(full_again), "--metadata", "full",
            )
            with gzip.open(full_again / "lsi_l1.jsonl.gz", "rt", encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.readline()), released)

    def test_opd_builder_keeps_privilege_out_of_student_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            l3 = root / "l3.jsonl"
            vsti = root / "vsti.jsonl"
            traces = root / "traces.jsonl"
            write_jsonl(l3, [choice_row("l3", "l3.mp4", "roomtour3d_simple_path_shape")])
            write_jsonl(
                vsti,
                [
                    {
                        **choice_row(
                            "direction",
                            "direction.mp4",
                            "camera_movement_direction_interval",
                        ),
                    },
                    {
                        "id": "distance",
                        "conversations": [
                            {"from": "human", "value": "<video>\nHow far did the camera move?"},
                            {"from": "gpt", "value": "2.0"},
                        ],
                        "videos": ["distance.mp4"],
                        "metadata": {
                            "task_type": "camera_displacement_interval",
                            "original_value_m": 2.0,
                        },
                    },
                ],
            )
            trace_text = "\n".join(
                f"[SEGMENT_{index}] Frames {start:02d}-{end:02d}: Visible anchors change as the camera moves."
                for index, (start, end) in enumerate(((1, 8), (9, 16), (17, 24), (25, 32)), 1)
            )
            write_jsonl(
                traces,
                [
                    {
                        "id": f"trace-{video}",
                        "video": video,
                        "segmented_video_description": trace_text,
                        "validation": {"ok": True},
                    }
                    for video in ("l3.mp4", "direction.mp4", "distance.mp4")
                ],
            )
            output = root / "opd"
            self.run_script(
                "data_preparation/build_opd_dataset.py",
                "--l3", str(l3),
                "--vsti", str(vsti),
                "--traces", str(traces),
                "--data-root", str(root),
                "--output-dir", str(output),
                "--vsti-limit", "2",
                "--jsonl-only",
            )
            rows = []
            for split in ("train", "val"):
                rows.extend(json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines())
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertNotIn("PRIVILEGED_STATE_TRANSITIONS", row["prompt"])
                self.assertIn("PRIVILEGED_STATE_TRANSITIONS", row["privileged_prompt"])
                self.assertFalse(Path(row["videos"][0]).is_absolute())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["source_counts"]["vsti_long_horizon"], 2)

    def test_release_source_is_not_inferred_from_campaign_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for source in ("SIMS-V", "ScanNetV2", "ScanNet++", "ARKitScenes", "roomtour3d"):
                row = choice_row(source, "clip.mp4", "simple_path_shape")
                row["metadata"].update(
                    source=source,
                    dataset="camera_pose_qa_visual_strict_intervals_user_arkit_cw90",
                )
                rows.append(row)
            extra = choice_row("extra", "extra.mp4", "simple_path_shape")
            extra["metadata"].update(source="SIMS-V", dataset="videoqa_extra_natural_candidates")
            rows.append(extra)
            write_jsonl(root / "input.jsonl", rows)
            self.run_script(
                "data_preparation/prepare_release_dataset.py",
                "--input", str(root / "input.jsonl"), "--output-dir", str(root / "out"),
            )
            with gzip.open(root / "out/lsi_l3.jsonl.gz", "rt") as handle:
                result = [json.loads(line) for line in handle]
            self.assertEqual([r["metadata"]["source"] for r in result],
                             ["SIMS-V", "ScanNet", "ScanNet++", "ARKitScenes", "RoomTour3D", "SIMS-V"])
            self.assertEqual([r["conversations"] for r in result], [r["conversations"] for r in rows])

    def test_trace_format_validator(self):
        module_path = ROOT / "data_generation/generate_privileged_traces.py"
        spec = importlib.util.spec_from_file_location("trace_generator", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        text = "\n".join(
            f"[SEGMENT_{index}] Frames {start:02d}-{end:02d}: " + "visible motion " * 30
            for index, (start, end) in enumerate(((1, 8), (9, 16), (17, 24), (25, 32)), 1)
        )
        self.assertTrue(module.validate_trace(text)["ok"])

    def test_trace_manifest_uses_portable_video_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source.jsonl"
            video = root / "videos" / "example.mp4"
            write_jsonl(source, [choice_row("trace-row", str(video), "simple_path_shape")])
            output = root / "manifest.jsonl"
            self.run_script(
                "data_generation/build_trace_manifest.py",
                "--input", str(source),
                "--data-root", str(root),
                "--output", str(output),
            )
            row = json.loads(output.read_text().strip())
            self.assertEqual(row["video"], "videos/example.mp4")
            self.assertNotIn("video_abs", row)

    def test_opd_video_key_normalizes_relative_and_absolute_paths(self):
        module_path = ROOT / "data_preparation/build_opd_dataset.py"
        spec = importlib.util.spec_from_file_location("opd_builder", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            relative = "videos/example.mp4"
            absolute = root / relative
            self.assertEqual(
                module.canonical_video_key(relative, root),
                module.canonical_video_key(str(absolute), root),
            )
            self.assertEqual(module.canonical_video_key(relative, root), relative)


if __name__ == "__main__":
    unittest.main()
