# Data construction

## LSI-108K

LSI-108K contains 107,518 verifiable spatial QA pairs organized into three
curriculum levels:

- **L1, passive world-state transitions:** object motion, attribute changes,
  visibility changes, and single- or multi-stage robot-arm operations under a
  fixed observer.
- **L2, active self-state transitions:** ego-motion direction and magnitude,
  and their effects on distance, position, overlap, and visibility.
- **L3, long-horizon interaction trajectories:** global path understanding,
  key-motion localization, revisiting, and reverse-path reasoning.

Simulation sources are AI2-THOR, ProcTHOR, HSSD, and ReplicaCAD. Pose-aligned
real-scene sources are ScanNet, ScanNet++, MultiScan, 3RScan, and ARKitScenes.
RoomTour and SIMS-V provide additional camera trajectories; BridgeData V2
provides robot-arm observations. Source data retain their own terms and attribution.

## Scene pipeline

The Hugging Face `default` configuration contains all annotations; `images`
embeds available original images. Video archives preserve the relative paths
used by the annotations. See the dataset card for current media coverage and
source-specific access conditions. Not all sources permit public redistribution.
`prepare_release_dataset.py --parquet` creates portable JSONL and Parquet exports.

The default annotation export retains only source, scene, task, curriculum level,
and the SFT inclusion flag alongside the conversation and media references.
Use `prepare_release_dataset.py --metadata full` with original generated records
to retain structured geometry for construction/debugging. The compact release is
not a substitute for the original geometric records used by the OPD data builder.

To package images you have permission to redistribute:

```bash
python data_preparation/export_image_subset.py \
  --input /data/LSI-108K/lsi_l1.jsonl.gz /data/LSI-108K/lsi_l2.jsonl.gz \
  --media-root /data/media --source AI2-THOR --output-dir /data/ai2thor_parquet
```

Install `requirements-data.txt` first. Images are validated and embedded without
resizing or re-encoding; source attribution must accompany any distribution.
For licensed video clips, `export_video_subset.py` accepts the same input,
media-root, source, and output-dir arguments. It requires FFmpeg, checks video
stream headers, deduplicates paths, and writes TAR shards with a checksum manifest.
It does not re-encode videos or replace the annotations.

`data_generation/scene/` contains the complete source-specific pipeline. The
common simulated flow is:

1. `build_state_bank.py` samples valid simulator states.
2. `enumerators/` proposes task-independent state/action/state transitions.
3. `execute_*_bank.py` executes actions and renders observations.
4. `augment_generated_qas.py` and the top-level aggregation scripts standardize
   prompts, answers, and simulator-derived ground truth.
5. Habitat and real-scene candidates pass additional visual-answerability and
   consistency filters before packaging.

The original source-specific defaults have been anonymized as
`/path/to/workspace`. Pass the corresponding CLI roots or replace this
placeholder for a local source installation. The stable release-facing data
interfaces are the L1/L2/L3 JSONL files, not the development-machine layout.

## Long-horizon pipeline

`data_generation/long_horizon/roomtour/` builds metric path, path-shape,
turning-interval, and revisiting QA from RoomTour camera trajectories.
`data_generation/long_horizon/bridgedata/` builds end-effector motion,
action-to-image, temporal sorting, multi-stage manipulation, and reverse-path
QA. Selection scripts enforce motion purity, answer uniqueness, and balanced
answer labels where candidates permit.

`data_preparation/assign_curriculum_levels.py` is the bridge from generated
JSONL files to `lsi_l1.jsonl`, `lsi_l2.jsonl`, and `lsi_l3.jsonl`. It first uses
explicit `metadata.curriculum_level`, then the paper task mapping. Unrecognized
task types fail closed and can be supplied through `--mapping`. With
`--paper-counts`, it verifies 15,109 L1 rows, 69,487 L2 rows, 22,922 L3 rows,
and 107,518 rows overall.

## Training mixtures

The reported SFT run uses 13,109 single-transition L1 records and all 69,487 L2
records, plus Public-80K. The 2,000 multi-stage operation records remain part
of the released L1 curriculum but were not included in that run.

| Source | Rows |
| --- | ---: |
| VSI | 50,000 |
| MindCube | 10,000 |
| VSTI | 20,000 |
| Public total | 80,000 |

LSI-108K contains 22,922 L3 rows. The paper's OPD/GRPO mixture uses a
selected 10,912-row RoomTour-L3 manifest and all 11,000 VSTI camera-movement
and camera-displacement rows before splitting. This produces 21,495 training
records (10,712 L3 and 10,783 VSTI) and 417 validation records. The builder
normalizes exports that append `_interval` to the two VSTI task names.
Train/validation splitting is performed by video key, so a video cannot occur
in both splits.

## Privileged traces

Each unique video is uniformly sampled to 32 frames and divided into fixed
continuous intervals 01-08, 09-16, 17-24, and 25-32. A frozen strong VLM
describes the visible environment, visual change, and coarse camera motion in
each interval. The annotator is intentionally question-free and answer-free.
Multiple QA rows referring to the same video reuse one trace.
