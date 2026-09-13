# Spatial-Interactor

Official code release for **Spatial-Interactor: Learning Spatial Reasoning
through Interaction with the Observable Physical World**.

[[Project page](https://zju-omniai.github.io/Spatial-Interactor/)]
[[Dataset](https://huggingface.co/datasets/kagakouko/LSI-108K)]
[[3B model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen2.5-VL-3B)]
[[7B model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen2.5-VL-7B)]
[[4B model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-4B)]
[[8B model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-8B)]

This repository contains the LSI-108K construction code, curriculum and mixture
builders, full-parameter SFT launcher, and On-Policy Distillation (OPD)
implementation described in the paper. The code archive excludes datasets,
source media, model weights, generated traces, experiment outputs, credentials,
and machine-specific paths.

## Layout

```text
data_generation/       simulated, real-scene, and long-trajectory construction
data_preparation/      curriculum, SFT-mixture, and OPD-data builders
training/sft/          LLaMA-Factory snapshot and full SFT launcher
training/opd/          modified EasyR1, reward, OPD, and GRPO launchers
docs/                  method-to-code and reproduction notes
tests/                 lightweight data and objective tests
```

To create the portable Hugging Face annotation release from generated source
files, run:

```bash
python data_preparation/prepare_release_dataset.py \
  --input /data/generated/*.jsonl \
  --output-dir /data/LSI-108K \
  --paper-counts
```

The exporter keeps relative media references and geometry-derived targets while
removing machine paths and development-only provenance fields. Source media are
not redistributed and must be obtained under their original licenses.

## 1. Build the SFT data

The complete LSI-108K release contains 15,109 L1, 69,487 L2, and 22,922 L3
records. The reported SFT mixture uses 13,109 single-transition L1 records, all
69,487 L2 records, VSI (50,000), MindCube (10,000), and VSTI (20,000). The
remaining 2,000 multi-stage operation records are retained in L1 but were not
part of the reported SFT run.

```bash
python data_preparation/assign_curriculum_levels.py \
  --input /data/generated/*.jsonl \
  --output-dir /data/lsi_curriculum \
  --paper-counts

python data_preparation/build_sft_mixture.py \
  --l1 /data/lsi_curriculum/lsi_l1.jsonl \
  --l2 /data/lsi_curriculum/lsi_l2.jsonl \
  --vsi /data/vsi_50k.jsonl \
  --mindcube /data/mindcube_10k.jsonl \
  --vsti /data/vsti_20k.jsonl \
  --output-dir /data/spatial_interactor_sft
```

Run one-epoch BF16 full-parameter SFT. The vision tower is frozen; the language
model and multimodal projector are updated. On eight GPUs, the default global
batch is 64.

```bash
MODEL_PATH=/models/qwen-vl \
DATA_DIR=/data/spatial_interactor_sft \
OUTPUT_DIR=/outputs/spatial_interactor_sft \
MODEL_FAMILY=qwen25vl \
GPUS=0,1,2,3,4,5,6,7 \
bash training/sft/train_sft.sh
```

Use `MODEL_FAMILY=qwen3vl` for Qwen3-VL. The launcher derives gradient
accumulation from `GLOBAL_BATCH_SIZE=64` and the number of visible GPUs.

## 2. Build privileged traces

The paper's online stage selects 10,912 RoomTour-based L3 records and 11,000
VSTI long-horizon records before a video-disjoint split. This yields 10,712 L3
and 10,783 VSTI training records (21,495 total), plus 417 validation records.
The annotator receives only 32 uniformly sampled chronological frames, divided
into four fixed eight-frame intervals. It never receives a question, answer,
reward target, or task metadata. One trace is generated per unique video and
reused by all associated questions.

```bash
python data_generation/build_trace_manifest.py \
  --input /data/lsi_curriculum/lsi_l3_roomtour_10912.jsonl /data/vsti_11k.jsonl \
  --data-root /data/media \
  --output /data/traces/manifest.jsonl

export VLM_API_KEY='your-key'
export VLM_BASE_URL='https://your-openai-compatible-endpoint'
export VLM_MODEL='your-frozen-strong-vlm'

python data_generation/generate_privileged_traces.py \
  --manifest /data/traces/manifest.jsonl \
  --data-root /data/media \
  --output /data/traces/traces.jsonl \
  --frame-cache /data/traces/frame_cache \
  --workers 4
```

## 3. Build and train OPD

```bash
python data_preparation/build_opd_dataset.py \
  --l3 /data/lsi_curriculum/lsi_l3_roomtour_10912.jsonl \
  --vsti /data/vsti_11k.jsonl \
  --traces /data/traces/traces.jsonl \
  --data-root /data/media \
  --output-dir /data/spatial_interactor_opd \
  --paper-counts

python data_preparation/validate_data.py \
  --format opd \
  --input /data/spatial_interactor_opd/train.parquet \
          /data/spatial_interactor_opd/val.parquet

MODEL_PATH=/outputs/spatial_interactor_sft \
DATA_DIR=/data/spatial_interactor_opd \
MEDIA_ROOT=/data/media \
OUTPUT_ROOT=/outputs/opd \
PYTHON_BIN=/path/to/easyr1-env/bin/python \
GPUS=0,1,2,3,4,5,6,7 \
bash training/opd/scripts/train_opd.sh
```

The paper defaults are 16 prompts per step, 8 rollouts per prompt, 32 frames,
temperature 0.8, top-p 0.9, a 384-token response limit, reference KL weight
0.01, and OPD weight 0.05. The OPD weight is held for 100 steps and decays to
zero over the next 1,100 steps. No vocabulary-wise pointwise KL clipping is
used.

The matched answer-only GRPO ablation uses the same checkpoint, data, prompts,
rollouts, and rewards:

```bash
MODEL_PATH=/outputs/spatial_interactor_sft \
DATA_DIR=/data/spatial_interactor_opd \
MEDIA_ROOT=/data/media \
OUTPUT_ROOT=/outputs/grpo \
PYTHON_BIN=/path/to/easyr1-env/bin/python \
GPUS=0,1,2,3,4,5,6,7 \
bash training/opd/scripts/train_grpo.sh
```

## 4. Verify the release

```bash
bash scripts/check_release.sh
```

See [DATA.md](docs/DATA.md), [OPD.md](docs/OPD.md),
[ENVIRONMENT.md](docs/ENVIRONMENT.md), and
[METHOD_TO_CODE.md](docs/METHOD_TO_CODE.md) for the paper-to-code mapping and
environment details.
