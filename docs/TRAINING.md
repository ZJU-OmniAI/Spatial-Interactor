# Training

## Repository layout

```text
data_generation/       interaction and privileged-trace construction
data_preparation/      curriculum, SFT-mixture, and OPD-data builders
training/sft/          LLaMA-Factory snapshot and full SFT launcher
training/opd/          EasyR1-based OPD, reward, and GRPO implementation
docs/                  method-to-code and reproduction notes
tests/                 lightweight data and objective tests
```

## Prepare the data

Create the portable Hugging Face annotation release from generated records:

```bash
python data_preparation/prepare_release_dataset.py \
  --input /data/generated/*.jsonl \
  --output-dir /data/LSI-108K \
  --paper-counts
```

Build the reported SFT mixture:

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

The reported SFT mixture uses 13,109 single-transition L1 records, all 69,487
L2 records, VSI (50,000), MindCube (10,000), and VSTI (20,000). The remaining
2,000 multi-stage operation records are released in L1 but are not part of the
reported SFT run.

## Train Spatial-Interactor

Run one-epoch BF16 full-parameter SFT. The vision tower is frozen; the language
model and multimodal projector are updated. The default global batch is 64.

```bash
MODEL_PATH=/models/qwen-vl \
DATA_DIR=/data/spatial_interactor_sft \
OUTPUT_DIR=/outputs/spatial_interactor_sft \
MODEL_FAMILY=qwen25vl \
GPUS=0,1,2,3,4,5,6,7 \
bash training/sft/train_sft.sh
```

Use `MODEL_FAMILY=qwen3vl` for Qwen3-VL.

Build question-independent privileged traces from 32 chronological frames:

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

Build the OPD split and launch training:

```bash
python data_preparation/build_opd_dataset.py \
  --l3 /data/lsi_curriculum/lsi_l3_roomtour_10912.jsonl \
  --vsti /data/vsti_11k.jsonl \
  --traces /data/traces/traces.jsonl \
  --data-root /data/media \
  --output-dir /data/spatial_interactor_opd \
  --paper-counts

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

The matched answer-only GRPO ablation uses the same initialization, data,
prompts, rollouts, and rewards:

```bash
MODEL_PATH=/outputs/spatial_interactor_sft \
DATA_DIR=/data/spatial_interactor_opd \
MEDIA_ROOT=/data/media \
OUTPUT_ROOT=/outputs/grpo \
PYTHON_BIN=/path/to/easyr1-env/bin/python \
GPUS=0,1,2,3,4,5,6,7 \
bash training/opd/scripts/train_grpo.sh
```

## Verification and documentation

```bash
bash scripts/check_release.sh
```

See [Data](DATA.md), [OPD](OPD.md),
[Environment](ENVIRONMENT.md), [Reproducibility](REPRODUCIBILITY.md),
and [Method-to-code mapping](METHOD_TO_CODE.md) for implementation details.
