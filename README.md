<p align="center">
  <img src="assets/readme/icon.png" width="100" alt="Spatial-Interactor">
</p>

<h1 align="center">Spatial-Interactor</h1>

<p align="center">
  <strong>Learning Spatial Reasoning through Interaction with the Observable Physical World</strong>
</p>

<p align="center">
  <a href="https://zju-omniai.github.io/Spatial-Interactor/"><img src="https://img.shields.io/badge/Project-Page-A56F59?style=flat-square&amp;labelColor=54534D" alt="Project page"></a>
  <a href="https://zju-omniai.github.io/Spatial-Interactor/assets/paper.pdf?v=20260917"><img src="https://img.shields.io/badge/Paper-PDF-9B8255?style=flat-square&amp;labelColor=54534D" alt="Paper PDF"></a>
  <a href="https://huggingface.co/collections/kagakouko/spatial-interactor"><img src="https://img.shields.io/badge/Hugging_Face-Models_%26_Data-738363?style=flat-square&amp;labelColor=54534D" alt="Models and data on Hugging Face"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-887A9A?style=flat-square&amp;labelColor=54534D" alt="BibTeX"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/kagakouko/Spatial-Interactor-Qwen2.5-VL-3B">Qwen2.5-VL-3B</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/kagakouko/Spatial-Interactor-Qwen2.5-VL-7B">Qwen2.5-VL-7B</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-4B">Qwen3-VL-4B</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-8B">Qwen3-VL-8B</a>
</p>

Spatial-Interactor learns spatial reasoning from observable physical interaction,
progressing from local state transitions to long-horizon integration. This
repository provides **training and evaluation code**. The dataset and four
model checkpoints are available on Hugging Face.

## Overview

<p align="center">
  <img src="assets/readme/overview.webp" width="100%" alt="Spatial-Interactor overview: interaction trajectories, three-level curriculum, SFT and OPD, and spatial reasoning results">
</p>

<p align="center">
  <img src="assets/readme/introduction-preview.webp" width="100%" alt="Animated Spatial-Interactor introduction">
</p>

## Presentation

<p align="center">
  <img src="assets/readme/presentation-preview.webp" width="100%" alt="Animated Spatial-Interactor 20-page presentation">
</p>

## Training

SFT learns local state transitions from L1/L2 and external spatial QA. OPD then
combines answer rewards with training-only transition traces for long-horizon
reasoning. At inference, only the original visual input and question are needed.

<p align="center">
  <img src="assets/readme/opd.webp" width="100%" alt="On-Policy Distillation pipeline">
</p>

After [environment setup](docs/ENVIRONMENT.md) and
[data preparation](docs/TRAINING.md#prepare-the-data):

```bash
# SFT
MODEL_PATH=/models/qwen-vl DATA_DIR=/data/sft OUTPUT_DIR=/outputs/sft \
  MODEL_FAMILY=qwen25vl bash training/sft/train_sft.sh

# OPD, initialized from the SFT checkpoint
MODEL_PATH=/outputs/sft DATA_DIR=/data/opd MEDIA_ROOT=/data/media \
  OUTPUT_ROOT=/outputs/opd PYTHON_BIN=/path/to/opd-env/bin/python \
  bash training/opd/scripts/train_opd.sh
```

See [Training](docs/TRAINING.md) for the Qwen3-VL setting and full commands.
Dataset downloads, fields, and media coverage are documented on
[Hugging Face](https://huggingface.co/datasets/kagakouko/LSI-108K).

## Evaluation

Use upstream benchmark prompts and scorers through one entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py \
  --bench vsi --family qwen25vl \
  --model kagakouko/Spatial-Interactor-Qwen2.5-VL-7B \
  --toolkit ./VLMEvalKit --data-root /data/benchmarks \
  --output ./outputs/qwen25vl7b/vsi
```

[Evaluation setup and benchmark commands](docs/EVALUATION.md) cover VSI, VSTI,
MindCube, SPBench-MV, MMSI, ViewSpatial, SAT-Real, and SAT-Syn. The launcher
records settings and preserves raw predictions for inspecting model errors.
Use `--dry-run` to check a command before loading weights.

## Code layout

```text
training/sft/       SFT launcher and LLaMA-Factory
training/opd/       OPD / GRPO training and rewards
evaluation/        benchmark launchers
data_preparation/  training data preparation and image export
data_generation/   interaction QA and privileged traces
```

Run the lightweight code checks:

```bash
bash scripts/check_release.sh
```

Licenses and source attributions are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{yao2026spatialinteractor,
  title = {Spatial-Interactor: Learning Spatial Reasoning through Interaction with the Observable Physical World},
  author = {Yao, Kaixiang and Wang, Xu and Pan, Miao and Hu, Xiyue and Wang, Weishi and Dahlmeier, Daniel and Chen, Jintao and Shen, Yongliang and Zhang, Xuhong and Zhang, Wenqi},
  year = {2026},
  note = {Preprint}
}
```
