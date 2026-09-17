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
repository contains the data-construction and training code; the dataset and
four model checkpoints are available on Hugging Face.

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

## Learning from observable change

Spatial reasoning is not only about recognizing relations in a static frame. An
agent must follow how objects, viewpoints, and locations change through
interaction, then integrate those local transitions into a coherent spatial
state. Spatial-Interactor turns observable physical interaction into direct
supervision for this process.

<p align="center">
  <img src="assets/readme/paradigm.webp?v=20260917" width="100%" alt="Spatial-Interactor learning paradigm">
</p>

## A three-level spatial interaction curriculum

The curriculum progresses from **passive world-state transitions (L1)**, to
**active self-state transitions (L2)**, and finally to **long-horizon state
transition integration (L3)**. Together, these levels connect local physical
change with global path understanding.

<p align="center">
  <img src="assets/readme/curriculum.webp" width="100%" alt="LSI-108K three-level spatial interaction curriculum">
</p>

## From interaction trajectories to verifiable QA

LSI-108K is constructed from simulator actions, scene states, camera poses,
object tracks, and robot trajectories. Geometric signals are converted into
textual ground truth before task templates produce question-answer pairs,
keeping supervision tied to observable state changes.

<p align="center">
  <img src="assets/readme/construction.webp?v=20260917" width="100%" alt="LSI-108K construction pipeline">
</p>

| Level | Samples | Spatial supervision |
|:---:|---:|:---|
| **L1** | 15,109 | Passive world-state transitions |
| **L2** | 69,487 | Active self-state transitions |
| **L3** | 22,922 | Long-horizon transition integration |
| **Total** | **107,518** | **Interaction-derived spatial QA** |

## On-Policy Distillation

Training first uses L1 and L2 to establish local state-transition modeling.
For L3, On-Policy Distillation (OPD) combines verifiable answer rewards with a
training-only privileged transition trace. The teacher and student evaluate the
same student-generated prefixes; process distillation is applied to reasoning
tokens, while answer rewards supervise the final result. At inference, the
model receives only the original image or video and question.

<p align="center">
  <img src="assets/readme/opd.webp" width="100%" alt="On-Policy Distillation pipeline">
</p>

## Get started

- **Models:** choose one of the four checkpoints above and use its Transformers interface.
- **Dataset:** [LSI-108K](https://huggingface.co/datasets/kagakouko/LSI-108K), with L1/L2/L3 annotations and an image-enabled AI2-THOR subset.
- **Training:** [Setup](docs/ENVIRONMENT.md) and [SFT / OPD commands](docs/TRAINING.md).
- **Data construction:** [Sources and pipeline](docs/DATA.md).

```bash
bash scripts/check_release.sh
```

`data_generation/` builds interaction QA and privileged traces;
`data_preparation/` prepares the training data; `training/` contains SFT and OPD.
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
