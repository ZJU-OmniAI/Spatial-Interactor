<div align="center">
  <h1><img src="assets/readme/icon.png" alt="Spatial-Interactor icon" height="64" align="absmiddle">&nbsp; Spatial-Interactor</h1>
  <p><strong>Learning Spatial Reasoning through Interaction with the Observable Physical World</strong></p>
  <p>
    <a href="https://zju-omniai.github.io/Spatial-Interactor/assets/paper.pdf?v=20260918baseline"><img src="https://img.shields.io/badge/-Paper-B31B1B?style=for-the-badge&amp;logo=adobeacrobatreader&amp;logoColor=white" alt="Paper"></a>
    <a href="https://zju-omniai.github.io/Spatial-Interactor/"><img src="https://img.shields.io/badge/-Project_Page-0F5354?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
    <a href="https://huggingface.co/collections/kagakouko/spatial-interactor"><img src="https://img.shields.io/badge/HF-Models_%26_Data-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=3A3B45" alt="Models and data on Hugging Face"></a>
    <a href="https://github.com/ZJU-OmniAI/Spatial-Interactor"><img src="https://img.shields.io/badge/-Code-171B1F?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Code"></a>
  </p>
  <p>
    <a href="requirements-data.txt"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2F855A" alt="Apache-2.0 license"></a>
  </p>
</div>

<p align="center">
  <a href="assets/presentation/spatial-interactor-intro.mp4">
    <img src="assets/readme/introduction-preview.webp" width="90%" alt="Spatial-Interactor project introduction">
  </a>
</p>

## 🎞️ Visual Presentation

The full 20-slide visual presentation is included in the repository. Select
the preview to open the presentation video, or download the
[editable deck](assets/presentation/Spatial-Interactor-Visual-Presentation.pptx).

<p align="center">
  <a href="assets/presentation/spatial-interactor-presentation.mp4">
    <img src="assets/readme/presentation-preview.webp" width="90%" alt="Spatial-Interactor visual presentation">
  </a>
</p>

## 📖 Overview

Spatial reasoning is not only a matter of recognizing static relations. An
agent must also understand how the observable world changes after an action,
how its own viewpoint changes during movement, and how successive transitions
compose into a complete trajectory.

**Spatial-Interactor** learns these capabilities from observable physical
interaction. It organizes interaction records into a progressive curriculum,
uses supervised fine-tuning to learn local state transitions, and applies
On-Policy Distillation (OPD) to integrate spatial evidence over long horizons.
The released model receives only the original visual input and question at
inference time.

<p align="center">
  <img src="assets/readme/overview.webp" width="94%" alt="Overview of Spatial-Interactor">
</p>

## 🧭 Spatial Interaction Curriculum

The curriculum follows the increasing reasoning horizon of physical
interaction:

- **L1, passive world-state transitions:** infer how objects and scenes change
  under an external action.
- **L2, active self-state transitions:** reason about ego-motion and its effect
  on position, distance, overlap, and visibility.
- **L3, long-horizon interaction trajectories:** integrate ordered local
  changes into key-motion events and global paths.

<p align="center">
  <img src="assets/readme/curriculum.webp" width="92%" alt="Three-level spatial interaction curriculum">
</p>

## 🧠 On-Policy Distillation

OPD combines verifiable answer rewards with training-only process supervision.
For each student rollout, a privileged branch reads an ordered state-transition
trace while the student branch sees only the video and question. Both branches
score the same student-generated prefixes, and distillation is applied only to
reasoning tokens. The privileged trace, teacher branch, reward functions, and
reference policy are removed after training.

<p align="center">
  <img src="assets/readme/opd.webp" width="92%" alt="On-Policy Distillation pipeline">
</p>

## 📦 Models and Data

The complete release is collected on
[Hugging Face](https://huggingface.co/collections/kagakouko/spatial-interactor).

| Resource | Description | Access |
|:---|:---|:---:|
| **LSI-108K** | 107,518 interaction-derived spatial QA pairs across L1, L2, and L3 | [🤗 Dataset](https://huggingface.co/datasets/kagakouko/LSI-108K) |
| **Spatial-Interactor-3B** | Qwen2.5-VL-3B checkpoint | [🤗 Model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen2.5-VL-3B) |
| **Spatial-Interactor-7B** | Qwen2.5-VL-7B checkpoint | [🤗 Model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen2.5-VL-7B) |
| **Spatial-Interactor-4B** | Qwen3-VL-4B checkpoint | [🤗 Model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-4B) |
| **Spatial-Interactor-8B** | Qwen3-VL-8B checkpoint | [🤗 Model](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-8B) |

Source media are governed by their upstream licenses. When redistribution is
not permitted, LSI-108K provides source and episode identifiers for obtaining
the corresponding media from the original dataset.

## 🚀 Quick Start

### 1. Clone and validate the release

```bash
git clone https://github.com/ZJU-OmniAI/Spatial-Interactor.git
cd Spatial-Interactor
bash scripts/check_release.sh
```

### 2. Download LSI-108K

```bash
hf download kagakouko/LSI-108K \
  --repo-type dataset \
  --local-dir data/LSI-108K
```

The annotation schema, curriculum assignment, source-media manifest, and media
access conditions are documented in [Data](docs/DATA.md).

### 3. Prepare the environment

The data tools, SFT stage, and OPD stage use separate environments. Follow the
tested setup in [Environment](docs/ENVIRONMENT.md) before launching training.

## ⚙️ Training

SFT learns local state transitions from L1/L2 and public spatial QA. OPD starts
from the SFT checkpoint and learns long-horizon integration from L3 and VSTI
trajectories.

```bash
# SFT
MODEL_PATH=/models/qwen-vl DATA_DIR=/data/sft OUTPUT_DIR=/outputs/sft \
  MODEL_FAMILY=qwen25vl bash training/sft/train_sft.sh

# OPD, initialized from the SFT checkpoint
MODEL_PATH=/outputs/sft DATA_DIR=/data/opd MEDIA_ROOT=/data/media \
  OUTPUT_ROOT=/outputs/opd PYTHON_BIN=/path/to/opd-env/bin/python \
  bash training/opd/scripts/train_opd.sh
```

See [Training](docs/TRAINING.md) for data preparation and multi-model commands,
[On-Policy Distillation](docs/OPD.md) for the objective, and the
[paper-to-code map](docs/METHOD_TO_CODE.md) for implementation ownership.

## 🧪 Evaluation

The evaluation entrypoint delegates prompting and scoring to the corresponding
upstream benchmark implementation while recording the run configuration and
raw predictions.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py \
  --bench vsi --family qwen25vl \
  --model kagakouko/Spatial-Interactor-Qwen2.5-VL-7B \
  --toolkit ./VLMEvalKit --data-root /data/benchmarks \
  --output ./outputs/qwen25vl7b/vsi
```

[Evaluation instructions](docs/EVALUATION.md) cover VSI, VSTI, MindCube,
SPBench-MV, MMSI, ViewSpatial, SAT-Real, and SAT-Syn. Use `--dry-run` to inspect
a resolved command before loading model weights.

## 🏗️ Repository Layout

```text
Spatial-Interactor/
├── data_generation/       interaction QA and privileged-trace construction
├── data_preparation/      curriculum, SFT-mixture, and OPD-data builders
├── training/
│   ├── sft/               LLaMA-Factory-based supervised fine-tuning
│   └── opd/               EasyR1/verl-based OPD, GRPO, and rewards
├── evaluation/            benchmark launchers and run manifests
├── docs/                  method, data, training, and evaluation guides
├── tests/                 lightweight data and objective tests
└── assets/                README figures and project media
```

## 🙏 Acknowledgements

We sincerely thank the authors of
[AI2-THOR](https://ai2thor.allenai.org/),
[ProcTHOR](https://procthor.allenai.org/),
[SIMS-V](https://arxiv.org/abs/2511.04668),
[HSSD](https://3dlg-hcvc.github.io/hssd/),
[Replica](https://github.com/facebookresearch/Replica-Dataset),
[ScanNet](http://www.scan-net.org/),
[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/),
[ARKitScenes](https://github.com/apple/ARKitScenes),
[MultiScan](https://3dlg-hcvc.github.io/multiscan/),
[RoomTour3D](https://roomtour3d.github.io/), and
[BridgeData V2](https://rail-berkeley.github.io/bridgedata/) for their public
environments, trajectories, and annotations. We also appreciate
[VSI-590K](https://arxiv.org/abs/2412.14171),
[MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube), and
[STI-Bench](https://huggingface.co/datasets/MINT-SJTU/STI-Bench) for the public
spatial QA used in the supervised training mixture.

The training implementation builds on
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[EasyR1](https://github.com/hiyouga/EasyR1), and
[verl](https://github.com/volcengine/verl). We are grateful to their authors
and maintainers for making these frameworks publicly available.

<a id="citation"></a>

## 📝 Citation

```bibtex
@misc{yao2026spatialinteractor,
  title  = {Spatial-Interactor: Learning Spatial Reasoning through Interaction with the Observable Physical World},
  author = {Yao, Kaixiang and Wang, Xu and Pan, Miao and Hu, Xiyue and Wang, Weishi and Dahlmeier, Daniel and Chen, Jintao and Shen, Yongliang and Zhang, Xuhong and Zhang, Wenqi},
  year   = {2026},
  note   = {Preprint}
}
```

## ⚖️ License

Spatial-Interactor source code is released under the
[Apache License 2.0](LICENSE). Third-party datasets, model checkpoints, and
media remain subject to their original licenses and terms. See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for the vendored training
frameworks.
