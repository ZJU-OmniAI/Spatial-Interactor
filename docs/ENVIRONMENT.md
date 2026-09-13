# Environment

## Data tools

Python 3.10 or newer and FFmpeg are required.

```bash
python3 -m venv .venv-data
source .venv-data/bin/activate
pip install -r requirements-data.txt
```

## SFT

The full LLaMA-Factory source used by the experiments is vendored under
`training/sft/LLaMA-Factory`. Create a CUDA-enabled PyTorch environment, then
install it in editable mode:

```bash
pip install -e training/sft/LLaMA-Factory
```

## OPD and GRPO

The modified EasyR1 source is vendored under `training/opd/EasyR1`. The tested
B200 software versions are recorded in
`training/opd/requirements-tested.txt`. The key versions were PyTorch 2.8.0,
Transformers 4.57.6, Ray 2.53.0, vLLM 0.11.0, and CUDA 12.8.
The paper experiments used one node with eight NVIDIA B200 GPUs.

Install EasyR1 and then align the tested package versions with the CUDA build
available on the target cluster. CUDA compiler headers are needed only when the
selected vLLM/FlashInfer wheel compiles kernels at runtime.

```bash
pip install -e training/opd/EasyR1
pip install pandas pyarrow tensorboard qwen-vl-utils==0.0.14
```

Before allocating GPUs, both launchers support a path/configuration check:

```bash
CHECK_ONLY=1 MODEL_PATH=/models/sft DATA_DIR=/data/opd \
MEDIA_ROOT=/data/media OUTPUT_ROOT=/outputs \
PYTHON_BIN=/absolute/path/to/python \
bash training/opd/scripts/train_opd.sh
```

Set `RUN_PREFLIGHT=true` for the full data/media audit before a training run.
It checks schemas, video-level split isolation, trace alignment, and exact
32-frame decodability. This check is optional because it decodes every unique
training video.

Training writes scalar logs to TensorBoard and sampled rollout diagnostics to
the checkpoint directory's `rollout_audit/` subdirectory.
