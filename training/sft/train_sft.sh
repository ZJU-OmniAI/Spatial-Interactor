#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-${ROOT_DIR}/LLaMA-Factory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5-VL or Qwen3-VL checkpoint}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the build_sft_mixture.py output directory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the SFT checkpoint}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen25vl}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/qwen_vl_full_sft.yaml}"

for required in \
  "${PYTHON_BIN}" \
  "${MODEL_PATH}/config.json" \
  "${DATA_DIR}/dataset_info.json" \
  "${DATA_DIR}/spatial_interactor_sft.jsonl" \
  "${LLAMAFACTORY_DIR}/src/train.py" \
  "${CONFIG}"; do
  if [[ ! -e "${required}" ]] && ! command -v "${required}" >/dev/null 2>&1; then
    echo "Missing required path or command: ${required}" >&2
    exit 2
  fi
done

case "${MODEL_FAMILY}" in
  qwen25vl)
    TEMPLATE=qwen2_vl
    WRAP_CLASS=Qwen2_5_VLDecoderLayer
    ;;
  qwen3vl)
    TEMPLATE=qwen3_vl_nothink
    WRAP_CLASS=Qwen3VLTextDecoderLayer
    ;;
  *)
    echo "MODEL_FAMILY must be qwen25vl or qwen3vl" >&2
    exit 2
    ;;
esac

NGPUS="$(${PYTHON_BIN} -c "print(len('${GPUS}'.split(',')))")"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
WORLD_MICRO_BATCH=$((NGPUS * PER_DEVICE_BATCH_SIZE))
if [[ -z "${GRAD_ACCUM:-}" ]]; then
  if (( GLOBAL_BATCH_SIZE % WORLD_MICRO_BATCH != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by GPUs x PER_DEVICE_BATCH_SIZE." >&2
    exit 2
  fi
  GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / WORLD_MICRO_BATCH))
fi
EFFECTIVE_GLOBAL_BATCH=$((WORLD_MICRO_BATCH * GRAD_ACCUM))
export PYTHONPATH="${LLAMAFACTORY_DIR}/src:${LLAMAFACTORY_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "== Spatial-Interactor SFT =="
echo "model:  ${MODEL_PATH}"
echo "data:   ${DATA_DIR}/spatial_interactor_sft.jsonl"
echo "output: ${OUTPUT_DIR}"
echo "gpus:   ${GPUS}"
echo "batch:  ${PER_DEVICE_BATCH_SIZE} per GPU x ${NGPUS} GPUs x ${GRAD_ACCUM} accumulation = ${EFFECTIVE_GLOBAL_BATCH} global"
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "Configuration check passed; training was not started."
  exit 0
fi

cd "${LLAMAFACTORY_DIR}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NGPUS}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29572}" \
  src/train.py "${CONFIG}" \
  model_name_or_path="${MODEL_PATH}" \
  dataset_dir="${DATA_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  template="${TEMPLATE}" \
  fsdp_config.transformer_layer_cls_to_wrap="${WRAP_CLASS}" \
  per_device_train_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  gradient_accumulation_steps="${GRAD_ACCUM}" \
  disable_gradient_checkpointing=true \
  gradient_checkpointing=false \
  fsdp_config.activation_checkpointing=false \
  "$@"
