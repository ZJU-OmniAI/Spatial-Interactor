#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/path_validation.sh"
if [[ -f "${ROOT_DIR}/train.env" ]]; then
  set -a
  source "${ROOT_DIR}/train.env"
  set +a
fi

ENV_PREFIX="${ENV_PREFIX:-${HOME}/.cache/spatial_interactor/easyr1_env}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
EASYR1_DIR="${EASYR1_DIR:-${ROOT_DIR}/EasyR1}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a supported Qwen-VL SFT checkpoint}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/all}"
MEDIA_ROOT="${MEDIA_ROOT:-${DATA_ROOT:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/runs}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.parquet}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

for path_name in MODEL_PATH DATA_DIR MEDIA_ROOT OUTPUT_ROOT ENV_PREFIX EASYR1_DIR PYTHON_BIN; do
  spatial_interactor_normalize_accidental_dollar_path "$path_name"
done

if [[ -z "${MEDIA_ROOT}" ]]; then
  echo "Set MEDIA_ROOT (or DATA_ROOT) to the dataset root used by relative video paths." >&2
  exit 2
fi
for path_name in MODEL_PATH DATA_DIR MEDIA_ROOT OUTPUT_ROOT ENV_PREFIX EASYR1_DIR PYTHON_BIN; do
  spatial_interactor_require_clean_absolute_path "$path_name" "${!path_name}"
done

for required in "${TRAIN_FILE}" "${VAL_FILE}" "${MODEL_PATH}/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ ! -f "${EASYR1_DIR}/verl/trainer/opd.py" ]]; then
  echo "The bundled EasyR1 OPD implementation is missing: ${EASYR1_DIR}" >&2
  exit 2
fi

NGPUS="$(${PYTHON_BIN} -c "print(len('${GPUS}'.split(',')))")"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-200}"
VAL_FREQ="${VAL_FREQ:-200}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VALIDATE_AT_END="${VALIDATE_AT_END:-true}"
SAVE_AT_END="${SAVE_AT_END:-true}"
DISABLE_KL="${DISABLE_KL:-false}"
USE_KL_LOSS="${USE_KL_LOSS:-true}"
KL_COEF="${KL_COEF:-0.01}"
OPD_ENABLED="${OPD_ENABLED:-true}"
OPD_LAMBDA="${OPD_LAMBDA:-0.05}"
OPD_DIVERGENCE="${OPD_DIVERGENCE:-forward_kl}"
OPD_TEACHER_MODE="${OPD_TEACHER_MODE:-latest_pre_update}"
OPD_HOLD_STEPS="${OPD_HOLD_STEPS:-100}"
OPD_DECAY_STEPS="${OPD_DECAY_STEPS:-1100}"
OPD_TOP_K="${OPD_TOP_K:-100}"
OPD_TEMPERATURE="${OPD_TEMPERATURE:-1.0}"
OPD_TOKEN_KL_CLIP="${OPD_TOKEN_KL_CLIP:-null}"
OPD_MASK_SPECIAL_TOKENS="${OPD_MASK_SPECIAL_TOKENS:-true}"
REQUIRE_NONEMPTY_COT="${REQUIRE_NONEMPTY_COT:-true}"
ALLOW_LEGACY_COT="${ALLOW_LEGACY_COT:-false}"
RESPONSE_STARTS_IN_COT="${RESPONSE_STARTS_IN_COT:-true}"
ROLLOUT_AUDIT_ENABLED="${ROLLOUT_AUDIT_ENABLED:-true}"
ROLLOUT_AUDIT_FREQ="${ROLLOUT_AUDIT_FREQ:-10}"
ROLLOUT_AUDIT_GROUPS="${ROLLOUT_AUDIT_GROUPS:-2}"
ROLLOUT_AUDIT_INCLUDE_PRIVILEGED="${ROLLOUT_AUDIT_INCLUDE_PRIVILEGED:-true}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-false}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are in long-video qualitative analysis mode. First understand the whole video, then localize the question-relevant interval or path phase, and only then give evidence-based visual-spatial reasoning before the final answer.}"
ASSISTANT_PREFILL="${ASSISTANT_PREFILL:-Reasoning: Across the full video,}"
LR="${LR:-5.0e-7}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.10}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"
PPO_CLIP_LOW="${PPO_CLIP_LOW:-0.2}"
PPO_CLIP_HIGH="${PPO_CLIP_HIGH:-0.28}"
PPO_DUAL_CLIP="${PPO_DUAL_CLIP:-3.0}"
TRAIN_SEED="${TRAIN_SEED:-1}"
VIDEO_FPS="${VIDEO_FPS:-8.0}"
VIDEO_NFRAMES="${VIDEO_NFRAMES:-32}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-32}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
ROLLOUT_GPU_UTILIZATION="${ROLLOUT_GPU_UTILIZATION:-0.45}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-spatial_interactor_opd}"
PROJECT_NAME="${PROJECT_NAME:-spatial_interactor}"

if [[ "${OPD_ENABLED}" == "true" ]]; then
  PRIVILEGED_PROMPT_KEY="privileged_prompt"
  TRAINING_LABEL="OPD + GRPO"
elif [[ "${OPD_ENABLED}" == "false" ]]; then
  PRIVILEGED_PROMPT_KEY="null"
  REQUIRE_NONEMPTY_COT="false"
  ROLLOUT_AUDIT_INCLUDE_PRIVILEGED="false"
  TRAINING_LABEL="answer-only GRPO"
else
  echo "OPD_ENABLED must be true or false; got ${OPD_ENABLED}." >&2
  exit 2
fi

if (( ROLLOUT_BATCH_SIZE % ACTOR_GLOBAL_BATCH_SIZE != 0 )); then
  echo "ROLLOUT_BATCH_SIZE must be divisible by ACTOR_GLOBAL_BATCH_SIZE." >&2
  exit 2
fi
if (( ROLLOUT_TP > NGPUS )); then
  echo "ROLLOUT_TP cannot exceed the number of visible GPUs (${NGPUS})." >&2
  exit 2
fi
if [[ "${OPD_DIVERGENCE}" != "forward_kl" ]]; then
  echo "This bundle currently supports OPD_DIVERGENCE=forward_kl only." >&2
  exit 2
fi
if [[ "${OPD_TEACHER_MODE}" != "latest_pre_update" ]]; then
  echo "This bundle currently supports OPD_TEACHER_MODE=latest_pre_update only." >&2
  exit 2
fi
if [[ "${VIDEO_NFRAMES}" != "32" ]]; then
  echo "OPD privileged traces require VIDEO_NFRAMES=32; got ${VIDEO_NFRAMES}." >&2
  exit 2
fi
if [[ "${DATA_SHUFFLE}" != "true" && "${DATA_SHUFFLE}" != "false" ]]; then
  echo "DATA_SHUFFLE must be true or false." >&2
  exit 2
fi
if [[ "${NGPUS}" == "1" && "${DISABLE_KL}" != "true" ]]; then
  echo "Warning: one-GPU actor + rollout + reference KL can OOM; run smoke first." >&2
fi

echo "== Spatial-Interactor ${TRAINING_LABEL} training =="
echo "model: ${MODEL_PATH}"
echo "train: ${TRAIN_FILE}"
echo "media: ${MEDIA_ROOT}"
echo "gpus:  ${GPUS}"
echo "OPD: ${OPD_ENABLED}; group size: ${ROLLOUT_N}; fixed frames: ${VIDEO_NFRAMES}"
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "Configuration check passed; training was not started."
  exit 0
fi

if [[ "${RUN_PREFLIGHT}" == "true" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/preflight.py" \
    --easyr1-dir "${EASYR1_DIR}" \
    --model-path "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" \
    --media-root "${MEDIA_ROOT}" \
    --video-nframes "${VIDEO_NFRAMES}"
elif [[ "${RUN_PREFLIGHT}" != "false" ]]; then
  echo "RUN_PREFLIGHT must be true or false." >&2
  exit 2
fi

export PYTHONPATH="${EASYR1_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export FLASHINFER_WORKSPACE_BASE="${OUTPUT_ROOT}/runtime/${EXPERIMENT_NAME}/flashinfer_workspace"
export TORCH_EXTENSIONS_DIR="${OUTPUT_ROOT}/runtime/${EXPERIMENT_NAME}/torch_extensions"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/si_ray_${UID}}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard}"
export SPATIAL_INTERACTOR_FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT}"
mkdir -p "${OUTPUT_ROOT}" "${FLASHINFER_WORKSPACE_BASE}" "${TORCH_EXTENSIONS_DIR}" "${RAY_TMPDIR}" "${TENSORBOARD_DIR}"

TOKENIZER_OVERLAY="${TOKENIZER_OVERLAY:-${OUTPUT_ROOT}/tokenizer_overlays/${EXPERIMENT_NAME}}"
spatial_interactor_normalize_accidental_dollar_path TOKENIZER_OVERLAY
spatial_interactor_require_clean_absolute_path TOKENIZER_OVERLAY "${TOKENIZER_OVERLAY}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_tokenizer_overlay.py" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${TOKENIZER_OVERLAY}" >/dev/null
export SPATIAL_INTERACTOR_VLLM_TOKENIZER_PATH="${TOKENIZER_OVERLAY}"
# Optional fixed KV cache size avoids vLLM profiling races on shared GPUs.
export SPATIAL_INTERACTOR_VLLM_KV_CACHE_MEMORY_BYTES="${SPATIAL_INTERACTOR_VLLM_KV_CACHE_MEMORY_BYTES:-}"

cd "${EASYR1_DIR}"
"${PYTHON_BIN}" -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.privileged_prompt_key="${PRIVILEGED_PROMPT_KEY}" \
  data.answer_key=answer \
  data.image_key=__images__ \
  data.video_key=videos \
  data.image_dir="${MEDIA_ROOT}" \
  data.video_fps="${VIDEO_FPS}" \
  data.video_nframes="${VIDEO_NFRAMES}" \
  data.video_max_frames="${VIDEO_MAX_FRAMES}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
  data.val_batch_size="${ROLLOUT_BATCH_SIZE}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.seed="${TRAIN_SEED}" \
  data.format_prompt="${ROOT_DIR}/prompts/spatial_interactor_opd.jinja" \
  data.system_prompt="'${SYSTEM_PROMPT}'" \
  data.assistant_prefill="'${ASSISTANT_PREFILL}'" \
  data.min_pixels=3136 \
  data.max_pixels=65536 \
  data.filter_overlong_prompts=false \
  algorithm.adv_estimator=grpo \
  algorithm.disable_kl="${DISABLE_KL}" \
  algorithm.use_kl_loss="${USE_KL_LOSS}" \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_coef="${KL_COEF}" \
  algorithm.opd.enabled="${OPD_ENABLED}" \
  algorithm.opd.loss_weight="${OPD_LAMBDA}" \
  algorithm.opd.weight_hold_steps="${OPD_HOLD_STEPS}" \
  algorithm.opd.weight_decay_steps="${OPD_DECAY_STEPS}" \
  algorithm.opd.top_k="${OPD_TOP_K}" \
  algorithm.opd.temperature="${OPD_TEMPERATURE}" \
  algorithm.opd.token_kl_clip="${OPD_TOKEN_KL_CLIP}" \
  algorithm.opd.mask_special_tokens="${OPD_MASK_SPECIAL_TOKENS}" \
  algorithm.opd.cot_open_tag='<COT>' \
  algorithm.opd.cot_close_tag='</COT>' \
  algorithm.opd.allow_legacy_cot="${ALLOW_LEGACY_COT}" \
  algorithm.opd.response_starts_in_cot="${RESPONSE_STARTS_IN_COT}" \
  algorithm.opd.require_nonempty_cot="${REQUIRE_NONEMPTY_COT}" \
  worker.actor.global_batch_size="${ACTOR_GLOBAL_BATCH_SIZE}" \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.clip_ratio_low="${PPO_CLIP_LOW}" \
  worker.actor.clip_ratio_high="${PPO_CLIP_HIGH}" \
  worker.actor.clip_ratio_dual="${PPO_DUAL_CLIP}" \
  worker.actor.loss_avg_mode=token \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.trust_remote_code=true \
  worker.actor.model.freeze_vision_tower="${FREEZE_VISION_TOWER}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.optim.lr="${LR}" \
  worker.actor.optim.weight_decay=0.01 \
  worker.actor.optim.strategy=adamw_bf16 \
  worker.actor.optim.lr_warmup_ratio=0.03 \
  worker.actor.fsdp.enable_full_shard=true \
  worker.actor.fsdp.enable_cpu_offload=false \
  worker.actor.offload.offload_params=true \
  worker.actor.offload.offload_optimizer=true \
  worker.rollout.n="${ROLLOUT_N}" \
  worker.rollout.tensor_parallel_size="${ROLLOUT_TP}" \
  worker.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTILIZATION}" \
  worker.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS}" \
  worker.rollout.max_model_len="${MAX_MODEL_LEN}" \
  worker.rollout.enforce_eager="${ENFORCE_EAGER}" \
  worker.rollout.limit_images=0 \
  worker.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  worker.rollout.top_p="${ROLLOUT_TOP_P}" \
  worker.rollout.top_k="${ROLLOUT_TOP_K}" \
  worker.ref.fsdp.enable_full_shard=true \
  worker.ref.fsdp.enable_cpu_offload=true \
  worker.ref.offload.offload_params=true \
  worker.reward.reward_function="${ROOT_DIR}/rewards/spatial_video_reward.py:compute_score" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='["file","tensorboard"]' \
  trainer.rollout_audit.enabled="${ROLLOUT_AUDIT_ENABLED}" \
  trainer.rollout_audit.log_freq="${ROLLOUT_AUDIT_FREQ}" \
  trainer.rollout_audit.groups_per_step="${ROLLOUT_AUDIT_GROUPS}" \
  trainer.rollout_audit.include_privileged_context="${ROLLOUT_AUDIT_INCLUDE_PRIVILEGED}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.validate_at_end="${VALIDATE_AT_END}" \
  trainer.val_freq="${VAL_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.save_limit=2 \
  trainer.save_at_end="${SAVE_AT_END}" \
  trainer.save_checkpoint_path="${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.find_last_checkpoint=true \
  trainer.val_generations_to_log=4 \
  "$@"
