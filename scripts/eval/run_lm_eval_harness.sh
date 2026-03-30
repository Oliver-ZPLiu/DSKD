#!/bin/bash
set -euo pipefail

# =========================
# Built-in config (edit here)
# =========================
# Optional override via env vars is supported for all *_DEFAULT items.

MODE_DEFAULT="all"  # mc | ifeval | all
CUDA_DEVICES_DEFAULT="0,1,2,3"
MASTER_PORT_DEFAULT=29601
GPUS_PER_NODE_DEFAULT=4

BASE_PATH_DEFAULT="/path/to/DSKD"
MODEL_PATH_DEFAULT="/path/to/base_model"
OUTPUT_DIR_DEFAULT="/path/to/output_dir"

BATCH_SIZE_MC_DEFAULT="auto"
BATCH_SIZE_IFEVAL_DEFAULT=1
DTYPE_DEFAULT="bfloat16"
TRUST_REMOTE_CODE_DEFAULT="True"
EXTRA_MODEL_ARGS_DEFAULT=""
IFEVAL_GEN_KWARGS_DEFAULT="max_gen_toks=1280"
LIMIT_DEFAULT=""
NUM_FEWSHOT_DEFAULT=""

# =========================
# Offline HF cache config
# =========================
HF_HOME_DEFAULT="/docker/l00625974/.cache/huggingface"
HUGGINGFACE_HUB_CACHE_DEFAULT="${HF_HOME_DEFAULT}/hub"
HF_DATASETS_CACHE_DEFAULT="${HF_HOME_DEFAULT}/datasets"
HF_MODULES_CACHE_DEFAULT="${HF_HOME_DEFAULT}/modules"

# =========================
# Resolve runtime config
# =========================
MODE="${MODE:-$MODE_DEFAULT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-$CUDA_DEVICES_DEFAULT}"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$GPUS_PER_NODE_DEFAULT}"
BASE_PATH="${BASE_PATH:-$BASE_PATH_DEFAULT}"
MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
BATCH_SIZE_MC="${BATCH_SIZE_MC:-$BATCH_SIZE_MC_DEFAULT}"
BATCH_SIZE_IFEVAL="${BATCH_SIZE_IFEVAL:-$BATCH_SIZE_IFEVAL_DEFAULT}"
DTYPE="${DTYPE:-$DTYPE_DEFAULT}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-$TRUST_REMOTE_CODE_DEFAULT}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-$EXTRA_MODEL_ARGS_DEFAULT}"
IFEVAL_GEN_KWARGS="${IFEVAL_GEN_KWARGS:-$IFEVAL_GEN_KWARGS_DEFAULT}"
LIMIT="${LIMIT:-$LIMIT_DEFAULT}"
NUM_FEWSHOT="${NUM_FEWSHOT:-$NUM_FEWSHOT_DEFAULT}"

export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE_DEFAULT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_DATASETS_CACHE_DEFAULT}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_MODULES_CACHE_DEFAULT}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH="${BASE_PATH}"

# =========================
# Preflight checks
# =========================
if [[ "${MODE}" != "mc" && "${MODE}" != "ifeval" && "${MODE}" != "all" ]]; then
  echo "[ERROR] MODE must be one of: mc | ifeval | all, got: ${MODE}"
  exit 1
fi

if [ ! -d "${HF_HOME}" ]; then
  echo "[ERROR] HF_HOME not found: ${HF_HOME}"
  exit 1
fi
if [ ! -d "${HUGGINGFACE_HUB_CACHE}" ]; then
  echo "[ERROR] HUGGINGFACE_HUB_CACHE not found: ${HUGGINGFACE_HUB_CACHE}"
  exit 1
fi
mkdir -p "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${OUTPUT_DIR}/logs"

echo "[INFO] MODE=${MODE}"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "[INFO] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[INFO] HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "[INFO] Offline mode: HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE}"
if [ -n "${EXTRA_MODEL_ARGS}" ]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

TASKS_MC="piqa,arc_challenge,boolq,mmlu,agieval_en,agieval_cn"
TASKS_IFEVAL="ifeval"

run_eval () {
  local tasks="$1"
  local batch_size="$2"
  local output_path="$3"
  local gen_kwargs="${4:-}"

  local cmd=(
    accelerate launch
    --num_processes "${GPUS_PER_NODE}"
    --main_process_port "${MASTER_PORT}"
    -m lm_eval
    --model hf
    --model_args "${MODEL_ARGS}"
    --tasks "${tasks}"
    --device cuda
    --batch_size "${batch_size}"
    --output_path "${output_path}"
  )

  if [ -n "${NUM_FEWSHOT}" ]; then
    cmd+=(--num_fewshot "${NUM_FEWSHOT}")
  fi
  if [ -n "${LIMIT}" ]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [ -n "${gen_kwargs}" ]; then
    cmd+=(--gen_kwargs "${gen_kwargs}")
  fi

  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

status_mc=0
status_ifeval=0

if [[ "${MODE}" == "mc" || "${MODE}" == "all" ]]; then
  set +e
  run_eval "${TASKS_MC}" "${BATCH_SIZE_MC}" "${OUTPUT_DIR}/run_mc"
  status_mc=$?
  set -e
  if [ ${status_mc} -ne 0 ]; then
    echo "[WARN] MC evaluation failed with code ${status_mc}"
  fi
fi

if [[ "${MODE}" == "ifeval" || "${MODE}" == "all" ]]; then
  set +e
  run_eval "${TASKS_IFEVAL}" "${BATCH_SIZE_IFEVAL}" "${OUTPUT_DIR}/run_ifeval" "${IFEVAL_GEN_KWARGS}"
  status_ifeval=$?
  set -e
  if [ ${status_ifeval} -ne 0 ]; then
    echo "[WARN] IFEval evaluation failed with code ${status_ifeval}"
  fi
fi

if [[ "${MODE}" == "mc" ]]; then
  exit ${status_mc}
fi
if [[ "${MODE}" == "ifeval" ]]; then
  exit ${status_ifeval}
fi

if [[ ${status_mc} -ne 0 || ${status_ifeval} -ne 0 ]]; then
  exit 1
fi

echo "[DONE] Completed mode=${MODE}. Outputs under: ${OUTPUT_DIR}"
