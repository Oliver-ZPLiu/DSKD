#!/bin/bash
set -e

# =========================
# Built-in config (edit here)
# =========================
# Mirrors:
# 0,1,2,3 29601 4 \
# /path/to/DSKD \
# /path/to/base_model \
# /path/to/lora_ckpt \
# /path/to/output_dir \
# auto 1 bfloat16

CUDA_DEVICES_DEFAULT="0,1,2,3"
MASTER_PORT_DEFAULT=29601
GPUS_PER_NODE_DEFAULT=4
BASE_PATH_DEFAULT="/path/to/DSKD"
MODEL_PATH_DEFAULT="/path/to/base_model"
PEFT_PATH_DEFAULT="/path/to/lora_ckpt"
OUTPUT_DIR_DEFAULT="/path/to/output_dir"
BATCH_SIZE_MC_DEFAULT="auto"
BATCH_SIZE_IFEVAL_DEFAULT=1
DTYPE_DEFAULT="bfloat16"

# Optional override via env vars:
# CUDA_DEVICES, MASTER_PORT, GPUS_PER_NODE, BASE_PATH, MODEL_PATH,
# PEFT_PATH, OUTPUT_DIR, BATCH_SIZE_MC, BATCH_SIZE_IFEVAL, DTYPE
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-$CUDA_DEVICES_DEFAULT}"
MASTER_PORT="${MASTER_PORT:-$MASTER_PORT_DEFAULT}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$GPUS_PER_NODE_DEFAULT}"
BASE_PATH="${BASE_PATH:-$BASE_PATH_DEFAULT}"
MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
PEFT_PATH="${PEFT_PATH:-$PEFT_PATH_DEFAULT}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
BATCH_SIZE_MC="${BATCH_SIZE_MC:-$BATCH_SIZE_MC_DEFAULT}"
BATCH_SIZE_IFEVAL="${BATCH_SIZE_IFEVAL:-$BATCH_SIZE_IFEVAL_DEFAULT}"
DTYPE="${DTYPE:-$DTYPE_DEFAULT}"

OPTS=""
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --peft-path ${PEFT_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --dtype ${DTYPE}"
OPTS+=" --device cuda"
OPTS+=" --batch-size-mc ${BATCH_SIZE_MC}"
OPTS+=" --batch-size-ifeval ${BATCH_SIZE_IFEVAL}"
OPTS+=" --launcher accelerate"
OPTS+=" --num-processes ${GPUS_PER_NODE}"
OPTS+=" --main-process-port ${MASTER_PORT}"

export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}

CMD="python ${BASE_PATH}/scripts/eval/run_lm_eval_harness.py ${OPTS}"
echo ${CMD}
${CMD}
