#!/bin/bash
set -euo pipefail

# Online machine (network available, no GPU required):
# Pre-fetch lm-eval task data into local Hugging Face cache.

HF_HOME_DEFAULT="/docker/l00625974/Rebuttal/huggingface_cache"
HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"

TASKS_DEFAULT="piqa,arc_challenge,boolq,mmlu,agieval_en,agieval_cn,ifeval"
TASKS="${TASKS:-$TASKS_DEFAULT}"

MODEL_ARGS_DEFAULT="pretrained=sshleifer/tiny-gpt2,dtype=float32,trust_remote_code=True"
MODEL_ARGS="${MODEL_ARGS:-$MODEL_ARGS_DEFAULT}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "[INFO] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[INFO] HF_MODULES_CACHE=$HF_MODULES_CACHE"
echo "[INFO] TASKS=$TASKS"

python -m lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --device cpu \
  --batch_size 1 \
  --limit 1

echo "[DONE] Prefetch finished. You can now package: $HF_HOME"
