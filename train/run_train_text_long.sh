#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412
# Keep HF cache off /home (often near full on this host).
export HF_HOME="${HF_HOME:-/tmp/zh/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"
# Silence numexpr thread-cap warning seen at startup.
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"

NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" && -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[run_train_text_long] resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# Example:
#   DATASET=HuggingFaceH4/ultrachat_200k \\
#   DATASET_TEXT_FIELD=messages \\
#   LONG_DATASET=THUDM/LongBench \\
#   LONG_DATASET_TEXT_FIELD=context \\
#   bash train/run_train_text_long.sh
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_TEXT_FIELD="${DATASET_TEXT_FIELD:-text}"
DATASET_LIMIT="${DATASET_LIMIT:-4096}"

LONG_DATASET="${LONG_DATASET:-}"
LONG_DATASET_SPLIT="${LONG_DATASET_SPLIT:-train}"
LONG_DATASET_TEXT_FIELD="${LONG_DATASET_TEXT_FIELD:-text}"
LONG_DATASET_LIMIT="${LONG_DATASET_LIMIT:-1024}"

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 12368 \
  train_text_budget.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --output_dir /tmp/qwen3vl-sparse-attn-text \
  --dataset "${DATASET}" \
  --dataset_split "${DATASET_SPLIT}" \
  --dataset_text_field "${DATASET_TEXT_FIELD}" \
  --dataset_limit "${DATASET_LIMIT}" \
  --max_length 8192 \
  --num_train_epochs 0.5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --train_layer_ids 0-17 \
  --budget_granularity head \
  --budget_init_ratio 1 \
  --budget_lambda 0 \
  --budget_ste_temperature 1.0 \
  --attn_implementation flash_attention_2 \
  --learning_rate 0.01 \
  --warmup_ratio 0.08 \
  --logging_steps 1 \
  --bf16 \
  --report_to none \
  --save_every_epoch_fraction 0.2 \
  --save_at_end \
  "${RESUME_ARGS[@]}" \
  $( [[ -n "${LONG_DATASET}" ]] && printf '%s' "--long_dataset ${LONG_DATASET} --long_dataset_split ${LONG_DATASET_SPLIT} --long_dataset_text_field ${LONG_DATASET_TEXT_FIELD} --long_dataset_limit ${LONG_DATASET_LIMIT}" ) \
  "$@"
