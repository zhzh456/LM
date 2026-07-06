#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412
NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Reduce CUDA allocator fragmentation (helps large contiguous backward allocs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

# Optional: RESUME_FROM_CHECKPOINT=/tmp/qwen3vl-sparse-attn/checkpoint-1670 bash train/run_train.sh
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" && -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[run_train] resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# STE surrogate strength presets (smaller -> stronger ratio gradient):
#   STE_TEMP=1.0 (recommended stable default)
#   STE_TEMP=0.7 (balanced, previous default)
#   STE_TEMP=0.5 (aggressive)
STE_TEMP="${STE_TEMP:-1.0}"
echo "[run_train] budget_ste_temperature=${STE_TEMP}"

# max_pixels=12845056 (video clips to 786432/frame, seq~6K)
# train target: layer 0 per-head budget ratio r (keep ceil(r*n_valid) top-QK keys)
# loss: final task CE + budget regularization (no distill)
# checkpoint every 0.05 epoch -> checkpoint-{step}/sparse_rel_pos_bias.pt
accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 12348 \
  train.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --output_dir /tmp/qwen3vl-sparse-attn \
  --max_pixels 12845056 \
  --min_pixels 200704 \
  --num_frames 16 \
  --rel_pos_buckets 16384 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 0.1 \
  --train_layer_ids 0 \
  --training_target budget \
  --loss_mode task_ce \
  --budget_granularity head \
  --budget_init_ratio 0.01 \
  --budget_lambda 0 \
  --budget_ste_temperature "${STE_TEMP}" \
  --attn_implementation flash_attention_2 \
  --learning_rate 0.1 \
  --warmup_ratio 0.08 \
  --logging_steps 1 \
  --bf16 \
  --distill_every_n_steps 1 \
  "${RESUME_ARGS[@]}" \
  --report_to none \
  --save_every_epoch_fraction 0.2 \
  --save_at_end \
  "$@"
