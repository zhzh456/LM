#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412
# Reduce CUDA allocator fragmentation (helps large contiguous backward allocs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"

cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

# Optional: RESUME_FROM_CHECKPOINT=/tmp/qwen3vl-sparse-attn/checkpoint-1670 bash train/run_train.sh
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" && -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[run_train] resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# Union sparse: content top-500 ∪ distance top-1000 (STE) → RoPE sparse attn; gap-recall loss only.
accelerate launch --config_file ./accelerate_single_gpu.yaml train.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --output_dir /tmp/qwen3vl-sparse-attn \
  --max_pixels 12845056 \
  --min_pixels 200704 \
  --num_frames 16 \
  --rel_pos_buckets 16384 \
  --content_topk_k 500 \
  --sparse_topk_k 1000 \
  --ste_tau 0.25 \
  --sparse_gap_recall_weight 1.0 \
  --sparse_dist_score_scale 0.75 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --train_layer_id 0 \
  --attn_implementation flash_attention_2 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --bf16 \
  --distill_every_n_steps 1 \
  "${RESUME_ARGS[@]}" \
  --report_to none \
  --save_every_epoch_fraction 0.25 \
  --save_at_end \
  "$@"

# bash train/run_train.sh --rel_pos_init_path /tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt
# layer0 backup: /tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias_layer0.pt
# layer18 backup: /tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias_layer18.pt
# layer35 backup: /tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias_layer35.pt