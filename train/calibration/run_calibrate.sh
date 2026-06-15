#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"

# Baseline calibration: pre-RoPE vs post-RoPE top-p; text decoder stops at --layer_id (no lm_head).
# query_scope last -> {output_dir}/last/ ; all -> {output_dir}/all/
# Override: bash run_calibrate.sh --query_scope all --dataset_fraction 0.1
python top_p_distance_calibrate.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --output_dir /tmp/qwen3vl-calib-test/update \
  --max_pixels 12845056 \
  --min_pixels 200704 \
  --num_frames 16 \
  --dataset_fraction 1 \
  --batch_size 1 \
  --layer_id 0 \
  --query_scope all \
  --save_interval 100 \
  --top_p 0.95 \
  --attn_implementation eager \
  --bf16 \
  "$@"
