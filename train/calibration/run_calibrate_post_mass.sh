#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"

python top_p_post_mass_calibrate.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --distance_excel_dir /tmp/qwen3vl-calib-test/update/all/excel/layer_00 \
  --distance_topk_ratio 0 \
  --pre_qk_topk_ratio 0.2 \
  --max_pixels 12845056 \
  --min_pixels 200704 \
  --num_frames 16 \
  --dataset_fraction 0.01 \
  --batch_size 1 \
  --layer_id 0 \
  --stage all \
  --print_mode mean \
  --attn_implementation eager \
  --bf16 \
  "$@"
  