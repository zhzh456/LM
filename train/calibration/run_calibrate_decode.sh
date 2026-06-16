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

# First decode-step calibration: prefill + first decode token (teacher-forced answer[0]).
# Output: {output_dir}/decode/
python top_p_distance_calibrate_decode.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --output_dir /tmp/qwen3vl-calib-test/update \
  --max_pixels 12845056 \
  --min_pixels 200704 \
  --num_frames 16 \
  --dataset_fraction 1 \
  --batch_size 1 \
  --layer_id 0 \
  --save_interval 0 \
  --top_p 0.95 \
  --attn_implementation eager \
  --bf16 \
  "$@"
