#!/usr/bin/env bash
# STEP 1: 逐层 sensitivity（每层单独稀疏，其余 dense）
# 4 卡 DP；循环 ratio: 1.0, 0.9, ... , 0.0；每轮先测 Acc_full 再逐层 profiling

set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/budget-opt/step1/independent_limit1484}"

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$(pwd)/train:${PYTHONPATH:-}"

RATIOS=(0.3)

for RATIO in "${RATIOS[@]}"; do
  OUT_DIR="${OUTPUT_ROOT}/r_${RATIO}"
  echo "[step1] running ratio=${RATIO}, output_dir=${OUT_DIR}"

  accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=12346 \
    train/budget_opt/step1_layer_sensitivity.py \
    --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
    --tasks tomato \
    --batch_size 1 \
    --limit 1484 \
    --max_pixels 12845056 \
    --max_num_frames 16 \
    --attn_implementation eager \
    --measure_acc_full \
    --ratios "${RATIO}" \
    --output_dir "${OUT_DIR}" \
    "$@"
done



