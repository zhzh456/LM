#!/usr/bin/env bash
# STEP 1: 全层联合 profiling（--all，prefill 保留率 0.5）
# 评测参数对齐 qwen3vl.sh；attn 用 eager（flash 无法在 kernel 内按分数动态选 KV）

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412
export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$(pwd)/train:${PYTHONPATH:-}"

python train/budget_opt/step1_layer_sensitivity.py \
  --model_path /home/zhanghao360/model/Qwen3-VL-4B-Instruct \
  --tasks tomato \
  --batch_size 1 \
  --limit 128 \
  --max_pixels 12845056 \
  --max_num_frames 16 \
  --attn_implementation eager \
  --all \
  --ratios 0.5 \
  "$@"