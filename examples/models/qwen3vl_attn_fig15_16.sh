#!/usr/bin/env bash
# 2 样本：baseline + sparse，第一个 decode token 的 pre-softmax 注意力
# 图：/tmp/Figure/1/5（sample0）、/tmp/Figure/1/6（sample1），前 2 层各 head

set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

BASELINE_DUMP=/tmp/qwen3vl-baseline-attn-scores-fig
SPARSE_DUMP=/tmp/qwen3vl-sparse-attn-scores-fig
SPARSE_WEIGHTS=/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt
MODEL=/home/zhanghao360/model/Qwen3-VL-4B-Instruct

COMMON="max_pixels=65536,min_pixels=3136,max_num_frames=16,interleave_visuals=False,attn_implementation=flash_attention_2,log_input_length=True,print_generation=True"

echo "[1/3] baseline eval (limit 2) -> ${BASELINE_DUMP}"
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
  --model qwen3_vl \
  --model_args=pretrained=${MODEL},${COMMON},save_attn_scores_dir=${BASELINE_DUMP} \
  --tasks tomato \
  --batch_size 1 \
  --limit 2

echo "[2/3] sparse eval (limit 2) -> ${SPARSE_DUMP}"
echo "  weights: ${SPARSE_WEIGHTS}"
accelerate launch --num_processes=1 --main_process_port=12347 -m lmms_eval \
  --model qwen3_vl_sparse \
  --model_args=pretrained=${MODEL},sparse_rel_pos_path=${SPARSE_WEIGHTS},rel_pos_buckets=4096,${COMMON},save_attn_scores_dir=${SPARSE_DUMP} \
  --tasks tomato \
  --batch_size 1 \
  --limit 2

echo "[3/3] plot -> /tmp/Figure/1/5 and /tmp/Figure/1/6"
python train/plot_attn_scores_compare.py \
  --baseline-dir "${BASELINE_DUMP}" \
  --sparse-dir "${SPARSE_DUMP}" \
  --out-root /tmp/Figure/1 \
  --sample1-subdir 5 \
  --sample2-subdir 6 \
  --max-layers 2 \
  --max-distance 511

echo "done. figures: /tmp/Figure/1/5 /tmp/Figure/1/6"
