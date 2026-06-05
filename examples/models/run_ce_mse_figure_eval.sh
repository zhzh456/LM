#!/usr/bin/env bash
# 每个 TASK 各 2 样本：baseline + sparse（ce copy.pt / mse .pt）
# 输出：/tmp/Figure/{ce_model,mse_model}/${TASK}/sample_{1,2}/{attention,metrics}
# 默认 TASKS="tomato mmbench_en_dev"；仅 tomato：TASKS=tomato
# 推理用 --predict_only，不走 MMBench GPT 打分（避免 API 403 卡住）

set -euo pipefail

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

TASKS="${TASKS:-tomato mmbench_en_dev}"
MODEL=/home/zhanghao360/model/Qwen3-VL-4B-Instruct
COMMON="max_pixels=65536,min_pixels=3136,max_num_frames=16,interleave_visuals=False,attn_implementation=flash_attention_2,log_input_length=True,print_generation=True"
SPARSE_COPY=/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias_copy.pt
SPARSE_MSE=/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt
DUMP_ROOT=/tmp/attn-dump-ce-mse
FIG_ROOT=/tmp/Figure
PREDICT_OUT=/tmp/lmms-eval-predict-only

run_baseline() {
  local task=$1
  local out="${DUMP_ROOT}/baseline-${task}"
  echo "[baseline] ${task} -> ${out}"
  accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=${MODEL},${COMMON},save_attn_scores_dir=${out} \
    --tasks "${task}" \
    --batch_size 1 \
    --limit 2 \
    --predict_only \
    --output_path "${PREDICT_OUT}"
}

run_sparse() {
  local task=$1 weights=$2 tag=$3 port=$4
  local out="${DUMP_ROOT}/sparse-${tag}-${task}"
  echo "[sparse ${tag}] ${task} weights=${weights} -> ${out}"
  accelerate launch --num_processes=1 --main_process_port="${port}" -m lmms_eval \
    --model qwen3_vl_sparse \
    --model_args=pretrained=${MODEL},sparse_rel_pos_path=${weights},rel_pos_buckets=4096,${COMMON},save_attn_scores_dir=${out} \
    --tasks "${task}" \
    --batch_size 1 \
    --limit 2 \
    --predict_only \
    --output_path "${PREDICT_OUT}"
}

postprocess() {
  local task=$1 model_tag=$2 sparse_tag=$3
  local bdir="${DUMP_ROOT}/baseline-${task}"
  local sdir="${DUMP_ROOT}/sparse-${sparse_tag}-${task}"
  local base="${FIG_ROOT}/${model_tag}/${task}"
  for i in 0 1; do
    local sid=$(printf '%05d' "${i}")
    local human=$((i + 1))
    echo "[plot] ${model_tag}/${task} sample_${human}"
    python train/plot_attn_scores_compare.py \
      --baseline-dir "${bdir}" --sparse-dir "${sdir}" \
      --sample-id "${sid}" --out-dir "${base}/sample_${human}/attention" --max-layers 2
    python train/compute_attn_ndcg_metrics.py \
      --baseline-dir "${bdir}" --sparse-dir "${sdir}" \
      --sample-id "${sid}" --out-dir "${base}/sample_${human}/metrics" --max-layers 2
  done
}

# 若 dump 已存在可跳过评测，只跑 postprocess：
#   SKIP_EVAL=1 bash examples/models/run_ce_mse_figure_eval.sh
# 只补 mmbench：TASKS=mmbench_en_dev bash examples/models/run_ce_mse_figure_eval.sh
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  mkdir -p "${FIG_ROOT}"
  for TASK in ${TASKS}; do
    [[ -d "${DUMP_ROOT}/baseline-${TASK}" ]] || run_baseline "${TASK}"
    [[ -d "${DUMP_ROOT}/sparse-ce-${TASK}" ]] || run_sparse "${TASK}" "${SPARSE_COPY}" ce 12347
    [[ -d "${DUMP_ROOT}/sparse-mse-${TASK}" ]] || run_sparse "${TASK}" "${SPARSE_MSE}" mse 12357
  done
fi

for TASK in ${TASKS}; do
  postprocess "${TASK}" ce_model ce
  postprocess "${TASK}" mse_model mse
done

echo "done."
for TASK in ${TASKS}; do
  echo "  ${FIG_ROOT}/ce_model/${TASK}/sample_{1,2}/{attention,metrics}  (copy.pt)"
  echo "  ${FIG_ROOT}/mse_model/${TASK}/sample_{1,2}/{attention,metrics}  (.pt)"
done
