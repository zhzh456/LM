#!/usr/bin/env bash
# TOMATO 评测：训练好的纯位置 attention（rel_pos_bias），与 qwen3vl.sh 对齐

export http_proxy=http://10.229.18.27:8412
export https_proxy=http://10.229.18.27:8412
export HTTP_PROXY=http://10.229.18.27:8412
export HTTPS_PROXY=http://10.229.18.27:8412

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
  --model qwen3_vl_sparse \
  --model_args=pretrained=/home/zhanghao360/model/Qwen3-VL-4B-Instruct,sparse_rel_pos_path=/tmp/qwen3vl-sparse-attn/final/sparse_rel_pos_bias.pt,rel_pos_buckets=16384,sparse_layer_id=18,max_pixels=12845056,min_pixels=200704,max_num_frames=16,interleave_visuals=False,attn_implementation=flash_attention_2,log_input_length=True,print_generation=True,save_attn_scores_dir=/tmp/qwen3vl-sparse-attn-scores \
  --tasks "tomato" \
  --batch_size 1 \
  --limit 2
