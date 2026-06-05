# export HF_HOME="~/.cache/huggingface"
# # pip3 install transformers==4.57.1 (Qwen3VL models)
# # pip3 install ".[qwen]" (for Qwen's dependencies)

# # Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen3_vl \
#     --model_args=pretrained=Qwen/Qwen3-VL-4B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
#     --tasks "mmmu_val,mmbench_en_dev,ocrbench,realworldqa,mmstar" \
#     --batch_size 1
    # --tasks "tomato" 786432 \

export http_proxy=http://10.229.18.27:8412 export https_proxy=http://10.229.18.27:8412 export HTTP_PROXY=http://10.229.18.27:8412 export HTTPS_PROXY=http://10.229.18.27:8412

# TOMATO 评测：max_pixels=12845056（官方常用值）
# 视频时 qwen_vl_utils 可能打印 per-frame cap 786432 的 warning，会自动截断，不影响评测
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=/home/zhanghao360/model/Qwen3-VL-4B-Instruct,max_pixels=12845056,max_num_frames=16,interleave_visuals=False,attn_implementation=flash_attention_2,log_input_length=True,print_generation=True \
    --tasks "tomato" \
    --batch_size 1 \
    --limit 32

# # ---------------------------------------------------------------------------
# # 1 条样本：第一个 decode token 对历史 key 的 pre-softmax 注意力（按距离 d 聚合）
# # 供训练 --rel_pos_init_path 使用
# #   原始逐 head：/tmp/qwen3vl-baseline-decode-attn/sample_00000/
# #   训练初始化：/tmp/baseline_relpos_scores.pt
# # ---------------------------------------------------------------------------
# BASELINE_ATTN_DUMP=/tmp/qwen3vl-baseline-decode-attn
# BASELINE_RELPOS_INIT=/tmp/baseline_relpos_scores.pt

# accelerate launch --num_processes=1 --main_process_port=12347 -m lmms_eval \
#     --model qwen3_vl \
#     --model_args=pretrained=/home/zhanghao360/model/Qwen3-VL-4B-Instruct,max_pixels=524288,min_pixels=200704,max_num_frames=16,interleave_visuals=False,attn_implementation=flash_attention_2,log_input_length=True,print_generation=True,save_attn_scores_dir=${BASELINE_ATTN_DUMP} \
#     --tasks tomato \
#     --batch_size 1 \
#     --limit 1

# python /home/zhanghao360/work/lmms-eval/train/export_relpos_init_from_attn_dump.py \
#     --sample-dir "${BASELINE_ATTN_DUMP}/sample_00000" \
#     --output "${BASELINE_RELPOS_INIT}" \
#     --num-buckets 16384

# echo "[baseline init] rel_pos vectors: ${BASELINE_RELPOS_INIT}"
# echo "[train] use: bash train/run_train.sh --rel_pos_init_path ${BASELINE_RELPOS_INIT}"