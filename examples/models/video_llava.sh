# cd /path/to/lmms-eval
# python3 -m pip install -e .;

# python3 -m pip install transformers --upgrade;
# python3 -m pip install av sentencepiece;


# TASK=$1
# echo $TASK
# TASK_SUFFIX="${TASK//,/_}"
# echo $TASK_SUFFIX

export http_proxy=http://10.229.18.27:8412 export https_proxy=http://10.229.18.27:8412 export HTTP_PROXY=http://10.229.18.27:8412 export HTTPS_PROXY=http://10.229.18.27:8412


accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
    --model video_llava \
    --tasks "tomato" \
    --batch_size 1 \
    --limit 8