# # install lmms_eval without building dependencies
# cd lmms_eval;
# pip install --no-deps -U -e .

# # install LLaVA without building dependencies
# cd LLaVA
# pip install --no-deps -U -e .

# # install all the requirements that require for reproduce llava results
# pip install -r llava_repr_requirements.txt

# # Run and exactly reproduce llava_v1.5 results!
# # mme as an example
# accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,use_flash_attention_2=False,device_map=auto"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/

export http_proxy=http://10.229.18.27:8412 export https_proxy=http://10.229.18.27:8412 export HTTP_PROXY=http://10.229.18.27:8412 export HTTPS_PROXY=http://10.229.18.27:8412

accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/zhanghao360/model/llava-v1.5-7b,device_map=auto" \
    --tasks "mmbench_en_dev" \
    --batch_size 1 \
    --limit 8