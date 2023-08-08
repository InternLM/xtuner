CUDA_VISIBLE_DEVICES=1,2,3,4 deepspeed train_alpaca-qlora_with_mmchat_deepspeed.py \
    --model_name_or_path internlm/internlm-7b \
    --use_qlora True \
    --use_lora False \
    --output_dir work_dirs/mmchat_ds_internlm-7b_qlora
