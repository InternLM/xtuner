export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_razor/duanyanhui/data/math-500.jsonl"
# export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_razor/duanyanhui/model/qwen3-4B" 
export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--Qwen--Qwen3-30B-A3B/snapshots/4c446470ba0aec43e22ac1128f9ffd915f338ba3/"
export XTUNER_USE_VLLM=1 
export PYTHONPATH='/cpfs01/shared/llm_ddd/caoweihan/projects/Liger-Kernel/src/':'.':$PYTHONPATH 

python ci/scripts/test_ray_rl.py \
    --work-dir work_dirs/debug_ray_rl2 \
    --global-batch-size 4 \
    --prompt-repeat-k 8 \
    --pack-max-length 1024 #\
    # --debug-train-only
