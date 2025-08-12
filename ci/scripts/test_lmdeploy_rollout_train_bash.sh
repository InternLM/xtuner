set -ex

export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_ddd/opencompass/models/modelscope_hub/QwQ/Qwen3-30B-A3B-250425"
export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_razor/duanyanhui/data/math-500.jsonl"
export XTUNER_USE_LMDEPLOY=1 
export PYTHONPATH='/cpfs01/shared/llm_razor/chenchiyu/rl_dev/lmdeploy_rl_new':'/cpfs01/shared/llm_ddd/caoweihan/projects/Liger-Kernel/src/':'.':$PYTHONPATH 

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

TORCH_LOGS="recompiles" python ci/scripts/test_ray_rl_lmdeploy.py \
    --work-dir work_dirs/debug_lmdeploy \
    --rollout-global-batch-size 4 \
    --train-global-batch-size 8 \
    --prompt-repeat-k 8 \
    --pack-max-length 4096 \
    --optimizer-disable-foreach
