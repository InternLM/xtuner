set -ex

# export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_ddd/opencompass/models/modelscope_hub/QwQ/Qwen3-30B-A3B-250425"
export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_razor/huanghaian/new_model/Qwen3-8B"
export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_ddd/lishuaibin/ckpt/Qwen/Qwen2.5-Math-7B"

# export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_razor/huanghaian/code/refactor_xtuner/gsm8k/train.jsonl"
export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_ddd/lishuaibin/verl_dirs/data/dapo-math-17k_1.jsonl"
export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_ddd/lishuaibin/verl_dirs/data/gsm8k_1.jsonl"
export ROLLOUT_TEST_DATA_PATH="/cpfs01/shared/llm_razor/huanghaian/code/refactor_xtuner/gsm8k/test.jsonl"

# export PYTHONPATH='/cpfs01/shared/llm_razor/caoweihan/projects/lmdeploy':'/cpfs01/shared/llm_ddd/caoweihan/projects/Liger-Kernel/src/':'.':$PYTHONPATH 
export PYTHONPATH='/cpfs01/shared/llm_razor/duanyanhui/workspace/lmdeploy/lmdeploy':'/cpfs01/shared/llm_ddd/caoweihan/projects/Liger-Kernel/src/':'.':$PYTHONPATH 

export UVICORN_LOG_LEVEL="CRITICAl"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export PYTHONPATH='/cpfs01/user/lishuaibin/projects/202509/xtuner_github/_push/xtuner_github/':$PYTHONPATH

# OUTPUT_DIR='work_dirs/dense_8b_gsm8k_grpo_fix_shuaibin'
OUTPUT_DIR='work_dirs/debug_7B_lmdeploy-yanhui_vllm'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export XTUNER_USE_LMDEPLOY=1 
python ci/scripts/test_lmdeploy_vllm.py \
    --work-dir "$OUTPUT_DIR" \
    --global-batch-size 1 \
    --top-k 0 \
    --top-p 1.0 \
    --temperature 1.0 \
    --prompt-repeat-k 64 \
    --max-prompt-length 2048 \
    --max-response-length 8192 \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

# sleep 2

# export XTUNER_USE_VLLM=1
# python ci/scripts/test_lmdeploy_vllm.py \
#     --work-dir "$OUTPUT_DIR" \
#     --global-batch-size 1 \
#     --top-k -1 \
#     --top-p 1.0 \
#     --temperature 1.0 \
#     --prompt-repeat-k 64 \
#     --max-prompt-length 2048 \
#     --max-response-length 8192 \
#     --vllm \
#     2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

# sleep 2