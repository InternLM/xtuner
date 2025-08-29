set -ex

# export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_ddd/opencompass/models/modelscope_hub/QwQ/Qwen3-30B-A3B-250425"
export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_razor/huanghaian/new_model/Qwen3-8B"
export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_razor/huanghaian/code/refactor_xtuner/gsm8k/train.jsonl"
export ROLLOUT_TEST_DATA_PATH="/cpfs01/shared/llm_razor/huanghaian/code/refactor_xtuner/gsm8k/test.jsonl"
export XTUNER_USE_LMDEPLOY=1 
export PYTHONPATH='/cpfs01/shared/llm_razor/caoweihan/projects/lmdeploy':'/cpfs01/shared/llm_ddd/caoweihan/projects/Liger-Kernel/src/':'.':$PYTHONPATH 
export UVICORN_LOG_LEVEL="CRITICAl"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# OUTPUT_DIR='work_dirs/dense_8b_gsm8k_grpo_fix_shuaibin'
OUTPUT_DIR='work_dirs/debug'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python ci/scripts/test_grpo_trainer.py \
    --total-epochs 15 \
    --work-dir "$OUTPUT_DIR" \
    --num-workers 8 \
    --gpus-per-node 8 \
    --rollout-global-batch-size 1024 \
    --train-optimizer-steps 4 \
    --max-concurrent 64 \
    --prompt-repeat-k 5 \
    --pack-max-length 32768 \
    --max-prompt-length 512 \
    --max-response-length 1024 \
    --optimizer-disable-foreach \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
