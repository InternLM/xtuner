set -ex

export ROLLOUT_MODEL_PATH="/cpfs01/shared/llm_ddd/lishuaibin/ckpt/Qwen/Qwen2.5-Math-7B"
export ROLLOUT_DATA_PATH="/cpfs01/shared/llm_razor/caoweihan/dapo-math-17k.jsonl"
export ROLLOUT_TEST_DATA_PATH="/cpfs01/shared/llm_razor/huanghaian/code/refactor_xtuner/gsm8k/test.jsonl"
export XTUNER_USE_LMDEPLOY=1 
export XTUNER_USE_FA3=1
export PYTHONPATH='/cpfs01/shared/llm_razor/caoweihan/projects/lmdeploy':'/cpfs01/shared/llm_ddd/caoweihan/projects/Liger-Kernel/src/':'.':$PYTHONPATH 
export UVICORN_LOG_LEVEL="CRITICAl"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

OUTPUT_DIR='work_dirs/dapo_math_7B_newlmdeploy_nogroup'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python ci/scripts/test_dapo_trainer.py \
    --total-epochs 1 \
    --work-dir "$OUTPUT_DIR" \
    --num-workers 8 \
    --gpus-per-node 8 \
    --rollout-global-batch-size 512 \
    --train-optimizer-steps 16 \
    --max-concurrent 64 \
    --prompt-repeat-k 16 \
    --pack-max-length 32768 \
    --max-prompt-length 2048 \
    --max-response-length 8192 \
    --optimizer-disable-foreach \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
