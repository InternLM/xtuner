set -ex

export XTUNER_USE_SGLANG=1 # 最好训练用 fa3，暂时是 fa2
export PYTHONPATH=/mnt/shared-storage-user/huanghaian/code/lmdeploy/:$PYTHONPATH
export UVICORN_LOG_LEVEL="CRITICAl"
export ID_INPUT_OUTPUT=1

# 不支持
# export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

OUTPUT_DIR='work_dirs/dapo_math_7B_newlmdeploy_nogroup_sglang'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export ROLLOUT_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B'
export ROLLOUT_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/verl/data/dapo_math/dapo-math-17k.jsonl"
export ROLLOUT_TEST_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/verl/data/dapo_math/aime-2024.jsonl"

ray stop --force

#  --max-concurrent 如果开大会 oom
python ci/scripts/test_dapo_trainer.py \
    --total-epochs 1 \
    --work-dir "$OUTPUT_DIR" \
    --num-workers 8 \
    --gpus-per-node 8 \
    --rollout-global-batch-size 512 \
    --train-optimizer-steps 16 \
    --max-concurrent 32 \
    --prompt-repeat-k 16 \
    --pack-max-length 32768 \
    --max-prompt-length 2048 \
    --max-response-length 8192 \
    --optimizer-disable-foreach \
    --enable-evaluate \
    --evaluate-step 5 \
    --hf-interval 50 \
    --evaluate-ratio 1 \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
