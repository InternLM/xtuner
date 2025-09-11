set -ex

MODEL_PATH=$1
TRAIN_DATA_PATH=$2
TEST_DATA_PATH=$3

export XTUNER_USE_LMDEPLOY=1
export UVICORN_LOG_LEVEL="CRITICAL"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

OUTPUT_DIR='work_dirs/debug'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python ci/scripts/test_grpo_trainer.py \
    --total-epochs 15 \
    --work-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --test-data-path "$TEST_DATA_PATH" \
    --num-workers 8 \
    --gpus-per-node 8 \
    --rollout-global-batch-size 1024 \
    --train-optimizer-steps 4 \
    --max-concurrent 512 \
    --prompt-repeat-k 5 \
    --pack-max-length 32768 \
    --max-prompt-length 512 \
    --max-response-length 1024 \
    --enable-evaluate \
    --evaluate-step 10 \
    --optimizer-disable-foreach \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
