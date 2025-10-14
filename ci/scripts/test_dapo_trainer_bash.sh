set -ex

ROLLOUT_MODEL_PATH=$1
ROLLOUT_DATA_PATH=$2
ROLLOUT_TEST_DATA_PATH=$3

export XTUNER_USE_LMDEPLOY=1
export XTUNER_USE_FA3=1
export UVICORN_LOG_LEVEL="CRITICAl"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

#export ID_INPUT_OUTPUT=1

OUTPUT_DIR='work_dirs/dapo_math_7B_newlmdeploy_nogroup'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python ci/scripts/test_dapo_trainer.py \
    --total-epochs 1 \
    --work-dir "$OUTPUT_DIR" \
    --model-path "$ROLLOUT_MODEL_PATH" \
    --data-path "$ROLLOUT_DATA_PATH" \
    --eval-data-path "$ROLLOUT_TEST_DATA_PATH" \
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
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"