# Example usage:
# Full GSM8K dataset test (recommended for comprehensive evaluation):
# bash ci/scripts/test_grpo_trainer_bash.sh $ROLLOUT_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH 15 1024 2 1 4 5 true 10
#
# Quick partial GSM8K test (for rapid accuracy validation within ~30 minutes):
# - Initial eval accuracy: ~25%
# - After training: ~88% eval accuracy
# bash ci/scripts/test_grpo_trainer_bash.sh $ROLLOUT_MODEL_PATH $ROLLOUT_DEBUG_DATA_PATH $ROLLOUT_TEST_DATA_PATH 3 64 1 1 1 5 true 45

# Note: Ensure environment variables like $ROLLOUT_MODEL_PATH are set before running.

set -ex

MODEL_PATH=$1
TRAIN_DATA_PATH=$2
TEST_DATA_PATH=$3
EPOCHS=${4:-15}
GLOBAL_BATCH_SIZE=${5:-1024}
ROLLOUT_TP_SIZE=${6:-1}
ROLLOUT_EP_SIZE=${7:-1}
TRAIN_OPTIMIZER_STEPS=${8:-4}
PROMPT_REPEAT_K=${9:-5}
ENABLE_EVALUATE=${10:-true}
EVALUATE_STEP=${11:-10}

export XTUNER_USE_LMDEPLOY=1
export UVICORN_LOG_LEVEL="CRITICAL"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

OUTPUT_DIR='work_dirs/debug'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

EVAL_FLAG=""
if [ "$ENABLE_EVALUATE" = "true" ]; then
    EVAL_FLAG="--enable-evaluate"
fi

python ci/scripts/test_grpo_trainer.py \
    --total-epochs $EPOCHS \
    --work-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH" \
    --data-path "$TRAIN_DATA_PATH" \
    --eval-data-path "$TEST_DATA_PATH" \
    --rollout-global-batch-size $GLOBAL_BATCH_SIZE \
    --rollout-tp-size $ROLLOUT_TP_SIZE \
    --rollout-ep-size $ROLLOUT_EP_SIZE \
    --train-optimizer-steps $TRAIN_OPTIMIZER_STEPS \
    --prompt-repeat-k $PROMPT_REPEAT_K \
    $EVAL_FLAG \
    --evaluate-step $EVALUATE_STEP \
    --optimizer-disable-foreach \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

