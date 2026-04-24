#!/usr/bin/env bash
# Async checkpoint A/B benchmark script.
#
# Runs async then sync checkpoint saves for a given model config,
# removes checkpoints after the async run, then runs sync,
# and removes checkpoints again after the sync run.
#
# Usage:
#   bash examples/v1/scripts/bench_async_checkpoint.sh 8b
#   bash examples/v1/scripts/bench_async_checkpoint.sh 30b
#
# Env:
#   CKPT_INTERVAL - checkpoint interval in steps (default: 20)
#   TOTAL_STEP    - total training steps (default: 100)
#   WORK_DIR      - override checkpoint save path
#   NPROC         - number of GPUs (default: 8)

set -euo pipefail

MODEL="${1:?Usage: $0 <8b|30b>}"
NPROC="${NPROC:-8}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"

export CKPT_INTERVAL="${CKPT_INTERVAL:-20}"
export TOTAL_STEP="${TOTAL_STEP:-100}"

mkdir -p "$LOG_DIR"

case "$MODEL" in
    8b)  CONFIG="examples/v1/config/sft_qwen3_8b_async_verify.py" ;;
    30b) CONFIG="examples/v1/config/sft_qwen3_30b_async_verify.py" ;;
    *)   echo "Unknown model: $MODEL (use 8b or 30b)"; exit 1 ;;
esac

WORK_DIR="${WORK_DIR:-$(pwd)}"
export WORK_DIR
CKPT_DIR="${WORK_DIR}/checkpoints"

SYNC_LOG="$LOG_DIR/xtuner_sync_${MODEL}_${TIMESTAMP}_interval_${CKPT_INTERVAL}.log"
ASYNC_LOG="$LOG_DIR/xtuner_async_${MODEL}_${TIMESTAMP}_interval_${CKPT_INTERVAL}.log"

cleanup_checkpoints() {
    local tag="$1"

    echo "--------------------------------------------"
    echo " Cleaning checkpoints after ${tag} run"
    echo "--------------------------------------------"

    if [ -d "$CKPT_DIR" ]; then
        du -sh "$CKPT_DIR" 2>/dev/null || true
        rm -rf "$CKPT_DIR"
        echo "Removed checkpoint dir: $CKPT_DIR"
    else
        echo "No checkpoint dir found: $CKPT_DIR"
    fi

    if [ -f "${WORK_DIR}/meta.json" ]; then
        rm -f "${WORK_DIR}/meta.json"
        echo "Removed meta file: ${WORK_DIR}/meta.json"
    fi

    echo ""
}

echo "============================================"
echo " Async Checkpoint Benchmark — ${MODEL^^}"
echo "============================================"
echo "Config:         $CONFIG"
echo "GPUs:           $NPROC"
echo "Total steps:    $TOTAL_STEP"
echo "Save interval:  $CKPT_INTERVAL"
echo "Work dir:       $WORK_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo "Sync log:       $SYNC_LOG"
echo "Async log:      $ASYNC_LOG"
echo ""

# ---- Run 1/2: Async ----
echo "============================================"
echo " [1/2] Running ASYNC mode..."
echo "============================================"
env ASYNC_CKPT=1 \
    torchrun --nproc_per_node="$NPROC" xtuner/v1/train/cli/sft.py \
    --config "$CONFIG" \
    2>&1 | tee "$ASYNC_LOG"
echo ""

# ---- Delete async checkpoints before sync run ----
cleanup_checkpoints "ASYNC"

# ---- Run 2/2: Sync ----
echo "============================================"
echo " [2/2] Running SYNC mode..."
echo "============================================"
env ASYNC_CKPT=0 \
    torchrun --nproc_per_node="$NPROC" xtuner/v1/train/cli/sft.py \
    --config "$CONFIG" \
    2>&1 | tee "$SYNC_LOG"
echo ""

# ---- Delete sync checkpoints after sync run ----
cleanup_checkpoints "SYNC"

# ---- Extract & compare ----
echo "============================================"
echo " Results Summary"
echo "============================================"
echo ""

echo "--- Total Training Time ---"
echo "Sync:  $(grep 'Training finished in' "$SYNC_LOG" | head -1 || echo '(not found)')"
echo "Async: $(grep 'Training finished in' "$ASYNC_LOG" | head -1 || echo '(not found)')"
echo ""

echo "--- Checkpoint Breakdown ---"
echo "[Sync]"
grep -E '\[Checkpoint Breakdown\]' "$SYNC_LOG" | head -20 || echo "  (not found)"
echo ""
echo "[Async]"
grep -E '\[Checkpoint Breakdown\]' "$ASYNC_LOG" | head -20 || echo "  (not found)"
echo ""

echo "--- Async Staging/Upload Waits ---"
grep -E '\[Async Checkpoint\]' "$ASYNC_LOG" | head -20 || echo "  (not found)"
echo ""

echo "--- Final Loss & Grad Norm ---"
echo "Sync:  $(grep -E 'loss=|reduced_llm_loss:|grad_norm:' "$SYNC_LOG" | tail -1 || echo '(not found)')"
echo "Async: $(grep -E 'loss=|reduced_llm_loss:|grad_norm:' "$ASYNC_LOG" | tail -1 || echo '(not found)')"
echo ""

echo "--- GPU Memory ---"
echo "Sync:  $(grep -E 'max_memory' "$SYNC_LOG" | tail -1 || echo '(not found)')"
echo "Async: $(grep -E 'max_memory' "$ASYNC_LOG" | tail -1 || echo '(not found)')"
echo ""

echo "Logs saved to:"
echo "  $SYNC_LOG"
echo "  $ASYNC_LOG"