#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh — tb2-rl rule_grader entrypoint.
#
# The bench ships its own test harness at /tests/test.sh which:
#   - installs pytest + pytest-json-ctrf + /tests/test_requirements.txt
#   - runs /tests/test_outputs.py with --ctrf /logs/verifier/ctrf.json
#   - writes /logs/verifier/reward.txt (1 or 0)
#
# We invoke it and hand the resulting CTRF to grader.py so the stage's
# stdout is a single judger score JSON line (what ParseJudgerStdout expects).
#
# Env:
#   $TESTS_DIR    tests directory inside the sandbox (default: /tests)
#   $JUDGER_DIR   where grader.py lives (default: dirname of this script)
#   $JUDGER_NAME  name to emit in the score JSON (default: rule_grader)
# ---------------------------------------------------------------------------
set -uo pipefail

TESTS_DIR="${TESTS_DIR:-/tests}"
JUDGER_DIR="${JUDGER_DIR:-$(dirname "$0")}"
JUDGER_NAME="${JUDGER_NAME:-rule_grader}"

TEST_LOG=/tmp/tb2_rl_test.log
: > "$TEST_LOG"

chmod +x "$TESTS_DIR/test.sh" 2>/dev/null || true
bash "$TESTS_DIR/test.sh" > "$TEST_LOG" 2>&1
TEST_RC=$?

PY=/mnt/llm-ai-infra/miniconda3/envs/train/bin/python3
if [ ! -x "$PY" ]; then
    PY=$(command -v python3 || echo python)
fi

"$PY" "$JUDGER_DIR/grader.py" \
    --ctrf /logs/verifier/ctrf.json \
    --log "$TEST_LOG" \
    --pytest-rc "$TEST_RC" \
    --judger-name "$JUDGER_NAME"
