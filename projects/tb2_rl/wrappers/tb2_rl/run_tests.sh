#!/bin/bash
# ---------------------------------------------------------------------------
# run_tests.sh — tb2-rl verifier entrypoint.
#
# The bench ships its own test harness at /tests/test.sh which:
#   - installs pytest + pytest-json-ctrf + /tests/test_requirements.txt
#   - runs /tests/test_outputs.py with --ctrf /logs/verifier/ctrf.json
#   - writes /logs/verifier/reward.txt (1 or 0)
#
# We just invoke it and hand the resulting CTRF to our shared emitter so
# the stage's stdout is a single JudgerResult JSON line (what
# ParseJudgerStdout expects).
#
# Env:
#   $TESTS_DIR    tests directory inside the sandbox (default: /tests)
#   $WRAPPER_DIR  where emit_judger_result_from_ctrf.py lives
#   $JUDGER_NAME  name to emit in the JudgerResult (default: rule_grader)
# ---------------------------------------------------------------------------
set -uo pipefail

TESTS_DIR="${TESTS_DIR:-/tests}"
WRAPPER_DIR="${WRAPPER_DIR:-$(dirname "$0")}"
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

"$PY" "$WRAPPER_DIR/emit_judger_result_from_ctrf.py" \
    --ctrf /logs/verifier/ctrf.json \
    --log "$TEST_LOG" \
    --pytest-rc "$TEST_RC" \
    --judger-name "$JUDGER_NAME"
