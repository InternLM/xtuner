#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh — tb2-eval verifier entrypoint.
#
# The bench ships its own test harness at /tests/test.sh which:
#   - installs pytest + pytest-json-ctrf + /tests/test_requirements.txt
#   - runs /tests/test_outputs.py with --ctrf /logs/verifier/ctrf.json
#   - writes the authoritative 0/1 outcome to /logs/verifier/reward.txt
#     (for multi-pytest tasks like fix-code-vulnerability, this is the
#     AND of all pytest exit codes — matching official TB2 scoring)
#
# We invoke it and hand both reward.txt and the resulting CTRF to our
# shared emitter. reward.txt drives the JudgerResult `total`; CTRF is
# parsed for per-test observability only.
#
# Env:
#   $TESTS_DIR    tests directory inside the sandbox (default: /tests)
#   $JUDGER_DIR  where grader.py lives
#   $JUDGER_NAME  name to emit in the JudgerResult (default: rule_grader)
# ---------------------------------------------------------------------------
set -uo pipefail

TESTS_DIR="${TESTS_DIR:-/tests}"
JUDGER_DIR="${JUDGER_DIR:-$(dirname "$0")}"
JUDGER_NAME="${JUDGER_NAME:-rule_grader}"

TEST_LOG=/tmp/tb2_eval_test.log
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
    --reward-file /logs/verifier/reward.txt \
    --pytest-rc "$TEST_RC" \
    --judger-name "$JUDGER_NAME"