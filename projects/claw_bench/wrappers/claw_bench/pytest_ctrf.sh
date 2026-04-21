#!/bin/bash
# ---------------------------------------------------------------------------
# pytest_ctrf.sh — run pytest + emit JudgerResult from CTRF JSON
#
# Env:
#   $TASK_WORKSPACE     agent's workspace (passed to pytest --workspace)
#   $TASK_JUDGER_DIR    judger directory containing test files
#
# Optional:
#   $PYTEST_TARGET      path to the pytest file (default: $TASK_JUDGER_DIR/test_output.py)
#   $PYTEST_EXTRA_WITH  additional uvx "--with foo==x.y.z" args (space-separated)
#
# Output: single-line JudgerResult JSON on stdout.
# ---------------------------------------------------------------------------
set -uo pipefail

: "${TASK_WORKSPACE:?TASK_WORKSPACE not set}"
: "${TASK_JUDGER_DIR:?TASK_JUDGER_DIR not set}"

PYTEST_TARGET="${PYTEST_TARGET:-$TASK_JUDGER_DIR/test_output.py}"
PYTEST_EXTRA_WITH="${PYTEST_EXTRA_WITH:-}"
TARGET_DIR="$(dirname "$PYTEST_TARGET")"
WRAPPER_DIR="$(dirname "$0")"

# Ship the shared conftest if the task didn't provide one.  Keeps the
# --workspace option + weight marker available without per-task churn.
if [ ! -f "$TARGET_DIR/conftest.py" ] && [ -f "$WRAPPER_DIR/claw_conftest.py" ]; then
    cp "$WRAPPER_DIR/claw_conftest.py" "$TARGET_DIR/conftest.py"
fi

REPORT_DIR=/tmp/verifier_report
mkdir -p "$REPORT_DIR"

# Install uvx on demand (most images don't ship it).
if ! command -v uvx >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -qq -y curl ca-certificates >/dev/null 2>&1 || true
    curl -LsSf https://astral.sh/uv/0.9.5/install.sh 2>/dev/null | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"

# shellcheck disable=SC2086
uvx -p 3.13 \
    --with pytest==8.4.1 \
    --with numpy==2.3.3 \
    --with scipy==1.16.2 \
    --with pytest-json-ctrf==0.3.5 \
    $PYTEST_EXTRA_WITH \
    pytest --ctrf "$REPORT_DIR/ctrf.json" \
    --workspace "$TASK_WORKSPACE" \
    "$PYTEST_TARGET" -rA \
    > "$REPORT_DIR/pytest.log" 2>&1
PYTEST_RC=$?

/mnt/llm-ai-infra/miniconda3/envs/train/bin/python3 \
    "$WRAPPER_DIR/emit_judger_result_from_ctrf.py" \
    --ctrf "$REPORT_DIR/ctrf.json" \
    --log  "$REPORT_DIR/pytest.log" \
    --pytest-rc "$PYTEST_RC" \
    --judger-name "${JUDGER_NAME:-rule_grader}"
