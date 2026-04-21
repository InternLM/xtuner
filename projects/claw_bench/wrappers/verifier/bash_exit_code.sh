#!/bin/bash
# ---------------------------------------------------------------------------
# bash_exit_code.sh — the simplest judger
#
# Runs an arbitrary command and maps exit code to a JudgerResult:
#   exit 0      → total=1.0
#   exit != 0   → total=0.0 with the stderr tail as the error field
#
# Env:
#   $JUDGER_NAME   name field in the output (default: rule_grader)
#
# Args: the command + its arguments
# ---------------------------------------------------------------------------
set -uo pipefail

JUDGER_NAME="${JUDGER_NAME:-rule_grader}"

STDERR=$(mktemp)
trap 'rm -f "$STDERR"' EXIT

"$@" 2>"$STDERR"
RC=$?

if [ "$RC" -eq 0 ]; then
    printf '{"judger_name":"%s","total":1.0,"criteria":{"exit_code":{"score":1.0,"weight":1.0}}}\n' "$JUDGER_NAME"
else
    ERR_TAIL=$(tail -c 500 "$STDERR" | /mnt/llm-ai-infra/miniconda3/envs/train/bin/python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')
    printf '{"judger_name":"%s","total":0.0,"criteria":{"exit_code":{"score":0.0,"weight":1.0}},"error":%s}\n' "$JUDGER_NAME" "$ERR_TAIL"
fi
