#!/bin/bash
# ---------------------------------------------------------------------------
# claw_pre_entry.sh — claw-bench pre-agent environment setup.
#
# Runs inside the infer sandbox, after the task tree has been mirrored into
# $TASK_WORKSPACE and before the agent starts.  Mirrors upstream claw-bench's
# core/runner.py behavior:
#   1. Flatten $TASK_WORKSPACE/environment/data/* into $TASK_WORKSPACE root
#      (so agents see data files at conventional paths, not under environment/).
#   2. Invoke $TASK_WORKSPACE/environment/setup.sh with $TASK_WORKSPACE as $1.
#
# Both steps are no-ops if the corresponding file/dir is absent.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${TASK_WORKSPACE:?TASK_WORKSPACE not set}"

if [ -d "$TASK_WORKSPACE/environment/data" ]; then
    cp -r "$TASK_WORKSPACE/environment/data/." "$TASK_WORKSPACE/"
fi

if [ -f "$TASK_WORKSPACE/environment/setup.sh" ]; then
    bash "$TASK_WORKSPACE/environment/setup.sh" "$TASK_WORKSPACE"
fi
