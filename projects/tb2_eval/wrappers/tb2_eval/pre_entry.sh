#!/bin/bash
# ---------------------------------------------------------------------------
# pre_entry.sh — tb2-eval pre-agent environment setup.
#
# Runs inside the infer sandbox, after the task tree has been seeded under
# $TASK_WORKSPACE and before the agent starts.
#
# Each tb2-eval task uses its own pre-built docker image (specified in
# task.toml [environment] docker_image).  At runtime we seed the workspace
# ourselves: ``environment/files/*`` are already uploaded into
# $TASK_WORKSPACE/ by the pipeline's UploadHook — this script is a
# placeholder kept for parity and to run any task-level setup.sh if present.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${TASK_WORKSPACE:?TASK_WORKSPACE not set}"

# Optional: run upstream setup.sh if a task ships one.
if [ -f "$TASK_WORKSPACE/environment/setup.sh" ]; then
    (cd "$(dirname "$TASK_WORKSPACE")" && bash "$TASK_WORKSPACE/environment/setup.sh" "$TASK_WORKSPACE")
fi

exit 0