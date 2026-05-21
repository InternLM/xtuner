#!/bin/bash
# ---------------------------------------------------------------------------
# pre_entry.sh — tb2-rl pre-agent environment setup.
#
# Runs inside the infer sandbox, after the task tree has been seeded under
# $TASK_WORKSPACE and before the agent starts.
#
# The upstream Dockerfile convention is ``COPY files/ /app/`` at image-build
# time.  We use the pre-built ``t-data-processing-v1`` image instead, so at
# runtime we seed /app/ ourselves: ``environment/files/*`` are already
# uploaded into $TASK_WORKSPACE/ by the pipeline's UploadHook — this script
# is a no-op placeholder kept for parity with claw_bench and to host any
# future task-level setup.sh invocation.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${TASK_WORKSPACE:?TASK_WORKSPACE not set}"

# Optional: run upstream setup.sh if a task ever ships one.
if [ -f "$TASK_WORKSPACE/environment/setup.sh" ]; then
    (cd "$(dirname "$TASK_WORKSPACE")" && bash "$TASK_WORKSPACE/environment/setup.sh" "$TASK_WORKSPACE")
fi

exit 0
