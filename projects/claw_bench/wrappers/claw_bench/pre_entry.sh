#!/bin/bash
# ---------------------------------------------------------------------------
# claw_pre_entry.sh — claw-bench pre-agent environment setup.
#
# Runs inside the infer sandbox, after the task tree has been mirrored into
# $TASK_WORKSPACE and before the agent starts.  Mirrors upstream claw-bench's
# core/runner.py behavior:
#   0. Install any common deps upstream scripts assume (sqlite3, textblob, …).
#   1. Flatten $TASK_WORKSPACE/environment/data/* into $TASK_WORKSPACE root
#      (so agents see data files at conventional paths, not under environment/).
#   2. Invoke $TASK_WORKSPACE/environment/setup.sh with $TASK_WORKSPACE as $1.
#
# Every step is idempotent + best-effort.  The dep install in particular is
# guarded by a "is it there already?" check so it's a fast no-op when the
# base image already has the tool.  Long-term these should be baked into
# ``ubuntu2404-v1``; this script is the short-term patch.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${TASK_WORKSPACE:?TASK_WORKSPACE not set}"


# ── 0. Common deps ─────────────────────────────────────────────────────
#
# Any system package / python module that upstream scripts reference.
# Add as we find them.  Each line is idempotent and quiet on success.

_install_apt() {
    # $1 = command name, $2 = apt package (defaults to $1)
    local cmd="$1" pkg="${2:-$1}"
    command -v "$cmd" >/dev/null 2>&1 && return 0
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null 2>&1 || true
}

_install_pip() {
    # $1 = importable module, $2 = pip package (defaults to $1)
    local mod="$1" pkg="${2:-$1}"
    python3 -c "import $mod" 2>/dev/null && return 0
    pip install -q --no-input "$pkg" >/dev/null 2>&1 || true
}


# ── 1. Flatten upstream data dir into workspace root ───────────────────

if [ -d "$TASK_WORKSPACE/environment/data" ]; then
    cp -r "$TASK_WORKSPACE/environment/data/." "$TASK_WORKSPACE/"
fi


# ── 2. Run upstream setup.sh (upstream convention: cwd = task_dir) ─────

if [ -f "$TASK_WORKSPACE/environment/setup.sh" ]; then
    # cd to the parent of $TASK_WORKSPACE so upstream scripts that reference
    # ``workspace/<file>`` relative paths resolve correctly (matches upstream
    # claw-bench's core/runner.py ``cwd=task_dir`` behavior).
    (cd "$(dirname "$TASK_WORKSPACE")" && bash "$TASK_WORKSPACE/environment/setup.sh" "$TASK_WORKSPACE")
fi
