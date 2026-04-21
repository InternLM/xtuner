#!/bin/bash
# ---------------------------------------------------------------------------
# lagent_entry.sh — implements the Lagent Entry Contract (Pipeline.md §7.3)
#
# Usage:
#   lagent_entry.sh --config PATH \
#                   --instruction-file PATH \
#                   --response-out PATH \
#                   --trajectory-out PATH \
#                   [--max-turns N] \
#                   [--sock PATH]
#
# Internally: starts lagent AgentDaemon → sends one chat request → dumps
# state_dict → kills daemon.  Exits 0 on success, non-zero on failure.
#
# Relies on /tmp/lagent-py wrapper (shared conda python + PYTHONPATH), which
# runner bootstrap writes.  Agent config JSON is what daemon consumes.
# ---------------------------------------------------------------------------
set -uo pipefail

CONFIG=""
INSTRUCTION_FILE=""
RESPONSE_OUT=""
TRAJECTORY_OUT=""
MAX_TURNS=""
SOCK="/tmp/lagent_agent.sock"

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --instruction-file) INSTRUCTION_FILE="$2"; shift 2 ;;
        --response-out) RESPONSE_OUT="$2"; shift 2 ;;
        --trajectory-out) TRAJECTORY_OUT="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        --sock) SOCK="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

for v in CONFIG INSTRUCTION_FILE RESPONSE_OUT TRAJECTORY_OUT; do
    if [ -z "${!v}" ]; then
        echo "missing --${v,,}" >&2
        exit 2
    fi
done

LAGENT_PY=/tmp/lagent-py
if [ ! -x "$LAGENT_PY" ]; then
    echo "$LAGENT_PY not found — runner bootstrap did not run" >&2
    exit 3
fi

LOG=/tmp/agent_daemon.log
: > "$LOG"

# ── 1. Start AgentDaemon ──────────────────────────────────────────────
nohup "$LAGENT_PY" -m lagent.serving.sandbox.daemon start \
    --mode agent \
    --config "$CONFIG" \
    --sock "$SOCK" \
    >>"$" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 60); do
    [ -S "$SOCK" ] && break
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "daemon died before socket came up" >&2
        tail -n 100 "$LOG" >&2 || true
        exit 4
    fi
    sleep 1
done
if [ ! -S "$SOCK" ]; then
    echo "daemon did not open socket within 60s" >&2
    tail -n 100 "$LOG" >&2 || true
    kill "$DAEMON_PID" 2>/dev/null || true
    exit 4
fi

cleanup() {
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$DAEMON_PID" 2>/dev/null || true
    fi
    pkill -f "lagent.serving.sandbox.daemon.*$SOCK" 2>/dev/null || true
}
trap cleanup EXIT

# ── 2. Send chat request ──────────────────────────────────────────────
INSTRUCTION_JSON=$("$LAGENT_PY" -c '
import json, sys
payload = {"cmd": "chat", "messages": [open(sys.argv[1]).read()]}
print(json.dumps(payload, ensure_ascii=False))
' "$INSTRUCTION_FILE")

CHAT_RESP=$("$LAGENT_PY" -m lagent.serving.sandbox.daemon call \
    --sock "$SOCK" "$INSTRUCTION_JSON" 2>>"$LOG") || {
    echo "chat call failed" >&2
    tail -n 100 "$LOG" >&2 || true
    exit 5
}

# Extract final response content (plain text) to RESPONSE_OUT.
printf '%s' "$CHAT_RESP" | "$LAGENT_PY" -c '
import json, sys
obj = json.loads(sys.stdin.read() or "{}")
out_path = sys.argv[1]
content = obj.get("content", "")
if not isinstance(content, str):
    content = json.dumps(content, ensure_ascii=False)
open(out_path, "w", encoding="utf-8").write(content or "")
' "$RESPONSE_OUT"

# ── 3. Dump state_dict ────────────────────────────────────────────────
STATE_RESP=$("$LAGENT_PY" -m lagent.serving.sandbox.daemon call \
    --sock "$SOCK" '{"cmd": "state_dict"}' 2>>"$LOG") || {
    echo "state_dict call failed" >&2
    tail -n 100 "$LOG" >&2 || true
    exit 6
}

# Wrap lagent's native state into {"trajectory": [...]} if it isn't already.
printf '%s' "$STATE_RESP" | "$LAGENT_PY" -c '
import json, sys
raw = sys.stdin.read() or "{}"
try:
    obj = json.loads(raw)
except json.JSONDecodeError:
    obj = {"raw": raw}
state = obj.get("state_dict", obj)
if isinstance(state, dict) and "trajectory" in state:
    payload = state
elif isinstance(state, dict):
    flat = []
    for v in state.values():
        if isinstance(v, list):
            flat.extend(m for m in v if isinstance(m, dict))
    payload = {"trajectory": flat, "raw": state}
else:
    payload = {"trajectory": state if isinstance(state, list) else [], "raw": state}
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(payload, ensure_ascii=False, default=str))
' "$TRAJECTORY_OUT"

exit 0
