#!/usr/bin/env bash
set -euo pipefail

# Explicit local reset tool: restarting this process clears Jaeger in-memory traces.
# Do not use this against a shared Jaeger deployment.

ROOT="${XTUNER_OTEL_ROOT:-/tmp/xtuner_otel}"
JAEGER_BIN="${JAEGER_BIN:-${ROOT}/bin/jaeger}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/../jaeger/jaeger-memory.yaml}"
PID_FILE="${XTUNER_JAEGER_PID_FILE:-/tmp/xtuner_jaeger_memory.pid}"
LOG_FILE="${XTUNER_JAEGER_LOG_FILE:-/tmp/xtuner_jaeger_memory.log}"
QUERY_URL="${XTUNER_JAEGER_QUERY_URL:-http://127.0.0.1:16686}"
WAIT_TIMEOUT_S="${XTUNER_JAEGER_WAIT_TIMEOUT_S:-30}"

if ! command -v "${JAEGER_BIN}" >/dev/null 2>&1; then
  echo "Jaeger binary not found: ${JAEGER_BIN}" >&2
  echo "Install it first: bash recipe/trace/scripts/install_otel_tools.sh" >&2
  exit 1
fi

if [ ! -f "${CONFIG}" ]; then
  echo "Jaeger config not found: ${CONFIG}" >&2
  exit 1
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to detach Jaeger from the launcher process group." >&2
  exit 1
fi

stop_pid() {
  local pid="$1"
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return
  fi
  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 50); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return
    fi
    sleep 0.1
  done
  kill -9 "${pid}" >/dev/null 2>&1 || true
}

if [ -f "${PID_FILE}" ]; then
  old_pid="$(cat "${PID_FILE}")"
  if [ -n "${old_pid}" ]; then
    stop_pid "${old_pid}"
  fi
  rm -f "${PID_FILE}"
fi

if command -v pgrep >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [ -n "${pid}" ] || continue
    if [ "${pid}" = "$$" ]; then
      continue
    fi
    stop_pid "${pid}"
  done < <(pgrep -f "jaeger.*jaeger-memory.yaml" || true)
fi

mkdir -p "$(dirname "${PID_FILE}")" "$(dirname "${LOG_FILE}")"
setsid "${JAEGER_BIN}" --config "${CONFIG}" >"${LOG_FILE}" 2>&1 </dev/null &
new_pid="$!"
echo "${new_pid}" >"${PID_FILE}"

deadline=$((SECONDS + WAIT_TIMEOUT_S))
until curl -fsS "${QUERY_URL}/api/services" >/dev/null 2>&1; do
  if ! kill -0 "${new_pid}" >/dev/null 2>&1; then
    echo "Jaeger exited before becoming ready. Log: ${LOG_FILE}" >&2
    exit 1
  fi
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    echo "Timed out waiting for Jaeger Query API at ${QUERY_URL}. Log: ${LOG_FILE}" >&2
    exit 1
  fi
  sleep 0.5
done

echo "Jaeger in-memory storage restarted."
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Jaeger UI: ${QUERY_URL}"
echo "OTLP gRPC: http://127.0.0.1:14317"
echo "OTLP HTTP: http://127.0.0.1:14318/v1/traces"
