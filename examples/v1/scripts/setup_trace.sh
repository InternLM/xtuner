#!/usr/bin/env bash

SETUP_TRACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SETUP_TRACE_DIR}/../../.." && pwd)"

export XTUNER_OTEL_ROOT="${XTUNER_OTEL_ROOT:-/tmp/xtuner_otel}"
export PATH="${XTUNER_OTEL_ROOT}/bin:${PATH}"

OTEL_INSTALL_SCRIPT="${XTUNER_OTEL_INSTALL_SCRIPT:-${REPO_ROOT}/recipe/otle/install_otel_tools.sh}"
JAEGER_RESTART_SCRIPT="${XTUNER_JAEGER_RESTART_SCRIPT:-${REPO_ROOT}/recipe/otle/restart_jaeger_memory.sh}"
JAEGER_CONFIG="${XTUNER_JAEGER_CONFIG:-${REPO_ROOT}/recipe/otle/jaeger/jaeger-memory.yaml}"
TRACE_VIEWER_PORT="${XTUNER_TRACE_VIEWER_PORT:-18080}"

if pgrep -f "recipe.trace_viewer.server.*--port ${TRACE_VIEWER_PORT}" >/dev/null 2>&1 ||
   pgrep -f "recipe.trace_viewer.server.*--port=${TRACE_VIEWER_PORT}" >/dev/null 2>&1; then
  echo "Stopping previous XTuner trace viewer on port ${TRACE_VIEWER_PORT}"
  pkill -f "recipe.trace_viewer.server.*--port ${TRACE_VIEWER_PORT}" 2>/dev/null || true
  pkill -f "recipe.trace_viewer.server.*--port=${TRACE_VIEWER_PORT}" 2>/dev/null || true
  sleep 1
fi

if [ ! -d "$XTUNER_OTEL_ROOT" ]; then
  echo "Installing XTuner OTel tools to ${XTUNER_OTEL_ROOT}"
  bash "$OTEL_INSTALL_SCRIPT" "$XTUNER_OTEL_ROOT" || return 1 2>/dev/null || exit 1
fi

otel_collector="$(command -v otelcol-contrib || command -v otelcol || true)"
if [ -z "$otel_collector" ]; then
  echo "Error: XTuner trace collector not found after checking ${XTUNER_OTEL_ROOT}." >&2
  echo "Expected otelcol-contrib or otelcol under ${XTUNER_OTEL_ROOT}/bin." >&2
  return 1 2>/dev/null || exit 1
fi

echo "XTuner trace collector: ${otel_collector}"
bash "$JAEGER_RESTART_SCRIPT" "$JAEGER_CONFIG" || return 1 2>/dev/null || exit 1
