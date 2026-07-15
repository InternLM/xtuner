#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/tmp/xtuner_otel}"
BIN_DIR="${ROOT}/bin"
DOWNLOAD_DIR="${ROOT}/downloads"
OTEL_VERSION="${OTEL_VERSION:-0.128.0}"
JAEGER_VERSION="${JAEGER_VERSION:-2.19.0}"

mkdir -p "${BIN_DIR}" "${DOWNLOAD_DIR}"

download_and_extract() {
  local url="$1"
  local archive="$2"

  if [ ! -f "${DOWNLOAD_DIR}/${archive}" ]; then
    curl -L --fail --retry 3 --output "${DOWNLOAD_DIR}/${archive}" "${url}"
  fi
  tar -xzf "${DOWNLOAD_DIR}/${archive}" -C "${BIN_DIR}"
}

download_and_extract \
  "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTEL_VERSION}/otelcol_${OTEL_VERSION}_linux_amd64.tar.gz" \
  "otelcol_${OTEL_VERSION}_linux_amd64.tar.gz"

download_and_extract \
  "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTEL_VERSION}/otelcol-contrib_${OTEL_VERSION}_linux_amd64.tar.gz" \
  "otelcol-contrib_${OTEL_VERSION}_linux_amd64.tar.gz"

download_and_extract \
  "https://github.com/jaegertracing/jaeger/releases/download/v${JAEGER_VERSION}/jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz" \
  "jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz"

if [ ! -f "${BIN_DIR}/jaeger" ] && [ -f "${BIN_DIR}/jaeger-${JAEGER_VERSION}-linux-amd64/jaeger" ]; then
  ln -sfn "${BIN_DIR}/jaeger-${JAEGER_VERSION}-linux-amd64/jaeger" "${BIN_DIR}/jaeger"
fi

chmod +x "${BIN_DIR}/otelcol" "${BIN_DIR}/otelcol-contrib" "${BIN_DIR}/jaeger"

echo "Installed:"
"${BIN_DIR}/otelcol" --version
"${BIN_DIR}/otelcol-contrib" --version
"${BIN_DIR}/jaeger" version
echo "Add to PATH: export PATH=${BIN_DIR}:\$PATH"
