#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 2; }
exec python3 -m xtuner.tools.model_normalize "$@"
