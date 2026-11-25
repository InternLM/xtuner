#!/usr/bin/env bash
# Same launch flow as run_rl.sh, but uses the Probing-enabled RL entry point.
# PROBING=1 activates probing in the driver and Ray workers; PROBING_PORT
# enables the driver's Web UI. All arguments are forwarded to run_rl.sh.
set -ex

export PROBING=${PROBING:-1}
export PROBING_PORT=${PROBING_PORT:-8080}
export XTUNER_RL_ENTRY=${XTUNER_RL_ENTRY:-xtuner/v1/train/cli/rl_probing.py}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${ROOT_DIR}"
exec bash examples/v1/scripts/run_rl.sh "$@"
