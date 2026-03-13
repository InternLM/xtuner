#!/bin/bash

# Ray cluster startup script
#
# This script supports two modes:
#   - Head mode: Start Ray head node (when $1 is not provided or equals "0.0.0.0")
#   - Worker mode: Start Ray worker nodes that connect to head (when $1 is head IP address)
#
# Usage:
#   ./launch_ray.sh                  # Start as head node
#   ./launch_ray.sh 0.0.0.0          # Start as head node (explicit)
#   ./launch_ray.sh <head_ip>        # Start as worker node, connect to <head_ip>
#
# Arguments:
#   $1: HEAD_ADDR (optional) - IP address of Ray head node
#       - If not provided or "0.0.0.0": Start as head node
#       - If provided: Start as worker node and connect to this IP
#
# Environment variables:
#   - RAY_DASHBOARD_PORT: Dashboard port (default: 8265)

set -e

HEAD_ADDR=${1:-"0.0.0.0"}
PORT=23333
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

echo "Starting Ray with HEAD_ADDR=$HEAD_ADDR, PORT=$PORT"

if [ "$HEAD_ADDR" == "0.0.0.0" ]; then
    echo "Starting Ray head node..."
    ray start --head \
        --port=$PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --ray-client-server-port=10001 \
        --include-dashboard=true \
        --block --verbose
else
    echo "Starting Ray worker node, connecting to $HEAD_ADDR:$PORT..."
    ray start \
        --address="$HEAD_ADDR:$PORT" \
        --block --verbose
fi
