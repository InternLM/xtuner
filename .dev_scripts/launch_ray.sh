#!/bin/bash

# Ray cluster startup script
# Usage: This script should be launched on all nodes with appropriate environment variables set:
#   - RANK: Node rank (0 for master, 1+ for workers)
#   - MASTER_ADDR: IP address of the master node
#   - MASTER_PORT: Port for Ray head node (default: 6379)

set -e

# Default values
MASTER_PORT=23333
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RANK=${RANK:-0}

if [ -z "$MASTER_ADDR" ]; then
    echo "Error: MASTER_ADDR environment variable must be set"
    exit 1
fi

echo "Starting Ray with RANK=$RANK, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

if [ "$RANK" -eq 0 ]; then
    # Master node: start Ray head
    echo "Starting Ray head node..."
    ray start --head \
        --port=$MASTER_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --ray-client-server-port=10001 \
        --include-dashboard=true \
        --block
else
    # Worker node: connect to Ray head
    echo "Starting Ray worker node, connecting to $MASTER_ADDR:$MASTER_PORT..."
    ray start \
        --address="$MASTER_ADDR:$MASTER_PORT" \
        --block
fi
