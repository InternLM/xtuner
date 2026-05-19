set -ex
ray stop --force
# examples of usage:
# qwen3_8B_grpo_gsm8k training: bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_8B_grpo.py "sglang" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
# qwen2.5_7B_dapo_math training: bash examples/v1/scripts/run_rl.sh  examples/v1/config/rl_qwen25_7B_dapo.py "sglang" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH

CONFIG_PATH=$1
INFER_BACKEND=$2
MODEL_PATH=$3
DATA_PATH=$4
EVAL_DATA_PATH=${5:-""}
ACCELERATOR=${6:-"gpu"} # "gpu" or "npu"
ACCELERATOR=$(echo "$ACCELERATOR" | tr '[:lower:]' '[:upper:]')
if [ $ACCELERATOR != "GPU" ] && [ $ACCELERATOR != "NPU" ]; then
  echo "Error: ACCELERATOR must be either 'gpu' or 'npu'!"
  exit 1
fi
if [ "$ACCELERATOR" = "NPU" ]; then
  ACCELERATOR_PER_NODE=${7:-16}
else
  ACCELERATOR_PER_NODE=${7:-8}
fi

ulimit -n 65536  # OSError: [Errno 24] Too many open files

# Ray cluster bootstrap variables. Business/runtime variables are passed through
# ray job runtime_env below instead of relying on raylet shell inheritance.
MASTER_PORT=${MASTER_PORT:-6000}
WORLD_SIZE=${NODE_COUNT:-1}
RANK=${NODE_RANK:-0}
RAY_MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_RANK=${RANK:-0} # 0 代表主节点, >0 代表工作节点
RAY_HEAD_PORT=${RAY_HEAD_PORT:-"6379"}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-"8265"}

XTUNER_USE_FA3=${XTUNER_USE_FA3:-1}
XTUNER_LOG_LEVEL=${XTUNER_LOG_LEVEL:-"INFO"}
PYTHONPATH_VALUE="$(pwd):${PYTHONPATH:-}"
PYTORCH_CUDA_ALLOC_CONF_VALUE=""
XTUNER_USE_SGLANG=0
XTUNER_USE_LMDEPLOY=0
XTUNER_USE_VLLM=0
 
infer_backend_lower=$(echo "$INFER_BACKEND" | tr '[:upper:]' '[:lower:]')
if [ "$infer_backend_lower" = "sglang" ]; then
  XTUNER_USE_SGLANG=1
elif [ "$infer_backend_lower" = "lmdeploy" ]; then
  XTUNER_USE_LMDEPLOY=1
  PYTORCH_CUDA_ALLOC_CONF_VALUE='expandable_segments:True'
  PYTHONPATH_VALUE="${LMDEPLOY_PATH}:${PYTHONPATH_VALUE}"
elif [ "$infer_backend_lower" = "vllm" ]; then
  XTUNER_USE_VLLM=1
  PYTORCH_CUDA_ALLOC_CONF_VALUE='expandable_segments:True'
else
  echo "Error: INFER_BACKEND '$INFER_BACKEND' is not supported or not specified!"
  exit 1
fi 

current_time=$(date "+%m%d%H")
# 取模型路径的最后一级作为model_name，取数据路径的倒数第二级作为data_name
model_dir_name=$(basename "$MODEL_PATH")
data_dir_name=$(basename "$(dirname "$DATA_PATH")")
DIR=$(pwd)
WORK_DIR="${WORK_DIR:-${DIR}/work_dirs/${model_dir_name}_${data_dir_name}_${infer_backend_lower}}"
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi
WORK_DIR=$(realpath "$WORK_DIR")
LMDEPLOY_LOG_FILE="${WORK_DIR}/lmdeploy_log_${current_time}.txt"
XTUNER_RL_MEM_DIR=""
if [ "$ACCELERATOR" = "GPU" ]; then
    # TODO: support NPU RL Memory Monitor
    XTUNER_RL_MEM_DIR="${WORK_DIR}/mem_${current_time}"
fi

# 2. Launch Ray cluster
# 根据 NODE_COUNT 分配 num_cpus, 防止内存OOM
node_count=${NODE_COUNT:-1}

if [ "$RAY_RANK" -eq 0 ]; then
  rm -rf /tmp/ray_log
  RAY_LOG_DIR="${WORK_DIR}/ray_${current_time}"
  mkdir -p "$RAY_LOG_DIR"
  ln -sfn "$RAY_LOG_DIR" /tmp/ray_log
  ray start --head \
    --node-ip-address="$RAY_MASTER_ADDR" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --include-dashboard=true \
    --disable-usage-stats \
    --temp-dir="/tmp/ray_log"
else
  while true; do
    if curl --connect-timeout 2 "http://${RAY_MASTER_ADDR}:${RAY_DASHBOARD_PORT}" >/dev/null 2>&1; then
      echo "Successfully connected to Ray master at ${RAY_MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
      break
    else
      echo "Waiting for Ray master at ${RAY_MASTER_ADDR}:${RAY_DASHBOARD_PORT} to be available..."
      sleep 2
    fi
  done
  ray start --address="$RAY_MASTER_ADDR:$RAY_HEAD_PORT" --block --disable-usage-stats
fi

while true; do
  result=$(ray status | grep ${ACCELERATOR} | cut -d ' ' -f2 | cut -d '/' -f2)
  expected_accelerator_count=$((node_count * ${ACCELERATOR_PER_NODE}))
  if [ "$result" = "$expected_accelerator_count.0" ]; then
    break
  else
    echo "Waiting for ${ACCELERATOR} count to be $expected_accelerator_count, current: $result"
    sleep 2
  fi
done

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"
cp "$CONFIG_PATH" "${WORK_DIR}/config.py"
LOG_FILE="${WORK_DIR}/training_log_${current_time}.txt"

# 3. Submit training job on Head node
if [ "$RAY_RANK" -eq 0 ]; then
  # Keep the runtime env explicit and easy to review. This heredoc intentionally
  # mirrors the common Ray examples; values are expected to be simple paths,
  # flags, or numbers. If a value contains JSON-special characters such as
  # quotes, backslashes, or newlines, ray job submit should fail while parsing
  # --runtime-env-json instead of silently dropping the variable.
  RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "WORK_DIR": "${WORK_DIR}",
    "MODEL_PATH": "${MODEL_PATH}",
    "DATA_PATH": "${DATA_PATH}",
    "EVAL_DATA_PATH": "${EVAL_DATA_PATH}",
    "WORLD_SIZE": "${WORLD_SIZE}",
    "MASTER_ADDR": "${RAY_MASTER_ADDR}",
    "MASTER_PORT": "${MASTER_PORT}",
    "RAY_MASTER_ADDR": "${RAY_MASTER_ADDR}",
    "ACCELERATOR": "${ACCELERATOR}",
    "XTUNER_USE_FA3": "${XTUNER_USE_FA3}",
    "XTUNER_LOG_LEVEL": "${XTUNER_LOG_LEVEL}",
    "PYTHONPATH": "${PYTHONPATH_VALUE}",
    "PYTHONUNBUFFERED": "1",
    "XTUNER_USE_SGLANG": "${XTUNER_USE_SGLANG}",
    "XTUNER_USE_LMDEPLOY": "${XTUNER_USE_LMDEPLOY}",
    "XTUNER_USE_VLLM": "${XTUNER_USE_VLLM}",
    "PYTORCH_CUDA_ALLOC_CONF": "${PYTORCH_CUDA_ALLOC_CONF_VALUE}",
    "LMDEPLOY_LOG_FILE": "${LMDEPLOY_LOG_FILE}",
    "XTUNER_RL_MEM_DIR": "${XTUNER_RL_MEM_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "False"
  }
}
EOF_JSON
)

  ray job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
       --runtime-env-json="$RUNTIME_ENV_JSON" \
       -- python xtuner/v1/train/cli/rl.py \
       --config $CONFIG_PATH \
       2>&1 | tee -a "$LOG_FILE"

  echo "训练任务提交完成。日志文件: $LOG_FILE"
fi
