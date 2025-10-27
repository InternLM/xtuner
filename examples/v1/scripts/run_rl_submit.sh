set -ex
# examples of usage:
# qwen3_8B_grpo_gsm8k training: bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_8B_grpo.py "sglang" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
# qwen2.5_7B_dapo_math training: bash examples/v1/scripts/run_rl.sh  examples/v1/config/rl_qwen25_7B_dapo.py "sglang" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH

CONFIG_PATH=$1
INFER_BACKEND=$2
MODEL_PATH=$3
DATA_PATH=$4
EVAL_DATA_PATH=${5:-""}

# 1. 环境配置
# NOTE: if you add new env vars, please also add them to RUNTIME_ENV_JSON in step 4.
# master 节点的 IP 地址
export RAY_MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# 0 代表主节点, >0 代表工作节点
export RAY_RANK=${RANK:-0}
export RAY_HEAD_PORT=${RAY_HEAD_PORT:-"6379"}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-"8265"}

export MODEL_PATH=$MODEL_PATH
export DATA_PATH=$DATA_PATH
export EVAL_DATA_PATH=$EVAL_DATA_PATH

export XTUNER_USE_FA3=1
export XTUNER_MAX_CONCURRENCY=2048
export XTUNER_LOG_LEVEL="INFO"
export PYTHONPATH=$(pwd):$PYTHONPATH
 
infer_backend_lower=$(echo "$INFER_BACKEND" | tr '[:upper:]' '[:lower:]')
if [ "$infer_backend_lower" = "sglang" ]; then
  export XTUNER_USE_SGLANG=1
  unset PYTORCH_CUDA_ALLOC_CONF
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
elif [ "$infer_backend_lower" = "lmdeploy" ]; then
  export XTUNER_USE_LMDEPLOY=1
  export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
  export PYTHONPATH=$LMDEPLOY_PATH:$PYTHONPATH
elif [ "$infer_backend_lower" = "vllm" ]; then
  export XTUNER_USE_VLLM=1
  export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
else
  echo "Error: INFER_BACKEND '$INFER_BACKEND' is not supported or not specified!"
  exit 1
fi 

# 2. Launch Ray cluster
# 根据 NODE_COUNT 分配 num_cpus, 防止内存OOM
node_count=${NODE_COUNT:-1}
total_cpus=$((node_count * 128))

if [ "$RAY_RANK" -eq 0 ]; then
  ray start --head \
    --node-ip-address="$RAY_MASTER_ADDR" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --include-dashboard=true \
    --disable-usage-stats \
    --num-cpus=$total_cpus
else
  sleep 10
  ray start --address="$RAY_MASTER_ADDR:$RAY_HEAD_PORT" --block --disable-usage-stats
fi

sleep 10

# 3. Prepare work directory and log file
current_time=$(date "+%m%d%H")
# 取模型路径的最后一级作为model_name，取数据路径的倒数第二级作为data_name
model_dir_name=$(basename "$MODEL_PATH")
data_dir_name=$(basename "$(dirname "$DATA_PATH")")
export WORK_DIR="work_dirs/${model_dir_name}_${data_dir_name}_${infer_backend_lower}"

if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"
cp "$CONFIG_PATH" "${WORK_DIR}/config.py"
LOG_FILE="${WORK_DIR}/training_log_${current_time}.txt"

# 4. Submit training job on Head node
if [ "$RAY_RANK" -eq 0 ]; then
  RUNTIME_ENV_JSON="{
      \"env_vars\": {
        \"WORK_DIR\": \"${WORK_DIR}\",
        \"MODEL_PATH\": \"${MODEL_PATH}\",
        \"DATA_PATH\": \"${DATA_PATH}\",
        \"EVAL_DATA_PATH\": \"${EVAL_DATA_PATH}\",
        \"XTUNER_USE_FA3\": \"${XTUNER_USE_FA3}\",
        \"XTUNER_MAX_CONCURRENCY\": \"${XTUNER_MAX_CONCURRENCY}\",
        \"XTUNER_LOG_LEVEL\": \"${XTUNER_LOG_LEVEL}\",
        \"PYTHONPATH\": \"${PYTHONPATH}\",
        \"MASTER_ADDR\": \"${RAY_MASTER_ADDR}\",
        \"XTUNER_USE_SGLANG\": \"${XTUNER_USE_SGLANG:-}\",
        \"XTUNER_USE_LMDEPLOY\": \"${XTUNER_USE_LMDEPLOY:-}\",
        \"XTUNER_USE_VLLM\": \"${XTUNER_USE_VLLM:-}\",
        \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\"
      }
    }"

  ray job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
       --runtime-env-json="$RUNTIME_ENV_JSON" \
       -- python xtuner/v1/train/cli/rl.py \
       --config $CONFIG_PATH \
       2>&1 | tee -a "$LOG_FILE"

  echo "训练任务提交完成。日志文件: $LOG_FILE"
fi