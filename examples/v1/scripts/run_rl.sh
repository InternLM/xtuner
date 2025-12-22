set -ex
ray stop --force
# examples of usage:
# qwen3_8B_grpo_gsm8k training: 
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_8B_grpo.py "sglang" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
# qwen2.5_7B_dapo_math training: 
# bash examples/v1/scripts/run_rl.sh  examples/v1/config/rl_qwen25_7B_dapo.py "sglang" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH

CONFIG_PATH=$1
INFER_BACKEND=$2
MODEL_PATH=$3
DATA_PATH=$4
EVAL_DATA_PATH=${5:-""}

export PYTHONPATH=$(pwd):$PYTHONPATH

# ray 环境变量
export MASTER_PORT=6000
export WORLD_SIZE=$NODE_COUNT
export RANK=$NODE_RANK
export RAY_MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_RANK=${RANK:-0} # 0 代表主节点, >0 代表工作节点
export RAY_HEAD_PORT=${RAY_HEAD_PORT:-"6379"}
export RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-"10001"}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-"8265"}
# TODO: 提供非环境变量方式配置 ray_max_concurrency
export RAY_MAX_CONCURRENCY=${RAY_MAX_CONCURRENCY:-1024} # dataflow_max_concurrency * prompt_repeat_k

# xtuner 环境变量
export MODEL_PATH=$MODEL_PATH
export DATA_PATH=$DATA_PATH
export EVAL_DATA_PATH=$EVAL_DATA_PATH
export XTUNER_USE_FA3=${XTUNER_USE_FA3:-1}
export XTUNER_LOG_LEVEL=${XTUNER_LOG_LEVEL:-"INFO"}
export PYTHONUNBUFFERED=1

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

current_time=$(date "+%m%d%H")
# 取模型路径的最后一级作为model_name，取数据路径的倒数第二级作为data_name
model_dir_name=$(basename "$MODEL_PATH")
data_dir_name=$(basename "$(dirname "$DATA_PATH")")
DIR=$(pwd)
export WORK_DIR="${DIR}/work_dirs/${model_dir_name}_${data_dir_name}_${infer_backend_lower}"
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi
export LMDEPLOY_LOG_FILE="${WORK_DIR}/lmdeploy_log_${current_time}.txt"
export XTUNER_RL_MEM_DIR="${WORK_DIR}/mem_${current_time}"

# 2. Launch Ray cluster
# 根据 NODE_COUNT 分配 num_cpus, 防止内存OOM
node_count=${NODE_COUNT:-1}
total_cpus=$((node_count * 128))

if [ "$RAY_RANK" -eq 0 ]; then
  rm -rf /tmp/ray_log
  export RAY_LOG_DIR="${WORK_DIR}/ray_${current_time}/"
  mkdir -p ${RAY_LOG_DIR}
  ln -sfn "${RAY_LOG_DIR}" /tmp/ray_log
  ray start --head \
    --node-ip-address="$RAY_MASTER_ADDR" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --include-dashboard=true \
    --disable-usage-stats \
    --num-cpus=$total_cpus \
    --temp-dir="/tmp/ray_log/"
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
  result=$(ray status | grep GPU | cut -d ' ' -f2 | cut -d '/' -f2)
  expected_gpu_count=$((node_count * 8))
  if [ "$result" = "$expected_gpu_count.0" ]; then
    break
  else
    echo "Waiting for GPU count to be $expected_gpu_count, current: $result"
    sleep 2
  fi
done

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"
cp "$CONFIG_PATH" "${WORK_DIR}/config.py"
LOG_FILE="${WORK_DIR}/training_log_${current_time}.txt"

python xtuner/v1/train/cli/rl.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"