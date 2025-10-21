set -ex
# examples of usage:
# qwen3_8B_grpo_gsm8k training: bash examples/v1/scripts/run_rl.sh "sglang" examples/v1/config/rl_qwen3_8B_grpo.py $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
# qwen2.5_7B_dapo_math training: bash examples/v1/scripts/run_rl.sh "sglang" examples/v1/config/rl_qwen2.5_7B_dapo.py $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH

CONFIG_PATH=$1
INFER_BACKEND=$2
MODEL_PATH=$3
DATA_PATH=$4
EVAL_DATA_PATH=${5:-""}


export MODEL_PATH=$MODEL_PATH
export DATA_PATH=$DATA_PATH
export EVAL_DATA_PATH=$EVAL_DATA_PATH

export XTUNER_USE_FA3=1
export XTUNER_MAX_CONCURRENCY=2048
export XTUNER_LOG_LEVEL="INFO"
export PYTHONPATH=$(pwd):$PYTHONPATH

# rollout infer engine config 
infer_backend_lower=$(echo "$INFER_BACKEND" | tr '[:upper:]' '[:lower:]')
if [ "$infer_backend_lower" = "sglang" ]; then
  export XTUNER_USE_SGLANG=1
  unset PYTORCH_CUDA_ALLOC_CONF
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

# log dir
current_time=$(date "+%m%d%H")
# 取模型路径的最后一级作为model_name，取数据路径的倒数第二级作为data_name
model_dir_name=$(basename "$MODEL_PATH")
data_dir_name=$(basename "$(dirname "$MODEL_PATH")")
export WORK_DIR="work_dirs/${model_dir_name}_${data_dir_name}_${infer_backend_lower}"
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"
cp "$CONFIG_PATH" "${WORK_DIR}/config.py"

export RAY_CLUSTER_URL="auto"

python xtuner/v1/train/cli/rl.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"


  