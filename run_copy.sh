model_type=$1  # 或者设置为 "qwen"
train=$2
# 根据 model_type 设置 x 的值
# MAX_TIME=18000 

# echo "starting sleeping"
# sleep 14400

save_iters=(iter_1000)
# save_iters=(iter_250 iter_500 iter_750)
if [ "$model_type" == "llama" ]; then
    save_dir=./work_dirs/saved_model/llama3.1_loss_ablation
    file=llama3_8b_instruct_dpo.py
elif [ "$model_type" == "qwen" ]; then
    save_dir=./work_dirs/saved_model/qwen2.5_loss_ablation
    file=qwen2_instruct_dpo.py
else
  echo "未知的 model_type"
  exit 1
fi

if [ "$train" == "train" ]; then
    mkdir -p $save_dir
    NPROC_PER_NODE=8 xtuner train $file --deepspeed deepspeed_zero3_offload --seed 42 --work-dir $save_dir
elif [ "$train" == "convert" ]; then
    # for i in {0..2}; do
    for ((i = 0; i < ${#save_iters[@]}; i++)); do
        pth=$save_dir/${save_iters[$i]}.pth
        SAVE_PATH=$save_dir/${save_iters[$i]}

        mkdir -p ${SAVE_PATH}

        xtuner convert pth_to_hf $file \
        ${pth} \
        ${SAVE_PATH}
    done
fi

# PID=$!
# START_TIME=$(date +%s)

# while true; do
#     # 获取当前时间
#     CURRENT_TIME=$(date +%s)
    
#     # 计算运行的时间差（单位：秒）
#     ELAPSED_TIME=$((CURRENT_TIME - START_TIME))

#     # 如果运行超过 5 小时，杀死程序
#     if [ $ELAPSED_TIME -ge $MAX_TIME ]; then
#         echo "程序运行超过 5 小时，正在杀死进程 $PID ..."
#         kill -9 $PID
#         break
#     fi
    
#     # 检查程序是否仍在运行
#     if ! ps -p $PID > /dev/null; then
#         echo "程序已正常结束。"
#         break
#     fi

#     # 每隔 60 秒检查一次
#     sleep 60
# done
