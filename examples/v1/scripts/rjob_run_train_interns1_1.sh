JOB_NAME=xtuner_router_reply_insterns2_preview_ßtrain
NUM_NODES=8
WORKER_GPU=8
CPU_NUMS=12
NODE_MEMS=960
IMAGE=registry.h.pjlab.org.cn/ailab-puyu/xpuyu:torch-2.7.0-076676dd-0708
# rjob delete ${JOB_NAME}
clusterx run \
--job-name=${JOB_NAME} \
--gpus-per-task=${WORKER_GPU} \
--memory-per-task=$NODE_MEMS \
--cpus-per-task=$((CPU_NUMS * (WORKER_GPU > 0 ? WORKER_GPU : 1))) \
--image=${IMAGE} \
--num-nodes ${NUM_NODES} \
--priority  9 \
--partition puyullm_gpu \
--project-name ailab-puyullmgpu \
--no-env \
/mnt/shared-storage-user/llmit/user/liukuikun/workspace/xtuner/examples/v1/scripts/train_agentrl.sh
# "zsh -exc '/mnt/shared-storage-user/llmit/user/liukuikun/workspace/crg_rl_projects_router_reply/scripts/run_train_interns1_1.sh'"
