rjob_to_puyuvlm_gpu() {
    rjob submit \
        --name=sft-internvl3.5-8b \
        --gpu=8 --memory=1638400 --charged-group=puyuvlm_gpu --cpu=160 \
        --private-machine=group \
        -P 4 \
        --image registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
        --mount=gpfs://gpfs1/gaozhangwei:/mnt/shared-storage-user/gaozhangwei \
        --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
        --mount=gpfs://gpfs1/songdemin/:/mnt/shared-storage-user/songdemin/ \
        --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery/ \
        --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
        --host-network=true \
        --gang-start=true \
        --custom-resources rdma/mlnx_shared=8  \
        --custom-resources mellanox.com/mlnx_rdma=1 \
        -e DISTRIBUTED_JOB=true \
        -- bash -c '
        pip install decord boto3 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
        pip install /mnt/shared-storage-user/gaozhangwei/workspace_glx/petrel-oss-sdk-2.3.24.tar.gz -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
        cd /mnt/shared-storage-user/gaozhangwei/workspace_glx/xtuner
        bash scripts/sft_intern_s1_vl_8B.sh '
}


rjob_to_puyullmgpunew_gpu() {
    rjob submit \
        --name=sft-internvl3.5-8b \
        --gpu=8 --memory=1638400 --cpu=120 \
        --charged-group=puyullmgpunew_gpu \
        --namespace ailab-puyullmgpunew \
        --private-machine=group \
        -P 32 \
        --image registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
        --mount=gpfs://gpfs1/gaozhangwei:/mnt/shared-storage-user/gaozhangwei \
        --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
        --mount=gpfs://gpfs1/songdemin/:/mnt/shared-storage-user/songdemin/ \
        --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery/ \
        --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
        --host-network=true \
        --gang-start=true \
        --custom-resources rdma/mlnx_shared=8  \
        --custom-resources mellanox.com/mlnx_rdma=1 \
        -e DISTRIBUTED_JOB=true \
        -- bash -c '
        pip install decord boto3 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
        pip install /mnt/shared-storage-user/gaozhangwei/workspace_glx/petrel-oss-sdk-2.3.24.tar.gz -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
        cd /mnt/shared-storage-user/gaozhangwei/workspace_glx/xtuner
        bash scripts/sft_intern_s1_vl_8B.sh '
}


rjob_to_puyuvlm_gpu
# rjob_to_puyullmgpunew_gpu
