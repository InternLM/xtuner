# syntax=docker/dockerfile:1.10.0
# builder
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3

## build base env
FROM ${BASE_IMAGE} AS setup_env

ARG PPA_SOURCE
# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN sed -i "s@http://.*.ubuntu.com@${PPA_SOURCE}@g" /etc/apt/sources.list.d/ubuntu.sources && \
    apt update && \
    apt install --no-install-recommends ca-certificates -y && \
    apt install --no-install-recommends bc wget -y && \
    apt install --no-install-recommends build-essential sudo -y && \
    apt install --no-install-recommends git curl pkg-config tree unzip tmux \
    openssh-server openssh-client dnsutils iproute2 lsof net-tools zsh rclone \
    iputils-ping telnet netcat-openbsd htop bubblewrap socat pciutils -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN if [ -d /etc/pip ] && [ -f /etc/pip/constraint.txt ]; then echo > /etc/pip/constraint.txt; fi
RUN pip uninstall flash_attn opencv -y && rm -rf /usr/local/lib/python3.12/dist-packages/cv2
RUN git config --system --add safe.directory "*"

# torch
ARG TORCH_VERSION
ARG PYTORCH_WHEELS_URL
RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    --mount=type=secret,id=NO_PROXY,env=no_proxy \
    if [ -n "${TORCH_VERSION}" ]; then \
        pip install torchvision torch==${TORCH_VERSION} \
        -i ${PYTORCH_WHEELS_URL}/cu128 \
        --extra-index-url ${PYTORCH_WHEELS_URL}/cu126 \
        --no-cache-dir; \
    fi
# set reasonable default for CUDA architectures when building ngc image
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"

ARG FLASH_ATTN_DIR=/tmp/flash-attn
ARG CODESPACE=/root/codespace
ARG FLASH_ATTN3_DIR=/tmp/flash-attn3
ARG ADAPTIVE_GEMM_DIR=/tmp/adaptive_gemm
ARG GROUPED_GEMM_DIR=/tmp/grouped_gemm
ARG CAUSAL_CONV1D_DIR=/tmp/causal_conv1d
ARG DEEP_EP_DIR=/tmp/deep_ep
ARG DEEP_GEMM_DIR=/tmp/deep_gemm
ARG DEEPSELECT_DIR=/tmp/deepselect
ARG FLASHMLA_DIR=/tmp/flash_mla
ARG NVSHMEM_PREFIX=/usr/local/nvshmem

RUN mkdir -p $CODESPACE
WORKDIR ${CODESPACE}

# compile flash-attn
FROM setup_env AS flash_attn

ARG CODESPACE
ARG FLASH_ATTN_DIR
ARG FLASH_ATTN3_DIR
ARG FLASH_ATTN_URL
# force hopper for now, you change it throught build args
ARG FLASH_ATTN_CUDA_ARCHS="90"
ARG FLASH_ATTENTION_DISABLE_SM80="TRUE"

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${FLASH_ATTN_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/flash-attention && \
    git checkout $(echo ${FLASH_ATTN_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/flash-attention

RUN cd hopper && FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel -w ${FLASH_ATTN3_DIR} -v --no-deps .
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel -w ${FLASH_ATTN_DIR} -v --no-deps .

# compile adaptive_gemm
FROM setup_env AS adaptive_gemm

ARG CODESPACE
ARG ADAPTIVE_GEMM_DIR
ARG ADAPTIVE_GEMM_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${ADAPTIVE_GEMM_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/AdaptiveGEMM && \
    git checkout $(echo ${ADAPTIVE_GEMM_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/AdaptiveGEMM

RUN pip wheel -w ${ADAPTIVE_GEMM_DIR} -v --no-deps .

# compile grouped_gemm(permute and unpermute)
FROM setup_env AS grouped_gemm

ARG CODESPACE
ARG GROUPED_GEMM_DIR
ARG GROUPED_GEMM_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${GROUPED_GEMM_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/GroupedGEMM && \
    git checkout $(echo ${GROUPED_GEMM_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/GroupedGEMM

RUN pip wheel -w ${GROUPED_GEMM_DIR} -v --no-deps .

# compile causal_conv1d
FROM setup_env AS causal_conv1d

ARG CODESPACE
ARG CAUSAL_CONV1D_DIR
ARG CAUSAL_CONV1D_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${CAUSAL_CONV1D_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/causal-conv1d && \
    git checkout $(echo ${CAUSAL_CONV1D_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/causal-conv1d

RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip wheel -w ${CAUSAL_CONV1D_DIR} -v --no-deps --no-build-isolation .

# compile deepselect
FROM setup_env AS deepselect

ARG CODESPACE
ARG DEEPSELECT_DIR
ARG DEEPSELECT_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${DEEPSELECT_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/DeepSelect && \
    git checkout $(echo ${DEEPSELECT_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/DeepSelect

# DeepSelect 0f03b68 hard-codes Blackwell (SM100a/SM103a) and asserts NVCC>=12.9,
# but the base image (NGC 25.03) ships NVCC 12.8 which cannot build SM100a.
# Patch setup.py to target Hopper SM90a only: neutralize the version guard and
# swap the gencode flags to compute_90a/sm_90a. deep_select then runs on Hopper;
# on Blackwell the caller is expected to fall back to torch.topk.
# DEEP_SELECT_DISABLE_REG_SPILL_CHECK avoids the register-spill gate falsely
# failing when building for an architecture other than the default Blackwell.
RUN sed -i 's@if major < 12 or (major == 12 and minor <= 8):@if False:@' setup.py && \
    sed -i 's@compute_100a,code=sm_100a@compute_90a,code=sm_90a@' setup.py && \
    sed -i '/compute_103a,code=sm_103a/d' setup.py && \
    DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA DEEP_SELECT_DISABLE_REG_SPILL_CHECK=1 pip wheel -w ${DEEPSELECT_DIR} -v --no-deps --no-build-isolation .

# compile flash_mla
FROM setup_env AS flash_mla

ARG CODESPACE
ARG FLASHMLA_DIR
ARG FLASHMLA_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${FLASHMLA_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/FlashMLA && \
    git checkout $(echo ${FLASHMLA_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/FlashMLA

RUN TORCH_CUDA_ARCH_LIST=9.0 FLASH_MLA_DISABLE_SM100=1 pip wheel -w ${FLASHMLA_DIR} -v --no-deps --no-build-isolation .

# compile nvshmem and deepep
FROM setup_env AS deep_ep

ARG CODESPACE
ARG DEEP_EP_DIR
ARG DEEP_EP_URL

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
#     curl -LO https://github.com/NVIDIA/nvshmem/releases/download/v3.4.5-0/nvshmem_src_cuda-all-all-3.4.5.tar.gz && \
#     tar -zxvf nvshmem_src_cuda-all-all-3.4.5.tar.gz && \
#     cd ${CODESPACE}/nvshmem_src && \
#     NVSHMEM_SHMEM_SUPPORT=0 \
#     NVSHMEM_UCX_SUPPORT=0 \
#     NVSHMEM_USE_NCCL=0 \
#     NVSHMEM_MPI_SUPPORT=0 \
#     NVSHMEM_IBGDA_SUPPORT=1 \
#     NVSHMEM_USE_GDRCOPY=0 \
#     NVSHMEM_PMIX_SUPPORT=0 \
#     NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
#     NVSHMEM_BUILD_TESTS=0 \
#     NVSHMEM_BUILD_EXAMPLES=0 \
#     NVSHMEM_BUILD_HYDRA_LAUNCHER=0 \
#     NVSHMEM_BUILD_TXZ_PACKAGE=0 \
#     NVSHMEM_BUILD_PYTHON_LIB=OFF \
#     cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_PREFIX} -DMLX5_lib=/lib/x86_64-linux-gnu/libmlx5.so.1 && \
#     cmake --build build --target install --parallel 32 && \
RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    cd ${CODESPACE} && git clone $(echo ${DEEP_EP_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/DeepEP && \
    git checkout $(echo ${DEEP_EP_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/DeepEP

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    (pip show nvidia-nvshmem-cu12 >/dev/null 2>&1 || pip install nvidia-nvshmem-cu12==3.4.5) && \
    pip wheel -w ${DEEP_EP_DIR} -v --no-deps .

# compile deep_gemm
FROM setup_env AS deep_gemm

ARG CODESPACE
ARG DEEP_GEMM_DIR
ARG DEEP_GEMM_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    git clone $(echo ${DEEP_GEMM_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/DeepGEMM && \
    git checkout $(echo ${DEEP_GEMM_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/DeepGEMM

RUN pip wheel -w ${DEEP_GEMM_DIR} -v --no-deps .

# integration xtuner
FROM setup_env AS xtuner_dev

ARG PYTHON_SITE_PACKAGE_PATH=/usr/local/lib/python3.12/dist-packages
ARG CODESPACE

ARG FLASH_ATTN_DIR
ARG FLASH_ATTN3_DIR
ARG ADAPTIVE_GEMM_DIR
ARG GROUPED_GEMM_DIR
ARG DEEP_EP_DIR
ARG DEEP_GEMM_DIR
ARG CAUSAL_CONV1D_DIR
ARG DEEPSELECT_DIR
ARG FLASHMLA_DIR

COPY --from=flash_attn ${FLASH_ATTN3_DIR} ${FLASH_ATTN3_DIR}
COPY --from=flash_attn ${FLASH_ATTN_DIR} ${FLASH_ATTN_DIR}
COPY --from=adaptive_gemm ${ADAPTIVE_GEMM_DIR} ${ADAPTIVE_GEMM_DIR}
COPY --from=grouped_gemm ${GROUPED_GEMM_DIR} ${GROUPED_GEMM_DIR}
COPY --from=deep_ep ${DEEP_EP_DIR} ${DEEP_EP_DIR}
# COPY --from=deep_ep ${NVSHMEM_PREFIX} ${NVSHMEM_PREFIX}
COPY --from=deep_gemm ${DEEP_GEMM_DIR} ${DEEP_GEMM_DIR}
COPY --from=causal_conv1d ${CAUSAL_CONV1D_DIR} ${CAUSAL_CONV1D_DIR}
COPY --from=deepselect ${DEEPSELECT_DIR} ${DEEPSELECT_DIR}
COPY --from=flash_mla ${FLASHMLA_DIR} ${FLASHMLA_DIR}

RUN unzip ${FLASH_ATTN_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${FLASH_ATTN3_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${ADAPTIVE_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${GROUPED_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${DEEP_EP_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${DEEP_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${CAUSAL_CONV1D_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${DEEPSELECT_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${FLASHMLA_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}

ARG DEFAULT_PYPI_URL

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN pip install pystack py-spy --no-cache-dir -i ${DEFAULT_PYPI_URL}

# install rdkit, boto3, nvidia-cudnn-frontend
RUN pip install rdkit boto3 nvidia-cudnn-frontend==1.28.0 --no-cache-dir -i ${DEFAULT_PYPI_URL}

# install lmdeploy and its missing runtime requirements
ARG LMDEPLOY_VERSION
ARG LMDEPLOY_URL

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    # --mount=type=secret,id=NO_PROXY,env=no_proxy \
    pip install fastapi fire openai outlines \
        pyzmq aiohttp cloudpickle prometheus_client protobuf numpy pillow einops tiktoken sentencepiece \
        partial_json_parser 'ray[default]<3' shortuuid uvicorn pybase64 \
        'pydantic>2' openai_harmony dlblas --no-cache-dir -i ${DEFAULT_PYPI_URL} && \
    pip install xgrammar>=0.1.33 timm!=1.0.23 --no-cache-dir -i ${DEFAULT_PYPI_URL} --no-deps && \
    if [ -n "${LMDEPLOY_VERSION}" ]; then \
        pip install lmdeploy==${LMDEPLOY_VERSION} --no-deps --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
    else \
        git clone $(echo ${LMDEPLOY_URL} | cut -d '@' -f 1) && \
        cd ${CODESPACE}/lmdeploy && \
        git checkout $(echo ${LMDEPLOY_URL} | cut -d '@' -f 2) && \
        pip install . -v --no-deps --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
    fi

## install xtuner
ARG XTUNER_URL
ARG XTUNER_COMMIT
# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
#   git clone $(echo ${XTUNER_URL} | cut -d '@' -f 1) && \
#   cd ${CODESPACE}/xtuner && \
#   git checkout $(echo ${XTUNER_URL} | cut -d '@' -f 2) 
COPY . ${CODESPACE}/xtuner

WORKDIR ${CODESPACE}/xtuner

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN pip install .[all] -v --no-cache-dir -i ${DEFAULT_PYPI_URL}

WORKDIR ${CODESPACE}

# use custom fla
ARG FLA_URL
RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    if [ -n "${FLA_URL}" ]; then \
        pip uninstall -y fla-core flash-linear-attention && \
        pip install --no-cache-dir --no-deps --no-build-isolation "git+${FLA_URL}" -i ${DEFAULT_PYPI_URL}; \
    fi

# nccl update for torch 2.6.0
# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN if [ "x${TORCH_VERSION}" = "x2.6.0" ]; then \
        pip install nvidia-nccl-cu12==2.25.1 --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
    fi

# cudnn update for torch 2.9.1
# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN if [ "x${TORCH_VERSION}" = "x2.9.1" ]; then \
        pip install nvidia-cudnn-cu12==9.15.1.9 --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
    fi

# setup sysctl
RUN echo "fs.file-max=100000" >> /etc/sysctl.conf
RUN sysctl -p
