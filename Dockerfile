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
    iputils-ping telnet netcat-openbsd htop bubblewrap socat -y && \
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

RUN pip wheel -w ${DEEP_EP_DIR} -v --no-deps .

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

COPY --from=flash_attn ${FLASH_ATTN3_DIR} ${FLASH_ATTN3_DIR}
COPY --from=flash_attn ${FLASH_ATTN_DIR} ${FLASH_ATTN_DIR}
COPY --from=adaptive_gemm ${ADAPTIVE_GEMM_DIR} ${ADAPTIVE_GEMM_DIR}
COPY --from=grouped_gemm ${GROUPED_GEMM_DIR} ${GROUPED_GEMM_DIR}
COPY --from=deep_ep ${DEEP_EP_DIR} ${DEEP_EP_DIR}
# COPY --from=deep_ep ${NVSHMEM_PREFIX} ${NVSHMEM_PREFIX}
COPY --from=deep_gemm ${DEEP_GEMM_DIR} ${DEEP_GEMM_DIR}
COPY --from=causal_conv1d ${CAUSAL_CONV1D_DIR} ${CAUSAL_CONV1D_DIR}

RUN unzip ${FLASH_ATTN_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${FLASH_ATTN3_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${ADAPTIVE_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${GROUPED_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${DEEP_EP_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${DEEP_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${CAUSAL_CONV1D_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}

ARG DEFAULT_PYPI_URL

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN pip install pystack py-spy --no-cache-dir -i ${DEFAULT_PYPI_URL}

# install sglang and its runtime requirements
ENV XTUNER_SGLANG_ENVS_DIR=/envs/sglang

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN \
   pip install --target ${XTUNER_SGLANG_ENVS_DIR} \
   sglang==0.5.9 sgl-kernel==0.3.21 \
   apache-tvm-ffi==0.1.9 \
   anthropic==0.86.0 \
   build==1.4.0 \
   cuda-python==12.9.0 \
   decord2==3.2.0 \
   flashinfer_python==0.6.3 \
   flashinfer_cubin==0.6.3 \
   gguf==0.18.0 \
   modelscope==1.35.3 \
   nvidia-cutlass-dsl==4.4.2 \
   openai-harmony==0.0.4 \
   openai==2.6.1 \
   outlines==0.1.11 \
   quack-kernels==0.2.4 \
   timm==1.0.16 \
   torchao==0.9.0 \
   torchaudio==2.9.1 \
   torchcodec==0.8.0 \
   xgrammar==0.1.32 \
   smg-grpc-proto==0.4.5 \
   grpcio==1.78.1 \
   grpcio-reflection==1.78.1 \
   grpcio-health-checking==1.80.0 \
   pycryptodomex==3.23.0 \
   lxml==6.0.2 \
   cuda-bindings==12.9.6 \
   cuda-pathfinder==1.5.0 \
   nvidia-cudnn-frontend==1.21.0 \
   lark==1.3.1 \
   pycountry==26.2.16 \
   airportsdata==20260315 \
   outlines_core==0.1.26 \
   torch-c-dlpack-ext==0.1.5 \
   pyproject_hooks==1.2.0 \
   huggingface_hub==0.36.2 \
   torch_memory_saver==0.0.9 \
   llguidance==0.7.11 blobfile==3.0.0 \
   pybase64 orjson uvloop setproctitle msgspec \
   compressed_tensors python-multipart \
   hf_transfer interegular --no-cache-dir --no-deps -i ${DEFAULT_PYPI_URL}

# install lmdeploy and its missing runtime requirements
ARG LMDEPLOY_VERSION
ARG LMDEPLOY_URL
ENV XTUNER_LMDEPLOY_ENVS_DIR=/envs/lmdeploy

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
ARG LMDEPLOY_WHEELS=https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu128-cp312-cp312-manylinux2014_x86_64.whl
RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    --mount=type=secret,id=NO_PROXY,env=no_proxy \
    pip install fastapi fire openai outlines \
        partial_json_parser 'ray[default]<3' shortuuid uvicorn pybase64 \
        'pydantic>2' openai_harmony dlblas --target ${XTUNER_LMDEPLOY_ENVS_DIR} --no-cache-dir -i ${DEFAULT_PYPI_URL} && \
    pip install xgrammar==0.1.32 --no-cache-dir -i ${DEFAULT_PYPI_URL} --no-deps && \
    if [ -n "${LMDEPLOY_VERSION}" ]; then \
        # pip install lmdeploy==${LMDEPLOY_VERSION} --target ${XTUNER_LMDEPLOY_ENVS_DIR} --no-deps --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
        echo pip install ${LMDEPLOY_WHEELS} --target ${XTUNER_LMDEPLOY_ENVS_DIR} --no-deps --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
        pip install ${LMDEPLOY_WHEELS} --target ${XTUNER_LMDEPLOY_ENVS_DIR} --no-deps --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
    else \
        git clone $(echo ${LMDEPLOY_URL} | cut -d '@' -f 1) && \
        cd ${CODESPACE}/lmdeploy && \
        git checkout $(echo ${LMDEPLOY_URL} | cut -d '@' -f 2) && \
        pip install . -v --target ${XTUNER_LMDEPLOY_ENVS_DIR} --no-deps --no-cache-dir -i ${DEFAULT_PYPI_URL}; \
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

# Install custom .pth file for conditional lmdeploy and sglang path injection
RUN cp -r .dev_scripts/xtuner_rl_path* ${PYTHON_SITE_PACKAGE_PATH}/

# RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
RUN pip install .[all] -v --no-cache-dir -i ${DEFAULT_PYPI_URL}

WORKDIR ${CODESPACE}

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
