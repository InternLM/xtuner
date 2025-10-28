# syntax=docker/dockerfile:1.10.0
# builder
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3

## build args
FROM ${BASE_IMAGE} AS setup_env

ARG TORCH_VERSION
ARG PPA_SOURCE

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    sed -i "s@http://.*.ubuntu.com@${PPA_SOURCE}@g" /etc/apt/sources.list.d/ubuntu.sources && \
    apt update && \
    apt install --no-install-recommends ca-certificates -y && \
    apt install --no-install-recommends bc wget -y && \
    apt install --no-install-recommends build-essential sudo -y && \
    apt install --no-install-recommends git curl pkg-config tree unzip tmux \
    openssh-server openssh-client dnsutils iproute2 lsof net-tools zsh rclone -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN if [ -d /etc/pip ] && [ -f /etc/pip/constraint.txt ]; then echo > /etc/pip/constraint.txt; fi
RUN pip install pystack py-spy --no-cache-dir
RUN git config --system --add safe.directory "*"

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    if [ -n "${TORCH_VERSION}" ]; then \
        pip install torchvision torch==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --no-cache-dir; \
    fi

# set reasonable default for CUDA architectures when building ngc image
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0"

RUN pip uninstall flash_attn opencv -y && rm -rf /usr/local/lib/python3.12/dist-packages/cv2

ARG FLASH_ATTN_DIR=/tmp/flash-attn
ARG CODESPACE=/root/codespace
ARG FLASH_ATTN3_DIR=/tmp/flash-attn3
ARG ADAPTIVE_GEMM_DIR=/tmp/adaptive_gemm
ARG GROUPED_GEMM_DIR=/tmp/grouped_gemm
ARG DEEP_EP_DIR=/tmp/deep_ep
ARG NVSHMEM_WHL_DIR=/tmp/nvshmem

RUN mkdir -p $CODESPACE
WORKDIR ${CODESPACE}

# compile flash-attn
FROM setup_env AS flash_attn

ARG CODESPACE
ARG FLASH_ATTN_DIR
ARG FLASH_ATTN3_DIR
ARG FLASH_ATTN_URL

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

# pypi install nvshmem and compile deepep
FROM setup_env AS deep_ep

ARG CODESPACE
ARG DEEP_EP_DIR
ARG DEEP_EP_URL
ARG NVSHMEM_WHL_DIR
# build sm90 and sm100 for deep_ep for now
ARG TORCH_CUDA_ARCH_LIST="9.0 10.0"

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    pip wheel -w ${NVSHMEM_WHL_DIR} -v "nvidia-nvshmem-cu12>=3.4.5" && \
    pip install ${NVSHMEM_WHL_DIR}/*.whl && \
    git clone $(echo ${DEEP_EP_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/DeepEP && \
    git checkout $(echo ${DEEP_EP_URL} | cut -d '@' -f 2) && \
    git submodule update --init --recursive --force

WORKDIR ${CODESPACE}/DeepEP

RUN pip wheel -w ${DEEP_EP_DIR} -v --no-deps .

# integration xtuner
FROM setup_env AS xtuner_dev

ARG PYTHON_SITE_PACKAGE_PATH=/usr/local/lib/python3.12/dist-packages
ARG CODESPACE

ARG FLASH_ATTN_DIR
ARG FLASH_ATTN3_DIR
ARG ADAPTIVE_GEMM_DIR
ARG GROUPED_GEMM_DIR
ARG DEEP_EP_DIR
ARG NVSHMEM_WHL_DIR

COPY --from=flash_attn ${FLASH_ATTN3_DIR} ${FLASH_ATTN3_DIR}
COPY --from=flash_attn ${FLASH_ATTN_DIR} ${FLASH_ATTN_DIR}
COPY --from=adaptive_gemm ${ADAPTIVE_GEMM_DIR} ${ADAPTIVE_GEMM_DIR}
COPY --from=grouped_gemm ${GROUPED_GEMM_DIR} ${GROUPED_GEMM_DIR}
COPY --from=deep_ep ${DEEP_EP_DIR} ${DEEP_EP_DIR}
COPY --from=deep_ep ${NVSHMEM_WHL_DIR} ${NVSHMEM_WHL_DIR}

RUN unzip ${FLASH_ATTN_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${FLASH_ATTN3_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${ADAPTIVE_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${GROUPED_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${DEEP_EP_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${NVSHMEM_WHL_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}

# install sglang and its runtime requirements
ARG SGLANG_VERSION

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
   pip install sglang==${SGLANG_VERSION} sgl_kernel pybase64 orjson uvloop setproctitle msgspec \
   compressed_tensors python-multipart torch_memory_saver \
   grpcio-tools==1.75.1 hf_transfer interegular llguidance==0.7.11 \
   xgrammar==0.1.24 blobfile==3.0.0 flashinfer_python==0.4.0 --no-cache-dir --no-deps

# install lmdeploy and its missing runtime requirements
ARG LMDEPLOY_VERSION
ARG LMDEPLOY_URL

RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    pip install fastapi fire openai outlines \
        partial_json_parser ray[default] shortuuid uvicorn \
        'pydantic>2' openai_harmony --no-cache-dir && \
    if [ -n "${LMDEPLOY_VERSION}" ]; then \
        pip install lmdeploy==${LMDEPLOY_VERSION} --no-deps --no-cache-dir; \
    else \
        git clone $(echo ${LMDEPLOY_URL} | cut -d '@' -f 1) && \
        cd ${CODESPACE}/lmdeploy && \
        git checkout $(echo ${LMDEPLOY_URL} | cut -d '@' -f 2) && \
        pip install . -v --no-deps --no-cache-dir; \
    fi

## install xtuner
ARG XTUNER_URL
ARG XTUNER_COMMIT
#RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
#   git clone $(echo ${XTUNER_URL} | cut -d '@' -f 1) && \
#   cd ${CODESPACE}/xtuner && \
#   git checkout $(echo ${XTUNER_URL} | cut -d '@' -f 2) 
COPY . ${CODESPACE}/xtuner

WORKDIR ${CODESPACE}/xtuner
RUN --mount=type=secret,id=HTTPS_PROXY,env=https_proxy \
    pip install .[all] -v --no-cache-dir

WORKDIR ${CODESPACE}

# setup sysctl
RUN echo "fs.file-max=100000" >> /etc/sysctl.conf
RUN sysctl -p
