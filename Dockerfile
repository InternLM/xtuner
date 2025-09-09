# syntax=docker/dockerfile:1.10.0
# builder
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3

## build args
FROM ${BASE_IMAGE} AS setup_env

ARG CODESPACE=/root/codespace

ARG FLASH_ATTN_DIR=/tmp/flash-attn
ARG FLASH_ATTN3_DIR=/tmp/flash-attn3
ARG ADAPTIVE_GEMM_DIR=/tmp/adaptive_gemm
ARG GROUPED_GEMM_DIR=/tmp/grouped_gemm

ARG TORCH_VERSION

ARG PPA_SOURCE

RUN if [ -d /etc/pip ] && [ -f /etc/pip/constraint.txt ]; then echo > /etc/pip/constraint.txt; fi
RUN if [ -n "${TORCH_VERSION}" ]; then \
        pip install torchvision torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir; \
    fi

# set reasonable default for CUDA architectures when building ngc image
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0"

RUN sed -i "s@http://.*.ubuntu.com@${PPA_SOURCE}@g" /etc/apt/sources.list.d/ubuntu.sources && \
    apt update && \
    apt install --no-install-recommends ca-certificates -y && \
    apt install --no-install-recommends bc wget -y && \
    apt install --no-install-recommends build-essential sudo -y && \
    apt install --no-install-recommends git curl pkg-config tree unzip tmux \
    openssh-server openssh-client nmap dnsutils iproute2 lsof net-tools -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip uninstall flash_attn -y

# compile flash-attn
FROM setup_env AS flash_attn

ARG CODESPACE
ARG FLASH_ATTN_DIR
ARG FLASH_ATTN3_DIR
ARG FLASH_ATTN_URL

RUN mkdir -p $CODESPACE 
WORKDIR ${CODESPACE}

RUN git clone -c https.proxy=$HTTPS_PROXY $(echo ${FLASH_ATTN_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/flash-attention && \
    git checkout $(echo ${FLASH_ATTN_URL} | cut -d '@' -f 2)

WORKDIR ${CODESPACE}/flash-attention

RUN git submodule update --init --recursive --force
RUN cd hopper && FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel -w ${FLASH_ATTN3_DIR} -v --no-deps .
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel -w ${FLASH_ATTN_DIR} -v --no-deps .

# compile adaptive_gemm
FROM setup_env AS adaptive_gemm

ARG CODESPACE
ARG ADAPTIVE_GEMM_DIR
ARG ADAPTIVE_GEMM_URL

RUN mkdir -p $CODESPACE
WORKDIR ${CODESPACE}

RUN git clone -c https.proxy=$HTTPS_PROXY $(echo ${ADAPTIVE_GEMM_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/AdaptiveGEMM && \
    git checkout $(echo ${ADAPTIVE_GEMM_URL} | cut -d '@' -f 2)

WORKDIR ${CODESPACE}/AdaptiveGEMM

RUN git submodule update --init --recursive --force
RUN pip wheel -w ${ADAPTIVE_GEMM_DIR} -v --no-deps .

# compile grouped_gemm(permute and unpermute)
FROM setup_env AS grouped_gemm

ARG CODESPACE
ARG GROUPED_GEMM_DIR
ARG GROUPED_GEMM_URL

RUN mkdir -p $CODESPACE
WORKDIR ${CODESPACE}

RUN git clone -c https.proxy=$HTTPS_PROXY $(echo ${GROUPED_GEMM_URL} | cut -d '@' -f 1) && \
    cd ${CODESPACE}/GroupedGEMM && \
    git checkout $(echo ${GROUPED_GEMM_URL} | cut -d '@' -f 2)

WORKDIR ${CODESPACE}/GroupedGEMM

RUN git submodule update --init --recursive --force
RUN pip wheel -w ${GROUPED_GEMM_DIR} -v --no-deps .


# integration xtuner
FROM setup_env AS xtuner_dev

ARG PYTHON_SITE_PACKAGE_PATH=/usr/local/lib/python3.12/dist-packages
ARG CODESPACE

ARG FLASH_ATTN_DIR
ARG FLASH_ATTN3_DIR
ARG ADAPTIVE_GEMM_DIR
ARG GROUPED_GEMM_DIR

COPY --from=flash_attn ${FLASH_ATTN3_DIR} ${FLASH_ATTN3_DIR}
COPY --from=flash_attn ${FLASH_ATTN_DIR} ${FLASH_ATTN_DIR}
COPY --from=adaptive_gemm ${ADAPTIVE_GEMM_DIR} ${ADAPTIVE_GEMM_DIR}
COPY --from=grouped_gemm ${GROUPED_GEMM_DIR} ${GROUPED_GEMM_DIR}

RUN unzip ${FLASH_ATTN_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${FLASH_ATTN3_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${ADAPTIVE_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}
RUN unzip ${GROUPED_GEMM_DIR}/*.whl -d ${PYTHON_SITE_PACKAGE_PATH}

ARG XTUNER_URL
ARG XTUNER_COMMIT
ARG LMDEPLOY_VERSION
ARG LMDEPLOY_URL

## install xtuner
RUN mkdir -p $CODESPACE
WORKDIR ${CODESPACE}

#RUN git clone -c https.proxy=$HTTPS_PROXY $(echo ${XTUNER_URL} | cut -d '@' -f 1) && \
    #cd ${CODESPACE}/xtuner && \
    #git checkout $(echo ${XTUNER_URL} | cut -d '@' -f 2) 
COPY . ${CODESPACE}/xtuner

WORKDIR ${CODESPACE}/xtuner
RUN export HTTPS_PROXY=$HTTPS_PROXY \
  && export https_proxy=$HTTPS_PROXY \
  && pip install liger-kernel parametrize --no-cache-dir \
  && pip install . -v --no-cache-dir

RUN pip install pystack py-spy --no-cache-dir
RUN git config --system --add safe.directory "*"

# install lmdeploy and its missing runtime requirements
RUN pip install fastapi fire openai outlines \
    partial_json_parser ray[default] shortuuid uvicorn \
    'numpy<2.0.0' \
    python-sat[aiger,approxmc,cryptosat,pblib] distance Faker --no-cache-dir
WORKDIR ${CODESPACE}
RUN if [ -n "${LMDEPLOY_VERSION}" ]; then \
        pip install lmdeploy==${LMDEPLOY_VERSION} --no-deps --no-cache-dir; \
    else \
        git clone -c https.proxy=$HTTPS_PROXY $(echo ${LMDEPLOY_URL} | cut -d '@' -f 1) && \
        cd ${CODESPACE}/lmdeploy && \
        git checkout $(echo ${LMDEPLOY_URL} | cut -d '@' -f 2) && \
        pip install . -v --no-deps --no-cache-dir; \
    fi

# setup sysctl
RUN echo "fs.file-max=100000" >> /etc/sysctl.conf
RUN sysctl -p
