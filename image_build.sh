export HTTPS_PROXY=$HTTPS_PROXY

# NGC 26.03-py3 = CUDA 13.2.0.046 / cuDNN 9.20 / NCCL 2.29.7，匹配 pt121 实测栈（torch2.12.1+cu132）。
# 基镜像自带 torch 2.11a 会在下方被 pip 装的 torch==2.12.1(cu132) 覆盖；基镜像只需提供 cu13.2 工具链供各扩展编译。
export BASE_IMAGE=${BASE_IMAGE:-"nvcr.io/nvidia/pytorch:26.03-py3"}
export XTUNER_COMMIT=$(git rev-parse HEAD)
export XTUNER_URL=https://github.com/InternLM/xtuner@${XTUNER_COMMIT}
export FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention@8a8b2f10ddca88fd46db406c3e143e1ab0af977f # FA3 对齐 pt121 conda（flash_attn_3 8a8b2f, 2026-05-16）；FA2 2.x 在 Dockerfile 里不再编译/安装以匹配环境
export ADAPTIVE_GEMM_URL=https://github.com/InternLM/AdaptiveGEMM@10411e08b182e853d0f3ecec4c68bf90c90e309f # #7 fix: make k_grouped_gemm_dw deterministic for varlen MoE backward（含 dw 修复；Dockerfile 会补 cu13 头补丁）
export GROUPED_GEMM_URL=https://github.com/InternLM/GroupedGEMM@21c199dee72b0fb96025751e0dbc4ad35ef5a94f # #2 radix sort using torch current stream；对齐 pt121 conda（grouped_gemm 1.1.4）
export DEEP_EP_URL=https://github.com/deepseek-ai/DeepEP@9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee # v1.2.1
export DEEP_GEMM_URL=https://github.com/deepseek-ai/DeepGEMM@c9f8b34dcdacc20aa746b786f983492c51072870 # v2.1.1.post3
export CAUSAL_CONV1D_URL=https://github.com/Dao-AILab/causal-conv1d@da6dbaa9fd5a919967f14d3fd031da1288ad5025 # v1.6.0
export FLA_URL="${FLA_URL-https://github.com/HAOCHENYE/flash-linear-attention@72d2a8f3a06cefda6a3fc79b2fcbd0b41c34f238}" # 钉住 tmp-tensor-cache 分支到 pt121 conda 实装 commit（fla 0.4.2, 2026-05-22 "using queue for tensor cache"）

export TORCH_VERSION=${TORCH_VERSION:-"2.12.1"}
# export LMDEPLOY_VERSION="0.13.0dev"
export LMDEPLOY_URL=https://github.com/InternLM/lmdeploy@efe3b88607756a7ad9411b89627b5ac6ebaa540e
export PPA_SOURCE="https://mirrors.aliyun.com"
export DEFAULT_PYPI_URL=${DEFAULT_PYPI_URL:-"https://mirrors.aliyun.com/pypi/simple"}
# mirror https://download.pytorch.org/whl
export PYTORCH_WHEELS_URL=${PYTORCH_WHEELS_URL:-"https://download.pytorch.org/whl"}

image_name=${IMAGE_NAME:-"xtuner"}
image_tag=${IMAGE_TAG:-"pt$(echo ${TORCH_VERSION} | awk -F. '{print $1$2}')_$(date +%Y%m%d)_${XTUNER_COMMIT:0:7}"}

docker build . \
  -t "$image_name:$image_tag" \
  --secret id=HTTPS_PROXY \
  --secret id=NO_PROXY \
  --build-arg TORCH_VERSION=$TORCH_VERSION\
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg PPA_SOURCE="$PPA_SOURCE" \
  --build-arg DEFAULT_PYPI_URL="$DEFAULT_PYPI_URL" \
  --build-arg PYTORCH_WHEELS_URL="$PYTORCH_WHEELS_URL" \
  --build-arg ADAPTIVE_GEMM_URL="$ADAPTIVE_GEMM_URL" \
  --build-arg FLASH_ATTN_URL=$FLASH_ATTN_URL \
  --build-arg GROUPED_GEMM_URL=$GROUPED_GEMM_URL \
  --build-arg CAUSAL_CONV1D_URL=$CAUSAL_CONV1D_URL \
  --build-arg FLA_URL="$FLA_URL" \
  --build-arg DEEP_EP_URL=$DEEP_EP_URL \
  --build-arg DEEP_GEMM_URL=$DEEP_GEMM_URL \
  --build-arg XTUNER_URL=$XTUNER_URL \
  --build-arg XTUNER_COMMIT=$XTUNER_COMMIT \
  --build-arg LMDEPLOY_URL=$LMDEPLOY_URL \
  --progress=plain \
  --label "BASE_IMAGE=$BASE_IMAGE" \
  --label "XTUNER_URL=${XTUNER_URL/@/\/tree\/}" \
  --label "XTUNER_COMMIT=$XTUNER_COMMIT" \
  --label "ADAPTIVE_GEMM_URL=${ADAPTIVE_GEMM_URL/@/\/tree\/}" \
  --label "FLASH_ATTN_URL=${FLASH_ATTN_URL/@/\/tree\/}" \
  --label "GROUPED_GEMM_URL=${GROUPED_GEMM_URL/@/\/tree\/}" \
  --label "CAUSAL_CONV1D_URL=${CAUSAL_CONV1D_URL/@/\/tree\/}" \
  --label "FLA_URL=${FLA_URL/@/\/tree\/}" \
  --label "DEEP_EP_URL=${DEEP_EP_URL/@/\/tree\/}" \
  --label "DEEP_GEMM_URL=${DEEP_GEMM_URL/@/\/tree\/}" \
  --label "LMDEPLOY_URL=${LMDEPLOY_URL/@/\/tree\/}"
  # --label "LMDEPLOY_VERSION=$LMDEPLOY_VERSION"
