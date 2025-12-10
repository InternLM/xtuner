export HTTPS_PROXY=$HTTPS_PROXY

export BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
export XTUNER_COMMIT=$(git rev-parse HEAD)
export XTUNER_URL=https://github.com/InternLM/xtuner@${XTUNER_COMMIT}
export FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention@060c9188beec3a8b62b33a3bfa6d5d2d44975fab
export ADAPTIVE_GEMM_URL=https://github.com/InternLM/AdaptiveGEMM@374e68fb9ea7168c038cc92d89b438de2b3beb9a
export GROUPED_GEMM_URL=https://github.com/InternLM/GroupedGEMM@aa5ffb21cb626d6cd61d99fc42958127b0b99be7
export DEEP_EP_URL=https://github.com/deepseek-ai/DeepEP@9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee # v1.2.1
export DEEP_GEMM_URL=https://github.com/deepseek-ai/DeepGEMM@c9f8b34dcdacc20aa746b786f983492c51072870 # v2.1.1.post3

export TORCH_VERSION=${TORCH_VERSION:-"2.8.0"}
export LMDEPLOY_VERSION="0.11.0"
# export LMDEPLOY_URL=https://github.com/InternLM/lmdeploy@a9a24fbd8985374cb01ecb6021d1ce9668253c9c
export PPA_SOURCE="https://mirrors.aliyun.com"
export SGLANG_VERSION="0.5.3"

image_name=${IMAGE_NAME:-"xtuner"}
image_tag=${IMAGE_TAG:-"pt$(echo ${TORCH_VERSION} | awk -F. '{print $1$2}')_$(date +%Y%m%d)_${XTUNER_COMMIT:0:7}"}

docker build . \
  -t "$image_name:$image_tag" \
  --secret id=HTTPS_PROXY \
  --build-arg TORCH_VERSION=$TORCH_VERSION\
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg PPA_SOURCE=$PPA_SOURCE \
  --build-arg ADAPTIVE_GEMM_URL=$ADAPTIVE_GEMM_URL \
  --build-arg FLASH_ATTN_URL=$FLASH_ATTN_URL \
  --build-arg GROUPED_GEMM_URL=$GROUPED_GEMM_URL \
  --build-arg DEEP_EP_URL=$DEEP_EP_URL \
  --build-arg DEEP_GEMM_URL=$DEEP_GEMM_URL \
  --build-arg XTUNER_URL=$XTUNER_URL \
  --build-arg XTUNER_COMMIT=$XTUNER_COMMIT \
  --build-arg LMDEPLOY_VERSION=$LMDEPLOY_VERSION \
  --build-arg LMDEPLOY_URL=$LMDEPLOY_URL \
  --build-arg SGLANG_VERSION=$SGLANG_VERSION \
  --progress=plain \
  --label "BASE_IMAGE=$BASE_IMAGE" \
  --label "XTUNER_URL=${XTUNER_URL/@/\/tree\/}}" \
  --label "XTUNER_COMMIT=$XTUNER_COMMIT" \
  --label "ADAPTIVE_GEMM_URL=${ADAPTIVE_GEMM_URL/@/\/tree\/}" \
  --label "FLASH_ATTN_URL=${FLASH_ATTN_URL/@/\/tree\/}" \
  --label "GROUPED_GEMM_URL=${GROUPED_GEMM_URL/@/\/tree\/}" \
  --label "DEEP_EP_URL=${DEEP_EP_URL/@/\/tree\/}" \
  --label "DEEP_GEMM_URL=${DEEP_GEMM_URL/@/\/tree\/}" \
  --label "LMDEPLOY_VERSION=$LMDEPLOY_VERSION" \
  --label "SGLANG_VERSION=$SGLANG_VERSION"
