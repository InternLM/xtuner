export HTTPS_PROXY=$HTTPS_PROXY

export BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
export XTUNER_COMMIT=$(git rev-parse HEAD)
export XTUNER_URL=https://github.com/InternLM/xtuner@${XTUNER_COMMIT}
export FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention@060c9188beec3a8b62b33a3bfa6d5d2d44975fab
export ADAPTIVE_GEMM_URL=https://github.com/InternLM/AdaptiveGEMM@f0314fa6b6c54da0aa98b3718025ab8e860fdff4
export GROUPED_GEMM_URL=https://github.com/InternLM/GroupedGEMM@3ae328844bb13679ef2ae4f704a8eb615cca7571
export TORCH_VERSION=${TORCH_VERSION:-"2.8.0"}
export LMDEPLOY_VERSION="0.10.0"
export LMDEPLOY_URL=https://github.com/InternLM/lmdeploy@11b9726de4cef1fca132c47a4bb98f4003c7ae27
export PPA_SOURCE="https://mirrors.aliyun.com"

image_name=${IMAGE_NAME:-"xtuner"}
image_tag=${IMAGE_TAG:-"${XTUNER_COMMIT}"}

docker build . \
  -t "$image_name:$image_tag" \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg TORCH_VERSION=$TORCH_VERSION\
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg PPA_SOURCE=$PPA_SOURCE \
  --build-arg ADAPTIVE_GEMM_URL=$ADAPTIVE_GEMM_URL \
  --build-arg FLASH_ATTN_URL=$FLASH_ATTN_URL \
  --build-arg GROUPED_GEMM_URL=$GROUPED_GEMM_URL \
  --build-arg XTUNER_URL=$XTUNER_URL \
  --build-arg XTUNER_COMMIT=$XTUNER_COMMIT \
  --build-arg LMDEPLOY_VERSION=$LMDEPLOY_VERSION \
  --build-arg LMDEPLOY_URL=${LMDEPLOY_URL}\
  --progress=plain \
  --label "BASE_IMAGE=$BASE_IMAGE" \
  --label "XTUNER_URL=$XTUNER_URL" \
  --label "XTUNER_COMMIT=$XTUNER_COMMIT" \
  --label "ADAPTIVE_GEMM_URL=$ADAPTIVE_GEMM_URL" \
  --label "FLASH_ATTN_URL=$FLASH_ATTN_URL" \
  --label "GROUPED_GEMM_URL=$GROUPED_GEMM_URL" \
  --label "LMDEPLOY_VERSION=$LMDEPLOY_VERSION"
