"""Convert identified runtime image failures into discardable rollout
inputs."""

import copy
import os

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import get_image_path
from xtuner.v1.utils import get_logger


def failed_image_rollout(
    error: Exception,
    *,
    item: dict,
    messages: list[dict],
    image_paths: list[str],
    media_root: str,
    data_judger_mapping: dict | None,
) -> RolloutState:
    # 仅接受本样本图片路径对应的错误；processor/config 等其它文件缺失仍向外抛出。
    paths = [get_image_path(path, media_root) for path in image_paths if not path.startswith("data:")]
    filename = getattr(error, "filename", None)
    if filename is not None:
        matches_image = any(os.path.abspath(filename) == os.path.abspath(path) for path in paths)
    else:
        # Pillow 的 UnidentifiedImageError 不带 filename，但诊断包含 repr(image_path)。
        matches_image = any(repr(path) in str(error) for path in paths)
    if not matches_image:
        raise error
    source = item["data_source"]
    mapped_source = data_judger_mapping.get(source) if data_judger_mapping is not None else {source: 1.0}
    get_logger().warning(f"Discard RL input from {source!r}: {type(error).__name__}: {error}")
    # 不伪造 prompt IDs，也不复用可能只处理了一半的像素或训练字段。
    return RolloutState(
        message=copy.deepcopy(messages),
        data_source=mapped_source,
        status=Status.FAILED,
        error_msg=f"{type(error).__name__}: {error}",
    )
