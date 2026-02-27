from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, field_serializer
from typing_extensions import NotRequired, TypedDict

# ====================================
# ====== DataFlow 数据流 ==============
# ====================================
from xtuner.v1.utils.logger import get_logger


if TYPE_CHECKING:
    import ray

    RayObjectRef = ray.ObjectRef
else:
    RayObjectRef: TypeAlias = Any

logger = get_logger()


class SampleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = 1
    top_k: int = 0
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 2048
    stops: list[str] = []
    stop_token_ids: list[int] = []
    skip_special_tokens: bool = True
    stream: bool = False
    return_logprob: bool = True
    top_logprobs: int = 1
    return_token_ids: bool = True
    include_stop_str_in_output: bool = True
    no_stop_trim: bool = True
    spaces_between_special_tokens: bool = False
    return_routed_experts: bool = False


class Status(Enum):
    INIT = "init"
    COMPLETED = "completed"
    ABORTED = "aborted"
    EXPIRED = "expired"
    FAILED = "failed"
    FILTERED = "filtered"
    # 归档，这个状态还是要保留，用不用再说，用于表示这个数据已经用于一次训练了，但保留在数据库里以备查询
    ARCHIVED = "archived"


class MultimodalInfo(TypedDict):
    # 使用TypedDict给出pixel_values的类型提示
    pixel_values: NotRequired[torch.Tensor | RayObjectRef | None]  # type: ignore[valid-type]
    image_grid_thw: NotRequired[torch.Tensor]
    position_ids: NotRequired[torch.Tensor]


class RolloutState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- 数据 ---
    message_uid: int | None = None  # 通过计算原始的message的哈希值得到的id，一组的数据为同一个prompt_id
    message: list[dict[str, Any]]  # dataset输出，需要在AgentLoop中转换成input_ids
    prompt_ids: list[int] | None = None  # 原始 prompt的token ids
    data_source: dict[str, Any] | str | None = None
    mm_info: MultimodalInfo | None = None
    reward_model: dict[str, Any] | None = None
    num_tokens: int | None = None  # 用于 cache 管理

    # --- InferEngine 输入 ---
    session_uid: int | None = None
    tokens: list[int] | None = None  # 每一次推理引擎的实际输入
    tools: list | None = None
    tool_choice: str | None = None
    sample_params: SampleParams = SampleParams()

    # --- InferEngine 输出 ---
    response: str | None = None
    response_ids: list[int] | None = None
    logprobs: list[float] | None = None
    routed_experts: list[int] | RayObjectRef | None = None  # type: ignore[valid-type]
    finish_reason: str | None = None

    @field_serializer('routed_experts')
    def _serialize_routed_experts(self, value: list[int] | RayObjectRef | None) -> list[int] | None:
        """Dump 时跳过 ray.ObjectRef，序列化为 None，避免 PydanticSerializationError。"""
        if value is None:
            return None
        try:
            import ray
            if isinstance(value, ray.ObjectRef):
                return None
        except ImportError:
            pass
        if type(value).__name__ == 'ObjectRef' and 'ray' in getattr(
                type(value), '__module__', ''):
            return None
        return value  # list[int]

    #  --- Judger 输出 ---
    reward: float | list[float] | list[dict] | None = None

    #  --- 状态 ---
    task_name: str | None = None
    status: Status = Status.INIT
    error_msg: str | None = None
    seq_staleness: int = 0  # 整条序列的staleness，一般为最大的token_staleness
    token_staleness: list[int] | None = None  # 每一个token的staleness，长度和tokens保持一致
    loss_mask: list[int] | None = None  # tokens + response_ids的长度
    extra_fields: dict[str, Any] = {}


def update_status_from_finish_reason(finish_reason: str | None) -> Status:
    """Updates the internal status based on the inference engine's finish
    reason.

    State Transition Logic:
    -------------------------------------------------------------
    | Finish Reason (Input)          | Internal Status (Output) |
    | :----------------------------- | :----------------------- |
    | `stop`, `length`, `tool_calls` | `Status.COMPLETED`       |
    | `abort`                        | `Status.ABORTED`         |
    | `error` or `None`              | `Status.FAILED`          |
    | *Others*                       | *Raises ValueError*      |
    -------------------------------------------------------------

    Args:
        finish_reason (str | None): The raw finish reason string returned by
            the inference engine (e.g., vLLM, LMDeploy).

    Raises:
        ValueError: If the ``finish_reason`` is unknown and cannot be mapped.
    """
    if finish_reason is None:
        logger.error("finish_reason is None, setting status to FAILED.")
        return Status.FAILED

    reason = finish_reason.lower()
    if reason in ("stop", "length", "tool_calls"):
        return Status.COMPLETED
    elif reason == "abort":
        return Status.ABORTED
    elif reason == "error":
        logger.warning("finish_reason is 'error', setting status to FAILED.")
        return Status.FAILED
    else:
        logger.error(f"finish_reason '{finish_reason}' is unknown, setting status to FAILED.")
        return Status.FAILED


def update_group_status(rollout_states: list[RolloutState]) -> Status:
    """Updates the group status based on the individual rollout states.

    Group Status Logic:
    -------------------------------------------------------------
    | Individual Rollout States       | Group Status (Output)   |
    | :----------------------------- | :----------------------- |
    | All `Status.COMPLETED`          | `Status.COMPLETED`       |
    | Any `Status.FAILED`             | `Status.FAILED`          |
    | Any `Status.ABORTED`            | `Status.ABORTED`         |
    | Any `Status.EXPIRED`            | `Status.EXPIRED`         |
    | Any `Status.FILTERED`           | `Status.FILTERED`        |
    | *Others*                       | *Determined by priority*|
    -------------------------------------------------------------

    Priority Order (from highest to lowest):
    1. FAILED
    2. ABORTED
    3. EXPIRED
    4. FILTERED
    5. COMPLETED

    Args:
        rollout_states (list[RolloutState]): A list of individual rollout states.

    Returns:
        Status: The aggregated group status based on the individual states.
    """
    if all(state.status == Status.COMPLETED for state in rollout_states):
        return Status.COMPLETED
    elif any(state.status == Status.FAILED for state in rollout_states):
        return Status.FAILED
    elif any(state.status == Status.ABORTED for state in rollout_states):
        return Status.ABORTED
    elif any(state.status == Status.EXPIRED for state in rollout_states):
        return Status.EXPIRED
    elif any(state.status == Status.FILTERED for state in rollout_states):
        return Status.FILTERED
    else:
        # If there are other statuses, we can determine the group status based on a defined priority order.
        # For now, we will default to COMPLETED if none of the above conditions are met.
        return Status.COMPLETED
