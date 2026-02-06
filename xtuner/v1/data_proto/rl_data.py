from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict
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
    model_config = ConfigDict(extra="forbid")

    # --- 数据 ---
    message: list[dict[str, Any]]  # dataset输出，需要在AgentLoop中转换成input_ids
    prompt_ids: list[int]  # 原始 prompt的token ids
    data_source: dict[str, Any] | None = None
    mm_info: MultimodalInfo | None = None
    reward_model: dict[str, Any] | None = None
    message_uid: int | None = None  # 通过计算原始的message的哈希值得到的id，一组的数据为同一个prompt_id
    num_tokens: int | None = None  # 用于 cache 管理

    # --- InferEngine 输入 ---
    session_uid: int | None = None
    tokens: list[int]  # 每一次推理引擎的实际输入
    tools: list | None = None
    tool_choice: str | None = None
    sample_parms: SampleParams | None = None

    # --- InferEngine 输出 ---
    response: str | None = None
    response_ids: list[int] | None = None
    logprobs: list[float] | None = None
    routed_experts: list[int] | RayObjectRef | None = None  # type: ignore[valid-type]
    finish_reason: str | None = None

    #  --- Judger 输出 ---
    reward: float | list[float] | list[dict] | None = None

    #  --- 状态 ---
    state: Status = Status.INIT
    seq_staleness: int = 0  # 整条序列的staleness，一般为最大的token_staleness
    token_staleness: list[int] | None = None  # 每一个token的staleness，长度和tokens保持一致
    loss_mask: list[int] | None = None  # tokens + response_ids的长度
    extra_fields: dict[str, Any] = {}
