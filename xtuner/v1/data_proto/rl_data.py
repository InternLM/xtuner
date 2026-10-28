from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import NotRequired, Self, TypedDict

# ====================================
# ====== DataFlow 数据流 ==============
# ====================================
from xtuner.v1.data_proto.utils import calculate_seq_staleness
from xtuner.v1.utils.logger import get_logger


if TYPE_CHECKING:
    from ray import ObjectRef as RayObjectRef
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
    sampling_seed: int | None = None
    stream: bool = False
    return_logprob: bool = True
    top_logprobs: int = 1
    return_token_ids: bool = True
    include_stop_str_in_output: bool = True
    no_stop_trim: bool = True
    spaces_between_special_tokens: bool = False
    return_routed_experts: bool = True


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
    pixel_values: NotRequired[np.ndarray | RayObjectRef | None]
    image_grid_thw: NotRequired[np.ndarray | None]
    num_img_tokens: NotRequired[list[int]]
    pixel_values_videos: NotRequired[np.ndarray | RayObjectRef | None]
    video_grid_thw: NotRequired[np.ndarray | None]
    mm_token_type_ids: NotRequired[np.ndarray | None]


class RolloutFunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: Any = Field(default_factory=dict)
    raw_arguments_text: str | None = None


class RolloutToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: RolloutFunctionCall


class TeacherTargets(BaseModel):
    """Sampled-token or top-k Teacher outputs for one rollout response."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sampled", "topk"]
    tokens: list[int] | list[list[int]]
    logprobs: list[float] | list[list[float]]

    @model_validator(mode="after")
    def _check_shapes(self) -> Self:
        if self.kind == "sampled":
            if len(self.tokens) != len(self.logprobs):
                raise ValueError("Sampled-token Teacher tokens and logprobs must have equal lengths")
            return self

        token_rows = cast(list[list[int]], self.tokens)
        logprob_rows = cast(list[list[float]], self.logprobs)
        if len(token_rows) != len(logprob_rows):
            raise ValueError("Top-k Teacher tokens and logprobs must have equal row counts")
        widths = {len(row) for row in token_rows}
        if not widths or widths == {0} or len(widths) != 1:
            raise ValueError("Top-k Teacher token rows must have one consistent positive width")
        if any(len(token_row) != len(logprob_row) for token_row, logprob_row in zip(token_rows, logprob_rows)):
            raise ValueError("Top-k Teacher token and logprob rows must have equal widths")
        return self


@dataclass(frozen=True)
class TrainMeta:
    """Lightweight training summary owned and published by ``RolloutState``.

    ``num_tokens`` stays a ``RolloutState`` field (the single shifted training length);
    this record only carries prompt/response shape, supervision count, layout and
    presence flags. It is a plain frozen dataclass so Ray/debug snapshots can rebuild it
    with ``dataclasses.replace``.
    """

    prompt_length: int
    response_length: int
    supervised_tokens: int  # count(full labels[1:] != -100)
    position_layout: Literal["1d", "mrope_3d"]
    has_rollout_logprobs: bool
    has_routed_experts: bool
    segment_index: int | None


class RolloutState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- 数据 ---
    # Samples generated from the same prompt share one group_id.
    group_id: int | None = None
    message: list[dict[str, Any]]  # dataset输出，需要在AgentLoop中转换成input_ids
    # 原始 prompt 的 token ids；agentic trace 样本为 trace 首个 prompt 段（会话初始 prompt）。
    prompt_ids: list[int] | None = None
    num_tokens: int | None = None
    proxy_attn_flops: float | None = None
    data_source: dict[str, Any] | str | None = None
    mm_info: MultimodalInfo | None = None
    reward_model: dict[str, Any] | None = None

    # --- InferEngine 输入 ---
    # Used to route a multi-turn inference session to the same rollout worker.
    session_id: int | None = None
    tokens: list[int] | None = None  # 每一次推理引擎的实际输入
    tools: list | None = None
    tool_choice: str | dict[str, Any] | None = None
    sample_params: SampleParams = SampleParams()

    # --- InferEngine 输出 ---
    # 每一次推理引擎的实际输出, 在rollout worker中被覆盖写
    response: str | None = None
    tool_calls: list[RolloutToolCall] | None = None
    response_ids: list[int] | None = None
    logprobs: list[float] | None = None
    teacher_targets: TeacherTargets | None = None
    routed_experts: np.ndarray | RayObjectRef | list[RayObjectRef] | None = None
    finish_reason: str | None = None
    # response_model_steps：记录 response_ids 中每个 token 来自哪个 model_step，与 response_ids 长度相同。
    # 环境/工具等非模型生成位沿用所在 rollout cycle 的 model_step 占位（勿填 0，避免污染 min() 语义）。
    response_model_steps: list[int] | None = None
    # 记录该样本过期程度，即最早生成 token 的模型版本与当前训练步数的差值，数值越大表示越过期。
    seq_staleness: int = 0

    input_ids: list[int] | None = None
    labels: list[int] | None = None
    # 训练字段的唯一摘要；只有 set_training_fields / apply_response_mask / reset_response 维护它。
    train_meta: TrainMeta | None = None

    #  --- Judger 输出 ---
    reward: dict[str, Any] | None = None

    #  --- 状态 ---
    # Per-rollout identity. Different K-rollouts from the same prompt should have different rollout_id values.
    rollout_id: int | None = None
    # Deprecated compatibility field for downstream libraries.
    # TODO: remove after callers migrate to ``rollout_id``.
    uid: int | None = None
    task_name: str | None = None
    status: Status = Status.INIT
    error_msg: str | None = None
    position_ids: np.ndarray | None = None
    extra_fields: dict[str, Any] = {}

    def model_post_init(self, __context: Any) -> None:
        if self.rollout_id is None:
            self.rollout_id = self.uid
        else:
            self.uid = self.rollout_id

    def set_training_fields(self, *, input_ids: list[int], labels: list[int], logprobs: list[float] | None) -> None:
        """Validate and publish the full, unshifted training fields.

        Args:
            input_ids (list[int]): Full training sequence.
            labels (list[int]): Per-position targets, aligned with ``input_ids``.
            logprobs (list[float] | None): Full rollout logprobs, aligned with ``input_ids``.

        Returns:
            None
        """
        # 复制轻量 token lists，避免调用方随后修改别名污染已发布字段。
        ids = list(input_ids)
        target_labels = list(labels)
        full_logprobs = list(logprobs) if logprobs is not None else None
        meta = self._build_train_meta(input_ids=ids, labels=target_labels, logprobs=full_logprobs)
        # 验证成功后才赋值：训练字段、训练长度和摘要一起发布。
        self.input_ids = ids
        self.labels = target_labels
        self.logprobs = full_logprobs
        self.num_tokens = len(ids) - 1
        self.train_meta = meta

    def apply_response_mask(self, effective_mask: list[bool]) -> None:
        """Bake one effective response mask into ``labels``.

        The mask aligns with ``response_ids``; positions that are not kept become ignore
        tokens. Length, prompt length and existing semantic holes are preserved.

        Args:
            effective_mask (list[bool]): Per-response-token keep mask.

        Returns:
            None
        """
        if self.labels is None or self.train_meta is None:
            raise ValueError("Training fields must be published before applying a response mask")
        if len(effective_mask) != len(self.response_ids or []):
            raise ValueError("Response mask must align with response_ids")
        labels = list(self.labels)
        start = len(labels) - len(effective_mask)
        for index, keep in enumerate(effective_mask):
            if not keep:
                labels[start + index] = -100
        self.labels = labels
        self.train_meta = replace(self.train_meta, supervised_tokens=sum(label != -100 for label in labels[1:]))
        # 不重新推导 prompt_length：缺失显式 prompt 时，mask 后的非监督位计数会变大。

    def reset_response(self) -> None:
        """Invalidate the response and its training products before
        regeneration.

        Retry/expiry policy stays with the caller; this method only clears this sample.

        Returns:
            None
        """
        routed_experts = self.routed_experts
        if routed_experts is not None:
            from ray import ObjectRef

            from xtuner.v1.rl.utils.ray_utils import free_object_refs

            if isinstance(routed_experts, (ObjectRef, list)):
                free_object_refs(routed_experts)
        self.routed_experts = None
        prompt_ids = self.prompt_ids
        self.tokens = list(prompt_ids) if prompt_ids is not None else None
        self.response = ""
        self.response_ids = []
        self.logprobs = []
        self.teacher_targets = None
        self.finish_reason = None
        self.response_model_steps = []
        self.reward = None
        self.error_msg = None
        # input_ids/labels 是上一轮 canonicalize 的产物，必须一并清掉，否则重生成后
        # 基类 canonicalize 见 input_ids 非空会跳过重建，留下跨轮错位的训练字段。
        self.input_ids = None
        self.labels = None
        self.num_tokens = None
        self.train_meta = None
        # 保留 prompt / multimodal 上下文；不在此选择 status 或改 partial-rollout 策略。

    def validate_training(self) -> None:
        """Check this sample's training fields against its published summary.

        Read-only: it neither repairs data nor changes group status.

        Returns:
            None
        """
        if self.input_ids is None or self.labels is None or self.train_meta is None:
            raise ValueError("Training input_ids, labels and train_meta must be present")
        expected = self._build_train_meta(
            input_ids=self.input_ids,
            labels=self.labels,
            logprobs=self.logprobs,
            prompt_length=self.train_meta.prompt_length,
        )
        if self.num_tokens != len(self.input_ids) - 1 or self.train_meta != expected:
            raise ValueError("Training fields and metadata are inconsistent")
        if len(self.response_model_steps or []) != len(self.response_ids or []):
            raise ValueError(
                f"response_model_steps length={len(self.response_model_steps or [])}, "
                f"expected response_ids={len(self.response_ids or [])}"
            )

    def _build_train_meta(
        self,
        *,
        input_ids: list[int],
        labels: list[int],
        logprobs: list[float] | None,
        prompt_length: int | None = None,
    ) -> TrainMeta:
        if len(input_ids) < 2:
            raise ValueError("Training input_ids must contain at least 2 tokens")
        if len(input_ids) != len(labels):
            raise ValueError(f"labels length mismatch: {len(labels)} vs input_ids={len(input_ids)}")
        if logprobs is not None and len(logprobs) != len(input_ids):
            raise ValueError(f"logprobs length mismatch: {len(logprobs)} vs input_ids={len(input_ids)}")
        response_length = len(self.response_ids or [])
        if response_length > len(labels):
            raise ValueError("Response suffix exceeds full labels")
        kept = sum(label != -100 for label in labels[1:])
        if prompt_length is None:
            # prompt_len 统计原始输入 prompt 长度（VLM 的 prompt 在 train_prompt_ids）；
            # prompt_ids 缺失时退回 shifted_labels 的非监督位计数。
            prompt_ids = self.extra_fields.get("train_prompt_ids") or self.prompt_ids
            prompt_length = len(prompt_ids) if prompt_ids else len(input_ids) - 1 - kept
        if prompt_length < 0:
            raise ValueError("prompt_length must be nonnegative")
        position_ids = self.position_ids
        return TrainMeta(
            prompt_length=prompt_length,
            response_length=response_length,
            supervised_tokens=kept,
            position_layout="mrope_3d" if position_ids is not None and position_ids.ndim == 3 else "1d",
            has_rollout_logprobs=logprobs is not None,
            has_routed_experts=self.routed_experts is not None,
            segment_index=self.extra_fields.get("agent_trace_segment_index"),
        )


def free_rollout_state_refs(rollout_state: RolloutState) -> None:
    from ray import ObjectRef

    from xtuner.v1.rl.utils.ray_utils import free_object_refs

    refs: list[ObjectRef] = []

    def clear_object_refs(value: Any) -> Any:
        if isinstance(value, ObjectRef):
            refs.append(value)
            return None
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                field_value = getattr(value, field_name)
                cleared_value = clear_object_refs(field_value)
                if cleared_value is not field_value:
                    setattr(value, field_name, cleared_value)
            return value
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = clear_object_refs(item)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = clear_object_refs(item)
            return value
        if isinstance(value, tuple):
            return tuple(clear_object_refs(item) for item in value)
        if isinstance(value, set):
            return {clear_object_refs(item) for item in value}
        return value

    clear_object_refs(rollout_state)
    free_object_refs(refs)


def discard_rollout_state(rollout_state: RolloutState) -> RolloutState:
    """Release heavy references and clear fields before dropping a rollout."""

    free_rollout_state_refs(rollout_state)

    for field_name, field in type(rollout_state).model_fields.items():
        if field.is_required():
            continue
        setattr(rollout_state, field_name, field.get_default(call_default_factory=True))
    return rollout_state


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


def is_valid_for_training(group_data_items: list[RolloutState], logger) -> bool:
    """Check group status and the canonical training contract of every
    state."""
    is_abort = any(item.status == Status.ABORTED for item in group_data_items)
    is_filtered = any(item.status == Status.FILTERED for item in group_data_items)
    is_failed = any(item.status == Status.FAILED for item in group_data_items)
    if is_filtered or is_failed or is_abort:
        logger.warning(
            f"Invalid dataflow group found during training, rollout state filtered: {is_filtered}, failed: {is_failed}, aborted: {is_abort}."
        )
        return False
    for item in group_data_items:
        try:
            item.validate_training()
        except ValueError as exc:
            logger.warning(f"Invalid canonical training state for rollout_id={item.rollout_id}: {exc}")
            return False
    return True


def get_group_status(rollout_states: list[RolloutState]) -> Status:
    """Get the group status based on the individual rollout states.

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


def update_sample_version(rollout_state: RolloutState, model_step: int) -> RolloutState:
    """Append token source model version for newly generated response
    tokens."""
    response_len = len(rollout_state.response_ids or [])
    response_model_steps = list(rollout_state.response_model_steps or [])
    missing_response_steps = max(0, response_len - len(response_model_steps))
    if missing_response_steps:
        response_model_steps.extend([model_step] * missing_response_steps)
    rollout_state.response_model_steps = response_model_steps
    return rollout_state


def refresh_seq_staleness(group: list[RolloutState], current_train_step: int) -> list[RolloutState]:
    for rollout_state in group:
        # response_model_steps 记录每个 response token 的模型版本；
        # 最早版本决定整条样本的滞后程度。
        response_model_steps = rollout_state.response_model_steps or []
        if response_model_steps:
            rollout_state.seq_staleness = calculate_seq_staleness(min(response_model_steps), current_train_step)
        else:
            rollout_state.seq_staleness = 0
    return group


def _calculate_effective_response_mask(
    rollout_state: RolloutState,
    *,
    current_train_step: int,
    token_stale_threshold: int,
) -> list[bool]:
    """Calculate the effective response mask of one rollout state without
    mutating it.

    Every loop writes ``response_ids`` under one convention: the contiguous suffix of ``input_ids`` after
    the prompt (env-injected or tool tokens included), so response token ``j`` maps to labels position
    ``len(labels) - len(response_ids) + j``. The effective mask follows the reasoning-RL rule
    ``effective = semantic_mask * token_staleness_mask``: the semantic mask is recovered from
    ``labels != -100`` on the response region (semantic holes stay supervised-out regardless of staleness),
    and staleness is evaluated per token via ``response_model_steps``.

    Args:
        rollout_state (RolloutState): Rollout sample to inspect; left unmodified.
        current_train_step (int): Trainer step that will consume the sample.
        token_stale_threshold (int): Maximum token staleness, measured in trainer steps, allowed for training.

    Returns:
        list[int]: The effective mask aligned with ``response_ids``.
    """
    labels = cast(list[int], rollout_state.labels)
    response_ids = cast(list[int], rollout_state.response_ids)
    response_model_steps = cast(list[int], rollout_state.response_model_steps)

    # response_ids 是 input_ids 去掉 prompt 前缀的连续后缀，offset 即 prompt 长度。
    offset = len(labels) - len(response_ids)
    # semantic_mask: labels 中非 -100 的监督位；token_staleness_mask: 逐 token 新鲜度
    # （语义洞位的 staleness 值无意义，乘上 semantic_mask 后自然归零）。
    semantic_mask = [label != -100 for label in labels[offset:]]
    token_staleness_mask = [
        calculate_seq_staleness(response_model_step, current_train_step) < token_stale_threshold
        for response_model_step in response_model_steps
    ]
    return [
        semantic_mask_value and token_staleness_mask_value
        for semantic_mask_value, token_staleness_mask_value in zip(semantic_mask, token_staleness_mask)
    ]


def calculate_group_effective_response_masks(
    group: list[RolloutState],
    *,
    current_train_step: int,
    token_stale_threshold: int | None,
) -> list[list[bool] | None]:
    """Calculate one group's effective response masks under the token-staleness
    policy.

    The function is pure: ``group`` is inspected, not modified. Each returned mask is the effective
    response mask aligned with ``response_ids``; callers bake them into ``labels`` in ``take_train_batch``, clearing zero-mask supervised positions to ``-100``. The clearing only ever
    extends, so masks computed at growing trainer steps converge to the same labels. ``None`` means
    token staleness is disabled or does not apply to that state (no labels, or no ``response_ids`` to
    align with).

    Args:
        group (list[RolloutState]): Rollout group to inspect; left unmodified.
        current_train_step (int): Trainer step that will consume the group.
        token_stale_threshold (int | None): Maximum token staleness, measured in trainer
            steps, allowed for training. ``None`` disables token staleness.

    Returns:
        list[list[int] | None]: Per-state effective masks.
    """
    if token_stale_threshold is None:
        return [None] * len(group)

    masks: list[list[bool] | None] = []
    for item in group:
        if item.labels is None or not item.response_ids or len(item.labels) < len(item.response_ids):
            masks.append(None)
            continue
        masks.append(
            _calculate_effective_response_mask(
                item,
                current_train_step=current_train_step,
                token_stale_threshold=token_stale_threshold,
            )
        )
    return masks


def validate_rollout_group(group: list[RolloutState], *, task_name: str) -> None:
    """Check producer result structure, final status and existing version
    bounds.

    This function is read-only: it neither counts tokens nor fills model versions.
    """
    allowed = {Status.COMPLETED, Status.FAILED, Status.FILTERED, Status.ABORTED, Status.EXPIRED}
    for state in group:
        context = f"task={task_name!r}, group_id={state.group_id}, rollout_id={state.rollout_id}"
        if state.status not in allowed:
            raise ValueError(f"{context}: invalid generated status {state.status}")
        response_length = len(state.response_ids or [])
        version_length = len(state.response_model_steps or [])
        if version_length > response_length:
            raise ValueError(
                f"{context}: response_model_steps length={version_length}, expected at most response_ids={response_length}"
            )


def validate_train_group(group: list[RolloutState], *, task_name: str) -> None:
    """Check a completed group's fields immediately before training.

    Group-level conditions stay here; each sample's training-field consistency is
    owned by ``RolloutState.validate_training``. Raise ValueError so the training
    boundary can discard the entire group.

    Args:
        group (list[RolloutState]): Rollout group selected for training.
        task_name (str): Task name used for error attribution.

    Returns:
        None
    """
    if not group or any(not isinstance(state, RolloutState) or state.status != Status.COMPLETED for state in group):
        raise ValueError(f"task={task_name!r}: training requires a nonempty COMPLETED rollout group")
    for state in group:
        context = f"task={task_name!r}, group_id={state.group_id}, rollout_id={state.rollout_id}"
        try:
            state.validate_training()
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
