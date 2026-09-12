from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

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


class RolloutState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- 数据 ---
    # Samples generated from the same prompt share one group_id.
    group_id: int | None = None
    message: list[dict[str, Any]]  # dataset输出，需要在AgentLoop中转换成input_ids
    prompt_ids: list[int] | None = None  # 原始 prompt的token ids
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
    routed_experts: np.ndarray | RayObjectRef | list[RayObjectRef] | None = None
    finish_reason: str | None = None
    # response_mask: 记录response_ids中哪个token算loss, 与response_ids长度相同，每轮rollout在 agent_loop.generate 中覆盖写
    response_mask: list[int] | None = None
    # response_model_steps：记录 response_ids 中每个 token 来自哪个 model_step，与 response_ids 长度相同。
    response_model_steps: list[int] | None = None
    # 记录该样本过期程度，即最早生成 token 的模型版本与当前训练步数的差值，数值越大表示越过期。
    seq_staleness: int = 0

    input_ids: list[int] | None = None
    labels: list[int] | None = None

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


class RolloutMetadata(BaseModel):
    """Small rollout descriptor passed through the trainer pipeline.

    Large prepared training fields and agentic segment details remain in
    ``storage``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    rollout_id: int | None = None
    group_id: int | None = None
    session_id: int | None = None
    task_name: str | None = None
    status: Status = Status.INIT
    finish_reason: str | None = None
    reward: dict[str, Any] | None = None

    response_mask: list[int] | None = None
    token_staleness_mask: list[int] | None = None
    prompt_len: int | None = None
    response_len: int | None = None
    response_model_steps: list[int] | None = None
    total_len: int | None = None
    seq_staleness: int = 0
    tool_turns: int | None = None
    generate_time_s: float | None = None
    # ObjectRef to one complete prepared RolloutState in the Ray Object Store.
    storage: RayObjectRef | None = None

    @classmethod
    def from_rollout_state(
        cls,
        rollout_state: RolloutState,
        *,
        storage: RayObjectRef | None = None,
    ) -> RolloutMetadata:
        """Build metadata without copying large prepared training fields."""
        prompt_ids = rollout_state.prompt_ids
        if prompt_ids is None:
            prompt_ids = rollout_state.extra_fields.get("train_prompt_ids")

        response_ids = rollout_state.response_ids
        response_len = len(response_ids) if response_ids is not None else None
        response_model_steps = (
            list(rollout_state.response_model_steps) if rollout_state.response_model_steps is not None else None
        )
        input_ids = rollout_state.input_ids
        total_len = len(input_ids) if input_ids is not None else None
        if total_len is None and prompt_ids is not None and response_len is not None:
            total_len = len(prompt_ids) + max(response_len - 1, 0)

        extra_fields = rollout_state.extra_fields
        tool_turns = extra_fields.get("agent_tool_turns")
        if not isinstance(tool_turns, int):
            tool_turns = None
        generate_time_s = extra_fields.get("group_generate_time_s")
        if not isinstance(generate_time_s, (int, float)):
            generate_time_s = None
        return cls(
            rollout_id=rollout_state.rollout_id,
            group_id=rollout_state.group_id,
            session_id=rollout_state.session_id,
            task_name=rollout_state.task_name,
            status=rollout_state.status,
            finish_reason=rollout_state.finish_reason,
            reward=rollout_state.reward,
            response_mask=list(rollout_state.response_mask) if rollout_state.response_mask is not None else None,
            prompt_len=len(prompt_ids) if prompt_ids is not None else None,
            response_len=response_len,
            response_model_steps=response_model_steps,
            total_len=total_len,
            seq_staleness=rollout_state.seq_staleness,
            tool_turns=tool_turns,
            generate_time_s=float(generate_time_s) if generate_time_s is not None else None,
            storage=storage,
        )

    def to_rollout_state(self) -> RolloutState:
        """Restore and update the rollout state referenced by this metadata.

        ``RolloutMetadata`` is the authoritative owner of replay lifecycle
        fields after a state is inserted into the replay buffer. The complete
        state is fetched from the Object Store through ``storage`` and then
        updated locally; mutating it does not modify the Object Store
        snapshot.
        """
        if self.storage is None:
            raise ValueError(f"Rollout metadata {self.rollout_id} has no storage reference.")

        from ray import get as ray_get

        rollout_state = ray_get(self.storage)
        if not isinstance(rollout_state, RolloutState):
            raise TypeError("Rollout metadata storage must resolve to a RolloutState object.")

        rollout_state.response_model_steps = (
            [] if self.status == Status.EXPIRED else list(self.response_model_steps or [])
        )
        rollout_state.response_mask = list(self.response_mask) if self.response_mask is not None else None
        rollout_state.seq_staleness = self.seq_staleness
        rollout_state.status = self.status
        if self.status == Status.EXPIRED:
            reset_rollout_response(rollout_state)
        return rollout_state


def _release_object_refs(
    value: Any,
    *,
    clear: bool = False,
    best_effort: bool = False,
) -> Any:
    """Release ObjectRefs recursively and optionally clear them in-place.

    ``best_effort`` is used for routed experts because the TraceStore may have
    already released some of the same references.  The normal rollout-state
    path keeps the existing batch-release behavior.
    """
    from ray import ObjectRef

    from xtuner.v1.rl.utils.ray_utils import free_object_refs

    refs: list[Any] = []

    def visit(item: Any) -> Any:
        if isinstance(item, ObjectRef):
            refs.append(item)
            return None if clear else item
        if isinstance(item, BaseModel):
            for field_name in type(item).model_fields:
                field_value = getattr(item, field_name)
                cleared_value = visit(field_value)
                if clear and cleared_value is not field_value:
                    setattr(item, field_name, cleared_value)
            return item
        if isinstance(item, dict):
            for key, child in item.items():
                cleared_value = visit(child)
                if clear and cleared_value is not child:
                    item[key] = cleared_value
            return item
        if isinstance(item, list):
            for index, child in enumerate(item):
                cleared_value = visit(child)
                if clear and cleared_value is not child:
                    item[index] = cleared_value
            return item
        if isinstance(item, tuple):
            if clear:
                return tuple(visit(child) for child in item)
            for child in item:
                visit(child)
            return item
        if isinstance(item, set):
            if clear:
                return {visit(child) for child in item}
            for child in item:
                visit(child)
            return item
        return item

    result = visit(value)
    if not best_effort:
        free_object_refs(refs)
        return result

    seen: set[str] = set()
    for ref in refs:
        ref_key = ref.hex()
        if ref_key in seen:
            continue
        seen.add(ref_key)
        try:
            free_object_refs(ref)
        except Exception as exc:
            # A routed-expert ref may already have been released by the
            # TraceStore. Cleanup remains idempotent for the other refs.
            logger.debug("ObjectRef %s was already released: %s", ref_key, exc)
    return result


def discard_rollout_state(rollout_state: RolloutState) -> RolloutState:
    """Release heavy references and clear fields before dropping a rollout."""

    _release_object_refs(rollout_state, clear=True)

    for field_name, field in type(rollout_state).model_fields.items():
        if field.is_required():
            continue
        setattr(rollout_state, field_name, field.get_default(call_default_factory=True))
    return rollout_state


def discard_rollout_state_from_metadata(rollout_metadata: RolloutMetadata) -> RolloutMetadata:
    """Release a metadata-owned complete state and its Object Store ref.

    The complete state is fetched only for cleanup.  Routed-expert refs are
    detached before the generic state cleanup so they can be released
    independently and tolerantly: TraceStore may already have released the
    same refs when its session is discarded.
    """
    storage = rollout_metadata.storage
    if storage is None:
        return rollout_metadata

    # TODO: Replace this synchronous full-state fetch with a storage-side
    # deletion API when shared storage is available. Besides copying the state
    # header, ray.get can block the async cleanup caller while the object is
    # being resolved.
    from ray import get as ray_get

    rollout_state: RolloutState | None = None
    try:
        rollout_state = ray_get(storage)
        if not isinstance(rollout_state, RolloutState):
            raise TypeError("Rollout metadata storage must resolve to a RolloutState object.")

        routed_experts = rollout_state.routed_experts
        rollout_state.routed_experts = None
        _release_object_refs(routed_experts, best_effort=True)
        discard_rollout_state(rollout_state)
    finally:
        try:
            _release_object_refs(storage)
        finally:
            rollout_metadata.storage = None
            del rollout_state
    return rollout_metadata


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


def reset_rollout_metadata_response(rollout_metadata: RolloutMetadata) -> RolloutMetadata:
    """Clear response fields while preserving the prompt for rerollout.

    The complete state referenced by ``storage`` is reset and stored under a
    new ObjectRef; the old outer reference is then released. This keeps the
    replay-buffer metadata-only contract while preserving the original
    retryable-expiry cleanup semantics.
    """
    storage = rollout_metadata.storage
    if storage is not None:
        from ray import get as ray_get
        from ray import put as ray_put

        from xtuner.v1.rl.utils.ray_utils import free_object_refs

        stored_state = ray_get(storage)
        stored_state.status = rollout_metadata.status
        stored_state.seq_staleness = rollout_metadata.seq_staleness
        reset_rollout_response(stored_state)
        rollout_metadata.storage = ray_put(stored_state)
        free_object_refs(storage)

    rollout_metadata.finish_reason = None
    rollout_metadata.reward = None
    rollout_metadata.response_mask = None
    rollout_metadata.token_staleness_mask = None
    rollout_metadata.response_len = 0
    rollout_metadata.response_model_steps = []
    rollout_metadata.total_len = rollout_metadata.prompt_len
    return rollout_metadata


def reset_rollout_response(rollout_state: RolloutState) -> RolloutState:
    """Clear response fields while preserving the prompt for rerollout.

    ``RolloutState`` is reset in place. For ``RolloutMetadata``, the complete
    state referenced by ``storage`` is reset and stored under a new ObjectRef;
    the old outer reference is then released. This keeps the replay-buffer
    metadata-only contract while preserving the original retryable-expiry
    cleanup semantics.
    """
    routed_experts = getattr(rollout_state, "routed_experts", None)
    if routed_experts is not None:
        from ray import ObjectRef

        from xtuner.v1.rl.utils.ray_utils import free_object_refs

        if isinstance(routed_experts, (ObjectRef, list)):
            free_object_refs(routed_experts)
        rollout_state.routed_experts = None
    prompt_ids = getattr(rollout_state, "prompt_ids", None)
    rollout_state.tokens = list(prompt_ids) if prompt_ids is not None else None
    rollout_state.response = ""
    rollout_state.response_ids = []
    rollout_state.logprobs = []
    rollout_state.routed_experts = None
    rollout_state.finish_reason = None
    rollout_state.response_mask = None
    rollout_state.response_model_steps = []
    rollout_state.reward = None
    rollout_state.error_msg = None
    return rollout_state


def get_group_status(rollout_states: list[RolloutState | RolloutMetadata]) -> Status:
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
        rollout_states (list[RolloutState | RolloutMetadata]): A list of
            individual rollout states or metadata descriptors.

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


def update_sample_version(
    rollout_metadata: RolloutMetadata,
    model_step: int,
) -> RolloutMetadata:
    """Append token source model versions for newly generated response tokens.

    Replay-buffer lifecycle bookkeeping operates on metadata only. The
    prepared ``RolloutState`` in the Object Store is immutable and must not be
    fetched or modified here.
    """
    response_len = rollout_metadata.response_len or 0
    response_model_steps = list(rollout_metadata.response_model_steps or [])
    missing_response_steps = max(0, response_len - len(response_model_steps))
    if missing_response_steps:
        response_model_steps.extend([model_step] * missing_response_steps)
    rollout_metadata.response_model_steps = response_model_steps
    return rollout_metadata


def refresh_seq_staleness(
    group: list[RolloutMetadata],
    current_train_step: int,
) -> list[RolloutMetadata]:
    """Refresh sequence staleness on a metadata-only rollout group.

    The complete prepared rollout states remain immutable in the Object Store; only their lightweight metadata
    projections are updated.
    """
    for rollout_metadata in group:
        # response_model_steps 记录每个 response token 的模型版本；
        # 最早版本决定整条样本的滞后程度。
        response_model_steps = rollout_metadata.response_model_steps or []
        oldest_response_model_step = min(response_model_steps) if response_model_steps else None
        if oldest_response_model_step is not None:
            rollout_metadata.seq_staleness = calculate_seq_staleness(
                oldest_response_model_step,
                current_train_step,
            )
        else:
            rollout_metadata.seq_staleness = 0
    return group


def _calculate_effective_response_mask(
    rollout_metadata: RolloutMetadata,
    *,
    current_train_step: int,
    token_stale_threshold: int,
) -> list[int]:
    """Calculate the response mask after applying token staleness.

    Args:
        rollout_metadata (RolloutMetadata): Rollout metadata whose response
            token provenance and semantic mask are evaluated.
        current_train_step (int): Trainer step that will consume the sample.
        token_stale_threshold (int): Maximum token staleness, measured in trainer steps, allowed for training.

    Returns:
        list[int]: The semantic response mask intersected with the token-staleness mask.
    """
    response_model_steps = list(rollout_metadata.response_model_steps or [])

    # semantic mask: 在 agent_loop 中根据是否是 LLM 产生的 token 来 mask 的结果
    semantic_mask = rollout_metadata.response_mask
    if semantic_mask is None:
        # Every token reaching this helper has a model-step entry, so its
        # length is the authoritative response-mask length.
        semantic_mask = [1] * len(response_model_steps)

    # token_staleness_mask: 根据 token 的新鲜程度来 mask
    token_staleness_mask = [
        int(calculate_seq_staleness(response_model_step, current_train_step) < token_stale_threshold)
        for response_model_step in response_model_steps
    ]
    effective_mask = [
        semantic_mask_value * token_staleness_mask_value
        for semantic_mask_value, token_staleness_mask_value in zip(semantic_mask, token_staleness_mask)
    ]
    return effective_mask


def calculate_group_effective_response_masks(
    group: list[RolloutMetadata],
    *,
    current_train_step: int,
    token_stale_threshold: int | None,
) -> list[list[int] | None]:
    """Calculate effective response masks for a rollout group.

    Return the semantic response mask intersected with token freshness.
    ``response_ids`` is not needed here and remains in the Object Store.
    """
    if token_stale_threshold is None:
        return [None] * len(group)

    return [
        (
            None
            if not item.response_model_steps
            or (item.response_len is not None and item.response_len == 0)
            or (item.response_mask is not None and not any(item.response_mask))
            else _calculate_effective_response_mask(
                item,
                current_train_step=current_train_step,
                token_stale_threshold=token_stale_threshold,
            )
        )
        for item in group
    ]
