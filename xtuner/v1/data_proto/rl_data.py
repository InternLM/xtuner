from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated, NotRequired

from xtuner.v1.utils import StrEnum

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


class RolloutState(StrEnum):
    """

    1. State Transitions from finish_reason and RolloutState:
    - A new task starts as `INIT`.
    - A successful generation (finish_reason 'stop' or 'length') becomes `COMPLETED`.
    - A generation stopped by the dataflow (e.g., for partial rollout) becomes `ABORTED`.
    - A generation that fails due to an inference server error becomes `FAILED`.
    - A generation skipped due to client errors or timeout errors (e.g., invalid input) becomes `SKIPPED`.
    - Data used for training is marked as `ARCHIVED`.
    - Old data (rollout for morn than expiration step) in the replay buffer is marked as `EXPIRED`.

    2. Dataflow Handling Based on RolloutState:
    - `INIT`: Data is in progress; no special handling.
    - `COMPLETED`: Data is valid for filtering, replay buffer insertion and training.
    - `ABORTED`: Data may be partially valid; It's valid for replay buffer insertion but not for filtering and training.
    - `FAILED`: Data is invalid; not used for filtering, replay buffer or training.
    - `SKIPPED`: Data is invalid; not used for filtering, replay buffer or training.
    - `ARCHIVED`: Data is stored for historical purposes; not used for training.
    - `EXPIRED`: Data is removed from the replay buffer; not used for training.
    """

    INIT = "init"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    SKIPPED = "skipped"

    @staticmethod
    def from_str(state_str: str) -> RolloutState:
        for state in RolloutState:
            if state.value == state_str:
                return state
        raise ValueError(f"Unknown ReplayState string: {state_str}")


class RLUIDItem(BaseModel):
    """A unique identifier for tracking data items within the dataflow.

    Attributes:
        env (str): The environment name.
        root_id (int): The root ID for grouping related data items.
        action_id (int): The ID for a specific action in prompt.
        observation_id (int): The ID for a specific observation in response.
        version (int): The version number of the data item.
    """

    model_config = ConfigDict(extra="forbid")
    env: str = ""
    root_id: int = -1
    action_id: int = -1
    observation_id: int = -1
    version: int = -1


class MultimodalTrainInfo(TypedDict):
    pixel_values: NotRequired[torch.Tensor | RayObjectRef | None]  # type: ignore[valid-type]
    image_grid_thw: NotRequired[torch.Tensor]
    position_ids: NotRequired[torch.Tensor]


class RLDatasetItem(BaseModel):
    """Represents the data structure output from the dataset.

    Attributes:
        messages (Optional[List[Dict[str, Any]]]): The message list for the prompt.
        input_ids (Optional[List[int]]): The tokenized input IDs.
        num_tokens (Optional[int]): The number of tokens in the input.
        ability (Optional[str]): The ability or category of the data.
        reward_model (Optional[Dict[str, Any]]): Data required by the reward model, like ground truth.
        data_source (Optional[Dict[str, Any]]): The source of the data, used for weighting rewards.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    messages: list[dict[str, Any]] | None = None
    input_ids: list[int] | None = None
    num_tokens: int | None = None
    ability: str | None = None
    reward_model: dict[str, Any] | None = None
    data_source: dict[str, Any] | None = None
    extra_info: dict[str, Any] = dict()
    multimodal_train_info: MultimodalTrainInfo | None = None


class RolloutExtraInfo(TypedDict):
    routed_experts: NotRequired[list[int] | RayObjectRef]  # type: ignore[valid-type]


class RLRolloutResponseItem(BaseModel):
    """Represents the data structure output from the rollout process.

    Attributes:
        response (Optional[str]): The generated text response from the model.
        response_ids (Optional[List[int]]): The token IDs of the generated response.
        num_return_tokens (Optional[int]): The number of tokens in the response.
        finish_reason (Optional[str]): The reason why the generation finished (e.g., 'stop', 'length').
        logprobs (Optional[List[float]]): The log probabilities of the generated tokens.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid")
    response: str | None = None
    response_ids: list[int] | None = None
    num_return_tokens: int | None = None
    finish_reason: str | None = None  # "stop", "length", "abort", "failed", "skipped"
    logprobs: list[float] | None = None
    extra_info: RolloutExtraInfo = Field(default_factory=dict)
    state: RolloutState = RolloutState.INIT


class RLJudgerResponseItem(BaseModel):
    """Represents the data structure output from the judger.

    Attributes:
        uid (Optional[int]): A unique ID to identify which input the result corresponds to.
        reward (Dict[str, Any]): A dictionary of reward scores, e.g., {"judger_type": reward_score, "weighted_scores": score}.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid")
    uid: int | None = None
    reward: dict[str, Any] = Field(default_factory=lambda: {"score": 0.0, "val": 0.0})
    extra_info: dict[str, Any] = dict()


class RLAgentDataItem(BaseModel):
    # todo: define agent output data structure
    model_config = ConfigDict(extra="forbid")
    extra_info: dict[str, Any] = dict()


class RLEnvDataItem(BaseModel):
    """Contains the internal data structures of the environment, stored as an
    observation.

    Attributes:
        rollout (RLRolloutResponseItem): Data from the rollout stage.
        judger (RLJudgerResponseItem): Data from the judger stage.
        agent (RLAgentDataItem): Data from the agent stage.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid")
    rollout: RLRolloutResponseItem = RLRolloutResponseItem()
    judger: RLJudgerResponseItem = RLJudgerResponseItem()
    agent: RLAgentDataItem = RLAgentDataItem()
    extra_info: dict[str, Any] = dict()


class RLExtraDataItem(BaseModel):
    """Reserved for data that does not belong to a specific stage of the
    dataflow.

    Attributes:
        retry_times (int): The number of times the data processing has been retried.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid")
    retry_times: int = 0
    extra_info: dict[str, Any] = dict()


class RLDataFlowItem(BaseModel):
    """The core data structure that flows through the dataflow and environment.

    It encapsulates all information related to a single data point, including its
    unique ID, the original data, environment outputs, and extra metadata.

    Attributes:
        uid (RLUIDItem): The unique identifier for the data item.
        data (RLDatasetItem): The original data from the dataset.
        env (RLEnvDataItem): The collected outputs from the environment stages.
        extra_info (RLExtraDataItem): Additional reserved information.
    """

    model_config = ConfigDict(extra="forbid")
    uid: RLUIDItem = RLUIDItem()
    data: RLDatasetItem = RLDatasetItem()
    env: RLEnvDataItem = RLEnvDataItem()
    extra_info: RLExtraDataItem = RLExtraDataItem()


def is_valid_for_replaybuffer(group_data_items: list[RLDataFlowItem]) -> bool:
    """Checks if a group of data items is valid for insertion into the replay
    buffer.

    Args:
        group_data_items: A list of RLDataFlowItem objects.

    Returns:
        True if the group is valid, False otherwise.

    NOTE: Why this check is needed:
    - For system fault tolerance, this check is performed at rollout / dataflow
    time, but we still do it here to ensure replay buffer data integrity.
    - 'skipped' or 'failed' states indicate that the rollout process did not
      complete successfully or was intentionally bypassed.
    - 'aborted' states may still contain useful data for the replay buffer,
      as the rollout was started but not finished.
    - 'completed' states are valid and should be included in the replay buffer.
    """
    is_skipped = any(item.env.rollout.state == RolloutState.SKIPPED for item in group_data_items)
    is_failed = any(item.env.rollout.state == RolloutState.FAILED for item in group_data_items)
    if is_skipped or is_failed:
        logger.warning(
            "Invalid dataflow group found during replay buffer insertion, skipped: {is_skipped}, failed: {is_failed}."
        )
        return False
    return True


def is_valid_for_training(group_data_items: list[RLDataFlowItem]) -> bool:
    """Checks if a group of data items is valid for a training step.

    Args:
        group_data_items: A list of RLDataFlowItem objects.

    Returns:
        True if the group is valid, False otherwise.

    NOTE: Why this check is needed:
    - For system fault tolerance, this check is performed at rollout / dataflow
      time, but we still do it here to ensure training data integrity.
    - 'skipped'/'failed': These items are fundamentally broken or incomplete and
      should not be used for training.
    - 'aborted': These items represent rollouts that were stopped
      prematurely. Using such partial data could lead the model to learn
      undesirable behaviors (e.g., stopping generation too early).
    - Empty response/response_ids: The model's generated response is the core
      of the training data for RL algorithms like PPO. If the response is
      missing, there is nothing to compute rewards on or to train the model with.
    """
    is_abort = any(item.env.rollout.state == RolloutState.ABORTED for item in group_data_items)
    is_skipped = any(item.env.rollout.state == RolloutState.SKIPPED for item in group_data_items)
    is_failed = any(item.env.rollout.state == RolloutState.FAILED for item in group_data_items)
    if is_skipped or is_failed or is_abort:
        logger.warning(
            "Invalid dataflow group found during training, rollout state skipped: {is_skipped}, failed: {is_failed}, aborted: {is_abort}."
        )
        return False
    for item in group_data_items:
        rollout_info = item.env.rollout
        response_valid = True if rollout_info.response is not None and len(rollout_info.response) > 0 else False
        ids_valid = True if rollout_info.response_ids is not None and len(rollout_info.response_ids) > 0 else False
        if not ids_valid and not response_valid:
            # NOTE: `response_ids` is the critical field for token-in-token-out mode, so we ensure it's not empty.
            logger.error(
                "Invalid dataflow item found during training: no response_ids and no response and skip this item."
            )
            return False
        elif not ids_valid:
            logger.warning(
                "Dataflow item has no response_ids during training, but still use it based on response text."
            )
        elif not response_valid:
            logger.warning(
                "Dataflow item has no response text during training, but still use it based on response_ids."
            )
    return True


def update_dataflow_item(group_data_items, target_key, target_value):
    """Update a list of RLDataFlowItem objects by setting a nested attribute
    for each item.

    Args:
        group_data_items (List[RLDataFlowItem]): List of data items to update.
        target_key (str): Dot-separated path to the attribute to update (e.g., 'env.rollout.response').
        target_value (List[Any]): List of values to set, one for each data item.

    Returns:
        List[RLDataFlowItem]: The updated list of data items.

    Example:
        >>> # Suppose you want to update the 'response' field in env.rollout for each item
        >>> items = [RLDataFlowItem(), RLDataFlowItem()]
        >>> responses = ["hello", "world"]
        >>> update_dataflow_item(items, "env.rollout.response", responses)
        # Now items[0].env.rollout.response == "hello", items[1].env.rollout.response == "world"
    """

    group_length = len(group_data_items)
    assert group_length == len(target_value)

    keys = target_key.split(".")

    for i in range(group_length):
        parent_obj = group_data_items[i]
        for key in keys[:-1]:
            parent_obj = getattr(parent_obj, key)
        setattr(parent_obj, keys[-1], target_value[i])

    return group_data_items


# ==============================================
# ====== Rollout API Server 数据流 ==============
# ==============================================


class SampleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: Annotated[int, Parameter(help="Number of samples to generate.")] = 1
    top_k: Annotated[
        int, Parameter(help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    ] = 0
    top_p: Annotated[float, Parameter(help="The cumulative probability for nucleus sampling.")] = 1.0
    temperature: Annotated[float, Parameter(help="The value used to module the next token probabilities.")] = 1.0
    repetition_penalty: Annotated[float, Parameter(help="The parameter for repetition penalty.")] = 1.0
    presence_penalty: Annotated[float, Parameter(help="The parameter for presence penalty.")] = 0.0
    frequency_penalty: Annotated[float, Parameter(help="The parameter for frequency penalty.")] = 0.0
    min_tokens: Annotated[int, Parameter(help="Minimum number of tokens to generate.")] = 0
    max_tokens: Annotated[int, Parameter(help="Maximum number of tokens to generate.")] = 2048
    stops: Annotated[list[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[list[int], Parameter(help="List of stop token IDs.")] = []
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True


class RolloutExtraParams(TypedDict):
    stream: bool
    return_logprob: bool
    top_logprobs: int
    return_token_ids: bool
    include_stop_str_in_output: bool
    no_stop_trim: bool
    skip_special_tokens: bool
    spaces_between_special_tokens: bool


# 说明： 这里没定义API server情况数据格式，因为直接使用openai server的格式
class RLRolloutRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[dict[str, Any]]
    tools: list = Field(default_factory=list)
    tool_choice: str = "auto"
    sample_params: SampleParams = Field(default_factory=SampleParams)
    extra_params: dict[str, Any] = Field(default_factory=dict)
