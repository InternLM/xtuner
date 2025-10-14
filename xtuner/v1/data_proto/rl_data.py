from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field
from ray import ObjectRef
from typing_extensions import Annotated


# ====================================
# ====== DataFlow 数据流 ==============
# ====================================


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

    model_config = ConfigDict(extra="forbid")
    messages: Optional[List[Dict[str, Any]]] = None
    input_ids: Optional[List[int]] = None
    num_tokens: Optional[int] = None
    ability: Optional[str] = None
    reward_model: Optional[Dict[str, Any]] = None
    data_source: Optional[Dict[str, Any]] = None
    extra_info: Dict[str, Any] = dict()


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
    response: Optional[str] = None
    response_ids: Optional[List[int]] = None
    response_len: Optional[List[int]] = None
    num_return_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[List[float]] = None
    extra_info: Dict[str, Any] = dict()


class RLJudgerResponseItem(BaseModel):
    """Represents the data structure output from the judger.

    Attributes:
        uid (Optional[int]): A unique ID to identify which input the result corresponds to.
        reward (Dict[str, Any]): A dictionary of reward scores, e.g., {"judger_type": reward_score, "weighted_scores": score}.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid")
    uid: Optional[int] = None
    reward: Dict[str, Any] = Field(default_factory=lambda: {"score": 0.0, "val": 0.0})
    extra_info: Dict[str, Any] = dict()


class RLAgentDataItem(BaseModel):
    # todo: define agent output data structure
    model_config = ConfigDict(extra="forbid")
    extra_info: Dict[str, Any] = dict()


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
    extra_info: Dict[str, Any] = dict()


class RLExtraDataItem(BaseModel):
    """Reserved for data that does not belong to a specific stage of the
    dataflow.

    Attributes:
        retry_times (int): The number of times the data processing has been retried.
        extra_info (Dict[str, Any]): Additional user-defined information.
    """

    model_config = ConfigDict(extra="forbid")
    retry_times: int = 0
    extra_info: Dict[str, Any] = dict()


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
    stops: Annotated[List[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[List[int], Parameter(help="List of stop token IDs.")] = []
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
    messages: List[Dict[str, Any]]
    tools: List = Field(default_factory=list)
    tool_choice: str = "auto"
    sample_params: SampleParams = Field(default_factory=SampleParams)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


# ==============================================
# ====== ReplayBuffer 数据流  =====================
# ==============================================


@dataclass
class ReplayMeta:
    """ReplayMeta aggregates all versions of data related to a single prompt in
    the replay buffer.

    Attributes:
        env (str): Name or identifier of the environment.
        root_id (int): Identifier for grouping related prompts (e.g., for GRPO or multi-turn scenarios).
        action_id (int): Unique identifier for the prompt. If the prompt changes (such as in a multi-turn scenario), a new action_id is assigned.
        action_ref (ObjectRef): Ray object reference to the prompt data (corresponds to RLDatasetItem in RLDataFlowItem).
        observation_ids (List[int]): IDs for different responses to the same prompt. Each response has a unique observation_id.
        observation_refs (List[ObjectRef]): Ray object references to environment data for each observation (corresponds to RLEnvDataItem in RLDataFlowItem).
        observation_versions (List[int]): Version numbers for each observation, supporting async rollout.
        state (str): Overall state of the prompt (e.g., "paused" for partial rollout, or other rollout states).
        extra_info (Dict[str, Any]): Additional metadata or information.
    """

    env: str = ""
    root_id: int = 0
    action_id: int = 0  # same prompt share the same action_id
    action_ref: ObjectRef = None
    observation_ids: List[int] = field(default_factory=list)  # observation IDs for different versions
    observation_refs: List[ObjectRef] = field(default_factory=list)
    observation_versions: List[int] = field(default_factory=list)  # reserved for async rollout
    state: str = ""  # overall state, e.g., for partial rollout
    extra_info: Dict[str, Any] = field(default_factory=dict)


def mapping_dataitem_to_replaymeta(grouped_dataitem: List[RLDataFlowItem]) -> ReplayMeta:
    assert len(grouped_dataitem) > 0

    env_str = grouped_dataitem[0].uid.env
    root_id = grouped_dataitem[0].uid.root_id
    action_id = grouped_dataitem[0].uid.action_id
    data = grouped_dataitem[0].data
    observation_ids = []
    observation_refs = []
    observation_versions = []

    group_states = []
    for item in grouped_dataitem:
        version = item.uid.version
        observation_ids.append(item.uid.observation_id)
        observation_refs.append(ray.put(item.env))
        observation_versions.append(version)
        group_states.append(item.env.rollout.finish_reason)

    state_str = "paused" if "paused" in group_states else "returned"
    replay_meta = ReplayMeta(
        env=env_str,
        root_id=root_id,
        action_id=action_id,
        action_ref=ray.put(data),
        observation_ids=observation_ids,
        observation_refs=observation_refs,
        observation_versions=observation_versions,
        state=state_str,  # 指代一个prompt的整体状态，用于partial rollout
        extra_info={},
    )
    return replay_meta


def mapping_replaymeta_to_dataitem(replay_meta: ReplayMeta) -> List[RLDataFlowItem]:
    env_str = replay_meta.env
    root_id = replay_meta.root_id
    action_id = replay_meta.action_id
    data_ref = ray.get(replay_meta.action_ref)
    group_data_item = []
    for obs_id, obs_ref, version in zip(
        replay_meta.observation_ids, replay_meta.observation_refs, replay_meta.observation_versions
    ):
        env_data = ray.get(obs_ref)
        item = RLDataFlowItem(
            uid=RLUIDItem(env=env_str, root_id=root_id, action_id=action_id, observation_id=obs_id, version=version),
            data=data_ref,
            env=env_data,
            extra_info=RLExtraDataItem(),
        )
        group_data_item.append(item)
    return group_data_item
