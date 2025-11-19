from typing import Any, Dict, List, Optional, TypedDict

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field
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
    multimodal_train_info: Optional[Dict[str, Any]] = None


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
    num_return_tokens: Optional[int] = None
    finish_reason: Optional[str] = None  # "stop", "length", "abort", "failed", "skipped"
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


def check_dataflow_item(group_data_items):
    if not group_data_items or len(group_data_items) == 0:
        return False

    # 如果存在abort的状态，相当于跳过检查，下次会重新rollout
    is_abort = any(item.env.rollout.finish_reason == "abort" for item in group_data_items)
    is_skipped = any(item.env.rollout.finish_reason == "skipped" for item in group_data_items)
    if is_abort or is_skipped:
        return True

    no_failures = all(item.env.rollout.finish_reason != "failed" for item in group_data_items)
    if not no_failures:
        return False

    no_judger_failures = all(item.env.judger.extra_info.get("state", "") != "failed" for item in group_data_items)
    if not no_judger_failures:
        return False

    all_responses_valid = all(item.env.rollout.response for item in group_data_items)
    all_ids_valid = all(item.env.rollout.response_ids for item in group_data_items)

    return all_responses_valid or all_ids_valid


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
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]]
    tools: List = Field(default_factory=list)
    tool_choice: str = "auto"
    sample_params: SampleParams = Field(default_factory=SampleParams)
    extra_params: Dict[str, Any] = Field(default_factory=dict)
