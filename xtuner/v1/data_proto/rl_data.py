import copy
from typing import Any, Dict, List, Literal, Optional, TypedDict

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from xtuner.v1.utils.logger import get_logger


logger = get_logger()

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
    version: int = 0


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
    async_response: Optional[List[Any]] = None
    response_ids: Optional[List[int]] = None
    async_response_ids: Optional[List[Any]] = None
    logprobs: Optional[List[float]] = None
    async_logprobs: Optional[List[Any]] = None
    num_return_tokens: int = 0
    finish_reason: Optional[str] = None  # "stop", "length", "abort", "failed", "skipped"
    extra_info: Dict[str, Any] = dict()
    state: Literal["init", "completed", "interrupted", "skipped", "failed"] = "init"

    # def update(self, other: "RLRolloutResponseItem") -> None:
    #     """Updates another RLRolloutResponseItem into this one for partial
    #     rollout."""
    #     if not isinstance(other, RLRolloutResponseItem):
    #         raise TypeError("Can only update with another RLRolloutResponseItem instance.")

    #     logger.info("call update RLRolloutResponseItem function")
    #     init_response_ids_len = 0
    #     if self.response_ids is not None:
    #         init_response_ids_len = len(self.response_ids)
    #         if other.response_ids is not None:
    #             self.response_ids.extend(other.response_ids)
    #         else:
    #             self.response_ids = self.response_ids
    #     else:
    #         self.response_ids = other.response_ids

    #     init_logprobs_len = 0
    #     if self.logprobs is not None:
    #         init_logprobs_len = len(self.logprobs)
    #         if other.logprobs is not None:
    #             self.logprobs.extend(other.logprobs)
    #         else:
    #             self.logprobs = self.logprobs
    #     else:
    #         self.logprobs = other.logprobs

    #     init_response_len = 0
    #     if self.response is not None:
    #         init_response_len = len(self.response)
    #         if other.response is not None and len(other.response) > 0:
    #             self.response += other.response
    #         else:
    #             self.response = self.response
    #     else:
    #         self.response = other.response

    #     logger.info(
    #         f"Updated response_ids from {init_response_ids_len} to {len(self.response_ids)}, logprobs from {init_logprobs_len} to {len(self.logprobs)}. response from {init_response_len} to {len(self.response)}."
    #     )
    #     self.num_return_tokens = len(self.response_ids)
    #     self.finish_reason = other.finish_reason
    #     self.extra_info.update(other.extra_info)
    #     self.state = other.state

    def update(self, other: "RLRolloutResponseItem") -> None:
        """Updates another RLRolloutResponseItem into this one for partial
        rollout."""
        if not isinstance(other, RLRolloutResponseItem):
            raise TypeError("Can only update with another RLRolloutResponseItem instance.")

        if self.response_ids is not None:
            init_response_ids = copy.deepcopy(self.response_ids)
            other_response_ids = copy.deepcopy(other.response_ids)
            init_async_response_ids = copy.deepcopy(self.async_response_ids)
            if other.response_ids is not None:
                self.async_response_ids.append(other_response_ids.copy())
                self.response_ids.extend(other_response_ids.copy())
                logger.debug(
                    f"update response_ids from {init_response_ids} with {other_response_ids} to {self.response_ids}, async_response_ids from {init_async_response_ids} to {self.async_response_ids}."
                )
        else:
            if other.response_ids is not None:
                other_response_ids = copy.deepcopy(other.response_ids)
                self.response_ids = other_response_ids
                self.async_response_ids = [other_response_ids.copy()]
            else:
                self.async_response_ids = []

        if self.logprobs is not None:
            other_logprobs = copy.deepcopy(other.logprobs)
            if other.logprobs is not None:
                self.async_logprobs.append(other_logprobs.copy())
                self.logprobs.extend(other_logprobs.copy())
        else:
            if other.logprobs is not None:
                other_logprobs = copy.deepcopy(other.logprobs)
                self.async_logprobs = [other_logprobs.copy()]
                self.logprobs = other_logprobs
            else:
                self.async_logprobs = []

        if self.response is not None:
            init_response = copy.deepcopy(self.response)
            other_response = copy.deepcopy(other.response)
            if other.response is not None:
                self.response += other_response
                self.async_response.append(other_response)
                logger.debug(
                    f"update response from {repr(init_response)} with {repr(other_response)} to {repr(self.response)}, async_response_ids: {self.async_response}."
                )
        else:
            if other.response is not None:
                self.response = other.response
                self.async_response = [other.response]
            else:
                self.async_response = []

        response_ids_lens = []
        for response_ids in self.async_response_ids:
            response_ids_lens.append(len(response_ids))
        logprobs_lens = []
        for logprobs in self.async_logprobs:
            logprobs_lens.append(len(logprobs))
        response_lens = []
        for response in self.async_response:
            response_lens.append(len(response))
        logger.debug(
            f"update response_ids lengths: {response_ids_lens}, logprobs lengths: {logprobs_lens}, response lengths: {response_lens}."
        )

        if self.response_ids is not None:
            assert sum(response_ids_lens) == len(self.response_ids), "response_ids length mismatch after update."
        if self.logprobs is not None:
            assert sum(logprobs_lens) == len(self.logprobs), "logprobs length mismatch after update."
        if self.response is not None:
            assert sum(response_lens) == len(self.response), "response length mismatch after update."
        self.num_return_tokens = sum(response_ids_lens)
        self.finish_reason = other.finish_reason
        self.extra_info.update(other.extra_info)
        self.state = other.state


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
    state: str = ""
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


def check_valid_dataflow_item(group_data_items: List[RLDataFlowItem]) -> bool:
    """Validates a group of RLDataFlowItem objects based on their state and
    data integrity.

    The validation follows a priority order for finish reasons:
    1. 'abort' or 'skipped': The group is considered valid for retry (returns True).
    2. 'failed' (rollout or judger): The group is invalid (returns False).
    3. Data Integrity Checks:
        - At least one of `response` or `response_ids` must be present.
        - If `response_ids` is present, `logprobs` must also be present and have the same length.

    Args:
        group_data_items: A list of RLDataFlowItem to be checked.

    Returns:
        A tuple containing:
        - bool: True if the group is valid or can be retried, False otherwise.
        - str: A message explaining the validation result.
    """
    for item in group_data_items:
        rollout_info = item.env.rollout
        response_valid = True if rollout_info.response is not None and len(rollout_info.response) > 0 else False
        ids_valid = True if rollout_info.response_ids is not None and len(rollout_info.response_ids) > 0 else False
        logprobs_valid = True if rollout_info.logprobs is not None and len(rollout_info.logprobs) > 0 else False
        if item.env.rollout.state in ["skipped", "failed"]:
            logger.info(f"Invalid dataflow item found: rollout state is {item.env.rollout.state}. UID: {item.uid}")
            return False
        if not response_valid and not ids_valid and item.env.rollout.state != "interrupted":
            logger.info(
                f"Invalid dataflow item found: no response or response_ids. UID:{item.uid.action_id} with rollout response {item.env.rollout}"
            )
            return False
        if ids_valid and logprobs_valid and len(rollout_info.logprobs) != len(rollout_info.response_ids):  # type: ignore[arg-type]
            logger.info(f"Invalid dataflow item found: logprobs and response_ids length mismatch. UID: {item.uid}")
            return False
    return True


def update_rollout_item(group_data_items, target_value):
    """Update a list of RLDataFlowItem objects by merging another
    RLRolloutResponseItem into each item's env.rollout attribute.

    Args:
        group_data_items (List[RLDataFlowItem]): List of data items to update.
        target_value (RLRolloutResponseItem): The rollout response item to merge into each data item.

    Returns:
        List[RLDataFlowItem]: The updated list of data items.

    Example:
        >>> # Suppose you want to update the rollout response for each item
        >>> items = [RLDataFlowItem(), RLDataFlowItem()]
        >>> rollout_response = RLRolloutResponseItem(response="new response", response_ids=[1,2,3])
        >>> update_rollout_item(items, rollout_response)
        # Now each item's env.rollout has been updated with the new response and response_ids
    """

    for idx, item in enumerate(group_data_items):
        item.env.rollout.update(target_value[idx])

    return group_data_items


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

        if keys[-1] == "rollout":
            existing_rollout_item = getattr(parent_obj, keys[-1])
            existing_rollout_item.update(target_value[i])
        else:
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
