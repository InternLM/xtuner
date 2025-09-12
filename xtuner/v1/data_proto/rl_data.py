from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field
from ray import ObjectRef
from typing_extensions import Annotated


# ====================================
# ====== DataFlow 数据流 ==============
# ====================================


class RLUIDItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    env: Optional[str] = None
    root_id: Optional[int] = None
    action_id: Optional[int] = None
    version: Optional[int] = None


# dataset部分输出的数据结构
class RLDatasetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: Optional[List[Dict[str, Any]]] = None
    input_ids: Optional[List[int]] = None
    num_tokens: Optional[int] = None
    ability: Optional[str] = None
    reward_model: Optional[Dict[str, Any]] = None
    data_source: Optional[Dict[str, Any]] = None
    extra_info: Dict[str, Any] = dict()


# rollout部分输出的数据结构
class RLRolloutResponseItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: Optional[str] = None
    response_ids: Optional[List[int]] = None
    num_return_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    extra_info: Dict[str, Any] = dict()

# judger部分输出数据结构
class RLJudgerResponseItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: Optional[int] = None # 必须需要uid来标识是哪个输入数据的结果
    reward: Dict[str, Any] = dict() # example: {"judger_type": reward_score}
    extra_info: Dict[str, Any] = dict()


# agent部分输出数据结构
class RLAgentDataItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extra_info: Dict[str, Any] = dict()


# 包含env内部的数据结构，作为observation存储的字段
class RLEnvDataItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: Optional[int] = None
    rollout: RLRolloutResponseItem = RLRolloutResponseItem()
    judger: RLJudgerResponseItem = RLJudgerResponseItem()
    agent: RLAgentDataItem = RLAgentDataItem()
    extra_info: Dict[str, Any] = dict()


# 其他部分，不属于某个环节的数据。预留
class RLExtraDataItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry_times: int = 0
    extra_info: Dict[str, Any] = dict()


class RLDataFlowItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: RLUIDItem = RLUIDItem()
    data: RLDatasetItem = RLDatasetItem()
    env: RLEnvDataItem = RLEnvDataItem()
    extra_info: RLExtraDataItem = RLExtraDataItem()


def update_dataflow_item(group_data_items, target_key, target_value):
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
    logprobs: Annotated[int, Parameter(help="Number of log probabilities to return.")] = 0
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True
    do_sample: Annotated[bool, Parameter(help="Whether to sample or not.")] = True

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
    # replaymeta会包含一个prompt的所有版本的数据，在转为dataitem时会拆分
    env: str = ""
    root_id: int = 0  # designed for grpo
    action_id: int = 0  # same prompt share the same action_id
    action_ref: ObjectRef = None
    observation_ids: List[int] = field(default_factory=list)  # 一个prompt不同版本的observation
    observation_refs: List[ObjectRef] = field(default_factory=list)
    observation_versions: List[int] = field(default_factory=list)  # 为异步rollout预留
    state: str = ""  # 暂时还没定义，应该会包含rollout的states的定义
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
        observation_ids.append(uuid4().int)
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
            uid=RLUIDItem(env=env_str, root_id=root_id, action_id=action_id, version=version),
            data=data_ref,
            env=env_data,
            extra=RLExtraDataItem(),
        )
        group_data_item.append(item)
    return group_data_item
