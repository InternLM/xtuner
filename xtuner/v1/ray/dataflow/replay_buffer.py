import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy
import ray
import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field
from ray import ObjectRef
from typing_extensions import Annotated

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import (
    MultimodalTrainInfo,
    RLDataFlowItem,
    RLDatasetItem,
    RLEnvDataItem,
    RLExtraDataItem,
    RLUIDItem,
    RolloutState,
    is_valid_for_replaybuffer,
)
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
logger = get_logger()


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
    observation_ids: List[int] = field(default_factory=list)
    observation_refs: List[ObjectRef] = field(default_factory=list)
    observation_versions: List[int] = field(default_factory=list)  # 目前发数据为按组下发，暂时用不到
    state: RolloutState = RolloutState.INIT
    version: int = 0  # version for partial rollout
    extra_info: Dict[str, Any] = field(default_factory=dict)


def determine_group_state(group_data_items: List[RLDataFlowItem]) -> RolloutState:
    """Determines the processing strategy for a group of rollout samples based
    on their state."""
    # TODO(@duanyanhui): remove this function when send one request instead of group requests.
    if not group_data_items:
        return RolloutState.SKIPPED
    group_states = {item.env.rollout.state for item in group_data_items}
    if RolloutState.SKIPPED in group_states:
        return RolloutState.SKIPPED
    elif RolloutState.FAILED in group_states:
        return RolloutState.FAILED
    elif RolloutState.ABORTED in group_states:
        return RolloutState.ABORTED
    elif all(state == RolloutState.COMPLETED for state in group_states):
        return RolloutState.COMPLETED
    else:
        raise ValueError(f"Unknown group states: {group_states}")


def mapping_dataitem_to_replaymeta(grouped_dataitem: List[RLDataFlowItem]) -> ReplayMeta:
    assert len(grouped_dataitem) > 0

    env_str = grouped_dataitem[0].uid.env
    root_id = grouped_dataitem[0].uid.root_id
    action_id = grouped_dataitem[0].uid.action_id
    # !!! 注意：这里放的是第一个dataitem的data，因为一组数据的data是一样的 !!!
    data = grouped_dataitem[0].data
    # 现在是按组发送，那么一组里的dataitem的version是一样的，如果一组中的数据在某次rollout step中没有生成的数据，version也还是会+1
    group_version = grouped_dataitem[0].uid.version
    observation_ids = []
    observation_refs = []

    for item in grouped_dataitem:
        observation_ids.append(item.uid.observation_id)
        observation_refs.append(ray.put(item.env))

    group_state = determine_group_state(grouped_dataitem)
    logger.debug(
        f"Mapping data items to ReplayMeta {action_id} with group_state: {group_state}, group_version: {group_version}"
    )

    replay_meta = ReplayMeta(
        env=env_str,
        root_id=root_id,
        action_id=action_id,
        action_ref=ray.put(data),
        observation_ids=observation_ids,
        observation_refs=observation_refs,
        state=group_state,
        version=group_version,
        extra_info={},
    )
    return replay_meta


def mapping_replaymeta_to_dataitem(replay_meta: ReplayMeta) -> List[RLDataFlowItem]:
    env_str = replay_meta.env
    root_id = replay_meta.root_id
    action_id = replay_meta.action_id
    data_ref = ray.get(replay_meta.action_ref)
    group_data_item = []
    for obs_id, obs_ref in zip(replay_meta.observation_ids, replay_meta.observation_refs):
        env_data = ray.get(obs_ref)
        # NOTE: This mapping function used by both dump and get. ObjectRefs are kept during dump (for training continuity)
        # but released during get (via del replaymeta) when no longer needed. So we do not free them manually here.
        # ray._private.internal_api.free(obs_ref)

        item = RLDataFlowItem(
            uid=RLUIDItem(
                env=env_str, root_id=root_id, action_id=action_id, observation_id=obs_id, version=replay_meta.version
            ),
            data=data_ref,
            env=env_data,
            extra_info=RLExtraDataItem(),
        )
        group_data_item.append(item)
    return group_data_item


class ReplayBufferConfig(BaseModel):
    """Replay buffer configuration for XTuner.

    This class defines configuration parameters for the replay buffer system in XTuner,
    managing dataset handling, data loading, text processing, and post-processing
    operations for reinforcement learning experience replay.

    Args:
        dataset_cfg (List): Configuration for datasets used to sample initial prompts.
        dataloader_cfg (DataloaderConfig): Configuration for the PyTorch DataLoader
            that iterates over the dataset.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer for
            processing text data, including support for partial rollouts.
        postprocessor_func (Optional[Callable]): Optional function to filter or
            modify data groups after generation. Defaults to None.
        replay_ratio (float): Ratio of samples to replay from the buffer versus
            sampling new data. Defaults to 0.
        replay_weights (dict): Weights for different states in the replay buffer
            to control sampling priorities. Defaults to empty dict.

    **Examples:**

    Example configuration for ReplayBuffer with GSM8K dataset config and base dataloader config::

        from transformers import AutoTokenizer

        config = ReplayBufferConfig(
            dataset_cfg=[{
                "dataset": DatasetConfig(name="gsm8k", anno_path="path/to/data"),
                "tokenize_fn": RLTokenizeFnConfig(max_length=512)
            }],
            dataloader_cfg=DataloaderConfig(collator='fake_collator'),
            tokenizer=AutoTokenizer.from_pretrained("model_path"),
            postprocessor_func=None,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataset_cfg: Annotated[List, Parameter(help="The dataset object to sample initial prompts from.")]

    dataloader_cfg: Annotated[
        Optional[DataloaderConfig], Parameter(help="The PyTorch DataLoader for iterating over the dataset.")
    ] = None

    tokenizer: Annotated[
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str],
        Field(exclude=True),
        Parameter(help="The tokenizer for processing text data, e.g., for partial rollouts."),
    ]
    postprocessor_func: Annotated[
        Optional[Callable],
        Field(exclude=True),
        Parameter(help="An optional function to filter or modify data groups after they are generated."),
    ] = None
    replay_ratio: Annotated[
        float,
        Parameter(help="Ratio of samples to replay from the buffer."),
    ] = 0
    replay_weights: Annotated[
        dict,
        Parameter(help="Weights for different states in the replay buffer."),
    ] = {}
    # async rollout related configs, assigned from dataflow cfg
    enable_partial_rollout: Annotated[
        bool,
        Parameter(help="Whether to enable partial rollout for asynchronous data generation."),
    ] = False
    tail_batch_candidate_steps: Annotated[
        int,
        Parameter(
            help="Number of rollout steps after which a sample becomes a candidate for the tail batch. Set to 0 to disable."
        ),
    ] = 0
    tail_batch_trigger_size: Annotated[
        Optional[int],
        Parameter(
            help="Number of candidate samples needed in the queue to trigger a tail batch operation. Set to 0 to disable."
        ),
    ] = None
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"


class DatasetSampler:
    """Sampler for drawing new prompts from the configured dataset.

    This class is responsible for building a dataloader from the provided dataset configurations and sampling fresh
    data prompts upon request.
    """

    def __init__(self, dataset_cfg, dataloader_cfg, tokenizer):
        """Initializes the DatasetSampler.

        Args:
            dataset_cfg (List): Configuration for the datasets to sample from.
            dataloader_cfg (Optional[DataloaderConfig]): Configuration for the
                PyTorch DataLoader.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str]):
                The tokenizer for processing text data. Can be a path or an object.
        """
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        if dataloader_cfg is not None:
            self.dataloader_cfg = dataloader_cfg
            self.dataloader_cfg.dataset_config_list = dataset_cfg
        else:
            self.dataloader_cfg = DataloaderConfig(
                dataset_config_list=dataset_cfg,
                collator="fake_collator",
                pack_level="none",
            )
        self.dataloader = self.dataloader_cfg.build(
            tokenizer=self.tokenizer, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        self.dataloader_iter = iter(self.dataloader)
        self.sample_count = 0
        self.cur_epoch = 0
        self.reduced_consumed_samples = 0
        self.logger = get_logger()

    def sample(self, env: str, prompt_repeat_k: int) -> List[RLDataFlowItem]:
        """Samples a new prompt from the dataset and prepares it as a group.

        This method fetches the next item from the dataloader, assigns new
        unique IDs (root_id, action_id, observation_id), and formats it into
        a list of RLDataFlowItem objects, repeated `prompt_repeat_k` times.

        Args:
            env (str): The environment name to be associated with the new samples.
            prompt_repeat_k (int): The number of times to repeat the sampled
                prompt in the returned group.

        Returns:
            List[RLDataFlowItem]: A list of newly created data items for a rollout.
        """
        root_id = uuid4().int
        action_id = uuid4().int
        group_data_item: List[RLDataFlowItem] = [RLDataFlowItem() for _ in range(prompt_repeat_k)]
        try:
            data = next(self.dataloader_iter)[0]
        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]
        self.reduced_consumed_samples += 1

        multimodal_train_info = data.pop("multimodal_train_info", {})
        if "pixel_values" in multimodal_train_info:
            multimodal_train_info["pixel_values"] = ray.put(multimodal_train_info["pixel_values"])
            data["multimodal_train_info"] = multimodal_train_info

        for data_item in group_data_item:
            data_item.uid = RLUIDItem(
                env=env,
                root_id=root_id,
                action_id=action_id,
                observation_id=uuid4().int,
            )
            data_item.data = RLDatasetItem(**data)
            data_item.extra_info = RLExtraDataItem(retry_times=0)
        self.logger.debug(f"Sampling new prompt with action_id: {action_id} in env: {env}")
        return group_data_item


class ReplayBufferStorage:
    """Handles the storage of experiences for the replay buffer."""

    def __init__(self, replay_buffer_cfg):
        """Initializes the data structures for storing replay data."""
        self.enable_partial_rollout = replay_buffer_cfg.enable_partial_rollout
        self.tail_batch_candidate_steps = replay_buffer_cfg.tail_batch_candidate_steps
        self.tail_batch_trigger_size = replay_buffer_cfg.tail_batch_trigger_size

        self._completed_actions: Dict[int, List[int]] = defaultdict(list)
        self._aborted_actions: Dict[int, List[int]] = defaultdict(list)
        self._expired_actions: List[int] = []
        self._actions: Dict[int, ReplayMeta] = {}
        self._root2actions: Dict[int, List[int]] = {}
        self._observations: Dict[int, ReplayMeta] = {}
        self._observations2states: Dict[int, str] = {}
        self._states: Dict[str, List[int]] = defaultdict(list)
        self._action2observations: Dict[int, List[int]] = defaultdict(list)
        self._multimodal_train_infos: Dict[int, Dict[str, Any]] = {}
        self.logger = get_logger(log_dir=replay_buffer_cfg.worker_log_dir, tag="ReplayBuffer")
        self.sample_from_aborted_count = 0
        self.sample_from_expired_count = 0

    def add(self, grouped_dataitem: List[RLDataFlowItem]):
        """Adds a group of data items to the storage.

        Args:
            grouped_dataitem (List[RLDataFlowItem]): A list of data items
                belonging to the same group.
        """
        if (
            grouped_dataitem is None
            or len(grouped_dataitem) == 0
            or is_valid_for_replaybuffer(grouped_dataitem) is False
        ):
            return

        replay_meta = mapping_dataitem_to_replaymeta(grouped_dataitem)
        root_id = replay_meta.root_id
        action_id = replay_meta.action_id

        # 1. 更新版本
        if root_id in self._root2actions:
            # TODO: 考虑到非共卡的情况，version是否更新需要根据是否update_weights来判断
            replay_meta.version += 1
            self._root2actions[root_id].append(action_id)
            self.logger.debug(
                f"Existing root_id: {root_id} with action_id {action_id} found. Incrementing version to {replay_meta.version}."
            )
        else:
            self._root2actions[root_id] = [action_id]
        self._actions[action_id] = replay_meta

        # 2. 根据rollout_state更新completed/aborted/expired相关映射
        self._check_rollout_state_and_insert(replay_meta)

        # 3. 更新observations相关映射
        for observation_id in replay_meta.observation_ids:
            self._observations[observation_id] = replay_meta
            self._observations2states[observation_id] = replay_meta.state
            if observation_id not in self._action2observations[action_id]:
                self._action2observations[action_id].append(observation_id)
            if observation_id not in self._states[replay_meta.state]:
                self._states[replay_meta.state].append(observation_id)

    def get(self, global_batch_size: int) -> Tuple[List[List[RLDataFlowItem]], List[MultimodalTrainInfo | None]]:
        """Retrieves a batch of finished sample groups from the buffer.

        Args:
            global_batch_size (int): The number of sample groups to retrieve.

        Raises:
            ValueError: If there are not enough finished samples in the buffer
                to meet the `global_batch_size`.

        Returns:
            List[List[RLDataFlowItem]]: A list of sample groups. Each inner
            list contains a group of data items that were generated from the
            same initial prompt, repeated `repeat_prompt_k` times.
        """
        samples = []
        multimodal_train_infos = []
        target_batch_size = min(global_batch_size, self.completed_samples_count)
        self.logger.info(f"Retrieving {target_batch_size} completed samples from the replay buffer.")
        task_time = []
        for _ in range(target_batch_size):
            task_start_time = time.perf_counter()
            action_id = self._pop_highest_version_action(self._completed_actions)
            if action_id is None:
                self.logger.warning("Get action_id None from completed_actions and skip this iteration.")
                continue
            replay_meta = self._actions.pop(action_id)
            group_samples = mapping_replaymeta_to_dataitem(replay_meta)
            # 将这条数据彻底清除,不用再记录root_id对应的action_ids了
            self._clear_meta_for_root(replay_meta)
            multimodal_train_info = None
            # TODO: 是否需要额外返回不重复的 multimodal_train_infos？
            for data_item in group_samples:
                if hasattr(data_item.data, "multimodal_train_info"):
                    multimodal_train_info = data_item.data.multimodal_train_info
                    del data_item.data.multimodal_train_info
                if "partial_rollout_input_ids" in data_item.env.rollout.extra_info:
                    del data_item.env.rollout.extra_info["partial_rollout_input_ids"]
            samples.append(group_samples)
            multimodal_train_infos.append(multimodal_train_info)
            task_end_time = time.perf_counter()
            task_time.append(task_end_time - task_start_time)
        # 检查completed_samples中是否还有剩余的数据，并且检查其是否过期
        self.logger.info(
            f"Remaining completed samples in buffer: {self.completed_samples_count}, task_time: {sum(task_time)}s, avg_time: {sum(task_time) / len(task_time)}s"
        )
        self._check_completed_samples_expired()
        self._check_completed_samples_aborted()
        return samples, multimodal_train_infos

    def sample(self, sample_from_expired_states) -> List[RLDataFlowItem]:
        if sample_from_expired_states and self.expired_samples_count > 0:
            self.sample_from_expired_count += 1
            return self._sample_from_expired_storage()
        if self.aborted_samples_count > 0:
            self.sample_from_aborted_count += 1
            return self._sample_from_aborted_storage()
        return []

    def clear(self):
        attrs_to_clear = [
            "_aborted_actions",
            "_completed_actions",
            "_expired_actions",
            "_actions",
            "_root2actions",
            "_observations",
            "_observations2states",
            "_states",
            "_action2observations",
        ]
        for attr in attrs_to_clear:
            getattr(self, attr).clear()

    def resolve_ray_objects(self, data_item: RLDataFlowItem):
        """Resolves ray.ObjectRefs in a RLDataFlowItem to their actual values.

        Args:
            data_item (RLDataFlowItem): The data item containing ray.ObjectRefs.
        Returns:
            RLDataFlowItem: The data item with ray.ObjectRefs resolved.
        """

        # Resolve data.multimodal_train_info
        if hasattr(data_item.data, "multimodal_train_info"):
            multimodal_info = data_item.data.multimodal_train_info
            if multimodal_info and "pixel_values" in multimodal_info:
                pixel_values_ref = multimodal_info["pixel_values"]
                if isinstance(pixel_values_ref, ObjectRef):
                    multimodal_info["pixel_values"] = ray.get(pixel_values_ref)
                    data_item.data.multimodal_train_info = multimodal_info
        # Resolve rollout.extra_info.router_experts
        if "routed_experts" in data_item.env.rollout.extra_info:
            if isinstance(data_item.env.rollout.extra_info["routed_experts"], ObjectRef):
                data_item.env.rollout.extra_info["routed_experts"] = ray.get(
                    data_item.env.rollout.extra_info["routed_experts"]
                )
                self.logger.info("Resolved routed_experts ObjectRef in rollout.extra_info")

    def convert_to_ray_objref(self, data_item: RLDataFlowItem):
        """Converts large tensors in RLDataFlowItem to ray.ObjectRefs.

        Args:
            data_item (RLDataFlowItem): The data item containing large tensors.
        Returns:
            RLDataFlowItem: The data item with large tensors converted to ray.ObjectRefs.
        """
        # convert data.multimodal_train_info to ray.ObjectRef
        if hasattr(data_item.data, "multimodal_train_info"):
            multimodal_info = data_item.data.multimodal_train_info
            if multimodal_info and "pixel_values" in multimodal_info:
                pixel_values_ref = ray.put(multimodal_info["pixel_values"])
                del multimodal_info["pixel_values"]
                data_item.data.multimodal_train_info = pixel_values_ref
        # convert rollout.extra_info.router_experts to ray.ObjectRef
        if "routed_experts" in data_item.env.rollout.extra_info:
            routed_experts_ref = ray.put(data_item.env.rollout.extra_info["routed_experts"])
            del data_item.env.rollout.extra_info["routed_experts"]
            data_item.env.rollout.extra_info["routed_experts"] = routed_experts_ref

    def has_objectref(self, item: RLDataFlowItem) -> bool:
        def check(obj):
            if isinstance(obj, ray.ObjectRef):
                return True
            if isinstance(obj, BaseModel):
                return any(check(getattr(obj, f)) for f in obj.model_fields)
            if isinstance(obj, (list, tuple, set)):
                return any(check(x) for x in obj)
            if isinstance(obj, dict):
                return any(check(v) for v in obj.values())
            if isinstance(obj, (str, int, float, bool, type(None), torch.Tensor, numpy.ndarray)):
                return False
            # 如果不满足以上类型，抛出错误，防止意想不到的问题
            raise TypeError(
                f"Unsupported type: {type(obj)} in {obj} "
                f"Expected ray.ObjectRef, BaseModel, list/tuple/set, dict, or primitive types."
            )

        return check(item)

    def dump(self, file_path: Path):
        """Dumps the entire state of the replay buffer storage to a single
        file, resolving all ray.ObjectRefs to their actual values.

        Args:
            file_path (str): The path to the file where the state will be
                saved.
        """
        all_data_items = [mapping_replaymeta_to_dataitem(replay_meta) for replay_meta in self._actions.values()]

        for data_items in all_data_items:
            for item in data_items:
                self.resolve_ray_objects(item)
                res = self.has_objectref(item)
                assert not res, "ReplayBufferStorage.dump found unresolved ray.ObjectRef in RLDataFlowItem"

        state = {
            "_completed_actions": self._completed_actions,
            "_aborted_actions": self._aborted_actions,
            "_expired_actions": self._expired_actions,
            "_actions": all_data_items,
            "_root2actions": dict(self._root2actions),
            "_observations2states": self._observations2states,
            "_states": dict(self._states),
            "_action2observations": dict(self._action2observations),
        }

        torch.save(state, file_path)
        self.logger.info(f"ReplayBufferStorage state dumped to {file_path}")

    def resume(self, file_path: Path):
        """Resumes the replay buffer storage from a single file.

        Args:
            file_path (str): The path to the file from which to restore the
                state.
        """
        if len(self._actions) > 0:
            self.logger.warning("ReplayBufferStorage is not empty. Resuming will overwrite the existing state.")
            self.clear()

        state = torch.load(file_path, map_location="cpu", weights_only=False)

        self._completed_actions = state["_completed_actions"]
        self._aborted_actions = state["_aborted_actions"]
        self._expired_actions = state["_expired_actions"]
        self._root2actions = defaultdict(list, state["_root2actions"])
        self._observations2states = state["_observations2states"]
        self._states = defaultdict(list, state["_states"])
        self._action2observations = defaultdict(list, state["_action2observations"])

        dump_actions = state["_actions"]
        # 重建 _actions 和 _observations: 与replaymeta相关
        for group_dataitem in dump_actions:
            for data_item in group_dataitem:
                self.convert_to_ray_objref(data_item)
            replay_meta = mapping_dataitem_to_replaymeta(group_dataitem)
            action_id = replay_meta.action_id
            self._actions[action_id] = replay_meta
            for observation_id in self._action2observations[action_id]:
                self._observations[observation_id] = replay_meta

        self.logger.info(f"ReplayBufferStorage state successfully resumed from {file_path}")
        self.logger.info(
            f"ReplayBuffer Storage status: completed_samples_count={self.completed_samples_count}, aborted_samples_count={self.aborted_samples_count}, expired_samples_count={self.expired_samples_count}"
        )

    @property
    def completed_samples_count(self) -> int:
        return sum(len(bucket) for bucket in self._completed_actions.values())

    @property
    def aborted_samples_count(self):
        return sum(len(bucket) for bucket in self._aborted_actions.values())

    @property
    def expired_samples_count(self):
        return len(self._expired_actions)

    def _sample_from_expired_storage(self) -> List[RLDataFlowItem]:
        """Samples an item from the expired storage for re-rollout.

        This method takes an action_id from the expired queue, retrieves its
        original prompt data, cleans up all its previous rollout outputs, and
        prepares it as a new sample group with a fresh action_id and reset
        version (0) to be sent for a new generation attempt.

        Returns:
            List[RLDataFlowItem]: A list of data items ready for a new rollout.
        """
        assert len(self._expired_actions) > 0
        action_id = self._expired_actions.pop()
        replay_meta = self._actions.pop(action_id)
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)
        # 把这条数据上次的记录全部删掉,重新开始rollout,root2actions也要清除
        self._clear_meta_for_root(replay_meta)

        for sample in group_samples:
            assert sample.data.input_ids and sample.data.num_tokens, "input_ids or num_tokens is empty!"
            if "routed_experts" in sample.env.rollout.extra_info:
                del sample.env.rollout.extra_info["routed_experts"]
            del sample.env
            sample.env = RLEnvDataItem()  # 重置env数据
            sample.uid.action_id = action_id
            sample.uid.version = 0

        self.logger.debug(
            f"Sampling expired action_id: {action_id} from replay buffer, remain expired samples: {len(self._expired_actions)}"
        )
        return group_samples

    def _sample_from_aborted_storage(self) -> List[RLDataFlowItem]:
        """Samples an item from the aborted storage for re-rollout.

        This method retrieves an action with the highest version (oldest version) from the
        aborted buckets. It then cleans up its previous (aborted) rollout
        outputs and prepares it as a new sample group with a fresh action_id.
        The original version number is preserved to track its retry history.

        Returns:
            List[RLDataFlowItem]: A list of data items ready for a new rollout.
        """
        assert self.aborted_samples_count > 0
        action_id = self._pop_highest_version_action(self._aborted_actions)
        # 通过self.aborted_samples_count判断过这里不会返回None
        replay_meta = self._actions.pop(action_id)  # type: ignore[arg-type]
        replay_meta_version = replay_meta.version
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)
        # 把这条数据上次rollout产生的输出的记录都删掉，上次的数据已经记录在了RLEnvDataItem中了
        self._clear_meta_for_actions(replay_meta)

        sample_action_id = uuid4().int
        for sample in group_samples:
            assert sample.data.input_ids and sample.data.num_tokens, "input_ids or num_tokens is empty!"
            if not self.enable_partial_rollout:
                # 清除上次的response_ids等env数据
                if "routed_experts" in sample.env.rollout.extra_info:
                    del sample.env.rollout.extra_info["routed_experts"]
                del sample.env
                sample.env = RLEnvDataItem()
                sample.uid.version = 0
                sample.uid.action_id = action_id if action_id is not None else sample_action_id
            else:
                # 将异步的逻辑尽量放在replay buffer中处理，尽量不在env/rollout中进行处理
                history_response_ids = list(itertools.chain.from_iterable(sample.env.rollout.versioned_response_ids))
                sample.env.rollout.extra_info["partial_rollout_input_ids"] = (
                    sample.data.input_ids + history_response_ids
                )
                self.logger.debug(
                    f"partial rollout enabled, {sample_action_id} pass response_ids {len(history_response_ids)} to input_ids {len(sample.data.input_ids)} to data extra info when sampling."
                )
                sample.uid.version = replay_meta_version
                sample.uid.action_id = int(sample_action_id)

        self.logger.debug(
            f"Sampling aborted action_id: {sample_action_id}, root_id: {group_samples[0].uid.root_id} from replay buffer, remain aborted samples: {self.aborted_samples_count}"
        )
        return group_samples

    def _pop_highest_version_action(self, buckets: Dict[int, List[int]]) -> Optional[int]:
        if not buckets:
            return None

        highest_version = sorted(buckets.keys())[-1]
        action_list = buckets[highest_version]
        action_id = action_list.pop()
        if not action_list:
            del buckets[highest_version]

        return action_id

    def _check_completed_samples_expired(self):
        """Moves samples from completed buckets to the expired list if they are
        too old after get target completed samples from replay buffer.

        This method iterates through the `_completed_actions` buckets. If a
        bucket's version index is greater than or equal to the configured
        `tail_batch_candidate_steps`, all action_ids within that bucket are
        moved to the `_expired_actions` list, marking them as expired.
        """
        if self.tail_batch_candidate_steps <= 0:
            return

        expired_versions = [v for v in self._completed_actions if v >= self.tail_batch_candidate_steps]

        for version in expired_versions:
            bucket = self._completed_actions.pop(version)
            self._expired_actions.extend(bucket)
            self.logger.info(
                f"Moved {len(bucket)} completed samples with version {version} to expired samples due to exceeding tail_batch_candidate_steps."
            )

    def _check_completed_samples_aborted(self):
        if self.enable_partial_rollout:
            return

        for version, bucket in self._completed_actions.items():
            self._aborted_actions[0].extend(bucket)
            self.logger.info(
                f"Moved {len(bucket)} completed samples with version {version} to aborted samples due to partial rollout disabled."
            )
        self._completed_actions.clear()

    def _clear_meta_for_actions(self, replay_meta: ReplayMeta):
        """Completely removes an action and all its associated data from the
        storage.

        This is the single source of truth for deleting an action.
        """
        action_id = replay_meta.action_id

        for observation_id in replay_meta.observation_ids:
            self._observations.pop(observation_id, None)
            state = self._observations2states.pop(observation_id, None)
            if state and observation_id in self._states.get(state, []):
                self._states[state].remove(observation_id)

        self._action2observations.pop(action_id, None)
        del replay_meta

    def _clear_meta_for_root(self, replay_meta: ReplayMeta):
        """Clears all actions and associated metadata linked to the same
        root_id.

        This function is called after a sample group is successfully retrieved
        for training. It ensures that all historical versions of a prompt
        (linked by root_id) are purged from the storage to prevent them from
        being re-sampled or replayed.

        Args:
            replay_meta (ReplayMeta): The metadata of the action that was just
                retrieved. The root_id from this object will be used to find
                and clear all related actions.
        """
        root_id = replay_meta.root_id
        if root_id in self._root2actions:
            for action_id in self._root2actions[root_id]:
                new_replay_meta = self._actions.pop(action_id, None)
                if new_replay_meta:
                    self._clear_meta_for_actions(new_replay_meta)
            del self._root2actions[root_id]
        del replay_meta

    def _check_rollout_state_and_insert(self, replay_meta: ReplayMeta):
        """Checks the rollout state of a ReplayMeta object and inserts its
        action_id into the appropriate state bucket.

        This method acts as a router, directing action_ids to different storage
        lists (_aborted_actions, _completed_actions, _expired_actions) based on
        their final rollout state and version. It also handles the logic for
        when an aborted sample becomes expired due to too many retries.

        Args:
            replay_meta (ReplayMeta): The metadata object containing the final
                state and version of a rollout action.
        """
        state = replay_meta.state
        root_id = replay_meta.root_id
        action_id = replay_meta.action_id

        if state == RolloutState.ABORTED:
            if self.tail_batch_candidate_steps > 0 and replay_meta.version >= self.tail_batch_candidate_steps:
                # 过期的数据需要重置状态
                self._expired_actions.append(action_id)
                self.logger.debug(
                    f"Add expired sample with action_id: {action_id} to _expired_actions because version: {replay_meta.version} >= tail_batch_candidate_steps: {self.tail_batch_candidate_steps}."
                )
            else:
                self._aborted_actions[replay_meta.version].append(action_id)
                self.logger.debug(
                    f"Add aborted sample with action_id: {action_id} version: {replay_meta.version} to _aborted_actions."
                )
        elif state == RolloutState.COMPLETED:
            self._completed_actions[replay_meta.version].append(action_id)
            self.logger.debug(f"Add sample with root_id: {root_id}, action_id: {action_id} to finished_actions.")
        else:
            assert False, f"Unsupported rollout state {state} for action_id {action_id} in ReplayBufferStorage."


@ray.remote
class ReplayBuffer:
    """A Ray actor that manages experience replay for reinforcement
    learning."""

    def __init__(
        self,
        config: ReplayBufferConfig,
    ):
        """Initializes the ReplayBuffer actor.

        Args:
            config (ReplayBufferConfig): The configuration object.
        """
        self.config = config
        self.storage = ReplayBufferStorage(config)
        self.sampler = DatasetSampler(config.dataset_cfg, config.dataloader_cfg, config.tokenizer)
        self.post_processor_func = config.postprocessor_func
        self.sample_from_expired_states = False
        self.sample_from_dataset_count = 0
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="ReplayBuffer")

    def get_prerun_state(self, target_batch_size: int):
        remain_size = target_batch_size - self.storage.completed_samples_count
        if remain_size <= 0:
            self.sample_from_expired_states = False
        else:
            expired_threshold = (
                min(remain_size, self.config.tail_batch_trigger_size)
                if self.config.tail_batch_trigger_size
                else remain_size
            )
            if self.storage.expired_samples_count > expired_threshold:
                self.sample_from_expired_states = True
                self.logger.info(
                    f"Enable sampling from expired states. Expired samples: {self.storage.expired_samples_count}, threshold: {expired_threshold}, remain_size: {remain_size}"
                )
            else:
                self.sample_from_expired_states = False
        return self.sample_from_expired_states, self.storage.completed_samples_count

    def get_train_dataset_length(self):
        """Returns the length of the training dataloader."""
        return len(self.sampler.dataloader)

    def post_processor(self, group_samples):
        """Applies a post-processing function to a group of samples.

        Args:
            group_samples: A list of samples to process.

        Returns:
            The processed group of samples.
        """
        if self.post_processor_func:
            group_samples = self.post_processor_func(group_samples)
            return group_samples
        return group_samples

    def sample(self, env, prompt_repeat_k) -> List[RLDataFlowItem]:
        """Samples a batch of experiences from the replay buffer.

        Args:
            env: The environment name.
            enable_partial_rollout (int): Flag to enable partial rollouts.
            prompt_repeat_k (int): Number of times to repeat a prompt.

        Returns:
            A list of sampled data items.
        """
        storage_samples = self.storage.sample(self.sample_from_expired_states)
        if storage_samples:
            return storage_samples
        else:
            self.sample_from_dataset_count += 1
            return self.sampler.sample(env, prompt_repeat_k)

    def get_samples(
        self,
        global_batch_size: int,
    ):
        """Gets a batch of finished samples from the storage.

        Args:
            global_batch_size (int): The number of sample groups to retrieve.

        Returns:
            A list of sample groups.
        """
        return self.storage.get(global_batch_size)

    def add(self, grouped_dataitem: List[RLDataFlowItem]):
        """Adds a group of data items to the replay buffer storage.

        Args:
            grouped_dataitem (List[RLDataFlowItem]): A list of data items
                from the same group.
        """
        self.storage.add(grouped_dataitem)

    def status(self):
        return {
            "remain_completed_samples_count": self.storage.completed_samples_count,
            "remain_aborted_samples_count": self.storage.aborted_samples_count,
            "remain_expired_samples_count": self.storage.expired_samples_count,
            "sample_from_dataset_count": self.sample_from_dataset_count,
            "sample_from_aborted_count": self.storage.sample_from_aborted_count,
            "sample_from_expired_count": self.storage.sample_from_expired_count,
        }

    def save(self, file_path: Path | str):
        """Saves the replay buffer's storage to a file.

        Args:
            file_path (str): The path to the file for saving the data.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # save dataloader
        dataloader_path = file_path / "dataloader"
        dataloader_state = self.sampler.dataloader.get_state_dict(self.sampler.reduced_consumed_samples)
        torch.save(dataloader_state, dataloader_path)

        # save storage
        rb_storage_path = file_path / "replay_buffer_storage.pth"
        self.storage.dump(rb_storage_path)

    def resume(self, file_path: Path | str):
        """Resumes the replay buffer's storage from a file.

        Args:
            file_path (str): The path to the file from which to restore the
                state.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        dataloader_path = file_path / "dataloader"
        if dataloader_path.exists():
            dataloader_state = torch.load(dataloader_path, map_location=DEVICE)
            self.sampler.dataloader.load_state_dict(dataloader_state)
            self.sampler.reduced_consumed_samples = dataloader_state["sampler"]["step"]
            self.sampler.cur_epoch = dataloader_state["sampler"]["epoch"]
        else:
            self.logger.warning(f"Dataloader state file {dataloader_path} does not exist. Skipping dataloader resume.")
        # resume storage
        rb_storage_path = file_path / "replay_buffer_storage.pth"
        if rb_storage_path.exists():
            self.storage.resume(rb_storage_path)
        else:
            self.logger.warning(
                f"ReplayBufferStorage state file {rb_storage_path} does not exist. Skipping storage resume."
            )

    def get_completed_samples_count(self) -> int:
        """Returns the count of completed samples in the replay buffer.

        Returns:
            int: The number of completed samples.
        """
        return self.storage.completed_samples_count

    def clear(self):
        """Clears the replay buffer storage."""
        self.storage.clear()
