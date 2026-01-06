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
    observation_ids: List[int] = field(default_factory=list)  # observation IDs for different versions
    observation_refs: List[ObjectRef] = field(default_factory=list)
    observation_versions: List[int] = field(default_factory=list)  # reserved for async rollout
    state: RolloutState = RolloutState.INIT
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

    group_state = determine_group_state(grouped_dataitem)
    logger.debug(f"determined group_state: {group_state}, replay_state: {group_state}")
    replay_meta = ReplayMeta(
        env=env_str,
        root_id=root_id,
        action_id=action_id,
        action_ref=ray.put(data),
        observation_ids=observation_ids,
        observation_refs=observation_refs,
        observation_versions=observation_versions,
        state=group_state,
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
        # NOTE: This mapping function used by both dump and get. ObjectRefs are kept during dump (for training continuity)
        # but released during get (via del replaymeta) when no longer needed. So we do not free them manually here.
        # ray._private.internal_api.free(obs_ref)

        item = RLDataFlowItem(
            uid=RLUIDItem(env=env_str, root_id=root_id, action_id=action_id, observation_id=obs_id, version=version),
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
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"


class Sampler:
    """Sampler for drawing prompts from datasets or the replay buffer."""

    def __init__(self, dataloader, tokenizer, storage):
        """Initializes the Sampler.

        Args:
            dataloader: The dataloader for the dataset.
            tokenizer: The tokenizer for processing text.
            storage: The ReplayBufferStorage instance.
        """
        self.train_dataloader = dataloader
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
            else AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        )
        self.storage = storage
        self.sample_count = 0
        self.cur_epoch = 0
        self.reduced_consumed_samples = 0
        self.logger = get_logger()

    def sample_from_datasets(self, env: str, repeat_prompt_k: int) -> List[RLDataFlowItem]:
        """Samples a new group of prompts from the original dataset.

        Args:
            env (str): The environment name.
            repeat_prompt_k (int): The number of times to repeat the prompt.

        Returns:
            List[RLDataFlowItem]: A list of data items for the data group contains repeat_prompt_k samples from same data.
        """
        root_id = uuid4().int
        action_id = uuid4().int
        group_data_item: List[RLDataFlowItem] = [RLDataFlowItem() for _ in range(repeat_prompt_k)]

        try:
            data = next(self.train_dataloader_iter)[0]
        except StopIteration:
            self.cur_epoch += 1
            self.train_dataloader.set_epoch(self.cur_epoch)
            self.train_dataloader_iter = iter(self.train_dataloader)
            data = next(self.train_dataloader_iter)[0]
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

        return group_data_item

    def sample_from_unfinished_buffer(self) -> List[RLDataFlowItem]:
        """Samples a prompt from a partially completed (unfinished) rollout."""
        action_id = self.storage._paused.pop(0)
        self.logger.debug(f"Sampling unfinished action_id: {action_id} from replay buffer")
        replay_meta = self.storage._actions[action_id]
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)
        self.sample_count += 1
        if len(self.storage._paused) == 0:
            self.logger.info(f"Sampled {self.sample_count} unfinished samples from replay buffer")
        return group_samples

    def sample(self, env: str, enable_partial_rollout: int, prompt_repeat_k: int) -> List[RLDataFlowItem]:
        """Selects a sampling strategy and returns a group of samples.

        It decides whether to sample from the unfinished buffer (for partial
        rollouts greater than 0) or from the original dataset.

        Args:
            env (str): The environment name.
            enable_partial_rollout (int): Flag to enable partial rollout.
            prompt_repeat_k (int): Number of times to repeat the prompt.

        Returns:
            List[RLDataFlowItem]: A list of sampled data items.
        """
        # TODO(@duanyanhui): 考虑sampler结构的独立性，不要传入replay buffer storage,
        # sample_from_unfinished_buffer可以作为replay buffer的一个方法
        if enable_partial_rollout > 0 and len(self.storage._paused) > 0:
            return self.sample_from_unfinished_buffer()
        else:
            # note: Sample grouped sample at once. They share the same action_id
            return self.sample_from_datasets(env, prompt_repeat_k)


class ReplayBufferStorage:
    """Handles the storage of experiences for the replay buffer."""

    def __init__(self, worker_log_dir):
        """Initializes the data structures for storing replay data."""
        self._paused: List[int] = []  # List of paused action_id,
        self._returned: List[int] = []  # List of returned action_id,
        self._actions: Dict[int, ReplayMeta] = {}  # action_id: ReplayMeta
        self._root2actions: Dict[int, List[int]] = defaultdict(
            list
        )  # root_id: [action_id, action_id, ...], designed for grpo
        self._observations: Dict[int, ReplayMeta] = {}  # observation_id: ReplayMeta
        self._observations2states: Dict[int, str] = {}  # observation_id: state_str
        self._states: Dict[str, List[int]] = defaultdict(list)  # str: [observation_id, observation_id, ...]
        self._action2observations: Dict[int, List[int]] = defaultdict(
            list
        )  # action_id: [observation_id, observation_id, ...]
        self.logger = get_logger(log_dir=worker_log_dir, tag="ReplayBuffer")

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
        state = replay_meta.state

        if state == RolloutState.ABORTED:
            self._paused.append(action_id)
        elif state == RolloutState.COMPLETED:
            self._returned.append(action_id)
        self.logger.debug(
            f"Adding action_id: {action_id} with state: {state} to ReplayBufferStorage. Paused count: {len(self._paused)}, Returned count: {len(self._returned)}"
        )
        self._root2actions[root_id].append(action_id)
        self._actions[action_id] = replay_meta

        # observation
        for observation_id in replay_meta.observation_ids:
            self._observations[observation_id] = replay_meta
            self._observations2states[observation_id] = replay_meta.state
            if observation_id not in self._action2observations[action_id]:
                self._action2observations[action_id].append(observation_id)
            if observation_id not in self._states[replay_meta.state]:
                self._states[replay_meta.state].append(observation_id)

    def clear(self):
        attrs_to_clear = [
            "_paused",
            "_returned",
            "_actions",
            "_root2actions",
            "_observations",
            "_observations2states",
            "_states",
            "_action2observations",
        ]
        for attr in attrs_to_clear:
            getattr(self, attr).clear()

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
        if len(self._returned) < global_batch_size:
            self.logger.error("Not enough finished samples in replay buffer")
            return [], []
        else:
            self.logger.info(
                f"Retrieving global_batch_size {global_batch_size} from replay buffer, len of self.returned: {len(self._returned)}"
            )
            target_finished_list = self._returned[:global_batch_size]
            remain_finished_list = self._returned[global_batch_size:]
            for action_id in target_finished_list:
                replay_meta = self._actions.pop(action_id)
                observation_ids = self._action2observations.pop(action_id)
                for obs_id in observation_ids:
                    self._observations.pop(obs_id)
                    self._observations2states.pop(obs_id)

                # todo: add an unified state management
                replay_meta.state = RolloutState.ARCHIVED
                group_samples = mapping_replaymeta_to_dataitem(replay_meta)
                del replay_meta
                multimodal_train_info = None
                # TODO: 是否需要额外返回不重复的 multimodal_train_infos？
                for data_item in group_samples:
                    if hasattr(data_item.data, "multimodal_train_info"):
                        multimodal_train_info = data_item.data.multimodal_train_info
                        del data_item.data.multimodal_train_info
                samples.append(group_samples)
                multimodal_train_infos.append(multimodal_train_info)
            self._returned = remain_finished_list

            return samples, multimodal_train_infos

    def get_finished_samples(self):
        """Returns the number of finished sample groups."""
        return len(self._returned)

    def get_unfinished_samples(self):
        """Returns the number of unfinished sample groups."""
        return len(self._paused)

    def get_prompt_num(self):
        return len(self._action2observations)

    def status(self):
        return {
            "rollout_finished_count": len(self._returned),
            "rollout_paused_count": len(self._paused),
            "action_count": len(self._actions),
            "observation_count": len(self._observations),
        }

    def print(self):
        rollout_finished_count = len(self._returned)
        rollout_paused_count = len(self._paused)
        action_count = len(self._actions)
        observation_count = len(self._observations)

        log_message = (
            "[ReplayBuffer] ReplayBufferStorage states:\n"
            f"  - Rollout States: Finished={rollout_finished_count}, Paused={rollout_paused_count}\n"
            f"  - History Actions: {action_count}\n"
            f"  - History Observations: {observation_count}"
        )
        self.logger.info(log_message)

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
            "_paused": self._paused,
            "_returned": self._returned,
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

        self._paused = state["_paused"]
        self._returned = state["_returned"]
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
        self.storage = ReplayBufferStorage(config.worker_log_dir)
        self.tokenizer = config.tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, trust_remote_code=True)

        if config.dataloader_cfg is not None:
            self.dataloader_cfg = config.dataloader_cfg
            self.dataloader_cfg.dataset_config_list = config.dataset_cfg
        else:
            self.dataloader_cfg = DataloaderConfig(
                dataset_config_list=config.dataset_cfg,
                collator="fake_collator",
                pack_level="none",
            )
        self._dataloader = self.dataloader_cfg.build(
            tokenizer=self.tokenizer, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        self.sampler = Sampler(
            self._dataloader,
            self.tokenizer,
            self.storage,
        )
        self.post_processor_func = config.postprocessor_func
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="ReplayBuffer")

    def get_train_dataset_length(self):
        """Returns the length of the training dataloader."""
        return len(self._dataloader)

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

    def sample(self, env, enable_partial_rollout: int, prompt_repeat_k: int) -> List[RLDataFlowItem]:
        """Samples a batch of experiences from the replay buffer.

        Args:
            env: The environment name.
            enable_partial_rollout (int): Flag to enable partial rollouts.
            prompt_repeat_k (int): Number of times to repeat a prompt.

        Returns:
            A list of sampled data items.
        """

        return self.sampler.sample(env, enable_partial_rollout, prompt_repeat_k)

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

    def print(self):
        """Prints the current state of the replay buffer storage."""
        self.storage.print()

    def status(self):
        return self.storage.status()

    def save(self, file_path: Path | str):
        """Saves the replay buffer's storage to a file.

        Args:
            file_path (str): The path to the file for saving the data.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # save dataloader
        dataloader_path = file_path / "dataloader"
        dataloader_state = self._dataloader.get_state_dict(self.sampler.reduced_consumed_samples)
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
            self._dataloader.load_state_dict(dataloader_state)

            # resume dataloader
            self.sampler = Sampler(
                self._dataloader,
                self.tokenizer,
                self.storage,
            )
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

    def get_finished_samples(self):
        """Returns the number of finished sample groups in the storage."""
        return self.storage.get_finished_samples()

    def get_unfinished_samples(self):
        """Returns the number of unfinished sample groups in the storage."""
        return self.storage.get_unfinished_samples()

    def clear(self):
        """Clears the replay buffer storage."""
        self.storage.clear()
