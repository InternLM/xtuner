import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from ray import ObjectRef
from typing_extensions import Annotated

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLDatasetItem,
    RLExtraDataItem,
    RLUIDItem,
    RolloutState,
    is_valid_for_replaybuffer,
)
from xtuner.v1.datasets import build_dataloader, build_datasets
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.utils import get_logger


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
        ray._private.internal_api.free(obs_ref)

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
        Parameter(help="The tokenizer for processing text data, e.g., for partial rollouts."),
    ]
    postprocessor_func: Annotated[
        Optional[Callable],
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
        self.datasets = build_datasets(dataset_cfg, self.tokenizer)
        if dataloader_cfg is not None:
            self.dataloader_cfg = dataloader_cfg
        else:
            self.dataloader_cfg = DataloaderConfig(
                collator="fake_collator",
                pack_level="none",
            )
        self.dataloader = build_dataloader(
            dataloader_config=self.dataloader_cfg,
            datasets=self.datasets,
            global_batch_size=1,
            micro_batch_size=1,
            seed=1,
        )
        self.dataloader_iter = iter(self.dataloader)
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
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]

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

    def resume(self, num: int) -> None:
        self.dataloader_iter = itertools.islice(self.dataloader, num, None)


class ReplayBufferStorage:
    """Handles the storage of experiences for the replay buffer."""

    def __init__(self, replay_buffer_cfg):
        """Initializes the data structures for storing replay data."""
        self._aborted_actions: List[int] = []
        self._completed_actions: List[int] = []
        self._actions: Dict[int, ReplayMeta] = {}
        self._root2actions: Dict[int, List[int]] = defaultdict(list)
        self._observations: Dict[int, ReplayMeta] = {}
        self._observations2states: Dict[int, str] = {}
        self._states: Dict[str, List[int]] = defaultdict(list)
        self._action2observations: Dict[int, List[int]] = defaultdict(list)
        self.logger = get_logger(log_dir=replay_buffer_cfg.worker_log_dir, tag="ReplayBuffer")
        self._multimodal_train_infos: Dict[int, Dict[str, Any]] = {}
        self.sample_from_aborted_count = 0

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
            self._aborted_actions.append(action_id)
        elif state == RolloutState.COMPLETED:
            self._completed_actions.append(action_id)
        self.logger.debug(
            f"Adding action_id: {action_id} with state: {state} to ReplayBufferStorage. Paused count: {len(self._aborted_actions)}, Returned count: {len(self._completed_actions)}"
        )
        self._root2actions[root_id].append(action_id)
        self._actions[action_id] = replay_meta

        # observation
        for observation_id in replay_meta.observation_ids:
            self._action2observations[action_id].append(observation_id)
            self._observations[observation_id] = replay_meta
            self._observations2states[observation_id] = replay_meta.state
            self._states[replay_meta.state].append(observation_id)

    def clear(self):
        attrs_to_clear = [
            "_aborted_actions",
            "_completed_actions",
            "_actions",
            "_root2actions",
            "_observations",
            "_observations2states",
            "_states",
            "_action2observations",
        ]
        for attr in attrs_to_clear:
            getattr(self, attr).clear()

    def get(self, global_batch_size: int) -> Tuple[List[List[RLDataFlowItem]], List[Dict[str, Any] | None]]:
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
        if len(self._completed_actions) < global_batch_size:
            self.logger.error("Not enough finished samples in replay buffer")
            return [], []
        else:
            self.logger.info(
                f"Retrieving global_batch_size {global_batch_size} from replay buffer, len of self.returned: {len(self._completed_actions)}"
            )
            target_finished_list = self._completed_actions[:global_batch_size]
            remain_finished_list = self._completed_actions[global_batch_size:]
            for action_id in target_finished_list:
                replay_meta = self._actions.pop(action_id)
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
            self._completed_actions = remain_finished_list

            return samples, multimodal_train_infos

    @property
    def completed_samples_count(self) -> int:
        return len(self._completed_actions)

    @property
    def aborted_samples_count(self):
        return len(self._aborted_actions)

    def dump(self, file_path: str):
        """Dumps the entire state of the replay buffer storage to a single
        file, resolving all ray.ObjectRefs to their actual values.

        Args:
            file_path (str): The path to the file where the state will be
                saved.
        """
        import os
        import pickle

        self.logger.info(f"Starting to dump ReplayBufferStorage state to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        all_data_items = []
        for replay_meta in self._actions.values():
            group_data_items = mapping_replaymeta_to_dataitem(replay_meta)
            all_data_items.append(group_data_items)

        with open(file_path, "wb") as f:
            pickle.dump(all_data_items, f)
        self.logger.info(f"ReplayBufferStorage state dumped to {file_path}")

    def resume(self, file_path: str):
        """Resumes the replay buffer storage from a single file.

        Args:
            file_path (str): The path to the file from which to restore the
                state.
        """
        import os
        import pickle

        self.logger.info(f"Starting to resume ReplayBufferStorage state from {file_path}...")
        if not os.path.exists(file_path):
            self.logger.error(f"State file not found: {file_path}. Cannot resume.")
            return

        with open(file_path, "rb") as f:
            all_data_items = pickle.load(f)

        for group_data_items in all_data_items:
            replay_meta = mapping_dataitem_to_replaymeta(group_data_items)
            root_id = replay_meta.root_id
            action_id = replay_meta.action_id
            state_str = replay_meta.state
            if state_str == "abort":
                self._aborted_actions.append(action_id)
            elif state_str == "returned":
                self._completed_actions.append(action_id)
            self._root2actions[root_id].append(action_id)
            self._actions[action_id] = replay_meta
            for observation_id in replay_meta.observation_ids:
                self._action2observations[action_id].append(observation_id)
                self._observations[observation_id] = replay_meta
                self._observations2states[observation_id] = replay_meta.state
                self._states[replay_meta.state].append(observation_id)

        self.logger.info(f"ReplayBufferStorage state successfully resumed from {file_path}")

    def sample(self) -> List[RLDataFlowItem]:
        """Samples a group of data items from aborted actions in the storage.

        Returns:
            List[RLDataFlowItem]: A list of sampled data items.
        """
        if len(self._aborted_actions) == 0:
            return []
        action_id = self._aborted_actions.pop(0)
        replay_meta = self._actions[action_id]
        self.logger.debug(f"Sampling aborted action_id: {action_id} from ReplayBufferStorage.")
        self.sample_from_aborted_count += 1
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)
        return group_samples


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
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="ReplayBuffer")
        self.sample_from_dataset_count = 0

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

    def sample(self, env, prompt_repeat_k: int) -> List[RLDataFlowItem]:
        """Samples a batch of experiences from the replay buffer.

        Args:
            env: The environment name.
            enable_partial_rollout (int): Flag to enable partial rollouts.
            prompt_repeat_k (int): Number of times to repeat a prompt.

        Returns:
            A list of sampled data items.
        """
        storage_samples = self.storage.sample()
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

    def dump(self, file_path: str):
        """Dumps the replay buffer's storage to a file.

        Args:
            file_path (str): The path to the file for saving the data.
        """
        self.storage.dump(file_path)

    def status(self):
        return {
            "remain_completed_samples_count": self.storage.completed_samples_count,
            "remain_aborted_samples_count": self.storage.aborted_samples_count,
            "sample_from_dataset_count": self.sample_from_dataset_count,
            "sample_from_aborted_count": self.storage.sample_from_aborted_count,
        }

    def resume(self, file_path: str):
        """Resumes the replay buffer's storage from a file.

        Args:
            file_path (str): The path to the file from which to restore the
                state.
        """
        self.storage.resume(file_path)

    def get_completed_samples(self):
        """Returns the number of finished sample groups in the storage."""
        return self.storage.completed_samples_count

    def get_aborted_samples(self):
        """Returns the number of unfinished sample groups in the storage."""
        return self.storage.aborted_samples_count

    def clear(self):
        """Clears the replay buffer storage."""
        self.storage.clear()
