import itertools
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import (
    ReplayMeta,
    RLDataFlowItem,
    RLDatasetItem,
    RLExtraDataItem,
    RLUIDItem,
    mapping_dataitem_to_replaymeta,
    mapping_replaymeta_to_dataitem,
)
from xtuner.v1.datasets import build_dataloader, build_datasets
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.utils import get_logger


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
                "tokenize_fn": RLTextTokenizeFnConfig(max_length=512)
            }],
            dataloader_cfg=DataloaderConfig(collator='fake_collator'),
            tokenizer=AutoTokenizer.from_pretrained("model_path"),
            postprocessor_func=None,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class Sampler:
    """Sampler for drawing prompts from datasets or the replay buffer."""

    def __init__(self, dataset, dataloader, tokenizer, storage):
        """Initializes the Sampler.

        Args:
            dataset: The dataset to sample from.
            dataloader: The dataloader for the dataset.
            tokenizer: The tokenizer for processing text.
            storage: The ReplayBufferStorage instance.
        """
        self.train_dataset = dataset
        self.train_dataloader = dataloader
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
            else AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        )
        self.storage = storage

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
            self.train_dataloader_iter = iter(self.train_dataloader)
            data = next(self.train_dataloader_iter)[0]

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
        replay_meta = self.storage._actions[action_id]
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)
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
        if enable_partial_rollout > 0 and len(self.storage._paused) > 0:
            return self.sample_from_unfinished_buffer()
        else:
            # note: Sample grouped sample at once. They share the same action_id
            return self.sample_from_datasets(env, prompt_repeat_k)

    def resume(self, num: int) -> None:
        self.train_dataloader_iter = itertools.islice(self.train_dataloader, num, None)


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
        if len(grouped_dataitem) == 0:
            return

        replay_meta = mapping_dataitem_to_replaymeta(grouped_dataitem)
        root_id = replay_meta.root_id
        action_id = replay_meta.action_id
        state_str = replay_meta.state

        # Here, partial rollout is handled based on whether finish_reason is "paused".
        # The logic for "paused" is user-defined, indicating that this data was
        # interrupted before inference was completed. Other states are returned
        # by the inference engine.
        if state_str == "paused":
            self._paused.append(action_id)
        elif state_str == "returned":
            self._returned.append(action_id)

        # action
        self._root2actions[root_id].append(action_id)
        self._actions[action_id] = replay_meta

        # observation
        for observation_id in replay_meta.observation_ids:
            self._action2observations[action_id].append(observation_id)
            self._observations[observation_id] = replay_meta
            self._observations2states[observation_id] = replay_meta.state
            self._states[replay_meta.state].append(observation_id)

    def get(self, global_batch_size: int) -> List[List[RLDataFlowItem]]:
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
        if len(self._returned) < global_batch_size:
            raise ValueError("Not enough finished samples in replay buffer")
            return []
        else:
            target_finished_list = self._returned[:global_batch_size]
            remain_finished_list = self._returned[global_batch_size:]
            for action_id in target_finished_list:
                replay_meta = self._actions[action_id]
                # todo: add an unified state management
                replay_meta.state = "history"
                group_samples = mapping_replaymeta_to_dataitem(self._actions[action_id])
                samples.append(group_samples)
            self._returned = remain_finished_list
            return samples

    def get_finished_samples(self):
        """Returns the number of finished sample groups."""
        return len(self._returned)

    def get_unfinished_samples(self):
        """Returns the number of unfinished sample groups."""
        return len(self._paused)

    def get_prompt_num(self):
        return len(self._action2observations)

    def print(self):
        rollout_finished_count = len(self._returned)
        rollout_paused_count = len(self._paused)
        action_count = len(self._actions)
        observation_count = len(self._observations)

        log_message = (
            "[ReplayBuffer] ReplayBufferStorage states:\n"
            f"  - Rollout States: Returned={rollout_finished_count}, Paused={rollout_paused_count}\n"
            f"  - History Actions: {action_count}\n"
            f"  - History Observations: {observation_count}"
        )
        self.logger.info(log_message)

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
            if state_str == "paused":
                self._paused.append(action_id)
            elif state_str == "returned":
                self._returned.append(action_id)
            self._root2actions[root_id].append(action_id)
            self._actions[action_id] = replay_meta
            for observation_id in replay_meta.observation_ids:
                self._action2observations[action_id].append(observation_id)
                self._observations[observation_id] = replay_meta
                self._observations2states[observation_id] = replay_meta.state
                self._states[replay_meta.state].append(observation_id)

        self.logger.info(f"ReplayBufferStorage state successfully resumed from {file_path}")

        self.print()


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
        self.storage = ReplayBufferStorage(config.worker_log_dir)
        self.tokenizer = config.tokenizer
        self.datasets = build_datasets(config.dataset_cfg, self.tokenizer)

        if config.dataloader_cfg is not None:
            self.dataloader_cfg = config.dataloader_cfg
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

        self.sampler = Sampler(
            self.datasets,
            self.dataloader,
            self.tokenizer,
            self.storage,
        )
        self.post_processor_func = config.postprocessor_func

    def get_train_dataset_length(self):
        """Returns the length of the training dataloader."""
        return len(self.dataloader)

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

    def dump(self, file_path: str):
        """Dumps the replay buffer's storage to a file.

        Args:
            file_path (str): The path to the file for saving the data.
        """
        self.storage.dump(file_path)

    def resume(self, file_path: str):
        """Resumes the replay buffer's storage from a file.

        Args:
            file_path (str): The path to the file from which to restore the
                state.
        """
        self.storage.resume(file_path)
        num = self.storage.get_prompt_num()
        self.sampler.resume(num)

    def get_finished_samples(self):
        """Returns the number of finished sample groups in the storage."""
        return self.storage.get_finished_samples()

    def get_unfinished_samples(self):
        """Returns the number of unfinished sample groups in the storage."""
        return self.storage.get_unfinished_samples()
