import copy
import itertools
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from ray import ObjectRef
from typing_extensions import Annotated

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.config import DataloaderConfig
from xtuner.v1.datasets import build_dataloader, build_datasets
from xtuner.v1.datasets.data_item import RLTextDataItem
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
        DataloaderConfig, Parameter(help="The PyTorch DataLoader for iterating over the dataset.")
    ]

    tokenizer: Annotated[
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
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


@dataclass
class ReplayMeta:
    """A dataclass to store metadata for a single action-observation step in
    the replay buffer."""

    env: str = ""
    group_id: int = 0
    action_id: int = 0
    action_refs: List[ObjectRef] = field(default_factory=list)  # multi-turn action
    observation_ids: List[int] = field(default_factory=list)  # multi-turn action and partial rollout
    observation_refs: List[ObjectRef] = field(default_factory=list)  # multi-turn action and partial rollout
    observation_versions: List[int] = field(default_factory=list)  # partial rollout
    rewards: List[float] = field(default_factory=list)
    state: str = ""
    ground_truth: str = ""


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
        self.train_dataloader_iter = itertools.cycle(self.train_dataloader)
        self.tokenizer = tokenizer
        self.storage = storage

    def sample_from_datasets(self, env: str, repeat_prompt_k: int) -> List[RLTextDataItem]:
        """Samples a new group of prompts from the original dataset.

        Args:
            env (str): The environment name.
            repeat_prompt_k (int): The number of times to repeat the prompt.

        Returns:
            List[RLTextDataItem]: A list of data items for the data group contains repeat_prompt_k samples from same data.
        """
        group_id = uuid4().int
        group_samples: List[RLTextDataItem] = []
        try:
            data = next(self.train_dataloader_iter)[0]
        except StopIteration:
            return group_samples
        for _ in range(repeat_prompt_k):
            prompt_id = uuid4().int
            data_item = copy.deepcopy(data)
            data_item["env"] = env
            data_item["group_id"] = group_id
            data_item["prompt_id"] = prompt_id
            data_item["retry_times"] = 0
            group_samples.append(data_item)
        return group_samples

    def sample_from_unfinished_buffer(self):
        """Samples a prompt from a partially completed (unfinished) rollout."""
        prompt_id = self.storage._rollout_states["unfinished"].pop(0)
        group_replay_meta = [self.storage._actions[action_id] for action_id in self.storage._prompt2actions[prompt_id]]
        group_samples = []
        for replay_meta in group_replay_meta:
            latest_prompt = ray.get(replay_meta.action_refs[-1])
            latest_observation = ray.get(replay_meta.observation_refs[-1]) if replay_meta.observation_refs else ""
            messages = [
                {"role": "user", "content": f"{latest_prompt}"},
                {"role": "assistant", "content": f"{latest_observation}"},
            ]
            data_item = self.storage.replaymeta2dataitem(replay_meta, messages=messages)
            group_samples.append(data_item)
        return group_samples

    def sample(self, env: str, enable_partial_rollout: int, prompt_repeat_k: int) -> List[RLTextDataItem]:
        """Selects a sampling strategy and returns a group of samples.

        It decides whether to sample from the unfinished buffer (for partial
        rollouts greater than 0) or from the original dataset.

        Args:
            env (str): The environment name.
            enable_partial_rollout (int): Flag to enable partial rollout.
            prompt_repeat_k (int): Number of times to repeat the prompt.

        Returns:
            List[RLTextDataItem]: A list of sampled data items.
        """
        if (
            enable_partial_rollout
            and "unfinished" in self.storage._rollout_states
            and len(self.storage._rollout_states["unfinished"]) > 0
        ):
            return self.sample_from_unfinished_buffer()
        else:
            # note: Sample grouped sample at once. They share the same prompt and
            # prompt id but different action_id.
            return self.sample_from_datasets(env, prompt_repeat_k)


class ReplayBufferStorage:
    """Handles the storage of experiences for the replay buffer."""

    def __init__(self):
        """Initializes the data structures for storing replay data."""
        self._states: Dict[str, List[int]] = defaultdict(list)  # str: [observation_id, observation_id, ...]
        self._rollout_states: Dict[str, List[int]] = defaultdict(
            list
        )  # str: [group_id, group_id, ...], designed for partial rollout
        self._actions: Dict[int, ReplayMeta] = {}  # action_id: ReplayMeta
        self._observations: Dict[int, ReplayMeta] = {}  # observation_id: ReplayMeta
        self._prompt2actions: Dict[int, List[int]] = defaultdict(list)  # group_id: [action_id, action_id, ...]
        self._action2observations: Dict[int, List[int]] = defaultdict(
            list
        )  # action_id: [observation_id, observation_id, ...]
        self._observations2states: Dict[int, str] = {}  # observation_id: state_str
        self.logger = get_logger()

    def replaymeta2dataitem(self, replay_meta: ReplayMeta, messages=None, input_ids=None) -> RLTextDataItem:
        """Converts a ReplayMeta object to an RLTextDataItem.

        Args:
            replay_meta: The ReplayMeta object.
            prompt_str (Optional[str]): The prompt string. If None, it's
                retrieved from the replay_meta.
            input_ids (Optional[list]): The input IDs. Defaults to an empty list.

        Returns:
            RLTextDataItem: The converted data item.
        """
        messages = messages or (
            ray.get(replay_meta.action_refs[-1])
            if replay_meta.action_refs and len(replay_meta.action_refs) > 0
            else ""
        )
        input_ids = input_ids or []
        num_tokens = len(input_ids)
        response_str = (
            ray.get(replay_meta.observation_refs[0])
            if replay_meta.observation_refs and len(replay_meta.observation_refs) > 0
            else ""
        )
        return RLTextDataItem(
            env=replay_meta.env,
            group_id=replay_meta.group_id,
            prompt_id=replay_meta.action_id,
            messages=messages,
            input_ids=input_ids,
            num_tokens=num_tokens,
            response_str=response_str,
            reward_model={"ground_truth": replay_meta.ground_truth},
            reward=replay_meta.rewards[-1] if replay_meta.rewards and len(replay_meta.rewards) > 0 else None,
            state=replay_meta.state,
        )

    def dataitem2replaymeta(self, data_item: RLTextDataItem) -> ReplayMeta:
        """Converts an RLTextDataItem to a ReplayMeta object.

        Args:
            data_item: The RLTextDataItem to convert.

        Returns:
            ReplayMeta: The converted metadata object.
        """
        return ReplayMeta(
            env=data_item["env"],
            group_id=data_item["group_id"],
            action_id=data_item["prompt_id"],
            action_refs=[ray.put(data_item["messages"])] if "messages" in data_item else [],
            observation_ids=[uuid.uuid4().int],
            observation_refs=[ray.put(data_item["response_str"])] if "response_str" in data_item else [],
            observation_versions=[1],
            state=data_item["state"] if "state" in data_item else "",
            ground_truth=data_item["reward_model"]["ground_truth"],
            rewards=[data_item["reward"]] if "reward" in data_item and data_item["reward"] is not None else [],
        )

    def add(self, grouped_dataitem: List[RLTextDataItem]):
        """Adds a group of data items to the storage.

        Args:
            grouped_dataitem (List[RLTextDataItem]): A list of data items
                belonging to the same group.
        """
        if len(grouped_dataitem) == 0:
            return

        rollout_state = (
            "unfinished" if any(data_item["state"] == "unfinished" for data_item in grouped_dataitem) else "finished"
        )
        group_id = grouped_dataitem[0]["group_id"]
        self._rollout_states[rollout_state].append(group_id)

        for data_item in grouped_dataitem:
            replay_meta = self.dataitem2replaymeta(data_item)
            group_id = replay_meta.group_id
            action_id = replay_meta.action_id
            if action_id not in self._prompt2actions[group_id]:
                self._prompt2actions[group_id].append(action_id)
            self._actions[action_id] = replay_meta
            for observation_id in replay_meta.observation_ids:
                if observation_id not in self._states[replay_meta.state]:
                    self._states[replay_meta.state].append(observation_id)
                if observation_id not in self._action2observations[action_id]:
                    self._action2observations[action_id].append(observation_id)
                self._observations[observation_id] = replay_meta
                self._observations2states[observation_id] = replay_meta.state

    def get(self, global_batch_size: int) -> List[List[RLTextDataItem]]:
        """Retrieves a batch of finished sample groups from the buffer.

        Args:
            global_batch_size (int): The number of sample groups to retrieve.

        Raises:
            ValueError: If there are not enough finished samples in the buffer
                to meet the `global_batch_size`.

        Returns:
            List[List[RLTextDataItem]]: A list of sample groups. Each inner
            list contains a group of data items that were generated from the
            same initial prompt, repeated `repeat_prompt_k` times.
        """
        samples = []
        if len(self._rollout_states["finished"]) < global_batch_size:
            raise ValueError("Not enough finished samples in replay buffer")
        else:
            target_finished_list = self._rollout_states["finished"][:global_batch_size]
            remain_finished_list = self._rollout_states["finished"][global_batch_size:]
            for group_id in target_finished_list:
                group_replay_meta = [self._actions[action_id] for action_id in self._prompt2actions[group_id]]
                group_samples = [self.replaymeta2dataitem(replay_meta) for replay_meta in group_replay_meta]
                samples.append(group_samples)
            self._rollout_states["finished"] = remain_finished_list
            return samples

    def print(self):
        """Prints the current state and statistics of the storage."""
        self.logger.info("ReplayBufferStorage states: ")
        self.logger.info(
            f"rollout_state: finished: {len(self._rollout_states['finished'])}, unfinished: {len(self._rollout_states['unfinished'])}"
        )
        self.logger.info(f"group: {len(self._prompt2actions)}")
        self.logger.info(f"action: {len(self._actions)}")
        self.logger.info(f"observations: {len(self._observations)}")

    def dump(self):
        """Dumps the state of the replay buffer to a file.

        (Not implemented)
        """
        pass

    def get_finished_samples(self):
        """Returns the number of finished sample groups."""
        return len(self._rollout_states["finished"])

    def get_unfinished_samples(self):
        """Returns the number of unfinished sample groups."""
        return len(self._rollout_states["unfinished"])


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
        self.storage = ReplayBufferStorage()
        self.tokenizer = config.tokenizer
        self.datasets = build_datasets(config.dataset_cfg, self.tokenizer)

        self.dataloader = build_dataloader(
            dataloader_config=config.dataloader_cfg,
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

    def sample(self, env, enable_partial_rollout: int, prompt_repeat_k: int):
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

    def add(self, grouped_dataitem: List[RLTextDataItem]):
        """Adds a group of data items to the replay buffer storage.

        Args:
            grouped_dataitem (List[RLTextDataItem]): A list of data items
                from the same group.
        """
        self.storage.add(grouped_dataitem)

    def print(self):
        """Prints the current state of the replay buffer storage."""
        self.storage.print()

    def dump(self):
        """Dumps the replay buffer state.

        (Not implemented)
        """
        # todo: support dump replaybuffer
        self.storage.dump()

    def get_finished_samples(self):
        """Returns the number of finished sample groups in the storage."""
        return self.storage.get_finished_samples()

    def get_unfinished_samples(self):
        """Returns the number of unfinished sample groups in the storage."""
        return self.storage.get_unfinished_samples()
