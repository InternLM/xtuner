import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
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
    RLEnvDataItem,
    RLExtraDataItem,
    RLUIDItem,
    check_valid_dataflow_item,
)
from xtuner.v1.datasets import build_dataloader, build_datasets
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.utils import get_logger


logger = get_logger()


class ReplayState(str, Enum):
    INIT = "init"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    ARCHIVED = "archived"
    EXPIRED = "expired"  # only in replay buffer
    SKIPPED = "skipped"  # only in dataflow

    @staticmethod
    def from_str(state_str: str) -> "ReplayState":
        for state in ReplayState:
            if state.value == state_str:
                return state
        raise ValueError(f"Unknown ReplayState string: {state_str}")


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
    action_id: int = 0
    action_ref: ObjectRef = None
    observation_ids: List[int] = field(default_factory=list)
    observation_refs: List[ObjectRef] = field(default_factory=list)
    observation_versions: List[int] = field(default_factory=list)
    state: ReplayState = ReplayState.INIT
    version: int = 0
    extra_info: Dict[str, Any] = field(default_factory=dict)


def determine_group_state(group_data_items: List[RLDataFlowItem]) -> str:
    """Determines the processing strategy for a group of rollout samples based
    on their state."""
    # TODO(@duanyanhui): remove this function when send one request instead of group requests.
    if not group_data_items or len(group_data_items) == 0:
        return "skipped"
    group_states = {item.env.rollout.state for item in group_data_items}
    if "skipped" in group_states or "failed" in group_states:
        return "skipped"
    elif "interrupted" in group_states:
        return "interrupted"
    elif all(state == "completed" for state in group_states):
        return "completed"
    else:
        raise ValueError(f"Unknown group states: {group_states}")


def mapping_dataitem_to_replaymeta(grouped_dataitem: List[RLDataFlowItem]) -> ReplayMeta:
    # TODO: 单独管理每一条query，而不是一组query，提高效率
    assert len(grouped_dataitem) > 0

    total_response_len = 0
    total_prefill_len = 0
    env_str = grouped_dataitem[0].uid.env
    root_id = grouped_dataitem[0].uid.root_id
    action_id = grouped_dataitem[0].uid.action_id
    data = grouped_dataitem[0].data
    group_rollout_finish_reason = []
    observation_ids = []
    observation_refs = []
    observation_versions = []

    for item in grouped_dataitem:
        version = item.uid.version
        observation_ids.append(item.uid.observation_id)
        total_response_len += (
            len(item.env.rollout.async_response_ids[-1]) if item.env.rollout.async_response_ids else 0
        )
        total_prefill_len += item.data.extra_info.get("prefill_token_len", 0)
        observation_refs.append(ray.put(item.env))
        observation_versions.append(version)
        group_rollout_finish_reason.append(item.env.rollout.finish_reason)

    version = max(observation_versions)

    group_state = determine_group_state(grouped_dataitem)

    replay_state = ReplayState.from_str(group_state)
    logger.debug(f"determined group_state: {group_state}, replay_state: {replay_state}, version: {version}")
    logger.debug(f"total_response_len: {total_response_len} for action_id: {action_id}")
    replay_meta = ReplayMeta(
        env=env_str,
        root_id=root_id,
        action_id=action_id,
        action_ref=ray.put(data),
        observation_ids=observation_ids,
        observation_refs=observation_refs,
        observation_versions=observation_versions,
        state=replay_state,
        version=version,
        extra_info={"response_tokens": total_response_len, "prefill_tokens": total_prefill_len},
    )
    return replay_meta


def mapping_replaymeta_to_dataitem(replay_meta: ReplayMeta) -> List[RLDataFlowItem]:
    env_str = replay_meta.env
    root_id = replay_meta.root_id
    action_id = replay_meta.action_id
    data_ref = ray.get(replay_meta.action_ref)
    state_str = str(replay_meta.state)
    group_data_item = []
    for obs_id, obs_ref, version in zip(
        replay_meta.observation_ids, replay_meta.observation_refs, replay_meta.observation_versions
    ):
        item = RLDataFlowItem(
            uid=RLUIDItem(
                env=env_str, root_id=root_id, action_id=action_id, observation_id=obs_id, version=replay_meta.version
            ),
            extra_info=RLExtraDataItem(state=state_str, retry_times=0),
        )
        if data_ref is not None:
            item.data = data_ref
        if obs_ref is not None:
            item.env = ray.get(obs_ref)
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


class DatasetSampler:
    """Sampler for drawing prompts from datasets or the replay buffer."""

    def __init__(self, dataset_cfg, dataloader_cfg, tokenizer):
        """Initializes the Sampler.

        Args:
            dataset: The dataset to sample from.
            dataloader: The dataloader for the dataset.
            tokenizer: The tokenizer for processing text.
            storage: The ReplayBufferStorage instance.
        """
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
        start_time = time.monotonic()
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
            data_item.extra_info = RLExtraDataItem(state=str(ReplayState.INIT), retry_times=0)
        # self.logger.debug(f"Sample from dataloader with data: {data_item}")
        elapsed_time = time.monotonic() - start_time
        self.logger.debug(f"Sample each sample from dataloader cost: {elapsed_time:.2f} seconds.")
        return group_data_item

    def resume(self, num: int) -> None:
        self.dataloader_iter = itertools.islice(self.dataloader, num, None)


class ReplayBufferStorage:
    """Handles the storage of experiences for the replay buffer."""

    def __init__(self, worker_log_dir, partial_rollout_step: int = 0):
        """Initializes the data structures for storing replay data."""

        num_buckets = partial_rollout_step + 1
        self._completed_actions: List[List[int]] = [[] for _ in range(num_buckets)]
        self._interrupted_actions: List[List[int]] = [[] for _ in range(num_buckets)]
        self._expired_actions: List[int] = []

        self._actions: Dict[int, ReplayMeta] = {}
        self._root2actions: Dict[int, List[int]] = {}
        self._observations: Dict[int, ReplayMeta] = {}
        self._observations2states: Dict[int, str] = {}
        self._states: Dict[str, List[int]] = defaultdict(list)
        self._action2observations: Dict[int, List[int]] = defaultdict(list)
        self.logger = get_logger(log_dir=worker_log_dir, tag="ReplayBuffer")
        self._multimodal_train_infos: Dict[int, Dict[str, Any]] = {}

    def add(self, grouped_dataitem: List[RLDataFlowItem], partial_rollout_step: int = 0):
        """Adds a group of data items to the storage.

        Args:
            grouped_dataitem (List[RLDataFlowItem]): A list of data items
                belonging to the same group.
        """
        start_time = time.monotonic()
        if (
            grouped_dataitem is None
            or len(grouped_dataitem) == 0
            or check_valid_dataflow_item(grouped_dataitem) is False
        ):
            return

        replay_meta = mapping_dataitem_to_replaymeta(grouped_dataitem)
        root_id = replay_meta.root_id
        action_id = replay_meta.action_id

        # 1. 跟prompt相关的action_id记录
        if root_id in self._root2actions:
            # TODO: version 更新需要 根据是否update_weights来判断，需要考虑到非共卡的情况
            replay_meta.version += 1 if partial_rollout_step > 0 else 0
            self.logger.info(
                f"Existing root_id: {root_id} with action_id {action_id} found. Incrementing version to {replay_meta.version}."
            )
            self._root2actions[root_id].append(action_id)
        else:
            self._root2actions[root_id] = [action_id]

        self._actions[action_id] = replay_meta

        # 2. 根据rollout状态加到finished, abort, abort_over_version队列中；Partial rollout is handled based on whether finish_reason is "abort".
        if replay_meta.state == ReplayState.INTERRUPTED and (
            replay_meta.version < partial_rollout_step or partial_rollout_step == 0
        ):
            self._interrupted_actions[replay_meta.version].append(action_id)
            self.logger.debug(
                f"Add aborted sample with action_id: {action_id} version: {replay_meta.version} to _interrupted_actions."
            )
        elif replay_meta.state == ReplayState.INTERRUPTED and replay_meta.version >= partial_rollout_step:
            self._expired_actions.append(action_id)
            replay_meta.version = 0
            replay_meta.state = ReplayState.EXPIRED
            self.logger.info(
                f"Action_id: {action_id} has exceeded partial_rollout_step {partial_rollout_step}. Add this sample with root_id: {root_id} to _expired_actions list."
            )
        elif replay_meta.state == ReplayState.COMPLETED:
            self._completed_actions[replay_meta.version].append(action_id)
            self.logger.info(f"Add sample with root_id: {root_id}, action_id: {action_id} to finished_actions.")
        elif replay_meta.state == ReplayState.FAILED:
            assert False, "Currently, failed samples are not supported in the replay buffer."

        # 3. observation
        for observation_id in replay_meta.observation_ids:
            self._action2observations[action_id].append(observation_id)
            self._observations[observation_id] = replay_meta
            self._observations2states[observation_id] = replay_meta.state
            self._states[str(replay_meta.state)].append(observation_id)
        elapsed_time = time.monotonic() - start_time
        self.logger.debug(f"Replay Buffer add each sample cost: {elapsed_time:.2f} seconds.")

    def get(self, global_batch_size: int) -> Tuple[List[List[RLDataFlowItem]], List[Dict[str, Any]]]:
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
        target_batch_size = min(global_batch_size, self.get_completed_samples())
        self.logger.info(f"Retrieving {target_batch_size} completed samples from the replay buffer.")
        for _ in range(target_batch_size):
            action_id = self._pop_highest_version_action(self._completed_actions)
            replay_meta = self._actions[action_id]  # type: ignore[index]
            group_samples = mapping_replaymeta_to_dataitem(replay_meta)
            multimodal_train_info = None
            # TODO: 是否需要额外返回不重复的 multimodal_train_infos？
            for data_item in group_samples:
                if hasattr(data_item.data, "multimodal_train_info"):
                    multimodal_train_info = data_item.data.multimodal_train_info
                    del data_item.data.multimodal_train_info
            samples.append(group_samples)
            if multimodal_train_info is not None:
                multimodal_train_infos.append(multimodal_train_info)
        return samples, multimodal_train_infos

    def refresh_completed_states_on_step(self, sample_from_expired_states, partial_rollout_step):
        if sample_from_expired_states:
            for bucket in self._completed_actions:
                while bucket:
                    action_id = bucket.pop()
                    replay_meta = self._actions[action_id]
                    replay_meta.state = ReplayState.INTERRUPTED
                    replay_meta.version += 1
                    if replay_meta.version >= partial_rollout_step:
                        replay_meta.state = ReplayState.EXPIRED
                        self._expired_actions.append(action_id)
                    else:
                        self._interrupted_actions[replay_meta.version].append(action_id)
        else:
            updated_completed_buckets = [[] for _ in range(len(self._completed_actions))]
            for version_idx, bucket in enumerate(self._completed_actions):
                while bucket:
                    action_id = bucket.pop()
                    replay_meta = self._actions[action_id]
                    if partial_rollout_step > 0 and replay_meta.version >= partial_rollout_step:
                        replay_meta.state = ReplayState.EXPIRED
                        replay_meta.version = 0
                        self._expired_actions.append(action_id)
                    else:
                        updated_completed_buckets[version_idx].append(action_id)
            self._completed_actions = updated_completed_buckets

    def status(self):
        return {
            "rollout_completed_count": self.get_completed_samples(),
            "rollout_interrupted_count": self.get_interrupted_samples(),
            "rollout_expired_count": len(self._expired_actions),
            "prompt_count": len(self._root2actions),
            "action_count": len(self._actions),
            "observation_count": len(self._observations),
        }

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
            self.logger.info(f"state of replay_meta while dumping: {replay_meta}")
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

        self.logger.info(f"self._expired_actions: {len(self._expired_actions)} before resuming.")
        for group_data_items in all_data_items:
            replay_meta = mapping_dataitem_to_replaymeta(group_data_items)
            root_id = replay_meta.root_id
            action_id = replay_meta.action_id
            state = replay_meta.state
            version = replay_meta.version
            self.logger.info(f"state of replay_meta while resuming: {replay_meta}")
            if state == ReplayState.INTERRUPTED:
                self._interrupted_actions[version].append(action_id)
            elif state == ReplayState.EXPIRED:
                self._expired_actions.append(action_id)
            elif state == ReplayState.COMPLETED:
                self._completed_actions[version].append(action_id)
            if root_id not in self._root2actions:
                self._root2actions[root_id] = [action_id]
            else:
                self._root2actions[root_id].append(action_id)
            self._actions[action_id] = replay_meta
            for observation_id in replay_meta.observation_ids:
                self._action2observations[action_id].append(observation_id)
                self._observations[observation_id] = replay_meta
                self._observations2states[observation_id] = replay_meta.state
                self._states[replay_meta.state].append(observation_id)

        self.logger.info(f"self._expired_actions: {len(self._expired_actions)} after resuming.")
        self.logger.info(f"ReplayBufferStorage state successfully resumed from {file_path}")

    def clear(self):
        attrs_to_clear = [
            "_interrupted_actions",
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

    def get_completed_samples(self):
        return sum(len(bucket) for bucket in self._completed_actions)

    def get_interrupted_samples(self):
        return sum(len(bucket) for bucket in self._interrupted_actions)

    def sample_from_expired_storage(self) -> List[RLDataFlowItem]:
        start_time = time.monotonic()
        assert len(self._expired_actions) > 0
        action_id = self._expired_actions.pop()
        replay_meta = self._actions[action_id]
        replay_meta.version = 0
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)

        # update env for expired samples
        for sample in group_samples:
            # sample.data.input_ids = sample.data.input_ids[: sample.data.num_tokens]
            sample.env = RLEnvDataItem()
            sample.uid.version = 0
            sample.extra_info.state = str(ReplayState.INIT)

        self.logger.info(
            f"Sampling expired action_id: {action_id} from replay buffer, remain expired samples: {len(self._expired_actions)}"
        )
        elapsed_time = time.monotonic() - start_time
        self.logger.debug(f"Sample each sample from expired storage cost: {elapsed_time:.2f} seconds.")
        return group_samples

    def sample_from_interrupted_storage(self, tokenizer) -> List[RLDataFlowItem]:
        start_time = time.monotonic()
        assert self.get_interrupted_samples() > 0
        action_id = self._pop_highest_version_action(self._interrupted_actions)
        replay_meta = self._actions[action_id]  # type: ignore[index]
        group_samples = mapping_replaymeta_to_dataitem(replay_meta)

        # update env for interrupted samples
        for sample in group_samples:
            assert sample.data.input_ids and sample.data.num_tokens, "input_ids or num_tokens is empty!"
            # sample.data.input_ids = sample.data.input_ids[: sample.data.num_tokens]
            sample.uid.action_id = int(uuid4().int)
            sample.uid.version = replay_meta.version
            sample.extra_info.state = str(ReplayState.INIT)
            # if sample.env.rollout.response_ids and sample.data.input_ids:
            #     # TODO： response_ids 累加
            #     if "train_prompt_ids" in sample.data.extra_info:
            #         sample.data.input_ids = (
            #             sample.data.extra_info["train_prompt_ids"] + sample.env.rollout.response_ids
            #         )
            #     else:
            #         sample.data.input_ids.extend(sample.env.rollout.response_ids)
            # elif sample.env.rollout.response:
            #     sample.data.input_ids.extend(tokenizer.encode(sample.env.rollout.response, add_special_tokens=False))
        self.logger.info(
            f"Sampling interrupted action_id: {action_id} from replay buffer, remain interrupted samples: {self.get_interrupted_samples()}"
        )
        elapsed_time = time.monotonic() - start_time
        self.logger.debug(f"Sample each sample from interrupted storage cost: {elapsed_time:.2f} seconds.")
        return group_samples

    def _pop_highest_version_action(self, buckets: List[List[int]]) -> Optional[int]:
        for bucket in reversed(buckets):
            if bucket:
                return bucket.pop()
        return None

    def get_expired_tokens(self):
        response_tokens = 0
        prefill_tokens = 0
        for action_id in self._expired_actions:
            replay_meta = self._actions[action_id]
            response_tokens += replay_meta.extra_info.get("response_tokens", 0)
            prefill_tokens += replay_meta.extra_info.get("prefill_tokens", 0)
        return response_tokens, prefill_tokens

    def get_completed_tokens(self):
        response_tokens = 0
        prefill_tokens = 0
        for bucket in self._completed_actions:
            # self.logger.info(f"len(bucket): {len(bucket)}")
            for action_id in bucket:
                replay_meta = self._actions[action_id]
                response_tokens += replay_meta.extra_info.get("response_tokens", 0)
                prefill_tokens += replay_meta.extra_info.get("prefill_tokens", 0)
        return response_tokens, prefill_tokens

    def get_interrupted_tokens(self):
        response_tokens = 0
        prefill_tokens = 0
        for bucket in self._interrupted_actions:
            for action_id in bucket:
                replay_meta = self._actions[action_id]
                response_tokens += replay_meta.extra_info.get("response_tokens", 0)
                prefill_tokens += replay_meta.extra_info.get("prefill_tokens", 0)
        return response_tokens, prefill_tokens


@ray.remote
class ReplayBuffer:
    """A Ray actor that manages experience replay for reinforcement
    learning."""

    def __init__(
        self,
        config: ReplayBufferConfig,
        enable_partial_rollout: int = 0,
        partial_rollout_step: int = 0,
    ):
        """Initializes the ReplayBuffer actor.

        Args:
            config (ReplayBufferConfig): The configuration object.
        """
        self.config = config
        self.storage = ReplayBufferStorage(config.worker_log_dir, partial_rollout_step)
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        if isinstance(self.config.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = config.tokenizer

        self.sampler = DatasetSampler(
            config.dataset_cfg,
            config.dataloader_cfg,
            self.tokenizer,
        )
        self.post_processor_func = config.postprocessor_func
        self.partial_rollout_step = partial_rollout_step
        self.enable_partial_rollout = enable_partial_rollout
        self.sample_from_expired_count = 0
        self.sample_from_interrupted_count = 0
        self.sample_from_dataloader_count = 0
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="ReplayBuffer")

    # replay buffer related methods
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

    def sample(
        self, env, enable_partial_rollout: int, prompt_repeat_k: int, sample_from_expired_storage: bool
    ) -> List[RLDataFlowItem]:
        """Samples a batch of experiences from the replay buffer.

        Args:
            env: The environment name.
            enable_partial_rollout (int): Flag to enable partial rollouts.
            prompt_repeat_k (int): Number of times to repeat a prompt.

        Returns:
            A list of sampled data items.
        """
        if sample_from_expired_storage and self.get_expired_samples() > 0:
            self.sample_from_expired_count += 1
            return self.storage.sample_from_expired_storage()
        elif self.get_interrupted_samples() > 0:
            self.sample_from_interrupted_count += 1
            return self.storage.sample_from_interrupted_storage(self.tokenizer)
        else:
            self.sample_from_dataloader_count += 1
            return self.sampler.sample(env, prompt_repeat_k)

    def status(self):
        status = self.storage.status()
        status.update(
            {
                "sample_from_dataloader_count": self.sample_from_dataloader_count,
                "sample_from_interrupted_count": self.sample_from_interrupted_count,
                "sample_from_expired_count": self.sample_from_expired_count,
            }
        )
        return status

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
        status = self.storage.status()
        prompt_num = status["prompt_count"]
        self.sampler.resume(prompt_num)

    # dataset related methods
    def get_train_dataset_length(self):
        """Returns the length of the training dataloader."""
        return len(self.sampler.dataloader)

    # storage related methods
    def refresh_completed_states_on_step(self, sample_from_expired_states):
        self.storage.refresh_completed_states_on_step(sample_from_expired_states, self.partial_rollout_step)

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
        self.sample_from_dataloader_count = 0
        self.sample_from_interrupted_count = 0
        self.sample_from_expired_count = 0
        return self.storage.get(global_batch_size)

    def add(self, grouped_dataitem: List[RLDataFlowItem]):
        """Adds a group of data items to the replay buffer storage.

        Args:
            grouped_dataitem (List[RLDataFlowItem]): A list of data items
                from the same group.
        """
        self.storage.add(grouped_dataitem, self.partial_rollout_step)

    def get_expired_samples(self):
        return len(self.storage._expired_actions)

    def get_completed_samples(self):
        return self.storage.get_completed_samples()

    def get_interrupted_samples(self):
        return self.storage.get_interrupted_samples()

    def get_expired_tokens(self):
        return self.storage.get_expired_tokens()

    def get_completed_tokens(self):
        return self.storage.get_completed_tokens()

    def get_interrupted_tokens(self):
        return self.storage.get_interrupted_tokens()

    def get_generated_tokens(self):
        completed_response_tokens, completed_prefill_tokens = self.get_completed_tokens()
        interrupted_response_tokens, interrupted_prefill_tokens = self.get_interrupted_tokens()
        expired_response_tokens, expired_prefill_tokens = self.get_expired_tokens()
        total_response_tokens = completed_response_tokens + interrupted_response_tokens + expired_response_tokens
        total_prefill_tokens = completed_prefill_tokens + interrupted_prefill_tokens + expired_prefill_tokens

        return {
            "total_response_tokens": total_response_tokens,
            "total_prefill_tokens": total_prefill_tokens,
            "completed_response_tokens": completed_response_tokens,
            "completed_prefill_tokens": completed_prefill_tokens,
            "interrupted_response_tokens": interrupted_response_tokens,
            "interrupted_prefill_tokens": interrupted_prefill_tokens,
            "expired_response_tokens": expired_response_tokens,
            "expired_prefill_tokens": expired_prefill_tokens,
        }

    def clear(self):
        """Clears the replay buffer storage."""
        self.storage.clear()
