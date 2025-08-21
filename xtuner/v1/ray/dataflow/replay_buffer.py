import copy
import itertools
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from uuid import uuid4

import ray.util.queue
from ray import ObjectRef

from xtuner.v1.datasets.data_item import RLTextDataItem


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from xtuner.v1.datasets import JsonlDataset

from xtuner.v1.utils import get_logger


@dataclass
class ReplayMeta:
    env: str
    group_id: int
    action_id: int
    action_refs: List[ObjectRef]  # multi-turn action
    observation_ids: List[int]  # multi-turn action and partial rollout
    observation_refs: List[ObjectRef]  # multi-turn action and partial rollout
    observation_versions: List[int]  # partial rollout
    rewards: List[int]
    state: str = ""
    ground_truth: str = ""


class Sampler:
    def __init__(self, dataset, dataloader, storage, tokenizer):
        self.datasets = dataset
        self.dataloader = dataloader
        self.dataloader_iter = itertools.cycle(self.dataloader)
        self.tokenizer = tokenizer
        self.storage = storage

    def sample_from_datasets(self, env: str, repeat_prompt_k: int) -> List[RLTextDataItem]:
        data = next(self.dataloader_iter)[0]
        group_id = uuid4().int
        group_samples = []
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
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            data_item = self.storage.replaymeta2dataitem(replay_meta, prompt_str=prompt)
            group_samples.append(data_item)
        return group_samples

    def sample(
        self,
        env: str,
        enable_partial_rollout: int,
        prompt_repeat_k: int,
        replay_ratio: float,
        replay_weights: dict,
    ) -> List[RLTextDataItem]:
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
    def __init__(self):
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

    def replaymeta2dataitem(self, replay_meta, prompt_str=None, input_ids=None) -> RLTextDataItem:
        prompt_str = prompt_str or ray.get(replay_meta.action_refs[-1])
        input_ids = input_ids or []
        num_tokens = len(input_ids)
        response_str = ray.get(replay_meta.observation_refs[0])
        return RLTextDataItem(
            env=replay_meta.env,
            group_id=replay_meta.group_id,
            prompt_id=replay_meta.action_id,
            prompt_str=prompt_str,
            input_ids=input_ids,
            num_tokens=num_tokens,
            response_str=response_str,
            reward_model={"ground_truth": replay_meta.ground_truth},
            reward=replay_meta.rewards[-1],
            state=replay_meta.state,
        )

    def dataitem2replaymeta(self, data_item) -> ReplayMeta:
        return ReplayMeta(
            env=data_item["env"],
            group_id=data_item["group_id"],
            action_id=data_item["prompt_id"],
            action_refs=[ray.put(data_item["prompt_str"])],
            observation_ids=[uuid.uuid4().int],
            observation_refs=[ray.put(data_item["response_str"])],
            observation_versions=[1],
            state=data_item["state"],
            ground_truth=data_item["reward_model"]["ground_truth"],
            rewards=[data_item["reward"]],
        )

    def add(self, grouped_dataitem: List[RLTextDataItem]):
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

        return len(self._rollout_states["finished"])

    def get(self, global_batch_size: int) -> List[List[RLTextDataItem]]:
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
        self.logger.info("ReplayBufferStorage states: ")
        self.logger.info(
            f"rollout_state: finished: {len(self._rollout_states['finished'])}, unfinished: {len(self._rollout_states['unfinished'])}"
        )
        self.logger.info(f"group: {len(self._prompt2actions)}")
        self.logger.info(f"action: {len(self._actions)}")
        self.logger.info(f"observations: {len(self._observations)}")

    def dump(self):
        pass


@ray.remote
class ReplayBuffer:
    def __init__(
        self,
        dataset: "JsonlDataset",
        dataloader: "DataLoader",
        tokenizer,
        post_processor_func: Optional[Callable[..., Any]] = None,
    ):
        self.storage = ReplayBufferStorage()
        self.sampler = Sampler(dataset, dataloader, self.storage, tokenizer)
        self.post_processor_func = post_processor_func

    def post_processor(self, group_samples):
        if self.post_processor_func:
            group_samples = self.post_processor_func(group_samples)
            return group_samples
        return group_samples

    def sample(
        self,
        env,
        enable_partial_rollout: int,
        prompt_repeat_k: int,
        replay_ratio: float,
        replay_weights: dict,
    ):
        return self.sampler.sample(env, enable_partial_rollout, prompt_repeat_k, replay_ratio, replay_weights)

    def get_samples(
        self,
        global_batch_size: int,
    ):
        return self.storage.get(global_batch_size)

    def add(self, grouped_dataitem: List[RLTextDataItem]):
        return self.storage.add(grouped_dataitem)

    def print(self):
        self.storage.print()

    def dump(self):
        # todo: support dump replaybuffer
        self.storage.dump()
