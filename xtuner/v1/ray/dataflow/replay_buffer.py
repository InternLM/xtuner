import itertools
import random
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


@dataclass
class ReplayMeta:
    env: str
    group_id: int
    action_id: int
    action_refs: List[ObjectRef]  # multi-turn action
    observation_ids: List[int]  # multi-turn action and partial rollout
    observation_refs: List[ObjectRef]  # multi-turn action and partial rollout
    observation_versions: List[int]  # partial rollout
    state: str = ""
    ground_truth: str = ""


@dataclass
class SampleMeta:
    env: str
    prompt_id: int
    action_id: int
    prompt: str
    ground_truth: str
    response: str = ""
    reward: float = 0.0
    retry_times: int = 0
    state: str = ""

    def update(self, response: str = "", reward: float = 0.0, state: str = ""):
        self.response = response
        self.reward = reward
        self.state = state


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
            data["env"] = env
            data["group_id"] = group_id
            data["prompt_id"] = prompt_id
            data["retry_times"] = 0
            group_samples.append(data)
        return group_samples

    def sample_from_unfinished_buffer(self):
        # transform replaymeta to samplemeta
        prompt_id = self.storage._rollout_states["unfinished"].pop(0)
        group_replay_meta = self.storage._prompt2actions[prompt_id]
        group_samples = []
        for replay_meta in group_replay_meta:
            if replay_meta.state == "unfinished":
                latest_prompt = ray.get(replay_meta.action_refs[-1])
                latest_observation = ray.get(replay_meta.observation_refs[-1]) if replay_meta.observation_refs else ""
                messages = [
                    {"role": "user", "content": f"{latest_prompt}"},
                    {"role": "assistant", "content": f"{latest_observation}"},
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = model_inputs.pop("input_ids")[0]
                num_tokens = len(input_ids)
                data_item = {
                    "input_ids": input_ids,
                    "prompt_str": prompt,
                    "num_tokens": num_tokens,
                    "reward_model": {"ground_truth": replay_meta.reward},
                    "state": replay_meta.state,
                    "retry_times": 0,
                }
                group_samples.append(data_item)
        return group_samples

    def sample_from_replay_weights(self, replay_ratio, replay_weights):
        # todo: support repeat_prompt_k
        items = list(replay_weights.keys())
        probs = list(replay_weights.values())
        selected = random.choices(items, weights=probs, k=1)[0]
        tid = self._states[selected].pop(0)
        group_replay_meta = self.storage._prompt2actions[tid]
        group_samples = []
        for replay_meta in group_replay_meta:
            sample_meta = SampleMeta(
                prompt_id=replay_meta.prompt_id,
                action_id=replay_meta.action_ids[-1],
                prompt=ray.get(replay_meta.action_refs[-1]),
                ground_truth=replay_meta.ground_truth,
            )
            group_samples.append(sample_meta)
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
            and "unfinished" in self.storage._states
            and len(self.storage._states["unfinished"]) > 0
        ):
            return self.sample_from_unfinished_buffer()
        elif replay_weights and random.random() < replay_ratio:
            return self.sample_from_replay_weights(replay_ratio, replay_weights)
        else:
            # note: Sample grouped sample at once. They share the same prompt and
            # prompt id but different action_id.
            return self.sample_from_datasets(env, prompt_repeat_k)


class ReplayBufferStorage:
    def __init__(self):
        self._states: Dict[str, List[ReplayMeta.observation_id]] = defaultdict(list)
        self._rollout_states: Dict[str, List[ReplayMeta.prompt_id]] = defaultdict(list)  # designed for partial rollout
        self._actions: Dict[ReplayMeta.action_id, ReplayMeta] = {}
        self._observations: Dict[ReplayMeta.observation_id, ReplayMeta] = {}
        self._prompt2actions: Dict[ReplayMeta.prompt_id, List[ReplayMeta.action_ids]] = defaultdict(
            list
        )  # 1: prompt_repeat_k
        self._action2observations: Dict[ReplayMeta.action_id, List[ReplayMeta.observation_id]] = defaultdict(
            list
        )  # 1:n
        self._observations2states: Dict[ReplayMeta.observation_id, str] = {}  # 1:1

    def construct_from_rldata(self, grouped_metadata: List[RLTextDataItem]) -> List[ReplayMeta]:
        env = grouped_metadata[0]["env"]
        group_id = grouped_metadata[0]["group_id"]
        ground_truth = grouped_metadata[0]["reward_model"]["ground_truth"]
        state = grouped_metadata[0]["state"]

        group_replay_meta = []
        for metadata in grouped_metadata:
            action_id = metadata["prompt_id"]
            action_refs = ray.put(metadata["prompt_str"])
            observation_ids = [uuid.uuid4().int]
            observation_refs = [ray.put(metadata["response_str"])]
            replay_meta = ReplayMeta(
                env=env,
                group_id=group_id,
                action_id=action_id,
                action_refs=action_refs,
                observation_ids=observation_ids,
                observation_refs=observation_refs,
                observation_versions=[1] * len(observation_ids),
                state=state,
                ground_truth=ground_truth,
            )
            group_replay_meta.append(replay_meta)
        return group_replay_meta

    def add(self, grouped_metadata: List[RLTextDataItem]):
        if len(grouped_metadata) == 0:
            return

        group_replay_meta = self.construct_from_rldata(grouped_metadata)
        prompt_id = group_replay_meta[0].group_id
        rollout_state = "unfinished" if any(meta.state == "unfinished" for meta in group_replay_meta) else "finished"
        self._rollout_states[rollout_state].append(prompt_id)

        for replay_meta in group_replay_meta:
            state = replay_meta.state
            action_id = replay_meta.action_id
            self._prompt2actions[prompt_id].append(action_id)
            self._actions[action_id] = replay_meta
            for observation_id in replay_meta.observation_ids:
                self._observations[observation_id] = replay_meta
                self._observations2states[observation_id] = state
                self._states[state].append(observation_id)
                self._action2observations[action_id].append(observation_id)

    def finished_rollout_samples(self):
        return len(self._rollout_states["finished"])


@ray.remote
class ReplayBuffer:
    def __init__(
        self,
        dataset: "JsonlDataset",
        dataloader: "DataLoader",
        post_processor_func: Optional[Callable[..., Any]] = None,
    ):
        self.storage = ReplayBufferStorage()
        self.sampler = Sampler(dataset, dataloader, self.storage, tokenizer=None)
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

    def add(self, metadata: List[RLTextDataItem]):
        self.storage.add(metadata)
