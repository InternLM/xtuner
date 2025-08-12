import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

import ray.util.queue
import torch
from ray import ObjectRef

from xtuner.v1.utils import get_logger


logger = get_logger()


@dataclass
class ReplayMeta:
    environment: str
    action_id: int
    action_ref: ObjectRef
    observation_ids: List[int] = field(default_factory=list)
    observation_refs: List[ObjectRef] = field(default_factory=list)
    state: str = ""
    version: List[int] = field(default_factory=list)

    def update_version(self, version: int):
        """更新版本号 :param version: 新的版本号."""
        if version not in self.version:
            self.version.append(version)
            self.version.sort()

    def update_observations(
        self, observation_id_list: Optional[list[int]], observation_ref_list: Optional[list[ObjectRef]]
    ):
        if observation_id_list is None and observation_ref_list is not None:
            for observe in observation_ref_list:
                self.observation_ids.append(uuid4().int)
                self.observation_refs.append(observe)
        else:
            raise ValueError("observation_ref must be provided.")

    def update(
        self,
        observation_id_list: Optional[list[int]] = None,
        observation_ref_list: Optional[list[ObjectRef]] = None,
        state: Optional[str] = None,
        version: Optional[int] = None,
    ):
        if state:
            self.state = state
        if version:
            self.update_version(version)
        if observation_id_list or observation_ref_list:
            self.update_observations(observation_id_list, observation_ref_list)


@ray.remote
class ReplayBuffer:
    def __init__(self, dataset):
        self.datasets: torch.utils.data.Dataset = dataset
        self._datasets_iter = iter(self.datasets)
        self._states: Dict[str, List[ReplayMeta.observation_id]] = {}
        self._actions: Dict[ReplayMeta.action_id, List[ReplayMeta]] = {}
        self._observations: Dict[ReplayMeta.observation_id, ReplayMeta] = {}
        self._action2observations: Dict[ReplayMeta.action_id, List[ReplayMeta.observation_id]] = {}  # 1:n
        self._observations2states: Dict[ReplayMeta.observation_id, str] = {}  # 1:1

    def get_current_size(self, states: List[str]) -> int:
        """获取当前缓冲区的大小 :return: 当前缓冲区的大小."""
        return sum(len(self._states[state]) for state in states if state in self._states)

    def add(self, metadata: ReplayMeta):
        """将数据添加到指定键的缓冲区 :param key: 目标键 :param value: 要添加的值."""
        # update self._actions
        handle_meta = metadata
        if metadata.action_id not in self._actions:
            self._actions[metadata.action_id] = [metadata]
        else:
            self._actions[metadata.action_id].append(metadata)

        action_id = handle_meta.action_id
        observation_id_list = handle_meta.observation_ids
        state = handle_meta.state

        # init states and action2observations
        if state not in self._states:
            self._states[state] = []
        if action_id not in self._action2observations:
            self._action2observations[action_id] = []

        # update _states, _observations, _action2observations, _observations2states
        for observation_id in observation_id_list:
            self._observations[observation_id] = handle_meta
            self._states[state].append(observation_id)
            self._action2observations[action_id].append(observation_id)
            self._observations2states[observation_id] = state

    def sample(self, replay_ratio: float, replay_weights: Dict[str, float]) -> Optional[ReplayMeta]:
        if random.random() < replay_ratio:
            states = self._states
            items = list(replay_weights.keys())
            probs = list(replay_weights.values())
            selected = random.choices(items, weights=probs, k=1)[0]
            tid = states[selected].pop(0)
            sampled = self._observations[tid]
        else:
            try:
                action = next(self._datasets_iter)
            except StopIteration:
                logger.warning("ReplayBuffer: dataset is exhausted!")
                return None  # 或 raise RuntimeError("Dataset is empty!")
            sampled = ReplayMeta(
                action_id=uuid4().int,
                observation_ids=[],
                action_ref=ray.put(action),
                observation_refs=[],
                environment="",
            )
        return sampled

    def get_observations(self):
        return [observation.observation_refs for observation in self._observations.values()]

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass
