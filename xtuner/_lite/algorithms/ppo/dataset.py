# Copyright (c) OpenMMLab. All rights reserved.
import json

import numpy as np
import torch
from torch import nn

from xtuner._lite.chat.messages.chat import ChatMsg
from xtuner._lite.datasets import OPENAI_CONVERT_MAP

from ..sft import SftCollator, SftTokenizeFunction


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, responses):
        super().__init__()

        assert len(prompts) == len(responses)
        self.prompts = prompts
        self.responses = responses
        self.policies = None

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        response = self.responses[item]
        num_prefill_tokens = len(prompt)

        input_ids = prompt + response
        labels = [-100] * (num_prefill_tokens - 1) + response + [-100]

        return {"input_ids": input_ids, "labels": labels, "num_tokens": len(input_ids)}


FASTER = False


class RewardBuffer(torch.utils.data.Dataset):
    def __init__(self, clip_min=-5, clip_max=5, normalize=True, faster=False):
        super().__init__()

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.normalize = normalize

        if self.normalize:
            self.bn = nn.BatchNorm1d(1, momentum=None, affine=False)
        else:
            self.bn = None

        self._num_action_tokens = 0
        self._num_total_tokens = 0
        self._trajectories = []

        self._current_mean = 0

    @property
    def running_mean(self):
        return self.bn.running_mean.item()

    @property
    def current_mean(self):
        return self._current_mean

    @property
    def num_action_tokens(self):
        return self._num_action_tokens.item()

    @property
    def num_total_tokens(self):
        return self._num_total_tokens

    def update(self, trajectories):
        rewards = [data["reward"] for data in trajectories]

        for i in range(len(trajectories)):
            trajectories[i]["ori_reward"] = trajectories[i]["reward"]

        rewards = torch.tensor(rewards)

        self._current_mean = rewards.mean().item()

        rewards = rewards.clip(self.clip_min, self.clip_max)

        if self.normalize:
            self.bn.train()
            _ = self.bn(rewards.unsqueeze(-1))
            self.bn.eval()
            rewards = self.bn(rewards.unsqueeze(-1))

        for i in range(len(trajectories)):
            trajectories[i]["reward"] = rewards[i].item()

        num_total_tokens = 0
        num_action_tokens = 0
        for data in trajectories:
            labels = np.array(data["labels"])
            num_total_tokens += labels.size
            num_action_tokens += (labels >= 0).sum()

        self._num_action_tokens = num_action_tokens
        self._num_total_tokens = num_total_tokens

        self._trajectories = trajectories

    def dump_jsonl(self, path, tokenizer, debug=False):
        with open(path, "w", encoding="utf8") as f:
            for data in self._trajectories:
                json_line = {
                    "num_tokens": data["num_tokens"],
                    "reward": data["ori_reward"],
                    "sequence": tokenizer.decode(data["input_ids"]),
                }

                if debug:
                    json_line["input_ids"] = data["input_ids"]
                    json_line["labels"] = data["labels"]

                json_str = json.dumps(json_line, ensure_ascii=False)
                f.write(json_str + "\n")

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, item):
        return self._trajectories[item]


class PPOTokenizeFunction(SftTokenizeFunction):
    def __init__(self, tokenizer, chat_template, raw_format="openai", sys_prompt=None):
        super().__init__(tokenizer, chat_template, raw_format)
        self.sys_prompt = sys_prompt

    def __call__(self, item):
        formatter = OPENAI_CONVERT_MAP[self.raw_format]
        msg = formatter(item)
        if self.sys_prompt is not None:
            sys_msg = ChatMsg(role="system", content=self.sys_prompt)
            msg.messages = [sys_msg] + msg.messages
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)

        return tokenized


class RewardBufferCollator(SftCollator):
    def __call__(self, instances):
        data = super().__call__(instances)
        data["rewards"] = [item["reward"] for item in instances]

        return data
