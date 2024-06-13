import time
from copy import deepcopy

import torch
from loguru import logger
from torch.utils.data import IterableDataset

from ..model_server.base_model_server import BaseModelServer
from .prompt_utils import META_PROMPT


class TxtEnv:
    """A generic RL environment to generate textual sequences."""

    def __init__(
        self,
        dataloader: IterableDataset,
        max_new_tokens: int = 1024,
        actor_micro_bs: int = 32,
        reward_micro_bs: int = 32,
        reward_function: BaseModelServer = None,
        async_reward: bool = True,
        generate_kwargs: dict = None,
        **_ignored,
    ):
        """
        Args:
            dataloader (IterableDataset): generate rl data iteratively
            reward_function: reward function that computes scalar reward for each episode  # noqa: E501
        """
        self.dataloader: IterableDataset = iter(dataloader)
        self.reward_function: BaseModelServer = reward_function
        self._cur_messagess = []
        self.max_new_tokens = max_new_tokens
        self.actor_micro_bs = actor_micro_bs
        self.reward_micro_bs = reward_micro_bs
        self.async_reward = async_reward
        self.generate_kwargs: dict = generate_kwargs

    def rollout(self, policy_model: BaseModelServer, display=False):
        sample_data = deepcopy(next(self.dataloader))
        prompt_input_messages = []
        pretrain_input_messages = []
        for data in sample_data:
            if data.sys_meta != 'default':
                message = deepcopy([{
                    'role': 'system',
                    'content': META_PROMPT[data.sys_meta]
                }] + data.message)
            else:
                message = deepcopy(data.message)
            if data.mes_type == 'prompt':
                prompt_input_messages.append(message)
            elif data.mes_type == 'pretrain':
                pretrain_input_messages.append(message)
            else:
                raise TypeError(f'Wrong message type {data.mes_type}')
        # prompt data
        s_t = time.time()
        print(f'[For Generate]: {prompt_input_messages[0]}')
        trajectories = policy_model.generate(
            inputs=prompt_input_messages,
            micro_batch_size=self.actor_micro_bs,
            step=self.max_new_tokens,
            output_str=True,
            generate_kwargs=self.generate_kwargs)
        logger.info(
            f'[actor generate] duration: {round(time.time() - s_t, 2)} s, len(inputs): {len(prompt_input_messages)} '  # noqa: E501
        )

        if self.async_reward:
            reward_output_ref = self.get_reward_async(sample_data,
                                                      trajectories)
            trajectories['reward_output_ref'] = reward_output_ref
        else:
            rewards = self.get_reward(sample_data, trajectories)
            trajectories['rewards'] = rewards

        # pretrain data
        if len(pretrain_input_messages) > 0:
            from ..tokenizer import tokenizer_utils
            pretrain_input_ids, pretrain_attention_mask = tokenizer_utils.encode(
                pretrain_input_messages, policy_model.tokenizer)
            pretrain_labels = torch.nn.functional.pad(pretrain_input_ids[:, 1:], (0, 1), mode="constant", value=-100)

            trajectories.pretrain_data = {"input_ids": pretrain_input_ids,
                                          "labels": pretrain_labels,
                                          "attention_mask": pretrain_attention_mask}
            print(
                f'[TxtEnv & {policy_model.__class__.__name__}] gets {len(pretrain_input_messages)} pretrain episodes.'  # noqa: E501
            )
        else:
            trajectories.pretrain_data = None

        return trajectories

    # default get_reward() is blocking. get_reward_async() needs to call get_reward_collect()  # noqa: E501
    def get_reward_async(self, sample_data, policyout):
        s_t = time.time()
        rm_input_messages = []
        for i in range(len(sample_data)):
            if sample_data[i].mes_type != "prompt":
                continue
            if sample_data[i].rm_meta != 'default':
                cur_rm_data = [{
                    'role': 'system',
                    'content': META_PROMPT[sample_data[i].rm_meta]
                }] + sample_data[i].message + [{
                    'role':
                    'assistant',
                    'content':
                    policyout.output_ans_str[i]
                }]
            else:
                cur_rm_data = sample_data[i].message + [{
                    'role':
                    'assistant',
                    'content':
                    policyout.output_ans_str[i]
                }]
            rm_input_messages.append(cur_rm_data)

        print(f'[For Reward]: {rm_input_messages[0]}')
        reward_output_ref = self.reward_function.infer_async(
            rm_input_messages,
            output_logprobs=False,
            micro_batch_size=self.reward_micro_bs)
        logger.info(
            f'[reward infer] async duration: {round(time.time() - s_t, 2)} s')
        return reward_output_ref

    def get_reward_collect(self, reward_output_ref):
        s_t = time.time()
        rm_out = self.reward_function.infer_get(reward_output_ref)
        logger.info(
            f'[reward infer] async wait duration: {round(time.time() - s_t, 2)} s'  # noqa: E501
        )
        rewards = rm_out.logits.squeeze(-1)
        return rewards

    def get_reward(self, sample_data, policyout):
        s_t = time.time()
        rm_input_messages = []
        for i in range(len(sample_data)):
            if sample_data[i].mes_type != "prompt":
                continue
            if sample_data[i].rm_meta != 'default':
                cur_rm_data = [{
                    'role': 'system',
                    'content': META_PROMPT[sample_data[i].rm_meta]
                }] + sample_data[i].message + [{
                    'role':
                    'assistant',
                    'content':
                    policyout.output_ans_str[i]
                }]
            else:
                cur_rm_data = sample_data[i].message + [{
                    'role':
                    'assistant',
                    'content':
                    policyout.output_ans_str[i]
                }]
            rm_input_messages.append(cur_rm_data)

        print(f'[For Reward]: {rm_input_messages[0]}')
        rm_out = self.reward_function.infer(
            rm_input_messages,
            output_logprobs=False,
            micro_batch_size=self.reward_micro_bs)
        logger.info(
            f'[reward infer] duration: {round(time.time() - s_t, 2)} s')
        rewards = rm_out.logits.squeeze(-1)
        return rewards
