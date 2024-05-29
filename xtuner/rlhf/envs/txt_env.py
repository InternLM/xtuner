import time
from copy import deepcopy

import torch
from loguru import logger
from torch.utils.data import IterableDataset

from ..model_server.base_model_server import BaseModelServer

META_PROMPT = {
    'default':
    '',
    'helpful':
    'Consider the helpfulness of the given sentence. '
    'A helpful sentence is one that provides value, offers insight, or assists the user in some way. '  # noqa: E501
    'It may solve a problem, answer a question, provide relevant information, or offer guidance. '  # noqa: E501
    "It should directly address the user's needs or inquiries and is ideally actionable, "  # noqa: E501
    'meaning the user can use the information to make decisions or take steps towards their goal. '  # noqa: E501
    'The score should be higher for sentences that are more beneficial and informative to the user. '  # noqa: E501
    'Here is the conversation:\n\n',
    'harmless':
    'Evaluate the harmlessness of the provided sentence. '
    'A harmless sentence is one that does not offend, insult, or cause distress to the user, '  # noqa: E501
    "and does not mislead or give false information. It respects the user's feelings, beliefs, "  # noqa: E501
    'and values, and does not provoke conflict or negative emotions. '
    'Additionally, a harmless sentence should not incite harmful actions or behaviors. '  # noqa: E501
    'It should uphold the principles of respect, honesty, and fairness. '
    'Sentences that are more considerate, respectful, and accurate should receive higher scores on this scale. '  # noqa: E501
    'Here is the conversation:\n\n',
    'hhh':
    'Evaluate the given sentence based on its helpfulness, harmlessness, and honesty. '  # noqa: E501
    'A helpful sentence provides value, solves problems, or aids the user in an actionable manner. '  # noqa: E501
    "A harmless sentence respects the user's feelings and beliefs, avoids causing distress, "  # noqa: E501
    'and does not mislead or incite harmful behaviors. An honest sentence delivers reliable and true information, '  # noqa: E501
    'presents facts objectively, and demonstrates integrity and authenticity. Higher scores should be assigned '  # noqa: E501
    'to sentences that embody these characteristics more strongly. '
    'Here is the conversation:\n\n',
    'summarization':
    'As a language model performing a summarization task, your goal is to generate a summary that '  # noqa: E501
    'accurately, succinctly, and coherently encapsulates the key details of the source text. Ensure relevance to '  # noqa: E501
    'the original material, completeness of main points, and logical structure. Maintain conciseness and high '  # noqa: E501
    'linguistic standards. Ensure only the summary is outputted, refraining from adding extraneous comments or '  # noqa: E501
    'remarks. Here is the original material:\n\n',
    'reddit':
    'Imagine you are a knowledgeable and friendly Reddit user. '
    'A fellow Redditor has just shared a post seeking feedback, advice, or input. '  # noqa: E501
    'Please read the post and provide a thoughtful, informative, and respectful response, '  # noqa: E501
    'just as if you were replying on the platform. Here is the post:\n\n',
    'latex':
    'When mathematical content appears in the conversation, please use latex format to express the mathematical content. Here is the conversation:\n\n',  # noqa: E501
    'math_ci':
    "Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:\n- Just write jupyter code to solve the problem without giving your thought;\n- Present the final result in LaTeX using a '\\boxed\\{{}}' without any units. \n",  # noqa: E501
}


class TxtEnv:
    """A generic RL environment to generate textual sequences."""

    def __init__(
        self,
        dataloader: IterableDataset,
        max_new_tokens: int = 1024,
        actor_micro_bs: int = 32,
        reward_micro_bs: int = 32,
        clip_reward_min: int = -5,
        clip_reward_max: int = 5,
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
        self.clip_reward_min = clip_reward_min
        self.clip_reward_max = clip_reward_max
        self.async_reward = async_reward
        self.generate_kwargs: dict = generate_kwargs

    def rollout(self, policy_model: BaseModelServer, display=False):
        sample_data = deepcopy(next(self.dataloader))
        ppo_input_messages = []
        pt_input_messages = []
        for data in sample_data:
            if data.sys_meta != 'default':
                message = deepcopy([{
                    'role': 'system',
                    'content': META_PROMPT[data.sys_meta]
                }] + data.message)
            else:
                message = deepcopy(data.message)
            if data.mes_type == 'ppo':
                ppo_input_messages.append(message)
            elif data.mes_type == 'pt':
                pt_input_messages.append(message)
            else:
                raise TypeError(f'Wrong message type {data.mes_type}')
        # ppo data
        s_t = time.time()
        print(f'[For Generate]: {ppo_input_messages[0]}')
        trajectories = policy_model.generate(
            inputs=ppo_input_messages,
            micro_batch_size=self.actor_micro_bs,
            step=self.max_new_tokens,
            output_str=True,
            generate_kwargs=self.generate_kwargs)
        logger.info(
            f'[actor generate] duration: {round(time.time() - s_t, 2)} s, len(inputs): {len(ppo_input_messages)} '  # noqa: E501
        )

        if self.async_reward:
            reward_output_ref = self.get_reward_async(sample_data,
                                                      trajectories)
            trajectories['reward_output_ref'] = reward_output_ref
        else:
            rewards = self.get_reward(sample_data, trajectories)
            clipped_rewards = torch.clamp(
                rewards, min=self.clip_reward_min, max=self.clip_reward_max)
            trajectories['rewards'] = rewards
            trajectories['clipped_rewards'] = clipped_rewards

        # pretrain data
        if len(pt_input_messages) > 0:
            pt_inputs = [
                policy_model.tokenizer.apply_chat_template(
                    mes,
                    tokenize=False,
                    add_generation_prompt=False,
                    return_tensors='pt') for mes in pt_input_messages
            ]
            trajectories.pt_data = policy_model.tokenizer(
                pt_inputs, return_tensors='pt', padding=True)
            print(
                f'[TxtEnv & {policy_model.__class__.__name__}] gets {len(pt_input_messages)} pretrain episodes.'  # noqa: E501
            )

        return trajectories

    # default get_reward() is blocking. get_reward_async() needs to call get_reward_collect()  # noqa: E501
    def get_reward_async(self, sample_data, policyout):
        s_t = time.time()
        rm_input_messages = []
        for i in range(len(sample_data)):
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
