import torch

from ..model_server.base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from ..timer import Timer
from .base import RepeaterBase
from .utils import RunningStates

from loguru import logger
from xtuner.rlhf.envs.utils import SYSTEM_PROMPT

class KLGAERepeater(RepeaterBase):

    def __init__(
        self,
        ref_model: BaseModelServer,
        reward_model: BaseModelServer,
        ref_micro_bs: int = 8,
        reward_micro_bs: int = 8,
        kl_coeff=0.01,
        gamma=1.0,
        gae_lambda=0.99,
        clip_reward_min: int = -5,
        clip_reward_max: int = 5,
        norm_rewards=True,
        norm_adv=False,
        **_ignored,
    ):
        # models
        self.ref_model = ref_model
        self.reward_model = reward_model

        self.ref_micro_bs = ref_micro_bs
        self.reward_micro_bs = reward_micro_bs
        self.kl_coeff = kl_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # rewards
        self.clip_reward_min = clip_reward_min
        self.clip_reward_max = clip_reward_max
        self.norm_rewards = norm_rewards
        if self.norm_rewards:
            self.running_states = RunningStates(epsilon=0)
        self.norm_adv = norm_adv

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134  # noqa: E501
        """Function that computes advantages and returns from rewards and
        values. Calculated as in the original PPO paper:
        https://arxiv.org/abs/1707.06347 Note that rewards may include a KL
        divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            # Since old_rewards and old_values are masked with action_mask,
            # i.e. they have 0's at pad tokens,
            # delta will be 0 if current t is at a pad token,
            # so will lastgaelam
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    # get_reward_async() needs to call get_reward_collect()
    def get_reward_async(self, prompt_datas, policyout):
        rm_input_messages = []
        # for i in range(len(prompt_datas)):
        for i, req_id in enumerate(policyout.req_ids):
            if prompt_datas[req_id].mes_type != 'prompt':
                continue
            if (prompt_datas[req_id].rm_prompt !=
                    'default') or (prompt_datas[req_id].sys_prompt != 'default'):
                # Conditional Reward Model
                # for queries from different domains, use appropriate conditional system prompts  # noqa: E501
                # From Alignment section of the InternLM2 Technical Report:
                # https://arxiv.org/pdf/2403.17297
                if prompt_datas[req_id].rm_prompt != 'default':
                    prompt = prompt_datas[req_id].rm_prompt
                else:
                    prompt = prompt_datas[req_id].sys_prompt
                cur_rm_data = [
                    dict(role='system', content=SYSTEM_PROMPT[prompt])
                ] + prompt_datas[req_id].message + [
                    dict(
                        role='assistant', content=policyout.output_ans_str[i])
                ]
            else:
                cur_rm_data = prompt_datas[req_id].message + [
                    dict(
                        role='assistant', content=policyout.output_ans_str[i])
                ]
            rm_input_messages.append(cur_rm_data)

        logger.info(f'[For Reward]: {rm_input_messages[0]}')

        with Timer('reward_model.infer_async'):
            reward_output_ref = self.reward_model.infer_async(
                rm_input_messages,
                output_logprobs=False,
                micro_batch_size=self.reward_micro_bs)
        return reward_output_ref

    def get_reward_collect(self, reward_output_ref):
        with Timer('reward_model.infer_get'):
            rm_out = self.reward_model.infer_get(reward_output_ref)
        rewards = rm_out.logits.squeeze(-1)
        return rewards

    def get_reward_and_reference(self, prompt_datas, trajectories):
        reward_ref = self.get_reward_async(
            prompt_datas, 
            trajectories)
        
        with Timer('ref_model.infer_async'):
            reference_ref = self.ref_model.infer_async(
                inputs=trajectories.output_ids,
                micro_batch_size=self.ref_micro_bs,
                attention_mask=trajectories.attention_mask,
                output_logits=False,
                output_logprobs=True)

        with Timer('ref_model.infer_get'):
            reference_output = self.ref_model.infer_get(reference_ref)

        rewards = self.get_reward_collect(reward_ref)

        return rewards, reference_output.logprobs

    def process_kl_gae(self, rewards, ref_logprobs, values, policy_logprobs, trajectories):
        clipped_rewards = torch.clamp(
            rewards, min=self.clip_reward_min, max=self.clip_reward_max)
        trajectories['rewards'] = rewards
        trajectories['clipped_rewards'] = clipped_rewards

        if self.norm_rewards:
            self.running_states.update(clipped_rewards)
            norm_reward_score = (clipped_rewards -
                                 self.running_states.mean) / (
                                     self.running_states.var.sqrt() + 1e-8)
        else:
            norm_reward_score = clipped_rewards

        action_mask = trajectories.action_mask
        num_actions = action_mask.size(1)

        policy_logprobs = policy_logprobs[:, -num_actions:]
        ref_logprobs = ref_logprobs[:, -num_actions:]

        if self.kl_coeff <= 0.0:
            self.kl_coeff = 0.0
        # compute_approx_kl
        log_ratio = policy_logprobs - ref_logprobs
        kl_distance = log_ratio * action_mask
        kl_penalty = -self.kl_coeff * kl_distance

        eos_indices = action_mask.size(
            1) - 1 - action_mask.long().fliplr().argmax(
                dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl_distance).scatter_(
            dim=1,
            index=eos_indices,
            src=norm_reward_score.unsqueeze(1).to(kl_distance.dtype))

        kl_rewards = last_reward + kl_penalty

        entropy = -(policy_logprobs *
                    action_mask).sum(axis=-1) / action_mask.sum(axis=-1)

        trajectories['kl'] = (kl_distance * action_mask).sum(
            axis=-1) / action_mask.sum(axis=-1)
        trajectories['entropy'] = entropy
        trajectories['kl_rewards'] = kl_rewards
        trajectories['policy_logprobs'] = policy_logprobs
        trajectories['ref_logprobs'] = ref_logprobs

        old_values = values[:, -num_actions:]
        advantages, returns = self.get_advantages_and_returns(
            old_values, kl_rewards, action_mask)
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8)
        trajectories['advantages'] = advantages
        trajectories['returns'] = returns
        trajectories['old_values'] = old_values
        trajectories['orig_values'] = values

        return trajectories