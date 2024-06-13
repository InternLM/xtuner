import time

import torch
from loguru import logger

from ..model_server.base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from .running_mean_std import RunningStates


class BaseRepeater:

    def __init__(
        self,
        sft_model,
        actor_micro_bs: int = 8,
        ref_micro_bs: int = 8,
        critic_micro_bs: int = 32,
        kl_coeff=0.02,
        gamma=1.0,
        gae_lambda=0.95,
        norm_adv=False,
        clip_reward_min: int = -5,
        clip_reward_max: int = 5,
        norm_rewards=True,
        reward_scale: bool = False,
        fine_grained_rm: bool = False,
        **_ignored,
    ):
        self.sft_model = sft_model
        self.actor_micro_bs = actor_micro_bs
        self.ref_micro_bs = ref_micro_bs
        self.critic_micro_bs = critic_micro_bs
        self.kl_coeff = kl_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # rewards
        self.clip_reward_min = clip_reward_min
        self.clip_reward_max = clip_reward_max
        self.norm_rewards = norm_rewards
        if self.norm_rewards:
            self.running_states = RunningStates(epsilon=0)

    def process(
        self,
        trajectories: PolicyOutput,
        policy_model: BaseModelServer,
        value_model: BaseModelServer,
        sft_model: BaseModelServer = None,
        # only used for async reward model.infer_get() in _get_kl_rewards
        env=None,
    ):
        value_output_ref = self._get_values_async(trajectories, value_model)
        action_mask = trajectories['action_mask']
        num_actions = action_mask.size(1)
        if sft_model is not None:
            self.sft_model: BaseModelServer = sft_model
        kl_rewards, entropy, kl_distance, policy_logprobs, sft_logprobs = self._get_kl_rewards(  # noqa: E501
            trajectories, policy_model, env=env)
        trajectories['kl'] = (kl_distance * action_mask).sum(
            axis=-1) / action_mask.sum(axis=-1)
        trajectories['entropy'] = entropy
        trajectories['kl_rewards'] = kl_rewards
        trajectories['policy_logprobs'] = policy_logprobs
        trajectories['sft_logprobs'] = sft_logprobs

        values = self._get_values_collect(value_output_ref, value_model)
        old_values = values[:, -num_actions:]
        advantages, returns = self.get_advantages_and_returns(
            old_values, kl_rewards, action_mask)

        trajectories['advantages'] = advantages
        trajectories['returns'] = returns
        trajectories['old_values'] = old_values

        return trajectories

    def _get_kl_rewards(self,
                        trajectories: PolicyOutput,
                        policy_model: BaseModelServer,
                        env=None):
        s_t = time.time()
        policy_output = policy_model.infer_async(
            inputs=trajectories.output_ids,
            micro_batch_size=self.actor_micro_bs,
            attention_mask=trajectories.attention_mask,
            output_logits=False,
            output_logprobs=True)
        sft_output = self.sft_model.infer_async(
            inputs=trajectories.output_ids,
            micro_batch_size=self.ref_micro_bs,
            attention_mask=trajectories.attention_mask,
            output_logits=False,
            output_logprobs=True)
        policy_output = policy_model.infer_get(policy_output)
        sft_output = self.sft_model.infer_get(sft_output)
        logger.info(
            f'[actor & ref infer_async] duration: {round(time.time() - s_t, 2)} s'  # noqa: E501
        )

        # Experimental
        if env.async_reward:
            rewards = env.get_reward_collect(trajectories['reward_output_ref'])
            trajectories['reward_output_ref'] = None
            trajectories['rewards'] = rewards
        # Experimental

        clipped_rewards = torch.clamp(
            rewards, min=self.clip_reward_min, max=self.clip_reward_max)
        trajectories['clipped_rewards'] = clipped_rewards

        if self.norm_rewards:
            self.running_states.update(clipped_rewards)
            norm_reward_score = (clipped_rewards - self.running_states.mean) / (
                self.running_states.var.sqrt() + 1e-8)
        action_mask = trajectories.action_mask
        num_actions = action_mask.size(1)

        policy_logprobs = policy_output.logprobs[:, -num_actions:]
        sft_logprobs = sft_output.logprobs[:, -num_actions:]

        if self.kl_coeff <= 0.0:
            self.kl_coeff = 0.0
        # compute_approx_kl
        log_ratio = policy_logprobs - sft_logprobs
        kl = log_ratio * action_mask
        kl_reward = -self.kl_coeff * kl

        eos_indices = action_mask.size(
            1) - 1 - action_mask.long().fliplr().argmax(
                dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(
            dim=1,
            index=eos_indices,
            src=norm_reward_score.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward

        entropy = -(policy_logprobs *
                    action_mask).sum(axis=-1) / action_mask.sum(axis=-1)
        return reward, entropy, kl, policy_logprobs, sft_logprobs

    def _get_values(self, trajectories: PolicyOutput,
                    value_model: BaseModelServer):
        s_t = time.time()
        value_output = value_model.infer(
            inputs=trajectories.output_ids,
            attention_mask=trajectories.attention_mask,
            output_logits=True,
            micro_batch_size=self.critic_micro_bs,
        )
        logger.info(
            f'[critic infer] duration: {round(time.time() - s_t, 2)} s')
        raw_values = value_output.logits.squeeze(-1)
        return raw_values

    def _get_values_async(self, trajectories: PolicyOutput,
                          value_model: BaseModelServer):
        s_t = time.time()
        value_output_ref = value_model.infer_async(
            inputs=trajectories.output_ids,
            attention_mask=trajectories.attention_mask,
            output_logits=True,
            micro_batch_size=self.critic_micro_bs,
        )
        logger.info(
            f'[critic infer] async duration: {round(time.time() - s_t, 2)} s')
        return value_output_ref

    def _get_values_collect(self, value_output_ref,
                            value_model: BaseModelServer):
        s_t = time.time()
        value_output = value_model.infer_get(value_output_ref)
        raw_values = value_output.logits.squeeze(-1)
        logger.info(
            f'[critic infer] async wait duration: {round(time.time() - s_t, 2)} s'  # noqa: E501
        )
        return raw_values

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134  # noqa: E501
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            # Since old_rewards and old_values are masked with action_mask, i.e. they have
            # 0's at pad tokens, delta will be 0 if current t is at a pad token, so will lastgaelam
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
