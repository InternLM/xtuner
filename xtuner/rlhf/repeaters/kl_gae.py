import torch

from ..model_server.base_model_server import BaseModelServer
from ..policy_output import PolicyOutput
from ..timer import Timer
from .base import RepeaterBase
from .utils import RunningStates


class KLGAERepeater(RepeaterBase):

    def __init__(
        self,
        ref_model: BaseModelServer,
        policy_model: BaseModelServer,
        critic_model: BaseModelServer,
        policy_micro_bs: int = 8,
        ref_micro_bs: int = 8,
        critic_micro_bs: int = 32,
        kl_coeff=0.01,
        gamma=1.0,
        gae_lambda=0.99,
        clip_reward_min: int = -5,
        clip_reward_max: int = 5,
        norm_rewards=True,
        norm_adv=False,
        env=None,
        **_ignored,
    ):
        # models
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.critic_model = critic_model

        self.policy_micro_bs = policy_micro_bs
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
        self.norm_adv = norm_adv

        # only used for async reward model.infer_get() in _get_kl_rewards
        self.env = env

    def process(self, trajectories: PolicyOutput):
        critic_output_ref = self._get_values_async(trajectories)
        action_mask = trajectories['action_mask']
        num_actions = action_mask.size(1)
        (kl_rewards, entropy, kl_distance, policy_logprobs,
         ref_logprobs) = self._get_kl_rewards(trajectories)
        trajectories['kl'] = (kl_distance * action_mask).sum(
            axis=-1) / action_mask.sum(axis=-1)
        trajectories['entropy'] = entropy
        trajectories['kl_rewards'] = kl_rewards
        trajectories['policy_logprobs'] = policy_logprobs
        trajectories['ref_logprobs'] = ref_logprobs

        values = self._get_values_collect(critic_output_ref)
        old_values = values[:, -num_actions:]
        advantages, returns = self.get_advantages_and_returns(
            old_values, kl_rewards, action_mask)
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8)
        trajectories['advantages'] = advantages
        trajectories['returns'] = returns
        trajectories['old_values'] = old_values

        return trajectories

    def _get_kl_rewards(self, trajectories: PolicyOutput):
        with Timer('policy_model.infer_async'):
            policy_output = self.policy_model.infer_async(
                inputs=trajectories.output_ids,
                micro_batch_size=self.policy_micro_bs,
                attention_mask=trajectories.attention_mask,
                output_logits=False,
                output_logprobs=True)
        with Timer('ref_model.infer_async'):
            ref_output = self.ref_model.infer_async(
                inputs=trajectories.output_ids,
                micro_batch_size=self.ref_micro_bs,
                attention_mask=trajectories.attention_mask,
                output_logits=False,
                output_logprobs=True)
        with Timer('policy_model.infer_get'):
            policy_output = self.policy_model.infer_get(policy_output)
        with Timer('ref_model.infer_get'):
            ref_output = self.ref_model.infer_get(ref_output)

        # Experimental
        if self.env.async_reward:
            rewards = self.env.get_reward_collect(
                trajectories['reward_output_ref'])
            trajectories['reward_output_ref'] = None
            trajectories['rewards'] = rewards
        # Experimental

        clipped_rewards = torch.clamp(
            rewards, min=self.clip_reward_min, max=self.clip_reward_max)
        trajectories['clipped_rewards'] = clipped_rewards

        if self.norm_rewards:
            self.running_states.update(clipped_rewards)
            norm_reward_score = (clipped_rewards -
                                 self.running_states.mean) / (
                                     self.running_states.var.sqrt() + 1e-8)
        action_mask = trajectories.action_mask
        num_actions = action_mask.size(1)

        policy_logprobs = policy_output.logprobs[:, -num_actions:]
        ref_logprobs = ref_output.logprobs[:, -num_actions:]

        if self.kl_coeff <= 0.0:
            self.kl_coeff = 0.0
        # compute_approx_kl
        log_ratio = policy_logprobs - ref_logprobs
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
        return reward, entropy, kl, policy_logprobs, ref_logprobs

    def _get_values(self, trajectories: PolicyOutput):
        with Timer('critic_model.infer'):
            critic_output = self.critic_model.infer(
                inputs=trajectories.output_ids,
                attention_mask=trajectories.attention_mask,
                output_logits=True,
                micro_batch_size=self.critic_micro_bs,
            )
        raw_values = critic_output.logits.squeeze(-1)
        return raw_values

    def _get_values_async(self, trajectories: PolicyOutput):
        with Timer('critic_model.infer_async'):
            critic_output_ref = self.critic_model.infer_async(
                inputs=trajectories.output_ids,
                attention_mask=trajectories.attention_mask,
                output_logits=True,
                micro_batch_size=self.critic_micro_bs,
            )
        return critic_output_ref

    def _get_values_collect(self, critic_output_ref):
        with Timer('critic_model.infer_get'):
            critic_output = self.critic_model.infer_get(critic_output_ref)
        raw_values = critic_output.logits.squeeze(-1)
        return raw_values

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
