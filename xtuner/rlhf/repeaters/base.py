import time

import numpy as np
import torch
from loguru import logger

from ..model_server.base_model_server import BaseModelServer
from ..policy_output import PolicyOutput


def find_mask_begin(padded_datas, mask_id=0):
    """finding the mask id begin index and it's length."""
    begin_indexs = []
    lengths = []

    for padded_data in padded_datas:
        is_flag = 0
        for index, data in enumerate(padded_data):
            if data != mask_id:
                is_flag = 1
                begin_indexs.append(index)
                length = (np.array(padded_data) != mask_id).sum()
                lengths.append(length)
                break
        assert is_flag
    return begin_indexs, lengths


class RunningStates:
    # adopt from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py  # noqa: E501
    def __init__(self, epsilon: float = 1e-4):
        self.mean = torch.tensor(0, dtype=torch.float32)
        self.var = torch.tensor(0, dtype=torch.float32)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        x_var, x_mean = torch.var_mean(x.cpu(), unbiased=False)
        x_count = x.shape[0]
        self.update_from_moments(x_mean, x_var, x_count)

    def update_from_other(self, other: 'RunningStates'):
        self.update_from_moments(other.mean, other.var, other.count)

    def update_from_moments(self, mean: torch.Tensor, var: torch.Tensor,
                            count: int):
        delta = mean - self.mean
        tot_count = self.count + count
        m_a = self.var * self.count
        m_b = var * count
        m_2 = m_a + m_b + delta**2 * self.count * count / (self.count + count)
        new_var = m_2 / (self.count + count)

        self.mean += delta * count / tot_count
        self.var = new_var
        self.count = tot_count

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, states):
        self.mean = states['mean']
        self.var = states['var']
        self.count = states['count']


class BaseRepeater:

    def __init__(
        self,
        sft_model,
        reward_scale: bool = False,
        fine_grained_rm: bool = False,
        value_ema: bool = False,
        actor_micro_bs: int = 8,
        ref_micro_bs: int = 8,
        critic_micro_bs: int = 32,
        kl_coeff=0.02,
        gamma=1.0,
        gae_lambda=0.95,
        answer_end_id=92542,
        norm_adv=False,
        norm_rewards=True,
        **_ignored,
    ):
        self.sft_model = sft_model
        self.actor_micro_bs = actor_micro_bs
        self.ref_micro_bs = ref_micro_bs
        self.critic_micro_bs = critic_micro_bs
        self.reward_scale = reward_scale
        self.fine_grained_rm = fine_grained_rm
        self.value_ema = value_ema
        self.kl_coeff = kl_coeff
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.answer_end_id = answer_end_id
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
            clipped_rewards = torch.clamp(
                rewards, min=env.clip_reward_min, max=env.clip_reward_max)
            trajectories['rewards'] = rewards
            trajectories['clipped_rewards'] = clipped_rewards
        # Experimental
        rewards = trajectories.clipped_rewards
        if self.norm_rewards:
            self.running_states.update(rewards)
            norm_reward_score = (rewards - self.running_states.mean) / (
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

    def _get_advantages_and_returns(self, trajectories):
        output_ids = trajectories.output_ids
        answer_mask = trajectories.answer_mask
        values_with_last_value = trajectories.values_with_last_value
        kl_rewards = trajectories.kl_rewards

        begins_index, answers_length = find_mask_begin(answer_mask, 0)
        count = 0
        advantages_padded, returns_padded = torch.zeros_like(
            kl_rewards, dtype=values_with_last_value.dtype), torch.zeros_like(
                kl_rewards, dtype=values_with_last_value.dtype)
        for begin_index, ans_len, value_with_last_value, reward, output_id in zip(  # noqa: E501
                begins_index, answers_length, values_with_last_value,
                kl_rewards, output_ids):
            # shape :ans_len + 1
            value_with_last_value = value_with_last_value[begin_index -
                                                          1:begin_index +
                                                          ans_len]
            # shape :ans_len
            reward = reward[begin_index:begin_index + ans_len]
            last_gae_lam = torch.zeros((1), dtype=values_with_last_value.dtype)
            # shape :ans_len
            advantages = torch.zeros_like(
                reward, dtype=values_with_last_value.dtype)
            step_nums = advantages.shape[-1]
            # shape:ans_len + 1
            dones = self._build_dones(output_id[begin_index:begin_index +
                                                ans_len])
            for step in reversed(range(step_nums)):
                next_non_terminal = 1 - dones[step + 1]
                next_values = value_with_last_value[step + 1]
                # delta and last_gae_lam using value and reward
                delta = reward[
                    step] + self.gamma * next_values * next_non_terminal - value_with_last_value[  # noqa: E501
                        step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam  # noqa: E501
                advantages[step] = last_gae_lam[0]
            returns = advantages + value_with_last_value[:-1]
            advantages_padded[count,
                              begin_index:begin_index + ans_len] = advantages
            returns_padded[count, begin_index:begin_index + ans_len] = returns
            count += 1
        return advantages_padded, returns_padded

    # ans_len + 1: dones
    def _build_dones(self, answer_ids):
        dones = torch.tensor(
            (answer_ids == self.answer_end_id).numpy().astype(np.float32))
        # (1, )the first one is not done, so obs_0_dones=0
        obs_0_dones = torch.zeros((1), dtype=torch.float32)
        # (ans_len + 1)，
        dones = torch.concat((obs_0_dones, dones), axis=0)
        return dones

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134  # noqa: E501
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
