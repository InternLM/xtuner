import time

import torch
from loguru import logger

from ..loss.actor_loss import ActorLoss
from ..loss.critic_loss import CriticLoss
from ..loss.pretrain_loss import PretrainLoss
from ..model_server.base_model_server import BaseModelServer
from ..timer import Timer


class PPOTrainer:

    def __init__(
        self,
        actor_micro_bs=2,
        critic_micro_bs=2,
        policy_learn_time=1,
        value_learn_time=1,
        policy_minibatch=None,
        value_minibatch=None,
        ppo_loss_weight=1.0,
        pretrain_loss_weight=0.5,
        pretrain_criterion=PretrainLoss(label_smoothing=0),
        policy_criterion=ActorLoss(cliprange=0.2, loss_type='per_seq'),
        value_criterion=CriticLoss(cliprange_value=0.5, loss_type='per_seq'),
        **kwargs,
    ):

        self.actor_micro_bs = actor_micro_bs
        self.critic_micro_bs = critic_micro_bs
        # policy
        self.policy_learn_time = policy_learn_time
        self.policy_minibatch = policy_minibatch

        # value
        self.value_learn_time = value_learn_time
        self.value_minibatch = value_minibatch

        self.ppo_loss_weight = ppo_loss_weight
        self.pretrain_loss_weight = pretrain_loss_weight
        self.pretrain_criterion = pretrain_criterion
        self.policy_criterion = policy_criterion
        self.value_criterion = value_criterion

    def policy_learn(self, trajectories, policy_model: BaseModelServer):
        if self.policy_minibatch is None:
            self.policy_minibatch = len(trajectories.output_ids)
        policy_updates = len(trajectories.output_ids) // self.policy_minibatch
        ppo_loss = []
        pretrain_loss = []

        for _ in range(self.policy_learn_time):
            for i in range(policy_updates):
                logger.info(
                    '[Policy Train] start policy trains {}/{} | {}'.format(
                        i + 1, policy_updates, _ + 1))
                # prompt train data
                begin = i * self.policy_minibatch
                end = begin + self.policy_minibatch

                train_input_ids = [
                    trajectories.output_ids[begin:end, :],
                ]
                train_attention_mask = [
                    trajectories.attention_mask[begin:end, :],
                ]
                train_criterion = [
                    self.policy_criterion,
                ]
                loss_weights = [
                    self.ppo_loss_weight,
                ]
                micro_batch_size = [
                    self.actor_micro_bs,
                ]
                assert len(
                    trajectories.output_ids[begin:end, :]
                ) == self.policy_minibatch, '[Policy learn] make sure len(policy_batch_inputs) == self.policy_minibatch'  # noqa: E501

                loss_factor = 1.0
                train_lables = [
                    dict(
                        input_ids=trajectories.output_ids[begin:end, :],
                        old_logprobs=trajectories.policy_logprobs[
                            begin:end, :],
                        advantages=trajectories.advantages[begin:end, :],
                        mask=trajectories.action_mask[begin:end, :],
                        loss_factor=torch.tensor(loss_factor),
                    ),
                ]
                # pretrain data
                if trajectories.pretrain_data is not None:
                    logger.info(
                        f'[Policy Train] policy train with pretrain data {trajectories.pretrain_data["input_ids"].shape}'
                    )
                    train_input_ids.append(
                        trajectories.pretrain_data['input_ids'])
                    train_lables.append(trajectories.pretrain_data['labels'])
                    # train_position_ids.append(trajectories.pretrain_data["position_ids"])
                    train_attention_mask.append(
                        trajectories.pretrain_data['attention_mask'])
                    train_criterion.append(self.pretrain_criterion)
                    loss_weights.append(self.pretrain_loss_weight)
                    micro_batch_size.append(self.actor_micro_bs)

                s_t = time.time()
                p_loss = policy_model.train(
                    input_ids=train_input_ids,
                    labels=train_lables,
                    attention_mask=train_attention_mask,
                    # position_ids=train_position_ids,
                    criterion=train_criterion,
                    loss_weights=loss_weights,
                    micro_batch_size=micro_batch_size)
                if isinstance(p_loss, list):
                    ppo_loss.append(p_loss[0].item())
                    pretrain_loss.append(p_loss[1].item())
                    logger.info(
                        f'[Policy Train] duration: {round(time.time() - s_t, 2)} s, prompt data: {train_input_ids[0].shape}, ppo loss: {p_loss[0].item()}; pretrain data: {train_input_ids[1].shape}, pretrain loss: {p_loss[1].item()}'
                    )
                else:
                    ppo_loss.append(p_loss.item())
                    logger.info(
                        f'[Policy Train] duration: {round(time.time() - s_t, 2)} s, prompt data: {train_input_ids[0].shape}, ppo loss: {p_loss.item()}'
                    )

        with Timer('policy_model.sync_model'):
            policy_model.sync_model()
        return ppo_loss, pretrain_loss

    def value_learn_async(self, trajectories, value_model: BaseModelServer):
        if self.value_minibatch is None:
            self.value_minibatch = len(trajectories.output_ids)
        value_updates = len(trajectories.output_ids) // self.value_minibatch
        value_loss = []
        assert value_updates == 1 and self.policy_learn_time == 1, f'value_updates={value_updates} * self.policy_learn_time={self.policy_learn_time} > 1'  # noqa: E501
        s_t = time.time()
        value_batch_inputs, labels = self._value_learn_prepare(
            0, 0, trajectories, value_updates)
        v_loss_ref = value_model.train_async(
            input_ids=value_batch_inputs['input_ids'],
            labels=labels,
            attention_mask=value_batch_inputs['attention_mask'],
            criterion=self.value_criterion,
            micro_batch_size=self.critic_micro_bs,
        )
        logger.info(
            f'[critic train] async duration: {round(time.time() - s_t, 2)} s, {self.value_minibatch} batch'  # noqa: E501
        )
        value_loss.append(v_loss_ref)
        return value_loss

    def value_learn_get(self, value_loss_ref, value_model: BaseModelServer):
        with Timer('value_model.train_get'):
            return [
                value_model.train_get(ref).item() for ref in value_loss_ref
            ]

    def value_learn(self, trajectories, value_model: BaseModelServer):
        if self.value_minibatch is None:
            self.value_minibatch = len(trajectories.output_ids)
        value_updates = len(trajectories.output_ids) // self.value_minibatch
        value_loss = []

        for learn_i in range(self.policy_learn_time):
            for step_i in range(value_updates):
                s_t = time.time()
                value_batch_inputs, labels = self._value_learn_prepare(
                    step_i, learn_i, trajectories, value_updates)
                v_loss = value_model.train(
                    input_ids=value_batch_inputs['input_ids'],
                    labels=labels,
                    attention_mask=value_batch_inputs['attention_mask'],
                    criterion=self.value_criterion,
                    micro_batch_size=self.critic_micro_bs,
                )
                logger.info(
                    f'[critic train] duration: {round(time.time() - s_t, 2)} s, {self.value_minibatch} batch,value loss: {v_loss.item()}'  # noqa: E501
                )
                value_loss.append(v_loss.item())
        return value_loss

    def _value_learn_prepare(self, step_i, learn_i, trajectories,
                             value_updates):
        logger.info('[Value Train] start value trains {}/{} | {}'.format(
            step_i + 1, value_updates, learn_i + 1))
        begin = step_i * self.value_minibatch
        end = begin + self.value_minibatch
        value_batch_inputs = {
            'input_ids': trajectories.output_ids[begin:end, :],
            'old_values': trajectories.old_values[begin:end, :],
            'returns': trajectories.returns[begin:end, :],
            'action_mask': trajectories.action_mask[begin:end, :],
            'attention_mask': trajectories.attention_mask[begin:end, :]
        }
        assert len(
            value_batch_inputs['input_ids']
        ) == self.value_minibatch, '[Value learn] make sure len(value_batch_inputs) == self.value_minibatch'  # noqa: E501

        loss_factor = 1.0
        labels = dict(
            old_values=value_batch_inputs['old_values'],
            returns=value_batch_inputs['returns'],
            mask=value_batch_inputs['action_mask'],
            loss_factor=torch.tensor(loss_factor),
        )
        return value_batch_inputs, labels
