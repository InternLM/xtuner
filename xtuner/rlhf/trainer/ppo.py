from loguru import logger

from ..loss import CriticLoss, PPOPolicyLoss, PretrainLoss
from ..model_server.base_model_server import BaseModelServer
from ..timer import Timer


class PPOTrainer:

    def __init__(
            self,
            policy_model: BaseModelServer,
            critic_model: BaseModelServer,
            policy_micro_bs=2,
            critic_micro_bs=2,
            policy_learn_time=1,
            critic_learn_time=1,
            policy_minibatch=None,
            critic_minibatch=None,
            ppo_loss_weight=1.0,
            pretrain_loss_weight=0.5,
            pretrain_criterion=PretrainLoss(label_smoothing=0),
            policy_criterion=PPOPolicyLoss(cliprange=0.2),
            critic_criterion=CriticLoss(cliprange_value=0.5),
            **kwargs,
    ):

        # policy
        self.policy_model = policy_model
        self.policy_learn_time = policy_learn_time
        self.policy_minibatch = policy_minibatch
        self.policy_micro_bs = policy_micro_bs

        self.ppo_loss_weight = ppo_loss_weight
        self.pretrain_loss_weight = pretrain_loss_weight
        self.pretrain_criterion = pretrain_criterion
        self.policy_criterion = policy_criterion

        # critic
        self.critic_model = critic_model
        self.critic_learn_time = critic_learn_time
        self.critic_minibatch = critic_minibatch
        self.critic_micro_bs = critic_micro_bs

        self.critic_criterion = critic_criterion

    def policy_learn(self, trajectories):
        if self.policy_minibatch is None:
            self.policy_minibatch = len(trajectories.output_ids)
        assert len(trajectories.output_ids) % self.policy_minibatch == 0
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

                train_input_ids = [trajectories.output_ids[begin:end, :]]
                train_attention_mask = [
                    trajectories.attention_mask[begin:end, :]
                ]
                train_criterion = [self.policy_criterion]
                loss_weights = [self.ppo_loss_weight]
                micro_batch_size = [self.policy_micro_bs]

                train_lables = [
                    dict(
                        input_ids=trajectories.output_ids[begin:end, :],
                        old_logprobs=trajectories.policy_logprobs[
                            begin:end, :],
                        advantages=trajectories.advantages[begin:end, :],
                        mask=trajectories.action_mask[begin:end, :],
                    ),
                ]
                # pretrain data
                if trajectories.pretrain_data is not None:
                    logger.info(
                        '[Policy Train] pretrain data '
                        f'{trajectories.pretrain_data["input_ids"].shape}')
                    train_input_ids.append(
                        trajectories.pretrain_data['input_ids'])
                    train_lables.append(trajectories.pretrain_data['labels'])
                    # train_position_ids.append(trajectories.pretrain_data["position_ids"])
                    train_attention_mask.append(
                        trajectories.pretrain_data['attention_mask'])
                    train_criterion.append(self.pretrain_criterion)
                    loss_weights.append(self.pretrain_loss_weight)
                    micro_batch_size.append(self.policy_micro_bs)

                with Timer('policy_model.train'):
                    p_loss = self.policy_model.train(
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
                        f'[Policy Train] prompt data: {train_input_ids[0].shape}, ppo loss: {p_loss[0].item()}; pretrain data: {train_input_ids[1].shape}, pretrain loss: {p_loss[1].item()}'  # noqa: E501
                    )
                else:
                    ppo_loss.append(p_loss.item())
                    logger.info(
                        f'[Policy Train] prompt data: {train_input_ids[0].shape}, ppo loss: {p_loss.item()}'  # noqa: E501
                    )

        with Timer('policy_model.sync_model'):
            self.policy_model.sync_model()
        return ppo_loss, pretrain_loss

    def critic_learn(self, trajectories):
        if self.critic_minibatch is None:
            self.critic_minibatch = len(trajectories.output_ids)
        assert len(trajectories.output_ids) % self.critic_minibatch == 0
        critic_updates = len(trajectories.output_ids) // self.critic_minibatch
        critic_loss = []

        for learn_i in range(self.critic_learn_time):
            for step_i in range(critic_updates):
                logger.info(
                    '[Critic Train] start critic trains {}/{} | {}'.format(
                        step_i + 1, critic_updates, learn_i + 1))
                with Timer('critic_model.train'):
                    critic_batch_inputs, labels = self._critic_learn_prepare(
                        step_i, learn_i, trajectories, critic_updates)
                    v_loss = self.critic_model.train(
                        input_ids=critic_batch_inputs['input_ids'],
                        labels=labels,
                        attention_mask=critic_batch_inputs['attention_mask'],
                        criterion=self.critic_criterion,
                        micro_batch_size=self.critic_micro_bs,
                    )
                logger.info(f'[Critic train] {self.critic_minibatch} batch, '
                            f'critic loss: {v_loss.item()}')
                critic_loss.append(v_loss.item())
        return critic_loss

    def _critic_learn_prepare(self, step_i, learn_i, trajectories,
                              critic_updates):
        logger.info('[Critic Train] start critic trains {}/{} | {}'.format(
            step_i + 1, critic_updates, learn_i + 1))
        begin = step_i * self.critic_minibatch
        end = begin + self.critic_minibatch
        critic_batch_inputs = dict(
            input_ids=trajectories.output_ids[begin:end, :],
            old_values=trajectories.old_values[begin:end, :],
            returns=trajectories.returns[begin:end, :],
            action_mask=trajectories.action_mask[begin:end, :],
            attention_mask=trajectories.attention_mask[begin:end, :])

        labels = dict(
            old_values=critic_batch_inputs['old_values'],
            returns=critic_batch_inputs['returns'],
            mask=critic_batch_inputs['action_mask'],
        )
        return critic_batch_inputs, labels

    def critic_learn_async(self, trajectories):
        if self.critic_minibatch is None:
            self.critic_minibatch = len(trajectories.output_ids)
        assert len(trajectories.output_ids) % self.critic_minibatch == 0
        critic_updates = len(trajectories.output_ids) // self.critic_minibatch
        critic_loss = []
        assert critic_updates == 1 and self.policy_learn_time == 1, \
            '[WIP] `critic_learn_async` support learn async in loop'
        with Timer('critic_model.train_async'):
            critic_batch_inputs, labels = self._critic_learn_prepare(
                0, 0, trajectories, critic_updates)
            v_loss_ref = self.critic_model.train_async(
                input_ids=critic_batch_inputs['input_ids'],
                labels=labels,
                attention_mask=critic_batch_inputs['attention_mask'],
                criterion=self.critic_criterion,
                micro_batch_size=self.critic_micro_bs,
            )
        logger.info(f'[critic train] {self.critic_minibatch} batch')
        critic_loss.append(v_loss_ref)
        return critic_loss

    def critic_learn_get(self, critic_loss_ref):
        with Timer('critic_model.train_get'):
            return [
                self.critic_model.train_get(ref).item()
                for ref in critic_loss_ref
            ]
