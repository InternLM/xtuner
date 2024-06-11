# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023  # noqa
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.integrations import is_deepspeed_zero3_enabled

from .sft import SupervisedFinetune


def create_reference_model(model):
    if is_deepspeed_zero3_enabled():
        raise ValueError('DeepSpeed ZeRO-3 is enabled and is not compatible '
                         'with `create_reference_model()`. Please instantiate '
                         'your reference model directly with '
                         '`AutoCausalLM.from_pretrained()`.')

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    return ref_model.eval()


class DPO(SupervisedFinetune):
    """A general class of DPO and its variants."""

    def __init__(self,
                 llm,
                 ref_llm=None,
                 beta=0.1,
                 loss_type='sigmoid',
                 label_smoothing=0.0,
                 **kwargs):
        super().__init__(llm, **kwargs)
        self.ref_llm = ref_llm
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.beta = beta

        if not self.use_lora:
            self.ref_llm = create_reference_model(self.llm)

    def _gather_masked_logits(self, logits, labels, mask):
        logits = torch.gather(
            logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        return logits * mask

    def get_logps(
            self,
            all_logits,  # bs, seqlen,vocab_size
            all_ref_logits,  # bs, seqlen,vocab_size
            labels,  # bs, seqlen
            loss_mask,  # bs, seqlen
    ):
        all_logps = self._gather_masked_logits(all_logits, labels,
                                               loss_mask).sum(-1)
        all_ref_logps = self._gather_masked_logits(all_ref_logits, labels,
                                                   loss_mask).sum(-1)

        if self.loss_type == 'ipo':  # average_log_prob
            all_logps = all_logps / loss_mask.sum(-1)
            all_ref_logps = all_ref_logps / loss_mask.sum(-1)

        policy_chosen_logps = all_logps[::2]
        policy_rejected_logps = all_logps[1::2]
        reference_chosen_logps = all_ref_logps[::2]
        reference_rejected_logps = all_ref_logps[1::2]
        return (policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps)

    def get_var_len_atten_logps(
        self,
        all_logits,
        all_ref_logits,
        labels,
        loss_mask,
        cu_seqlens,
    ):
        masked_logps = self._gather_masked_logits(all_logits, labels,
                                                  loss_mask)
        masked_ref_logps = self._gather_masked_logits(all_ref_logits, labels,
                                                      loss_mask)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # unpack sequence
        unpacked_logps = torch.split(masked_logps, seqlens, dim=1)
        unpacked_ref_logps = torch.split(masked_ref_logps, seqlens, dim=1)
        unpacked_mask = torch.split(loss_mask, seqlens, dim=1)

        def compute_logps(logps, mask, idx, loss_type):
            logp = logps[idx].sum(-1)
            if loss_type == 'ipo':
                logp /= mask[idx].sum(-1)
            return logp

        policy_chosen_logps = [
            compute_logps(unpacked_logps, unpacked_mask, 2 * i, self.loss_type)
            for i in range(len(unpacked_logps) // 2)
        ]
        policy_rejected_logps = [
            compute_logps(unpacked_logps, unpacked_mask, 2 * i + 1,
                          self.loss_type)
            for i in range(len(unpacked_logps) // 2)
        ]
        reference_chosen_logps = [
            compute_logps(unpacked_ref_logps, unpacked_mask, 2 * i,
                          self.loss_type)
            for i in range(len(unpacked_ref_logps) // 2)
        ]
        reference_rejected_logps = [
            compute_logps(unpacked_ref_logps, unpacked_mask, 2 * i + 1,
                          self.loss_type)
            for i in range(len(unpacked_ref_logps) // 2)
        ]

        return (torch.stack(policy_chosen_logps),
                torch.stack(policy_rejected_logps),
                torch.stack(reference_chosen_logps),
                torch.stack(reference_rejected_logps))

    def compute_loss(self, data, data_samples=None):
        # modified from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py  # noqa
        all_logits = self.llm(**data).logits
        with torch.no_grad():
            if self.ref_llm is None:
                with self.llm.disable_adapter():
                    all_ref_logits = self.llm(**data).logits
            else:
                all_ref_logits = self.ref_llm(**data).logits

        labels = data['labels']
        labels[labels == -100] = 0
        loss_mask = labels != 0
        if not self.use_varlen_attn:
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_logps(
                 all_logits, all_ref_logits, labels, loss_mask)
        else:
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_var_len_atten_logps(
                 all_logits, all_ref_logits, labels, loss_mask, cu_seqlens)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        if self.loss_type == 'sigmoid':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) -
                    F.logsigmoid(-self.beta * logits) * self.label_smoothing)
        elif self.loss_type == 'robust':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) +
                    F.logsigmoid(-self.beta * logits) *
                    self.label_smoothing) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == 'hinge':
            loss = torch.relu(1 - self.beta * logits)
        elif self.loss_type == 'ipo':
            # eqn (17) of the paper where beta is the regularization
            # parameter for the IPO loss, denoted by tau in the paper.  # noqa
            loss = (logits - 1 / (2 * self.beta))**2
        elif self.loss_type == 'kto_pair':
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps -
                         reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps -
                           reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = \
                policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.  # noqa
            loss = torch.cat(
                (
                    1 - F.sigmoid(self.beta *
                                  (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta *
                                  (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == 'bco_pair':
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = (
                policy_rejected_logps - reference_rejected_logps)

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards),
                                0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            loss = -F.logsigmoid(
                (self.beta * chosen_logratios) - delta) - F.logsigmoid(
                    -(self.beta * rejected_logratios - delta))
        elif self.loss_type == 'sppo_hard':
            # In the paper (https://arxiv.org/pdf/2405.00675),
            # SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation
            # is conducted outside of the trainer class.
            # The version described here is the hard probability version,
            # where P in Equation (4.7) of Algorithm 1 is set to 1 for
            # the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.beta)**2 + (b + 0.5 / self.beta)**2
        elif self.loss_type == 'nca_pair':
            chosen_rewards = (policy_chosen_logps -
                              reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps -
                                reference_rejected_logps) * self.beta
            loss = (-F.logsigmoid(chosen_rewards) -
                    0.5 * F.logsigmoid(-chosen_rewards) -
                    0.5 * F.logsigmoid(-rejected_rewards))
        else:
            raise ValueError(
                f'Unknown loss type: {self.loss_type}. Should be one of '
                "['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', "
                "'sppo_hard', 'nca_pair', 'robust']")
        chosen_rewards = self.beta * (
            policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
            policy_rejected_logps - reference_rejected_logps)

        loss_dict = {
            'loss': loss,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards
        }
        return loss_dict
