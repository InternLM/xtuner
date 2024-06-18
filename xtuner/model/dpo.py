# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023  # noqa
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.parallel.sequence import (gather_forward_split_backward,
                                      get_sequence_parallel_group,
                                      get_sequence_parallel_world_size,
                                      split_for_sequence_parallel)
from .sft import SupervisedFinetune


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
            self.ref_llm = self.create_reference_model(ref_llm, **kwargs)

    def create_reference_model(self, ref_llm=None, **kwargs):
        ref_model = None
        if ref_llm is None:
            if is_deepspeed_zero3_enabled():
                raise ValueError(
                    'DeepSpeed ZeRO-3 is enabled and is not compatible '
                    'with `deepcopy(self.llm)`. Please instantiate '
                    'your reference model by modifying key `model.ref_llm` '
                    'in your config with `AutoCausalLM.from_pretrained()`.')
            ref_model = deepcopy(self.llm)
        else:
            ref_model = SupervisedFinetune(ref_llm, **kwargs).llm
        # freeze parameters
        parameter_names = [n for n, _ in ref_model.named_parameters()]
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

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
    ):
        labels = labels[:, 1:].clone()
        all_logits = all_logits[:, :-1, :]
        all_ref_logits = all_ref_logits[:, :-1, :]

        labels[labels == -100] = 0
        loss_mask = labels != 0
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

    def get_var_len_atten_logps(self, all_logits, all_ref_logits, labels,
                                cu_seqlens, attention_mask):
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # unpack sequence
        unpacked_logits = torch.split(all_logits, seqlens, dim=1)
        unpacked_ref_logits = torch.split(all_ref_logits, seqlens, dim=1)
        unpacked_labels = torch.split(labels, seqlens, dim=1)
        if attention_mask is not None:
            # It indicate that we pad the original sequence, labels,
            # position_ids and cumulative_len for sequence parallel if the
            # attention_mask is not None.
            # We then need to remove the padded segments.
            assert False in attention_mask
            unpacked_logits = unpacked_logits[:-1]
            unpacked_ref_logits = unpacked_ref_logits[:-1]
            unpacked_labels = unpacked_labels[:-1]
            assert len(unpacked_logits) % 2 == 0

        def compute_logps(_logits, _labels):
            _labels = _labels[:, 1:].clone()
            _logits = _logits[:, :-1, :]
            _labels[_labels == -100] = 0
            loss_mask = _labels != 0
            logps = self._gather_masked_logits(_logits, _labels, loss_mask)
            logps = logps.sum(-1)
            if self.loss_type == 'ipo':
                logps /= loss_mask.sum(-1)
            return logps

        (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
         reference_rejected_logps) = [], [], [], []
        for i in range(len(unpacked_logits) // 2):
            chosen = unpacked_logits[2 * i]
            rejected = unpacked_logits[2 * i + 1]
            chosen_ref = unpacked_ref_logits[2 * i]
            rejected_ref = unpacked_ref_logits[2 * i + 1]
            chosen_label = unpacked_labels[2 * i]
            rejected_label = unpacked_labels[2 * i + 1]
            policy_chosen_logps.append(compute_logps(chosen, chosen_label))
            policy_rejected_logps.append(
                compute_logps(rejected, rejected_label))
            reference_chosen_logps.append(
                compute_logps(chosen_ref, chosen_label))
            reference_rejected_logps.append(
                compute_logps(rejected_ref, rejected_label))

        return (torch.stack(policy_chosen_logps),
                torch.stack(policy_rejected_logps),
                torch.stack(reference_chosen_logps),
                torch.stack(reference_rejected_logps))

    @staticmethod
    def _split_for_sequence_parallel(data):
        # attention mask should not be split
        ARGS_NEED_TO_SPLIT = ('input_ids', 'position_ids')
        sp_group = get_sequence_parallel_group()
        for key in ARGS_NEED_TO_SPLIT:
            val = data.get(key, None)
            if val is not None:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                data[key] = split_for_sequence_parallel(
                    val, dim=1, sp_group=sp_group)
        return data

    def compute_loss(self, data, data_samples=None):
        # modified from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py  # noqa

        labels = data.pop('labels')

        if get_sequence_parallel_world_size() > 1:
            data = self._split_for_sequence_parallel(data)

        all_logits = self.llm(**data).logits
        with torch.no_grad():
            if self.ref_llm is None:
                with self.llm.disable_adapter():
                    all_ref_logits = self.llm(**data).logits
            else:
                all_ref_logits = self.ref_llm(**data).logits

        if get_sequence_parallel_world_size() > 1:
            all_logits = gather_forward_split_backward(
                all_logits,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')
            all_ref_logits = gather_forward_split_backward(
                all_ref_logits,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')

        if not self.use_varlen_attn:
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_logps(
                 all_logits, all_ref_logits, labels)
        else:
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_var_len_atten_logps(
                 all_logits, all_ref_logits, labels, cu_seqlens,
                 data['attention_mask'])

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
                "['sigmoid', 'hinge', 'ipo', 'kto_pair', "
                "'sppo_hard', 'nca_pair', 'robust']")
        # for logging
        chosen_rewards = self.beta * (
            policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
            policy_rejected_logps - reference_rejected_logps)
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        loss_dict = {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_acc': reward_acc,
            'reward_margin': (chosen_rewards - rejected_rewards).mean(),
        }
        return loss_dict
