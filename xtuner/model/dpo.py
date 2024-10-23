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


def disable_grad(model):
    # freeze parameters
    parameter_names = [n for n, _ in model.named_parameters()]
    for param_name in parameter_names:
        param = model.get_parameter(param_name)
        param.requires_grad = False
    return model.eval()


def create_reference_model(model):
    if is_deepspeed_zero3_enabled():
        raise ValueError('DeepSpeed ZeRO-3 is enabled and is not compatible '
                         'with `create_reference_model()`. Please instantiate '
                         'your reference model directly with '
                         '`AutoCausalLM.from_pretrained()`.')
    ref_model = deepcopy(model)
    ref_model = disable_grad(ref_model)
    return ref_model


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
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.beta = beta

        if ref_llm is not None:
            ref_llm = self.build_llm_from_cfg(
                ref_llm, kwargs.get('use_varlen_attn', False),
                kwargs.get('max_position_embeddings', None))
            self.ref_llm = disable_grad(ref_llm)
        else:
            self.ref_llm = None if self.use_lora else create_reference_model(
                self.llm)

    def _gather_masked_logits(self, logits, labels, mask):
        logits = torch.gather(
            logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        return logits * mask

    def get_logps(
            self,
            policy_logps,  # bs, seqlen,vocab_size
            ref_logps,  # bs, seqlen,vocab_size
            loss_mask,  # bs, seqlen
    ):
        policy_logps = policy_logps[:, :-1].sum(-1)
        ref_logps = ref_logps[:, :-1].sum(-1)
        loss_mask = loss_mask[:, :-1]

        if self.loss_type == 'ipo':  # average_log_prob
            policy_logps = policy_logps / loss_mask.sum(-1)
            ref_logps = ref_logps / loss_mask.sum(-1)

        policy_chosen_logps = policy_logps[::2]
        policy_rejected_logps = policy_logps[1::2]
        reference_chosen_logps = ref_logps[::2]
        reference_rejected_logps = ref_logps[1::2]
        return (policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps)

    def get_var_len_atten_logps(self, policy_logps, ref_logps, loss_mask,
                                cu_seqlens, attention_mask):
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # unpack sequence
        unpacked_policy_logps = torch.split(policy_logps, seqlens, dim=1)
        unpacked_ref_logps = torch.split(ref_logps, seqlens, dim=1)
        unpacked_loss_mask = torch.split(loss_mask, seqlens, dim=1)
        if attention_mask is not None:
            # It indicate that we pad the original sequence, labels,
            # position_ids and cumulative_len for sequence parallel if the
            # attention_mask is not None.
            # We then need to remove the padded segments.
            assert False in attention_mask
            unpacked_policy_logps = unpacked_policy_logps[:-1]
            unpacked_ref_logps = unpacked_ref_logps[:-1]
            unpacked_loss_mask = unpacked_loss_mask[:-1]
            assert len(unpacked_policy_logps) % 2 == 0

        def compute_logps(_logps, _mask):
            _logps = _logps[:, :-1].sum(-1)
            _mask = _mask[:, :-1]
            if self.loss_type == 'ipo':
                _logps /= _mask.sum(-1)
            return _logps

        (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
         reference_rejected_logps) = [], [], [], []
        for i in range(len(unpacked_policy_logps) // 2):
            chosen = unpacked_policy_logps[2 * i]
            rejected = unpacked_policy_logps[2 * i + 1]
            chosen_ref = unpacked_ref_logps[2 * i]
            rejected_ref = unpacked_ref_logps[2 * i + 1]
            chosen_mask = unpacked_loss_mask[2 * i]
            rejected_mask = unpacked_loss_mask[2 * i + 1]
            policy_chosen_logps.append(compute_logps(chosen, chosen_mask))
            policy_rejected_logps.append(
                compute_logps(rejected, rejected_mask))
            reference_chosen_logps.append(
                compute_logps(chosen_ref, chosen_mask))
            reference_rejected_logps.append(
                compute_logps(rejected_ref, rejected_mask))

        return (torch.stack(policy_chosen_logps),
                torch.stack(policy_rejected_logps),
                torch.stack(reference_chosen_logps),
                torch.stack(reference_rejected_logps))

    @staticmethod
    def _split_for_sequence_parallel(data):
        # attention mask should not be split
        ARGS_NEED_TO_SPLIT = ('input_ids', 'position_ids', 'labels')
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
        # shift labels first and add a dummy label at the end, to support sequence parallel  # noqa
        data['labels'] = torch.cat(
            (data['labels'][:, 1:], torch.zeros_like(data['labels'][:, :1])),
            dim=1)
        tmp_label = data['labels'].clone()
        tmp_label[tmp_label == 0] = -100
        all_loss_mask = data[
            'labels'] != -100  # loss mask of all tokens in all sp ranks  # noqa

        if get_sequence_parallel_world_size() > 1:
            data = self._split_for_sequence_parallel(data)

        all_logits = self.llm(**data).logits
        with torch.no_grad():
            if self.ref_llm is None:
                with self.llm.disable_adapter():
                    all_ref_logits = self.llm(**data).logits
            else:
                all_ref_logits = self.ref_llm(**data).logits

        labels = data['labels']
        labels[labels == -100] = 0
        loss_mask = labels != 0  # loss mask in a single sp rank
        policy_logps = self._gather_masked_logits(all_logits, labels,
                                                  loss_mask)
        ref_logps = self._gather_masked_logits(all_ref_logits, labels,
                                               loss_mask)

        if get_sequence_parallel_world_size() > 1:
            policy_logps = gather_forward_split_backward(
                policy_logps,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')
            ref_logps = gather_forward_split_backward(
                ref_logps,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')

        if not self.use_varlen_attn:
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_logps(
                 policy_logps, ref_logps, all_loss_mask)
        else:
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            (policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps,
             reference_rejected_logps) = self.get_var_len_atten_logps(
                 policy_logps, ref_logps, all_loss_mask, cu_seqlens,
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
