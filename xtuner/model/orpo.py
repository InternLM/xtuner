# ORPO Authors: Jiwoo Hong, Noah Lee, and James Thorne
# Official code: https://github.com/xfactlab/orpo
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from torch import nn

from xtuner.parallel.sequence import (gather_forward_split_backward,
                                      get_sequence_parallel_group,
                                      get_sequence_parallel_world_size,
                                      split_for_sequence_parallel)
from .sft import SupervisedFinetune


class ORPO(SupervisedFinetune):
    """ORPO: Monolithic Preference Optimization without Reference Model
    https://arxiv.org/abs/2403.07691

    Args:
        beta (float): Weight of the odds_ratio_loss. Defaults to 0.1.
    """

    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _gather_masked_logits(self, logits, labels, mask):
        logits = torch.gather(
            logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        return logits * mask

    def get_logps(
            self,
            all_logps,  # bs, seqlen
            average_log_prob,
            loss_mask,  # bs, seqlen
    ):
        all_logps = all_logps[:, :-1].sum(-1)
        loss_mask = loss_mask[:, :-1]

        if average_log_prob:  # average_log_prob
            all_logps = all_logps / loss_mask.sum(-1)

        chosen_logps = all_logps[::2]
        rejected_logps = all_logps[1::2]
        return chosen_logps, rejected_logps

    def get_var_len_atten_logps(self, all_logps, average_log_prob, loss_mask,
                                cu_seqlens, attention_mask):
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # unpack sequence
        unpacked_logps = torch.split(all_logps, seqlens, dim=1)
        unpacked_loss_mask = torch.split(loss_mask, seqlens, dim=1)
        if attention_mask is not None:
            # It indicate that we pad the original sequence, labels,
            # position_ids and cumulative_len for sequence parallel if the
            # attention_mask is not None.
            # We then need to remove the padded segments.
            assert False in attention_mask
            unpacked_logps = unpacked_logps[:-1]
            unpacked_loss_mask = unpacked_loss_mask[:-1]
            assert len(unpacked_logps) % 2 == 0

        def compute_logps(_logps, _mask):
            _logps = _logps[:, :-1].sum(-1)
            _mask = _mask[:, :-1]
            if average_log_prob:
                _logps /= _mask.sum(-1)
            return _logps

        chosen_logps, rejected_logps = [], []
        for i in range(len(unpacked_logps) // 2):
            chosen = unpacked_logps[2 * i]
            rejected = unpacked_logps[2 * i + 1]
            chosen_mask = unpacked_loss_mask[2 * i]
            rejected_mask = unpacked_loss_mask[2 * i + 1]
            chosen_logps.append(compute_logps(chosen, chosen_mask))
            rejected_logps.append(compute_logps(rejected, rejected_mask))

        return (torch.stack(chosen_logps), torch.stack(rejected_logps))

    def cross_entropy_loss(self, logits, labels):
        logits = logits[..., :-1, :].contiguous()
        # labels are already shifted, now we need to remove the last dummy label  # noqa
        labels = labels[..., :-1].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)
        return loss

    def odds_ratio_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
    ):
        # modified from https://github.com/huggingface/trl/blob/b031adfdb8708f1f295eab6c3f2cb910e8fe0c23/trl/trainer/orpo_trainer.py#L597  # noqa
        # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)  # noqa
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) -
            torch.log1p(-torch.exp(rejected_logps)))
        ratio = F.logsigmoid(log_odds)
        ratio = ratio[~torch.isnan(ratio)]  # select valid loss
        losses = self.beta * ratio

        chosen_rewards = self.beta * chosen_logps
        rejected_rewards = self.beta * rejected_logps

        return losses, chosen_rewards, rejected_rewards, torch.mean(
            ratio), torch.mean(log_odds)

    @staticmethod
    def _split_for_sequence_parallel(data):
        # attention mask should not be split
        ARGS_NEED_TO_SPLIT = ('input_ids', 'position_ids', 'labels',
                              'chosen_rejected_tag')
        sp_group = get_sequence_parallel_group()
        for key in ARGS_NEED_TO_SPLIT:
            val = data.get(key, None)
            if val is not None:
                # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                data[key] = split_for_sequence_parallel(
                    val, dim=1, sp_group=sp_group)
        return data

    def compute_loss(self, data, data_samples=None):
        # shift labels first and add a dummy label at the end, to support sequence parallel  # noqa
        data['labels'] = torch.cat(
            (data['labels'][:, 1:], torch.zeros_like(data['labels'][:, :1])),
            dim=1)
        tmp_label = data['labels'].clone()
        tmp_label[tmp_label == 0] = -100
        # loss mask of all tokens in all sp ranks
        all_loss_mask = data['labels'] != -100

        if self.use_varlen_attn:
            # create a chosen rejected tag for varlen_attn ce loss
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

            chosen_rejected_tag = torch.ones_like(data['labels'])
            unpacked_tag = list(
                torch.split(chosen_rejected_tag, seqlens, dim=1))
            # import pdb; pdb.set_trace()
            for i in range(len(unpacked_tag) // 2):
                # import pdb; pdb.set_trace()
                unpacked_tag[2 * i + 1] *= 0
            chosen_rejected_tag = torch.cat(unpacked_tag, dim=1)
            data['chosen_rejected_tag'] = chosen_rejected_tag

        if get_sequence_parallel_world_size() > 1:
            data = self._split_for_sequence_parallel(data)
        chosen_rejected_tag = data.pop('chosen_rejected_tag', None)
        all_logits = self.llm(**data).logits

        labels = data['labels'].clone()
        labels[labels == -100] = 0
        loss_mask = labels != 0  # loss mask in a single sp rank
        all_logps = self._gather_masked_logits(all_logits, labels, loss_mask)
        if get_sequence_parallel_world_size() > 1:
            all_logps = gather_forward_split_backward(
                all_logps,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')

        if not self.use_varlen_attn:
            chosen_nll_loss = self.cross_entropy_loss(all_logits[::2],
                                                      data['labels'][::2])
            chosen_logps, rejected_logps = self.get_logps(
                all_logps, True, all_loss_mask)
        else:
            chosen_idxs = chosen_rejected_tag == 1
            chosen_logits = all_logits[chosen_idxs]
            chosen_labels = data['labels'][chosen_idxs]
            chosen_nll_loss = self.cross_entropy_loss(chosen_logits,
                                                      chosen_labels)

            chosen_logps, rejected_logps = self.get_var_len_atten_logps(
                all_logps, True, all_loss_mask, cu_seqlens,
                data['attention_mask'])
        (losses, chosen_rewards, rejected_rewards, log_odds_ratio,
         log_odds_chosen) = self.odds_ratio_loss(chosen_logps, rejected_logps)
        losses = losses.mean()
        # skip nan loss
        if torch.isnan(chosen_nll_loss):
            chosen_nll_loss = all_logits.mean() * 0
        if torch.isnan(losses):
            losses = all_logits.mean() * 0
        loss = chosen_nll_loss - losses

        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        loss_dict = {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_acc': reward_acc,
            'reward_margin': (chosen_rewards - rejected_rewards).mean(),
            'log_odds_ratio': log_odds_ratio,
            'log_odds_chosen': log_odds_chosen,
            'nll_loss': chosen_nll_loss.detach().mean()
        }
        return loss_dict
