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
            all_logits,  # bs, seqlen,vocab_size
            average_log_prob,  # bs, seqlen,vocab_size
            labels,  # bs, seqlen
            loss_mask,  # bs, seqlen
    ):
        all_logps = self._gather_masked_logits(all_logits, labels,
                                               loss_mask).sum(-1)

        if average_log_prob:  # average_log_prob
            all_logps = all_logps / loss_mask.sum(-1)

        chosen_logps = all_logps[::2]
        rejected_logps = all_logps[1::2]
        return chosen_logps, rejected_logps

    def get_var_len_atten_logps(self, all_logits, average_log_prob, labels,
                                loss_mask, cu_seqlens, attention_mask):
        masked_logps = self._gather_masked_logits(all_logits, labels,
                                                  loss_mask)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # unpack sequence
        unpacked_logps = torch.split(masked_logps, seqlens, dim=1)
        unpacked_mask = torch.split(loss_mask, seqlens, dim=1)

        if attention_mask is not None:
            # It indicate that we pad the original sequence, labels,
            # position_ids and cumulative_len for sequence parallel if the
            # attention_mask is not None.
            # We then need to remove the padded segments.
            assert False in attention_mask
            unpacked_logps = unpacked_logps[:-1]
            unpacked_mask = unpacked_mask[:-1]
            assert len(unpacked_logps) % 2 == 0

        def compute_logps(logps, mask, idx):
            logp = logps[idx].sum(-1)
            if average_log_prob:
                logp /= mask[idx].sum(-1)
            return logp

        chosen_logps = [
            compute_logps(unpacked_logps, unpacked_mask, 2 * i)
            for i in range(len(unpacked_logps) // 2)
        ]
        rejected_logps = [
            compute_logps(unpacked_logps, unpacked_mask, 2 * i + 1)
            for i in range(len(unpacked_logps) // 2)
        ]

        return (torch.stack(chosen_logps), torch.stack(rejected_logps))

    def cross_entropy_loss(self, logits, labels):
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
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
        labels_ori = data.pop('labels')

        if get_sequence_parallel_world_size() > 1:
            data = self._split_for_sequence_parallel(data)

        all_logits = self.llm(**data).logits
        if get_sequence_parallel_world_size() > 1:
            all_logits = gather_forward_split_backward(
                all_logits,
                dim=1,
                sp_group=get_sequence_parallel_group(),
                grad_scale='up')

        if not self.use_varlen_attn:
            chosen_nll_loss = self.cross_entropy_loss(all_logits[::2],
                                                      labels_ori.clone()[::2])
            labels = labels_ori.clone()
            labels[labels == -100] = 0
            loss_mask = labels != 0
            chosen_logps, rejected_logps = self.get_logps(
                all_logits, True, labels, loss_mask)
        else:
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

            attention_mask = data['attention_mask']
            if attention_mask is not None:
                # It indicate that we pad the original sequence, labels,
                # position_ids and cumulative_len for sequence parallel if the
                # attention_mask is not None.
                # We then need to remove the padded segments.
                logits = torch.split(all_logits, seqlens, dim=1)[:-1]
                assert len(logits) % 2 == 0
                chosen_logits = logits[::2]
                labels = torch.split(labels_ori.clone(), seqlens, dim=1)[:-1]
                assert len(labels) % 2 == 0
                chosen_labels = labels[::2]
            else:
                chosen_logits = torch.split(all_logits, seqlens, dim=1)[::2]
                chosen_labels = torch.split(
                    labels_ori.clone(), seqlens, dim=1)[::2]

            chosen_logits = torch.cat(chosen_logits, dim=1)
            chosen_labels = torch.cat(chosen_labels, dim=1)
            chosen_nll_loss = self.cross_entropy_loss(chosen_logits,
                                                      chosen_labels)
            labels = labels_ori.clone()
            labels[labels == -100] = 0
            loss_mask = labels != 0
            chosen_logps, rejected_logps = self.get_var_len_atten_logps(
                all_logits, True, labels, loss_mask, cu_seqlens,
                attention_mask)
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
