# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from torch import nn

from .sft import SupervisedFinetune


class ORPO(SupervisedFinetune):

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

    def get_var_len_atten_logps(
        self,
        all_logits,
        average_log_prob,
        labels,
        loss_mask,
        cu_seqlens,
    ):
        masked_logps = self._gather_masked_logits(all_logits, labels,
                                                  loss_mask)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # unpack sequence
        unpacked_logps = torch.split(masked_logps, seqlens, dim=1)
        unpacked_mask = torch.split(loss_mask, seqlens, dim=1)

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

    def compute_loss(self, data, data_samples=None):
        all_logits = self.llm(**data).logits

        if not self.use_varlen_attn:
            chosen_nll_loss = self.cross_entropy_loss(
                all_logits[::2], data['labels'].clone()[::2])
            labels = data['labels'].clone()
            labels[labels == -100] = 0
            loss_mask = labels != 0
            chosen_logps, rejected_logps = self.get_logps(
                all_logits, True, labels, loss_mask)
        else:
            message_hub = MessageHub.get_instance('varlen_attn_args')
            rank = dist.get_rank()
            cu_seqlens = message_hub.get_info(f'cumulative_len_rank_{rank}')
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            chosen_logits = torch.split(all_logits, seqlens, dim=1)[::2]
            chosen_logits = torch.cat(chosen_logits, dim=1)
            chosen_labels = torch.split(
                data['labels'].clone(), seqlens, dim=1)[::2]
            chosen_labels = torch.cat(chosen_labels, dim=1)
            chosen_nll_loss = self.cross_entropy_loss(chosen_logits,
                                                      chosen_labels)
            labels = data['labels'].clone()
            labels[labels == -100] = 0
            loss_mask = labels != 0
            chosen_logps, rejected_logps = self.get_var_len_atten_logps(
                all_logits, True, labels, loss_mask, cu_seqlens)
        (losses, chosen_rewards, rejected_rewards, log_odds_ratio,
         log_odds_chosen) = self.odds_ratio_loss(chosen_logps, rejected_logps)
        losses = losses.mean()
        # skip nan loss
        if torch.isnan(chosen_nll_loss):
            chosen_nll_loss = torch.zeros_like(chosen_nll_loss)
        if torch.isnan(losses):
            losses = torch.zeros_like(losses)
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
