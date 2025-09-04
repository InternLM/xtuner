from unittest import TestCase
import random
import os
from pydantic import BaseModel, ConfigDict, model_validator
from typing import TYPE_CHECKING, Literal, Union, Any
import parametrize

from torch.testing._internal.common_distributed import DistributedTestBase
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
import torch.nn as nn
import torch.nn.functional as F
from xtuner.v1.rl.oreal.loss import OrealLossConfig
from xtuner.v1.rl.base import RLLossContextInputItem
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.rl.utils import sp_split, gather_logprobs
from xtuner.v1.rl.loss_fn import kl_penalty
from xtuner.v1.data_proto.utils import unpack_sequence
from xtuner.v1.utils.test_utils import init_data_mesh


def policy_loss(logprobs, old_logprobs, advantages, loss_weights, cliprange_low, cliprange_high):
    ratio = (logprobs - old_logprobs.detach()).exp()
    loss1 = -ratio * advantages
    loss2 = -ratio.clamp(1 - cliprange_low, 1 + cliprange_high) * advantages
    loss_max = torch.max(loss1, loss2)
    loss = (loss_max * loss_weights.to(loss_max.dtype)).sum()
    return loss


class TestOrealLoss(DistributedTestBase):

    @parametrize.parametrize(
        "grad_acc, sp_size, kl_loss_coef, loss_mode, chunk_size, atol, rtol",
        [
            (1, 1, 0, "eager", None, 1e-3, 1e-3),
            (2, 2, 0, "eager", None, 1e-3, 1e-3),
            (2, 2, 0, "chunk", 100, 1e-3, 1e-3),
            (1, 1, 1, "eager", None, 1e-3, 1e-3),
            (2, 2, 1, "eager", None, 1e-3, 1e-3),
        ],
    )
    def test_grpo_loss(self, grad_acc, sp_size, kl_loss_coef, loss_mode, chunk_size, atol, rtol):
        device = 'cuda'
        pg = self.create_pg(device)

        dtype = torch.bfloat16
        input_dim = 2
        vocab_size = 40
        cliprange_low = 0.2
        cliprange_high = 0.2
        positive_loss_factor=1.0
        pos_sft_loss_weight=1.0
        pos_policy_loss_weight=1.0
        negative_loss_factor=1.0
        torch.manual_seed(42)
        random.seed(42)
        emb1 = nn.Embedding(vocab_size, input_dim).to(device=device, dtype=dtype)
        emb2 = nn.Embedding(vocab_size, input_dim).to(device=device, dtype=dtype)
        emb2.weight.data = emb1.weight.data.clone()
        lm_head1 = nn.Linear(input_dim , vocab_size, bias=False).to(device=device, dtype=dtype)
        lm_head2 = nn.Linear(input_dim , vocab_size, bias=False).to(device=device, dtype=dtype)
        lm_head2.weight.data = lm_head1.weight.data.clone()

        noise = torch.randn(emb1.weight.shape, device='cuda', dtype=emb1.weight.dtype) * 0.01
        emb1_old = nn.Embedding(vocab_size, input_dim).to(device=device, dtype=dtype)
        emb1_old.weight.data = emb1.weight.data.clone() + noise
        emb2_old = nn.Embedding(vocab_size, input_dim).to(device=device, dtype=dtype)
        emb2_old.weight.data = emb2.weight.data.clone() + noise
        lm_head1_old = nn.Linear(input_dim , vocab_size, bias=False).to(device=device, dtype=dtype)
        lm_head1_old.weight.data = lm_head1.weight.data.clone() + noise
        lm_head2_old = nn.Linear(input_dim , vocab_size, bias=False).to(device=device, dtype=dtype)
        lm_head2_old.weight.data = lm_head2.weight.data.clone() + noise

        emb1.train()
        lm_head1.train()
        emb2.train()
        lm_head2.train()
        emb1_old.eval()
        lm_head1_old.eval()
        emb2_old.eval()
        lm_head2_old.eval()

        torch.manual_seed(42)
        world_size = dist.get_world_size()
        dp_size = world_size // sp_size
        data_mesh = init_data_mesh("cuda", sp_size)
        dp_mesh = data_mesh["dp"]
        sp_mesh = data_mesh["sp"]
        dp_rank = dp_mesh.get_local_rank()
        sp_rank = sp_mesh.get_local_rank()
        rank = dist.get_rank()
        prompt_list = [torch.randint(0, vocab_size, (1, random.randint(50, 100)), device=device, dtype=torch.long) for _ in range(dp_size * grad_acc)]
        response_list = [torch.randint(0, vocab_size, (1, random.randint(500, 1000)), device=device, dtype=torch.long) for _ in range(dp_size * grad_acc)]

        input_ids_list = []
        shifted_labels_list = []
        advantage_list = []
        for prompt, response in zip(prompt_list, response_list):
            input_ids = torch.cat([prompt, response], dim=1)
            shifted_labels = [-100] * (prompt.shape[1] - 1) + response[0].cpu().tolist() + [-100]
            shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64, device=device).unsqueeze(0)
            input_ids_list.append(input_ids)
            shifted_labels_list.append(shifted_labels)
            advantage_list.append(random.random() * (-1) ** random.randint(0, 1))
        
        # 1 gpu, pack inputs and labels
        num_tokens = [ids.shape[1] for ids in input_ids_list]
        num_grad_tokens = [(labels >= 0).sum() for labels in shifted_labels_list]
        global_grad_tokens = sum(num_grad_tokens)
        global_positive_tokens = sum([num if adv > 0 else 0 for adv, num in zip(advantage_list, num_grad_tokens)])
        global_negative_tokens = global_grad_tokens - global_positive_tokens

        input_ids_ref = torch.cat(input_ids_list, dim=1)
        shifted_labels_ref = torch.cat(shifted_labels_list, dim=1)

        with torch.no_grad():
            logits_ref = lm_head1_old(emb1_old(input_ids_ref)).float()
            old_logprobs_ref = gather_logprobs(logits_ref, shifted_labels_ref)
            if kl_loss_coef > 0:
                ref_logprobs_ref = old_logprobs_ref.clone()
        
        logits_ref = lm_head1(emb1(input_ids_ref)).float()
        sft_loss_ref = F.cross_entropy(
            logits_ref.squeeze(),
            shifted_labels_ref.squeeze(),
            ignore_index=-100,
            reduction="none",
        ).unsqueeze(0)

        logprobs_ref = gather_logprobs(logits_ref, shifted_labels_ref)
        logprobs_ref_list = unpack_sequence(logprobs_ref, num_tokens, dim=1)
        sft_loss_ref_list = unpack_sequence(sft_loss_ref, num_tokens, dim=1)
        old_logprobs_ref_list = unpack_sequence(old_logprobs_ref, num_tokens, dim=1)
        if kl_loss_coef > 0:
            ref_logprobs_ref_list = unpack_sequence(ref_logprobs_ref, num_tokens, dim=1)

        _losses = []

        for i in range(len(logprobs_ref_list)):
            assert shifted_labels_list[i].numel() == num_tokens[i]
            _num_grad_tokens = (shifted_labels_list[i] >= 0).sum()

            _logprobs = logprobs_ref_list[i][0, -_num_grad_tokens - 1 : -1]
            _old_logprobs = old_logprobs_ref_list[i][0, -_num_grad_tokens - 1 : -1]
            _judger_advantages = advantage_list[i]

            _sft_is_weight = 1
            _sft_loss = sft_loss_ref_list[i][0, -_num_grad_tokens - 1 : -1]
            _sft_loss = (_sft_loss * _sft_is_weight).sum()

            if _judger_advantages > 0:
                _pos_loss_factor = positive_loss_factor / global_positive_tokens
                _sft_loss = _sft_loss * _pos_loss_factor.to(_sft_loss.dtype)
                _loss_weights = torch.ones_like(_logprobs, dtype=torch.float32) * _pos_loss_factor
                _pos_policy = policy_loss(
                    _logprobs, _old_logprobs, _judger_advantages, _loss_weights, cliprange_low, cliprange_high
                )
                _positive_loss = (
                    _sft_loss * pos_sft_loss_weight
                    + _pos_policy * pos_policy_loss_weight
                )
                _negative_loss = _sft_loss * 0
            else:
                _positive_loss = _sft_loss * 0
                _neg_loss_factor = negative_loss_factor / global_negative_tokens
                _loss_weights = torch.ones_like(_logprobs, dtype=torch.float32) * _neg_loss_factor
                _negative_loss = policy_loss(
                    _logprobs, _old_logprobs, _judger_advantages, _loss_weights, cliprange_low, cliprange_high
                )

            _loss = _positive_loss + _negative_loss
            if kl_loss_coef > 0:
                _ref_logprobs = ref_logprobs_ref_list[i][0, -_num_grad_tokens - 1 : -1]
                _kl_loss_weight = torch.ones_like(_ref_logprobs, dtype=torch.float32) * kl_loss_coef / global_grad_tokens
                _kl_loss = kl_penalty(_logprobs, _ref_logprobs, _kl_loss_weight, "low_var_kl")
                _loss = _loss + _kl_loss
            _losses.append(_loss)
        
        loss_ref = sum(_losses)
        loss_ref.backward()

        # 8 gpus
        loss_cfg = OrealLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=cliprange_high,
                cliprange_low=cliprange_low,
                loss_type="vanilla",
            ),
            positive_loss_factor=positive_loss_factor,
            pos_sft_loss_weight=pos_sft_loss_weight,
            pos_policy_loss_weight=pos_policy_loss_weight,
            negative_loss_factor=negative_loss_factor,
            mode=loss_mode,
            chunk_size=chunk_size,
            use_kl_loss=kl_loss_coef > 0,
            kl_loss_coef=kl_loss_coef,
            kl_loss_type="low_var_kl",
        )

        input_ids_list_rank = input_ids_list[dp_rank::dp_size]
        shifted_labels_list_rank = shifted_labels_list[dp_rank::dp_size]
        advantages_list_rank = advantage_list[dp_rank::dp_size]
        for iter_idx in range(grad_acc):
            length = input_ids_list_rank[iter_idx].shape[1]
            advantage = advantages_list_rank[iter_idx]
            advantages = torch.tensor([advantage] * length, dtype=torch.float32, device=device).view(1, -1)
            advantages_list_rank[iter_idx] = advantages
        
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_input_list: list[RLLossContextInputItem] = []
        for iter_idx in range(grad_acc):
            input_ids = input_ids_list_rank[iter_idx]
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=device)
            loss_ctx_input = RLLossContextInputItem(
                shifted_labels=shifted_labels_list_rank[iter_idx],
                advantages=advantages_list_rank[iter_idx],
            )
            if sp_size > 1:
                seq_ctx = seq_ctx.split(sp_mesh)
                loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx_input_list.append(loss_ctx_input)

        with torch.no_grad():
            for iter_idx in range(grad_acc):
                seq_ctx = seq_ctx_list[iter_idx]
                loss_ctx_input = loss_ctx_input_list[iter_idx]
                logits = lm_head2_old(emb2_old(seq_ctx.input_ids)).float()
                old_logprobs = gather_logprobs(logits, loss_ctx_input.shifted_labels)
                loss_ctx_input.old_logprobs = old_logprobs
                if kl_loss_coef > 0:
                    loss_ctx_input.ref_logprobs = old_logprobs.clone()
        
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
        )

        for iter_idx in range(grad_acc):
            seq_ctx = seq_ctx_list[iter_idx]
            loss_kwargs = batches_loss_kwargs[iter_idx]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)

            hidden_states = emb2(seq_ctx.input_ids)
            head_weight = lm_head2.weight
            out = loss_ctx.forward(hidden_states, head_weight)
            loss = out[0]
            loss.backward()
        
        dist.all_reduce(emb2.weight.grad, op=dist.ReduceOp.AVG)
        dist.all_reduce(lm_head2.weight.grad, op=dist.ReduceOp.AVG)
        self.assertTrue(torch.allclose(lm_head1.weight.grad, lm_head2.weight.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(emb1.weight.grad, emb2.weight.grad, atol=atol, rtol=rtol))

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
