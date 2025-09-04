from unittest import TestCase
import random
import os
import parametrize

from torch.testing._internal.common_distributed import DistributedTestBase
import torch
import torch.distributed as dist
import torch.nn as nn
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.rl.base import RLLossContextInputItem
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.rl.utils import gather_logprobs
from xtuner.v1.rl.loss_fn import kl_penalty
from xtuner.v1.utils.test_utils import init_data_mesh


class TestGRPOLoss(DistributedTestBase):
    
    @parametrize.parametrize(
        "grad_acc, sp_size, kl_loss_coef, loss_mode, chunk_size, atol, rtol",
        [
            (1, 1, 0, "eager", None, 1e-4, 1e-4),
            (2, 2, 0, "eager", None, 1e-4, 1e-4),
            (2, 2, 0, "chunk", 100, 1e-4, 1e-4),
            (1, 1, 1, "eager", None, 1e-4, 1e-4),
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
        input_ids_ref = torch.cat(input_ids_list, dim=1)
        shifted_labels_ref = torch.cat(shifted_labels_list, dim=1)
        advantages_ref = torch.tensor(advantage_list, dtype=torch.float32, device=device).unsqueeze(0)
        num_tokens = [ids.shape[1] for ids in input_ids_list]
        num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=device)
        advantages_ref = torch.repeat_interleave(advantages_ref, num_tokens, dim=1)
        
        with torch.no_grad():
            logits_ref = lm_head1_old(emb1_old(input_ids_ref)).float()
            old_logprobs_ref = gather_logprobs(logits_ref, shifted_labels_ref)
            if kl_loss_coef > 0:
                ref_logprobs_ref = old_logprobs_ref.clone()

        logits_ref = lm_head1(emb1(input_ids_ref)).float()
        logprobs_ref = gather_logprobs(logits_ref, shifted_labels_ref.clip(0))
        ratio = (logprobs_ref - old_logprobs_ref.detach()).exp()
        loss1 = -ratio * advantages_ref
        loss2 = -ratio.clamp(1 - cliprange_low, 1 + cliprange_high) * advantages_ref
        loss_max_ref = torch.max(loss1, loss2)
        mask = (shifted_labels_ref != -100).int()
        loss = (loss_max_ref * mask.to(loss_max_ref.dtype)).sum()
        loss = loss / mask.sum()
        if kl_loss_coef > 0:
            kl_loss_weight = mask.clone().float()
            kl_loss_weight = kl_loss_weight * kl_loss_coef
            kl_loss = kl_penalty(logprobs_ref, ref_logprobs_ref, kl_loss_weight, "low_var_kl")
            kl_loss = kl_loss / mask.sum()
            loss = loss + kl_loss
        loss.backward()

        # 8 gpus
        loss_cfg = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=cliprange_high,
                cliprange_low=cliprange_low,
                loss_type="vanilla",
            ),
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
