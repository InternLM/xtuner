from unittest import TestCase
import random
import os

from torch.testing._internal.common_distributed import DistributedTestBase
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from xtuner.v1.loss import CELossContext
from xtuner.v1.rl.grpo.loss import GRPOLossContext
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils.test_utils import assert_verbose_allclose
from xtuner.v1.loss.utils import cal_global_grad_tokens
import parametrize


class TestGRPOLoss(DistributedTestBase):
    def _gather_logprobs(self, shifted_logits, shifted_labels):
        shift_logprobs = F.log_softmax(shifted_logits, dim=-1)
        shift_logprobs = shift_logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
        return shift_logprobs
    
    def test_grpo_loss(self):
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

        torch.manual_seed(42)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        prompt_list = [torch.randint(0, vocab_size, (1, random.randint(50, 100)), device=device, dtype=torch.long) for _ in range(world_size)]
        response_list = [torch.randint(0, vocab_size, (1, random.randint(500, 1000)), device=device, dtype=torch.long) for _ in range(world_size)]

        input_ids_list = []
        shift_labels_list = []
        advantage_list = []
        for prompt, response in zip(prompt_list, response_list):
            input_ids = torch.cat([prompt, response], dim=1)
            shift_labels = [-100] * (prompt.shape[1] - 1) + response[0].cpu().tolist() + [-100]
            shift_labels = torch.tensor(shift_labels, dtype=torch.int64, device=device).unsqueeze(0)
            input_ids_list.append(input_ids)
            shift_labels_list.append(shift_labels)
            advantage_list.append(random.random() * (-1) ** random.randint(0, 1))
        
        # 1 gpu, pack inputs and labels
        input_ids_ref = torch.cat(input_ids_list, dim=1)
        shift_labels_ref = torch.cat(shift_labels_list, dim=1)
        advantages_ref = torch.tensor(advantage_list, dtype=torch.float32, device=device).unsqueeze(0)
        num_tokens = [ids.shape[1] for ids in input_ids_list]
        num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=device)
        advantages_ref = torch.repeat_interleave(advantages_ref, num_tokens, dim=1)
        
        with torch.no_grad():
            logits_ref = lm_head1(emb1(input_ids_ref))
            old_logprobs_ref = self._gather_logprobs(logits_ref, shift_labels_ref.clip(0))

        logits_ref = lm_head1(emb1(input_ids_ref))
        logprobs_ref = self._gather_logprobs(logits_ref, shift_labels_ref.clip(0))
        ratio = (logprobs_ref - old_logprobs_ref.detach()).exp()
        loss1 = -ratio * advantages_ref
        loss2 = -ratio.clamp(1 - cliprange_low, 1 + cliprange_high) * advantages_ref
        loss_max_ref = torch.max(loss1, loss2)
        mask = (shift_labels_ref != -100).int()
        loss = (loss_max_ref * mask.to(loss_max_ref.dtype)).sum()
        loss = loss / mask.sum()
        loss.backward()

        # 8 gpus
        input_ids = input_ids_list[rank]
        shift_labels = shift_labels_list[rank]
        advantages = advantage_list[rank]
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device).view(1, 1)

        with torch.no_grad():
            logits = lm_head2(emb2(input_ids))
            old_logprobs = self._gather_logprobs(logits, shift_labels.clip(0))

        loss_ctx = GRPOLossContext(cliprange_high=cliprange_high, cliprange_low=cliprange_low)
        seq_ctx = SequenceContext.from_input_ids((input_ids,), device=device)
        out = loss_ctx.build_list_ctx(
            [{'seq_ctx': seq_ctx, 'shift_labels': shift_labels, 'old_logprobs': old_logprobs, 'advantage': advantages}],
        )
        seq_ctx = out[0]['seq_ctx']
        loss_ctx = out[0]['loss_ctx']

        hidden_states = emb2(input_ids)
        head_weight = lm_head2.weight
        out = loss_ctx.forward(hidden_states, head_weight)
        loss = out[0]
        loss.backward()

        dist.all_reduce(emb2.weight.grad, op=dist.ReduceOp.AVG)
        dist.all_reduce(lm_head2.weight.grad, op=dist.ReduceOp.AVG)
        self.assertTrue(torch.allclose(lm_head1.weight.grad, lm_head2.weight.grad, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(emb1.weight.grad, emb2.weight.grad, atol=1e-4, rtol=1e-4))
    
    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
