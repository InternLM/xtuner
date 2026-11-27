"""The no-loss dense VL path must enter the head module's FSDP hooks."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from xtuner.v1.model.dense.qwen3vl_text import Qwen3VLTextDense
from xtuner.v1.module.head.lm_head import LMHead
from xtuner.v1.module.head.value_head import ValueHead


class _ForwardHarness(nn.Module):
    forward = Qwen3VLTextDense.forward

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.config = SimpleNamespace(return_hidden_states=False)
        self.layers = nn.ModuleDict()
        self.norm = nn.Identity()
        self.lm_head = head

    def rotary_emb(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> None:
        return None

    def _mark_dynamic(self, seq_ctx: Any) -> None:
        return None

    def _maybe_forward_mtp(self, **kwargs: Any) -> None:
        return None


def _sequence(hidden_states: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=None,
        inputs_embeds=hidden_states,
        position_ids=torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0),
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
    )


class TestQwen3VLForwardHead:
    @pytest.mark.parametrize("value_head", [False, True])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_no_loss_enters_head_hooks(self, value_head: bool, bias: bool, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        head = (ValueHead(8, bias=bias) if value_head else LMHead(8, 16, bias=bias)).to(dtype=dtype)
        model = _ForwardHarness(head)
        hidden_states = torch.randn(2, 3, 8, dtype=dtype)
        calls = []
        handle = head.register_forward_pre_hook(lambda module, args: calls.append(args[1]))
        try:
            with torch.no_grad():
                actual = model(seq_ctx=_sequence(hidden_states), loss_ctx=None).logits
        finally:
            handle.remove()
        assert calls == [None]
        assert actual.dtype == torch.float32
        expected = F.linear(hidden_states, head.weight, head.bias).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FSDP is required")
    def test_no_loss_with_independently_sharded_head(self) -> None:
        """Run under torchrun with two or more ranks; no model assets required."""
        import os

        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            pytest.skip("Run this regression with torchrun --nproc-per-node=2")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        try:
            torch.manual_seed(42)
            mesh = init_device_mesh("cuda", (dist.get_world_size(),))
            # Share one process group: destroying/recreating NCCL groups between
            # parametrized pytest cases is not synchronized across torchrun ranks.
            for head in (LMHead(8, 16).cuda(), ValueHead(8).cuda()):
                model = _ForwardHarness(head).cuda()
                hidden_states = torch.randn(2, 3, 8, device="cuda")
                expected = F.linear(hidden_states, head.weight, head.bias).float().detach()
                fully_shard(head, mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy())
                fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy())
                for _ in range(2):
                    with torch.no_grad():
                        actual = model(seq_ctx=_sequence(hidden_states), loss_ctx=None).logits
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            dist.destroy_process_group()
