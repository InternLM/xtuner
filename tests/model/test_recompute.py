"""Gradient checkpointing regression tests.

TestCheckpointWrapper
    test_wrapper_is_transparent_to_state_dict_and_attributes: 包裹后参数名/state_dict/属性访问不变。
    test_non_tensor_signature_preserves_gradients: 关键字参数 + dict 返回值下梯度与不重算一致。
TestDominoEPRecompute
    test_recompute_matches_baseline_under_domino_ep: domino EP 下重算与不重算的 loss/梯度一致。
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.moe import MoE, MoEConfig, SequenceContext
from xtuner.v1.model.utils import apply_gradient_checkpointing
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import NoAuxRouterConfig


class _KeywordOnlyBlock(nn.Module):
    """A forward shape the legacy reentrant checkpoint could not support.

    Tensors arrive nested in a dict and behind a keyword-only argument, and the result is returned
    as a dict rather than a tensor or a tuple of tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.tag = "block"

    def forward(self, inputs: dict[str, torch.Tensor], *, scale: float) -> dict[str, torch.Tensor]:
        return {"out": self.linear(inputs["x"]) * scale}


class TestCheckpointWrapper:
    def test_wrapper_is_transparent_to_state_dict_and_attributes(self):
        # 包裹层不能出现在参数名里，否则 checkpoint 的存/取与非重算模型不兼容。
        plain = _KeywordOnlyBlock()
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())
        assert sorted(name for name, _ in wrapped.named_parameters()) == sorted(
            name for name, _ in plain.named_parameters()
        )
        torch.testing.assert_close(wrapped.state_dict()["linear.weight"], plain.state_dict()["linear.weight"])
        assert wrapped.tag == "block"

    def test_non_tensor_signature_preserves_gradients(self):
        # 非 tensor 签名下梯度必须与不重算完全一致；legacy reentrant 在这里会直接断梯度。
        torch.manual_seed(0)
        plain = _KeywordOnlyBlock()
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        plain({"x": x}, scale=2.0)["out"].square().sum().backward()
        baseline_input_grad, x.grad = x.grad.clone(), None

        wrapped({"x": x}, scale=2.0)["out"].square().sum().backward()

        torch.testing.assert_close(x.grad, baseline_input_grad)
        torch.testing.assert_close(wrapped.linear.weight.grad, plain.linear.weight.grad)


def _build_moe_config(ep_size: int, dispatcher: str) -> MoEConfig:
    router_config = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
    )
    attention_config = MHAConfig(num_attention_heads=32, num_key_value_heads=32, head_dim=16)
    return MoEConfig(
        vocab_size=10240,
        max_position_embeddings=2048,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=4,
        hidden_size=512,
        intermediate_size=2048,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=attention_config,
        tie_word_embeddings=False,
        n_routed_experts=32,
        n_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=1,
        hidden_factor=1.0,
        moe_intermediate_size=512,
        router=router_config,
        ep_size=ep_size,
        dispatcher=dispatcher,
        compile_cfg=False,
    )


class TestDominoEPRecompute(DeterministicDDPTestCase):
    """Regression guard for non-reentrant recompute under domino EP.

    ``checkpoint_wrapper`` used to pin ``CheckpointImpl.REENTRANT`` for the decoder layers. The
    reentrant implementation only tracks gradients for top-level ``torch.Tensor`` arguments, which
    is what forced the decoder layers to pass hidden states positionally and to return a flat tuple.
    This test asserts that, under domino EP (``intra_layer_micro_batch > 1``, the case that pinned
    the choice), enabling recompute reproduces the no-recompute baseline loss and gradients.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))

    @pytest.mark.gpu
    def test_recompute_matches_baseline_under_domino_ep(self):
        self.create_pg("cuda")
        ep_size = self.world_size

        loss_ref, grad_norm_ref, finite_ref = self._run_once(ep_size, "all2all", recompute_ratio=0.0)
        loss_rc, grad_norm_rc, finite_rc = self._run_once(ep_size, "all2all", recompute_ratio=1.0)

        # A broken checkpoint graph shows up as non-finite or missing gradients.
        self.assertTrue(finite_ref)
        self.assertTrue(finite_rc)

        # Recompute is mathematically equivalent to the baseline; only bf16 rounding and the
        # nondeterministic async EP reduction order separate them, so compare with a band that
        # is loose enough for that noise but tight enough to catch a corrupted gradient.
        self.assertTrue(
            torch.allclose(loss_rc, loss_ref, atol=5e-3, rtol=0.0),
            f"recompute loss {loss_rc.item()} diverged from baseline {loss_ref.item()}",
        )
        rel = abs(grad_norm_rc - grad_norm_ref) / (grad_norm_ref + 1e-8)
        self.assertLess(rel, 5e-2, f"recompute grad-norm rel diff {rel} too large")

    def _run_once(self, ep_size: int, dispatcher: str, recompute_ratio: float):
        num_mb = 2
        seq_len = 512
        config = _build_moe_config(ep_size, dispatcher)
        with torch.device("meta"):
            model = MoE(config=config)._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.fully_shard(fsdp_config=FSDPConfig(ep_size=ep_size, recompute_ratio=recompute_ratio, torch_compile=False))

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model.init_weights()

        loss_cfg = CELossConfig()
        seq_ctx_list = []
        loss_ctx_list = []
        # Fixed seed so the baseline and the recompute model consume identical data.
        gen = torch.Generator(device="cuda").manual_seed(1234)
        for _ in range(num_mb):
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len + 1), device="cuda", generator=gen)
            seq_ctx_list.append(SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1],)))
            loss_ctx_list.append(loss_cfg.build(data={"shifted_labels": input_ids[:, 1:]}, sp_mesh=None))
        loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches(loss_ctx_list)

        out = model(seq_ctx=seq_ctx_list, loss_ctx=[{"lm": lc} for lc in loss_ctx_list])
        loss = out["loss"]
        loss.backward()

        grad_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        all_finite = True
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            all_finite = all_finite and bool(torch.isfinite(g).all())
            grad_sq += g.float().pow(2).sum()
        dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)
        grad_norm = grad_sq.sqrt().item()

        loss_val = loss.detach().float()
        dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)

        del model, out, loss
        torch.cuda.empty_cache()
        return loss_val, grad_norm, all_finite
