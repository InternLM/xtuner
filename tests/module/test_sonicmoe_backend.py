import importlib.util
import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import ValidationError
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.moe import MoEConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer
from xtuner.v1.module.moe_backend import SonicMoEBackend, SonicMoEBackendConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.ops.moe.sonicmoe import SonicMoEOp
from xtuner.v1.ops.moe.token_rounding import build_token_rounding_metadata


SONICMOE_AVAILABLE = importlib.util.find_spec("sonicmoe") is not None


def _stub_sonicmoe_decoder_layer() -> MoEDecoderLayer:
    layer = MoEDecoderLayer.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.experts = types.SimpleNamespace(uses_sonicmoe=True)  # type: ignore[assignment]

    def pre_moe_forward(self, hidden_states, **_):
        marker = int(hidden_states.flatten()[0])
        router_results = {
            "logits": torch.full((1, 2), marker, dtype=torch.float32),
            "router_weights": torch.full((1, 2), marker + 1, dtype=torch.float32),
            "topk_ids": torch.full((1, 1), marker + 2, dtype=torch.int64),
        }
        return hidden_states + 10, hidden_states + 20, router_results

    def sonicmoe_forward(self, hidden_states, residual, router_results):
        del router_results
        return hidden_states + residual

    layer._pre_moe_forward = types.MethodType(pre_moe_forward, layer)  # type: ignore[method-assign]
    layer._sonicmoe_forward = types.MethodType(sonicmoe_forward, layer)  # type: ignore[method-assign]
    return layer


def test_sonicmoe_decoder_preserves_router_result_contract():
    layer = _stub_sonicmoe_decoder_layer()
    hidden_states = [torch.tensor([[[1.0]]]), torch.tensor([[[2.0]]])]
    position_embeddings = [(torch.empty(0), torch.empty(0))] * 2
    seq_ctx = [object(), object()]

    single_output = layer._forward(
        hidden_states[0],
        seq_ctx=seq_ctx[0],  # type: ignore[arg-type]
        position_embeddings=position_embeddings[0],
    )
    assert len(single_output) == 4
    torch.testing.assert_close(single_output[3], torch.tensor([[3]], dtype=torch.int64))

    micro_batch_output = layer._sonicmoe_micro_batch_forward(
        hidden_states,
        seq_ctx_list=seq_ctx,  # type: ignore[arg-type]
        position_embeddings_list=position_embeddings,
    )
    assert len(micro_batch_output) == 8
    torch.testing.assert_close(micro_batch_output[6], torch.tensor([[3]], dtype=torch.int64))
    torch.testing.assert_close(micro_batch_output[7], torch.tensor([[4]], dtype=torch.int64))


def _mock_backend(monkeypatch: pytest.MonkeyPatch, captured: dict | None = None) -> SonicMoEBackend:
    activation = object()

    def fake_forward(*args, **kwargs):
        if captured is not None:
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["activation"] = activation
        x, router_scores, _, expert_indices, w1, b1, w2, b2, num_experts, *_ = args
        zero = router_scores.sum().to(x.dtype) + w1.sum() + w2.sum()
        if b1 is not None and b2 is not None:
            zero = zero + b1.sum() + b2.sum()
        output = x + zero * 0
        frequency = expert_indices.long().bincount(minlength=num_experts).to(torch.int32)
        return output, frequency

    monkeypatch.setattr(
        SonicMoEOp,
        "_resolve_official_api",
        staticmethod(lambda: (fake_forward, activation)),
    )
    return SonicMoEBackend(SonicMoEBackendConfig())


def _reference_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1w3: torch.Tensor,
    w2: torch.Tensor,
    b1w3: torch.Tensor | None,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    for expert_idx in range(w1w3.shape[0]):
        token_idx, topk_slot = torch.where(topk_ids == expert_idx)
        if token_idx.numel() == 0:
            continue
        gate, up = F.linear(
            hidden_states[token_idx],
            w1w3[expert_idx],
            None if b1w3 is None else b1w3[expert_idx],
        ).chunk(2, dim=-1)
        expert_output = F.linear(
            F.silu(gate) * up,
            w2[expert_idx],
            None if b2 is None else b2[expert_idx],
        )
        expert_output = expert_output * topk_weights[token_idx, topk_slot, None]
        output = output.index_add(0, token_idx, expert_output.to(output.dtype))
    return output


def _reference_general_routing_moe(
    hidden_states: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    for expert_idx in range(w1w3.shape[0]):
        positions = expert_indices == expert_idx
        selected_tokens = token_indices[positions].long()
        if selected_tokens.numel() == 0:
            continue
        gate, up = F.linear(hidden_states[selected_tokens], w1w3[expert_idx]).chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, w2[expert_idx])
        output = output.index_add(
            0,
            selected_tokens,
            expert_output * router_scores[positions, None].to(expert_output.dtype),
        )
    return output


class _SonicMoECompileWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.backend = SonicMoEBackend(
            SonicMoEBackendConfig(
                routing_mode="token_rounding",
                rounding_mode="nearest",
                rounding_quantum=8,
            )
        )
        self.w1w3 = nn.Parameter(torch.randn(4, 256, 256, dtype=torch.bfloat16) * 0.02)
        self.w2 = nn.Parameter(torch.randn(4, 256, 128, dtype=torch.bfloat16) * 0.02)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        router_weights = router_logits.softmax(dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = router_weights.topk(self.top_k, dim=-1)
        output, _ = self.backend(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_weights=router_weights,
            fused_w1w3=self.w1w3,
            fused_w2=self.w2,
            training=True,
        )
        return output


def test_sonicmoe_config_rejects_unsupported_options():
    valid = Qwen3MoE30BA3Config(expert_backend="sonicmoe")
    assert valid.ep_size == 1
    assert valid.dispatcher is None
    assert valid.compile_cfg is None

    with pytest.raises(ValidationError, match="ep_size=1"):
        Qwen3MoE30BA3Config(expert_backend="sonicmoe", ep_size=2, compile_cfg=False)
    with pytest.raises(ValidationError, match="expert_tp_size=1"):
        Qwen3MoE30BA3Config(expert_backend="sonicmoe", expert_tp_size=2, compile_cfg=False)
    with pytest.raises(ValidationError, match="dispatcher must be None"):
        Qwen3MoE30BA3Config(expert_backend="sonicmoe", dispatcher="all2all", compile_cfg=False)

    compiled = Qwen3MoE30BA3Config(expert_backend="sonicmoe", compile_cfg=True)
    assert compiled.compile_cfg is True

    token_rounding = SonicMoEBackendConfig(
        routing_mode="token_rounding",
        rounding_mode="nearest",
        rounding_quantum=128,
    )
    rounded = Qwen3MoE30BA3Config(expert_backend="sonicmoe", sonicmoe_cfg=token_rounding)
    assert rounded.sonicmoe_cfg == token_rounding

    noaux = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
    )
    with pytest.raises(ValidationError, match="GreedyRouterConfig"):
        Qwen3MoE30BA3Config(
            expert_backend="sonicmoe",
            sonicmoe_cfg=token_rounding,
            router=noaux,
        )

    with pytest.raises(ValidationError, match="norm_topk_prob=True"):
        Qwen3MoE30BA3Config(
            expert_backend="sonicmoe",
            sonicmoe_cfg=token_rounding,
            router=GreedyRouterConfig(
                scoring_func="softmax",
                norm_topk_prob=False,
                router_scaling_factor=1.0,
            ),
        )


def test_sonicmoe_rejects_cpu_input(monkeypatch: pytest.MonkeyPatch):
    backend = _mock_backend(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA-only"):
        backend(
            hidden_states=torch.empty(2, 16, dtype=torch.bfloat16),
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            topk_weights=torch.ones(2, 1, dtype=torch.float32),
            fused_w1w3=torch.empty(2, 32, 16, dtype=torch.bfloat16),
            fused_w2=torch.empty(2, 16, 16, dtype=torch.bfloat16),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SonicMoE requires CUDA")
def test_sonicmoe_official_call_contract(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}
    backend = _mock_backend(monkeypatch, captured)
    num_tokens, hidden_size = 5, 32
    num_experts, topk, intermediate_size = 4, 2, 16
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.tensor([[3, 1], [0, 2], [1, 3], [2, 0], [0, 1]], device="cuda")
    topk_weights = torch.rand(num_tokens, topk, device="cuda", dtype=torch.bfloat16)
    w1w3 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b1w3 = torch.randn(num_experts, 2 * intermediate_size, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(num_experts, hidden_size, device="cuda", dtype=torch.bfloat16)

    output, frequency = backend(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        fused_w1w3=w1w3,
        fused_w2=w2,
        fused_w1w3_bias=b1w3,
        fused_w2_bias=b2,
        training=False,
    )

    args = captured["args"]
    assert args[0] is hidden_states
    torch.testing.assert_close(
        args[1],
        topk_weights.float().reshape(-1),
    )
    torch.testing.assert_close(
        args[2],
        torch.arange(num_tokens, device="cuda", dtype=torch.int32).repeat_interleave(topk),
    )
    torch.testing.assert_close(args[3], topk_ids.to(torch.int32).reshape(-1))
    assert args[4].shape == (2 * intermediate_size, hidden_size, num_experts)
    assert args[4].stride() == (hidden_size, 1, 2 * intermediate_size * hidden_size)
    assert args[5] is b1w3
    assert args[6].shape == (hidden_size, intermediate_size, num_experts)
    assert args[6].stride() == (intermediate_size, 1, hidden_size * intermediate_size)
    assert args[7] is b2
    assert args[8] == num_experts
    assert args[9] is None
    assert args[10] is captured["activation"]
    assert args[11] is True
    assert captured["kwargs"] == {"concat_layout": True}
    torch.testing.assert_close(output, hidden_states)
    torch.testing.assert_close(frequency, topk_ids.reshape(-1).bincount(minlength=num_experts).to(torch.int32))


class TestTokenRounding:
    @pytest.mark.parametrize("rounding_mode", ["nearest", "up", "down"])
    def test_metadata_matches_official_algorithm(self, rounding_mode: str):
        torch.manual_seed(7)
        num_tokens, num_experts, top_k, quantum = 17, 5, 2, 4
        router_weights = torch.randn(num_tokens, num_experts, dtype=torch.float32).softmax(dim=-1)
        topk_values, topk_ids = router_weights.topk(top_k, dim=-1)

        actual = build_token_rounding_metadata(
            router_weights,
            topk_ids,
            num_experts=num_experts,
            rounding_quantum=quantum,
            rounding_mode=rounding_mode,  # type: ignore[arg-type]
        )

        expert_frequency = torch.bincount(topk_ids.reshape(-1), minlength=num_experts).to(torch.int32)
        if rounding_mode == "up":
            rounded_frequency = torch.ceil(expert_frequency / quantum).to(torch.int32) * quantum
        elif rounding_mode == "down":
            rounded_frequency = expert_frequency // quantum * quantum
        else:
            rounded_frequency = torch.round(expert_frequency / quantum).to(torch.int32) * quantum
        rounded_frequency = rounded_frequency.clamp(max=num_tokens)

        topk_priority = topk_values / topk_values.sum(dim=-1, keepdim=True)
        combined_priority = router_weights.scatter(1, topk_ids, topk_priority).detach() - 1
        combined_priority.scatter_(1, topk_ids, topk_priority)
        ranked_tokens = combined_priority.argsort(dim=0, descending=True).to(torch.int32)
        mask = torch.arange(num_tokens, dtype=torch.int32)[:, None] < rounded_frequency[None, :]
        expected_tokens = ranked_tokens[mask]
        expected_experts = torch.arange(num_experts, dtype=torch.int32)[None, :].expand(num_tokens, -1)[mask]
        order = expected_tokens.argsort()
        expected_tokens = expected_tokens[order]
        expected_experts = expected_experts[order]
        expected_scores = router_weights[expected_tokens.long(), expected_experts.long()].float().contiguous()

        actual_scores, actual_tokens, actual_experts = actual
        torch.testing.assert_close(actual_tokens, expected_tokens)
        torch.testing.assert_close(actual_experts, expected_experts)
        torch.testing.assert_close(actual_scores, expected_scores)
        torch.testing.assert_close(
            actual_experts.long().bincount(minlength=num_experts).to(torch.int32),
            rounded_frequency,
        )
        assert torch.all(actual_tokens[:-1] <= actual_tokens[1:])

    def test_rounding_scores_keep_router_gradient(self):
        logits = torch.randn(12, 4, dtype=torch.float32, requires_grad=True)
        router_weights = logits.softmax(dim=-1)
        topk_ids = router_weights.topk(2, dim=-1).indices
        router_scores, _, _ = build_token_rounding_metadata(
            router_weights,
            topk_ids,
            num_experts=4,
            rounding_quantum=4,
            rounding_mode="up",
        )
        router_scores.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad) > 0

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="SonicMoE requires CUDA")
    @pytest.mark.skipif(not SONICMOE_AVAILABLE, reason="official sonic-moe package is not installed")
    def test_compiled_wrapper_matches_eager(self):
        torch.manual_seed(11)
        module = _SonicMoECompileWrapper().cuda().train()
        hidden_states = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        router_logits = torch.randn(32, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        grad_output = torch.randn_like(hidden_states)

        eager_output = module(hidden_states, router_logits)
        eager_output.backward(grad_output)
        eager_hidden_grad = hidden_states.grad.detach().clone()
        eager_router_grad = router_logits.grad.detach().clone()
        eager_parameter_grads = []
        for parameter in module.parameters():
            assert parameter.grad is not None
            eager_parameter_grads.append(parameter.grad.detach().clone())

        module.zero_grad(set_to_none=True)
        compiled_hidden = hidden_states.detach().clone().requires_grad_()
        compiled_router = router_logits.detach().clone().requires_grad_()
        compiled_module = torch.compile(module, fullgraph=False)
        compiled_output = compiled_module(compiled_hidden, compiled_router)
        compiled_output.backward(grad_output)

        torch.testing.assert_close(compiled_output, eager_output, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(compiled_hidden.grad, eager_hidden_grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(compiled_router.grad, eager_router_grad, rtol=1e-4, atol=1e-5)
        for parameter, eager_grad in zip(module.parameters(), eager_parameter_grads):
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, eager_grad, rtol=3e-2, atol=3e-2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="SonicMoE requires CUDA")
    @pytest.mark.skipif(not SONICMOE_AVAILABLE, reason="official sonic-moe package is not installed")
    def test_official_forward_backward_matches_rounded_reference(self):
        torch.manual_seed(23)
        num_tokens, hidden_size = 32, 256
        num_experts, top_k, intermediate_size = 4, 2, 128
        hidden_states = (
            torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
        ).requires_grad_()
        router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True)
        router_weights = router_logits.softmax(dim=-1)
        topk_weights, topk_ids = router_weights.topk(top_k, dim=-1)
        w1w3 = (
            torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_()
        w2 = (
            torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_()

        ref_hidden = hidden_states.detach().clone().requires_grad_()
        ref_logits = router_logits.detach().clone().requires_grad_()
        ref_w1w3 = w1w3.detach().clone().requires_grad_()
        ref_w2 = w2.detach().clone().requires_grad_()

        backend = SonicMoEBackend(
            SonicMoEBackendConfig(
                routing_mode="token_rounding",
                rounding_mode="nearest",
                rounding_quantum=8,
            )
        )
        output, expert_frequency = backend(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_weights=router_weights,
            fused_w1w3=w1w3,
            fused_w2=w2,
            training=True,
        )

        ref_router_weights = ref_logits.softmax(dim=-1)
        ref_scores, ref_tokens, ref_experts = build_token_rounding_metadata(
            ref_router_weights,
            topk_ids,
            num_experts=num_experts,
            rounding_quantum=8,
            rounding_mode="nearest",
        )
        reference = _reference_general_routing_moe(
            ref_hidden,
            ref_scores,
            ref_tokens,
            ref_experts,
            ref_w1w3,
            ref_w2,
        )

        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        reference.backward(grad_output)

        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            expert_frequency,
            ref_experts.long().bincount(minlength=num_experts).to(torch.int32),
        )
        for actual, expected in (
            (hidden_states, ref_hidden),
            (router_logits, ref_logits),
            (w1w3, ref_w1w3),
            (w2, ref_w2),
        ):
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SonicMoE requires CUDA")
def test_sonicmoe_empty_tokens_keep_parameters_in_autograd(monkeypatch: pytest.MonkeyPatch):
    backend = _mock_backend(monkeypatch)
    hidden_states = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    topk_ids = torch.empty(0, 2, device="cuda", dtype=torch.int64)
    topk_weights = torch.empty(0, 2, device="cuda", dtype=torch.float32, requires_grad=True)
    w1w3 = torch.randn(4, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(4, 32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output, frequency = backend(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        fused_w1w3=w1w3,
        fused_w2=w2,
    )
    output.sum().backward()

    assert output.shape == hidden_states.shape
    torch.testing.assert_close(frequency, torch.zeros(4, device="cuda", dtype=torch.int32))
    for tensor in (hidden_states, topk_weights, w1w3, w2):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SonicMoE requires CUDA")
@pytest.mark.skipif(not SONICMOE_AVAILABLE, reason="official sonic-moe package is not installed")
def test_official_sonicmoe_forward_backward_parity(use_bias: bool):
    torch.manual_seed(42)
    num_tokens, hidden_size = 32, 256
    num_experts, topk, intermediate_size = 4, 2, 128

    hidden_states = (torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")
    topk_weights = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32).softmax(dim=-1)
    topk_weights.requires_grad_()
    w1w3 = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    b1w3 = (
        torch.randn(num_experts, 2 * intermediate_size, device="cuda", dtype=torch.bfloat16).requires_grad_()
        if use_bias
        else None
    )
    b2 = (
        torch.randn(num_experts, hidden_size, device="cuda", dtype=torch.bfloat16).requires_grad_()
        if use_bias
        else None
    )

    inputs = [hidden_states, topk_weights, w1w3, w2]
    if use_bias:
        inputs.extend([b1w3, b2])
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    backend = SonicMoEBackend(SonicMoEBackendConfig())
    output, expert_frequency = backend(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        fused_w1w3=w1w3,
        fused_w2=w2,
        fused_w1w3_bias=b1w3,
        fused_w2_bias=b2,
        training=True,
    )

    ref_hidden, ref_topk_weights, ref_w1w3, ref_w2, *ref_biases = reference_inputs
    ref_b1w3, ref_b2 = ref_biases if use_bias else (None, None)
    reference = _reference_moe(
        ref_hidden,
        topk_ids,
        ref_topk_weights,
        ref_w1w3,
        ref_w2,
        ref_b1w3,
        ref_b2,
    )

    grad = torch.randn_like(output)
    output.backward(grad)
    reference.backward(grad)

    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    assert expert_frequency.shape == (num_experts,)
    assert int(expert_frequency.sum()) == num_tokens * topk
    for actual, expected in zip(inputs, reference_inputs):
        assert actual is not None and expected is not None
        assert actual.grad is not None and expected.grad is not None
        # The official grouped kernel and the per-expert reference accumulate
        # BF16 gradients in a different order. Keep FP32 routing gradients at
        # the tighter tolerance while allowing one additional BF16 rounding
        # interval for activations and expert parameters.
        atol = 6e-2 if actual.grad.dtype == torch.bfloat16 else 3e-2
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-2, atol=atol)


def _tiny_sonicmoe_config() -> MoEConfig:
    return MoEConfig(
        vocab_size=256,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=1,
        hidden_size=256,
        intermediate_size=512,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=128,
        ),
        tie_word_embeddings=False,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=128,
        router=GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        ),
        expert_backend="sonicmoe",
        ep_size=1,
        dispatcher=None,
        compile_cfg=False,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SonicMoE requires CUDA")
@pytest.mark.skipif(not SONICMOE_AVAILABLE, reason="official sonic-moe package is not installed")
class TestSonicMoEEightGpuFSDP(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def test_model_forward_backward_and_optimizer_step(self):
        self.create_pg("cuda")
        config = _tiny_sonicmoe_config()
        with torch.device("meta"):
            model = config.build()

        model.fully_shard(
            FSDPConfig(
                ep_size=1,
                recompute_ratio=0.0,
                torch_compile=False,
            )
        )
        model.init_weights()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        torch.manual_seed(2026 + self.rank)
        input_ids = torch.randint(1, config.vocab_size, (1, 17), dtype=torch.int64, device="cuda")
        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids[:, :-1],), device="cuda")
        loss_cfg = CELossConfig()
        loss_ctx = loss_cfg.build(data={"shifted_labels": input_ids[:, 1:]}, sp_mesh=None)
        assert loss_ctx is not None
        loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]

        output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        loss = output["loss"]
        assert loss is not None and torch.isfinite(loss)
        loss.backward()

        expert_grad_count = 0
        for name, parameter in model.named_parameters():
            if ".experts.fused_" not in name:
                continue
            assert parameter.grad is not None, f"missing expert gradient for {name}"
            grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
            assert torch.isfinite(grad).all(), f"non-finite expert gradient for {name}"
            expert_grad_count += 1
        assert expert_grad_count == 2

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        passed = torch.tensor(1, device="cuda", dtype=torch.int32)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)
        assert int(passed) == self.world_size
