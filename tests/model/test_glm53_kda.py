"""GLM-5.3-Flash 的 Kimi Delta Attention，见 doc/xtuner_glm5p3flash_design.md F3。

需要 GPU：FLA 的 Triton kernel 没有 CPU 后端；SP 用例需要 2 卡、4 卡或 8 卡。

TestKDAGate
    test_chunk_kda_signature_has_no_hf_style_gate_kwargs  钉住 fla 的签名，防上游漂移
    test_fused_kda_gate_matches_naive_reference           融合 gate 与朴素实现一致
TestKDAKernelDispatch
    test_grad_enabled_preserves_all_parameter_gradients  train/eval 均保留 37/64/65 token 梯度
    test_no_grad_matches_hf_at_kernel_boundary           阈值两侧无梯度前向与 HF 一致
    test_activation_checkpointing_matches_plain_training 重计算与普通训练的前反向一致
TestKDAModuleParity
    test_kda_module_matches_hf_single_document            单文档下与 HF 实现一致
    test_kda_chunk_backward_matches_hf                     chunk 前向和全部梯度与 HF 对齐
    test_kda_module_packed_multi_document_matches_concatenated_single_document_forwards
                                                          packed 多文档等价于逐文档前向
TestKDASequenceParallel
    test_forward_for_sp_matches_non_sp                    2 卡 SP 与非 SP 结果一致
TestKDAFLASequenceParallel / TestKDAFLASequenceParallel4 / TestKDAFLASequenceParallel8
    test_packed_forward_and_backward_match_non_sp          FLA SP 前后向与 SP1 比较
    test_compiled_forward_and_backward                    FLA SP fullgraph 前后向比较
    test_layout_validation_and_inference_offsets          分片校验与 inference offsets
"""

import inspect
import math
from copy import deepcopy

import pytest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from torch.utils.checkpoint import checkpoint

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.utils.checkpointing import apply_activation_checkpointing
from xtuner.v1.module.attention.kda import KDAConfig, KimiDeltaAttention, fused_kda_gate
from xtuner.v1.utils.test_utils import init_data_mesh


def _hf_glm53_kda(**overrides):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    kwargs = dict(
        hidden_size=64,
        linear_num_heads=4,
        linear_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    kwargs.update(overrides)
    config = Glm5NextTextConfig(**kwargs)
    module = Glm5NextTextLinearAttention(config, layer_idx=0)
    # A directly constructed HF attention module has not run PreTrainedModel.post_init().
    with torch.no_grad():
        module.forget_gate.A_log.zero_()
        module.forget_gate.dt_bias.uniform_(math.log(1e-3), math.log(1e-1))
        dt = module.forget_gate.dt_bias.exp().clamp_min(1e-4)
        module.forget_gate.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        module.o_norm.weight.fill_(1)
    return module, config


def _build_xtuner_kda(hidden_size=64, num_heads=4, head_dim=16, conv_kernel_size=4, sp_impl="ulysses"):
    cfg = KDAConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        use_full_rank_gate=False,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        sp_impl=sp_impl,
    )
    module = cfg.build(hidden_size=hidden_size, layer_idx=0)
    # F3 constructs dt_bias with torch.empty; initialize standalone test modules explicitly.
    with torch.no_grad():
        module.A_log.zero_()
        module.dt_bias.uniform_(math.log(1e-3), math.log(1e-1))
        dt = module.dt_bias.exp().clamp_min(1e-4)
        module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
    return module


def _copy_hf_weights_into_xtuner(hf_module, xtuner_module) -> None:
    """Bridge HF's fused ``conv1d``/``forget_gate`` layout into XTuner's published-checkpoint
    layout (separate q/k/v_conv1d, flat A_log/dt_bias), matching the mapping documented in
    doc/xtuner_glm5p3flash_design.md section 3.1."""
    with torch.no_grad():
        xtuner_module.q_proj.weight.copy_(hf_module.q_proj.weight)
        xtuner_module.k_proj.weight.copy_(hf_module.k_proj.weight)
        xtuner_module.v_proj.weight.copy_(hf_module.v_proj.weight)

        qkv_dim = hf_module.qkv_dim
        q_w, k_w, v_w = hf_module.conv1d.weight.split(qkv_dim, dim=0)
        xtuner_module.q_conv1d.weight.copy_(q_w)
        xtuner_module.k_conv1d.weight.copy_(k_w)
        xtuner_module.v_conv1d.weight.copy_(v_w)

        xtuner_module.f_a_proj.weight.copy_(hf_module.forget_gate.f_a_proj.weight)
        xtuner_module.f_b_proj.weight.copy_(hf_module.forget_gate.f_b_proj.weight)
        xtuner_module.dt_bias.copy_(hf_module.forget_gate.dt_bias)
        xtuner_module.A_log.copy_(hf_module.forget_gate.A_log)

        xtuner_module.b_proj.weight.copy_(hf_module.b_proj.weight)
        xtuner_module.g_a_proj.weight.copy_(hf_module.g_a_proj.weight)
        xtuner_module.g_b_proj.weight.copy_(hf_module.g_b_proj.weight)
        xtuner_module.o_norm.weight.copy_(hf_module.o_norm.weight)
        xtuner_module.o_proj.weight.copy_(hf_module.o_proj.weight)


class TestKDAGate:
    def test_chunk_kda_signature_has_no_hf_style_gate_kwargs(self):
        """Reverse guard for design doc section 3.5.1: the installed fla's ``chunk_kda`` must
        NOT accept ``A_log``/``dt_bias``/``use_beta_sigmoid_in_kernel`` -- if a future fla
        upgrade adds them back, passing the externally-precomputed gate through ``g=`` would
        silently double-apply the gate transform instead of raising. Catch that regression by
        asserting these names stay absent from the kernel's signature."""
        # 钉住 fla 的 chunk_kda 签名：它不接受 A_log/dt_bias，传进去会被静默吞掉。
        from fla.ops.kda import chunk_kda as fla_chunk_kda

        params = set(inspect.signature(fla_chunk_kda).parameters)
        assert "A_log" not in params
        assert "dt_bias" not in params
        assert "use_beta_sigmoid_in_kernel" not in params

    @pytest.mark.gpu
    def test_fused_kda_gate_matches_naive_reference(self):
        # 融合 gate kernel 必须与朴素公式一致。
        from fla.ops.kda.gate import naive_kda_lowerbound_gate

        torch.manual_seed(0)
        num_heads, head_dim = 4, 16
        g_raw = torch.randn(1, 8, num_heads, head_dim, device="cuda")
        a_log = torch.randn(num_heads, device="cuda")
        dt_bias = torch.randn(num_heads * head_dim, device="cuda")

        got = fused_kda_gate(g_raw, a_log, dt_bias=dt_bias, lower_bound=-5.0)
        expected = naive_kda_lowerbound_gate(g_raw, a_log, dt_bias=dt_bias, lower_bound=-5.0)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


class TestKDAKernelDispatch:
    """验证 kernel 长度阈值两侧的完整梯度、推理精度与重计算一致性。"""

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [37, 64, 65])
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    def test_grad_enabled_preserves_all_parameter_gradients(self, seq_len: int, training: bool) -> None:
        # eval 不等于 no_grad：阈值两侧都必须保留 q/k/v、卷积、forget gate、beta 等全部参数的梯度。
        torch.manual_seed(0)
        module = _build_xtuner_kda().cuda().train(training)
        hidden_states = torch.randn(1, seq_len, 64, device="cuda", requires_grad=True)
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")

        output = module(hidden_states, seq_ctx)["projected_output"]
        assert torch.isfinite(output).all()
        (output * torch.randn_like(output)).sum().backward()

        missing = [name for name, param in module.named_parameters() if param.grad is None]
        assert not missing, f"Missing parameter gradients: {missing}"
        for name, param in module.named_parameters():
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient: {name}"
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [37, 64, 65])
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    @pytest.mark.parametrize("inference", [False, True], ids=["no-grad", "inference-mode"])
    def test_no_grad_matches_hf_at_kernel_boundary(self, seq_len: int, training: bool, inference: bool) -> None:
        # 通过完整模块输出核验推理行为，不依赖内部 kernel 的具体分派或调用次数。
        torch.manual_seed(0)
        hf_module, _ = _hf_glm53_kda()
        hf_module = hf_module.cuda().train(training)
        module = _build_xtuner_kda().cuda().train(training)
        _copy_hf_weights_into_xtuner(hf_module, module)
        hidden_states = torch.randn(1, seq_len, 64, device="cuda")
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")

        with torch.inference_mode() if inference else torch.no_grad():
            expected = hf_module(hidden_states)
            output = module(hidden_states, seq_ctx)["projected_output"]

        assert not output.requires_grad
        assert torch.isfinite(expected).all()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [37, 64, 65])
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    @pytest.mark.parametrize("checkpoint_api", ["xtuner", "torch-reentrant", "torch-nonreentrant"])
    def test_activation_checkpointing_matches_plain_training(
        self, seq_len: int, training: bool, checkpoint_api: str
    ) -> None:
        # Eval mode must preserve the same computation in a checkpoint's first pass and replay.
        torch.manual_seed(0)
        plain_module = _build_xtuner_kda().cuda().train(training)
        replay_module = deepcopy(plain_module)
        hidden_states = torch.randn(1, seq_len, 64, device="cuda", requires_grad=True)
        checkpointed_hidden = hidden_states.detach().clone().requires_grad_()
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")

        expected = plain_module(hidden_states, seq_ctx)["projected_output"]
        if checkpoint_api == "xtuner":
            output = apply_activation_checkpointing(replay_module)(checkpointed_hidden, seq_ctx)["projected_output"]
        else:
            output = checkpoint(
                lambda x: replay_module(x, seq_ctx)["projected_output"],
                checkpointed_hidden,
                use_reentrant=checkpoint_api == "torch-reentrant",
            )
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
        # A nonlinear objective makes upstream gradients depend on the original forward values.
        expected.square().sum().backward()
        output.square().sum().backward()

        assert hidden_states.grad is not None
        assert checkpointed_hidden.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        assert torch.isfinite(checkpointed_hidden.grad).all()
        torch.testing.assert_close(checkpointed_hidden.grad, hidden_states.grad, atol=1e-6, rtol=1e-5)
        for name, param in plain_module.named_parameters():
            replay_param = replay_module.get_parameter(name)
            assert param.grad is not None, name
            assert replay_param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            assert torch.isfinite(replay_param.grad).all(), name
            torch.testing.assert_close(replay_param.grad, param.grad, atol=1e-6, rtol=1e-5, msg=name)


class TestKDAModuleParity:
    @pytest.mark.gpu
    def test_kda_module_matches_hf_single_document(self):
        # 单文档下整个 KDA 模块的输出与 HF 实现一致。
        torch.manual_seed(0)
        hf_module, _ = _hf_glm53_kda()
        hf_module = hf_module.cuda()
        xtuner_module = _build_xtuner_kda().cuda()
        _copy_hf_weights_into_xtuner(hf_module, xtuner_module)

        hidden_states = torch.randn(1, 37, 64, device="cuda")
        with torch.no_grad():
            hf_out = hf_module(hidden_states)  # returns a single tensor, not a tuple

            seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, 37, dtype=torch.long),), device="cuda")
            xtuner_out = xtuner_module(hidden_states, seq_ctx)["projected_output"]

        torch.testing.assert_close(xtuner_out, hf_out, atol=2e-2, rtol=2e-2)

    @pytest.mark.gpu
    def test_kda_chunk_backward_matches_hf(self):
        # HF uses chunked KDA for prefill; compare its public output and all gradients.
        seq_len = 80
        torch.manual_seed(0)
        hf_module, _ = _hf_glm53_kda()
        hf_module = hf_module.cuda()
        xtuner_module = _build_xtuner_kda().cuda()
        _copy_hf_weights_into_xtuner(hf_module, xtuner_module)

        hf_hidden = torch.randn(1, seq_len, 64, device="cuda", requires_grad=True)
        xtuner_hidden = hf_hidden.detach().clone().requires_grad_()
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")
        hf_out = hf_module(hf_hidden)
        xtuner_out = xtuner_module(xtuner_hidden, seq_ctx)["projected_output"]
        upstream = torch.randn_like(hf_out)
        hf_out.backward(upstream)
        xtuner_out.backward(upstream)

        torch.testing.assert_close(xtuner_out, hf_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(xtuner_hidden.grad, hf_hidden.grad, atol=2e-2, rtol=2e-2)
        # HF fuses the three causal convolutions; XTuner stores them separately.
        hf_grads = {name: param.grad for name, param in hf_module.named_parameters()}
        xtuner_grads = {name: param.grad for name, param in xtuner_module.named_parameters()}
        qkv_dim = hf_module.qkv_dim
        conv_grads = hf_grads.pop("conv1d.weight").split(qkv_dim, dim=0)
        for name, grad in zip(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"), conv_grads):
            torch.testing.assert_close(xtuner_grads.pop(name), grad, atol=2e-2, rtol=2e-2)
        for name, grad in hf_grads.items():
            xtuner_name = name.removeprefix("forget_gate.")
            torch.testing.assert_close(xtuner_grads.pop(xtuner_name), grad, atol=5e-2, rtol=2e-2, msg=name)
        assert not xtuner_grads

    @pytest.mark.gpu
    def test_kda_module_packed_multi_document_matches_concatenated_single_document_forwards(self):
        """Packed multi-document forward must equal the per-document forwards concatenated
        (design doc F3 test item 3: no cross-document leakage through the conv/recurrent state)."""
        # packed 多文档必须等价于逐文档单独前向再拼接，文档间不能串状态。
        torch.manual_seed(0)
        xtuner_module = _build_xtuner_kda().cuda()

        doc1 = torch.randn(1, 20, 64, device="cuda")
        doc2 = torch.randn(1, 33, 64, device="cuda")

        with torch.no_grad():
            seq_ctx1 = SequenceContext.from_input_ids((torch.zeros(1, 20, dtype=torch.long),), device="cuda")
            out1 = xtuner_module(doc1, seq_ctx1)["projected_output"]
            seq_ctx2 = SequenceContext.from_input_ids((torch.zeros(1, 33, dtype=torch.long),), device="cuda")
            out2 = xtuner_module(doc2, seq_ctx2)["projected_output"]

            packed_hidden = torch.cat([doc1, doc2], dim=1)
            packed_seq_ctx = SequenceContext.from_input_ids(
                (torch.zeros(1, 20, dtype=torch.long), torch.zeros(1, 33, dtype=torch.long)), device="cuda"
            )
            packed_out = xtuner_module(packed_hidden, packed_seq_ctx)["projected_output"]

        expected = torch.cat([out1, out2], dim=1)
        torch.testing.assert_close(packed_out, expected, atol=1e-4, rtol=1e-4)


class TestKDASequenceParallel(DistributedTestBase):
    @pytest.mark.gpu
    def test_forward_for_sp_matches_non_sp(self, device="cuda"):
        # 2 卡 Ulysses SP 的输出与非 SP 一致（head 切分不改变数学）。
        self.create_pg(device)
        torch.manual_seed(0)

        module = _build_xtuner_kda(hidden_size=64, num_heads=4, head_dim=16).to(device)
        for p in module.parameters():
            torch.distributed.broadcast(p.data, src=0)

        seq_len_per_rank = 40
        sp_size = self.world_size
        torch.manual_seed(1234)
        full_hidden = torch.randn(1, seq_len_per_rank * sp_size, 64, device=device)
        torch.distributed.broadcast(full_hidden, src=0)

        seq_ctx_non_sp = SequenceContext.from_input_ids(
            (torch.zeros(1, seq_len_per_rank * sp_size, dtype=torch.long),), device=device
        )
        with torch.no_grad():
            non_sp_out = module(full_hidden, seq_ctx_non_sp)["projected_output"]

        data_mesh = init_data_mesh(device, sp_size)
        sp_mesh = data_mesh["sp"]
        rank = sp_mesh.get_local_rank()
        local_hidden = full_hidden[:, rank * seq_len_per_rank : (rank + 1) * seq_len_per_rank, :].contiguous()

        seq_ctx_sp = SequenceContext.from_input_ids(
            (torch.zeros(1, seq_len_per_rank * sp_size, dtype=torch.long),), device=device
        )
        seq_ctx_sp = seq_ctx_sp.split(sequence_parallel_mesh=sp_mesh)

        with torch.no_grad():
            sp_out = module(local_hidden, seq_ctx_sp)["projected_output"]

        expected_local = non_sp_out[:, rank * seq_len_per_rank : (rank + 1) * seq_len_per_rank, :]
        torch.testing.assert_close(sp_out, expected_local, atol=2e-2, rtol=2e-2)

    @property
    def world_size(self) -> int:
        return 2


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [16, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_sp_norm_matches_existing_norm(head_dim, dtype):
    from xtuner.v1.module.attention.kda import FusedRMSNormGated

    torch.manual_seed(1234)
    norm = FusedRMSNormGated(head_dim, eps=1e-5, activation="sigmoid").cuda().to(dtype)
    wrapped_norm = deepcopy(norm)
    wrapped_norm.compile_friendly = True
    x = torch.randn(1, 37, 3, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    replay_x = x.detach().clone().requires_grad_()
    replay_gate = gate.detach().clone().requires_grad_()
    expected = norm(x, gate)
    actual = wrapped_norm(replay_x, replay_gate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    for replay, original in (
        (replay_x.grad, x.grad),
        (replay_gate.grad, gate.grad),
        (wrapped_norm.weight.grad, norm.weight.grad),
    ):
        torch.testing.assert_close(replay, original, atol=0, rtol=0)


def _assert_fla_sp_parity(actual, expected, dtype, name):
    """Relative budgets cover FLA TF32 transfers and BF16 chunk retiling.

    A tensor norm check avoids an absolute floor hiding small boundary gradients
    or rejecting cancellation elements near zero. The maximum error is also
    bounded relative to the tensor maximum. Missing state gradients fail both.
    """
    actual, expected = actual.detach().float(), expected.detach().float()
    difference = actual - expected
    if not bool(expected.any()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)
        return
    limit = 3e-3 if dtype == torch.float32 else 2e-2
    relative_l2 = float(difference.norm() / expected.norm())
    relative_max = float(difference.abs().max() / expected.abs().max())
    assert relative_l2 < limit and relative_max < 2 * limit, (
        f"{name}: relative_l2={relative_l2}, relative_max={relative_max}, "
        f"max_abs={float(difference.abs().max())}, reference_max={float(expected.abs().max())}"
    )


class TestKDAFLASequenceParallel(DistributedTestBase):
    parity_dtypes = (torch.float32, torch.bfloat16)

    @pytest.mark.gpu
    def test_layout_validation_and_inference_offsets(self, device="cuda"):
        self.create_pg(device)
        sp_mesh = init_data_mesh(device, self.world_size)["sp"]
        module = _build_xtuner_kda(sp_impl="fla").to(device)
        short_ctx = SequenceContext.from_input_ids(
            (torch.zeros(1, 2 * self.world_size, dtype=torch.long),), device=device
        ).split(sp_mesh)
        with pytest.raises(ValueError, match="at least conv_kernel_size - 1"):
            module(torch.randn(1, 2, 64, device=device), short_ctx)
        # These offsets do not expose a tensor version counter. Inference must
        # still build the same FLA local document context and enter collectives.
        with torch.inference_mode():
            ctx = SequenceContext.from_input_ids(
                (torch.zeros(1, 8 * self.world_size, dtype=torch.long),), device=device
            ).split(sp_mesh)
            hidden = torch.randn(1, 8, 64, device=device)
            assert torch.isfinite(module(hidden, ctx)["projected_output"]).all()
            ctx._shard_size = 7
            with pytest.raises(ValueError, match="equal contiguous shards"):
                module(hidden, ctx)

    @pytest.mark.gpu
    def test_packed_forward_and_backward_match_non_sp(self, device="cuda"):
        """Late-rank losses must differentiate through earlier conv and recurrent state.

        Packed cases include a document beginning two tokens before a rank boundary,
        an exact boundary, and padding introduced by SequenceContext.split().
        """
        self.create_pg(device)
        sp_mesh = init_data_mesh(device, self.world_size)["sp"]
        rank = sp_mesh.get_local_rank()
        for dtype in self.parity_dtypes:
            for lengths, conv_width, num_heads, head_dim in (
                ([512], 4, 4, 16),
                ([254, 5, 253], 4, 4, 16),
                ([256, 256], 4, 4, 16),
                ([37, 472], 4, 4, 16),
                ([512], 1, 3, 16),
                ([512], 4, 64, 128),
            ):
                # Native FLA's all-FP32 K=128 chunk-state kernel exceeds H200's
                # shared-memory limit. Production uses BF16 with FP32 masters.
                if dtype == torch.float32 and head_dim == 128:
                    continue
                torch.manual_seed(1234)
                module = _build_xtuner_kda(
                    sp_impl="fla", conv_kernel_size=conv_width, num_heads=num_heads, head_dim=head_dim
                ).to(device=device, dtype=dtype)
                # Preserve long-range state so missing recurrent boundary gradients cannot
                # be hidden by fast forget gates; conv weights exercise all history taps.
                with torch.no_grad():
                    module.A_log.zero_()
                    module.dt_bias.fill_(-6)
                    for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
                        conv.float()
                        conv.weight.fill_(0.25)
                    module.A_log.data = module.A_log.data.float()
                    module.dt_bias.data = module.dt_bias.data.float()
                reference = deepcopy(module)
                seq_ctx = SequenceContext.from_input_ids(
                    tuple(torch.zeros(1, length, dtype=torch.long) for length in lengths), device=device
                )
                local_ctx = seq_ctx.copy().split(sequence_parallel_mesh=sp_mesh)
                total = int(local_ctx.cu_seq_lens_q[-1])
                local_length = total // self.world_size
                # Use the padded global document layout for the SP1 reference too.
                reference_ctx = SequenceContext.from_input_ids(
                    tuple(torch.zeros(1, int(length), dtype=torch.long) for length in local_ctx.seq_lens_q),
                    device=device,
                )
                full_hidden = torch.randn(1, total, 64, device=device, dtype=dtype, requires_grad=True)
                local_hidden = (
                    full_hidden.detach()[:, rank * local_length : (rank + 1) * local_length].clone().requires_grad_()
                )
                expected = reference(full_hidden, reference_ctx)["projected_output"]
                output = module(local_hidden, local_ctx)["projected_output"]
                _assert_fla_sp_parity(
                    output, expected[:, rank * local_length : (rank + 1) * local_length], dtype, "output"
                )
                # Only the first few outputs on the last rank receive an upstream gradient.
                # Earlier ranks still must receive nonzero gradients through both states.
                gradient = torch.zeros_like(expected)
                torch.manual_seed(5678)
                gradient[:, -local_length : -local_length + 8] = torch.randn_like(gradient[:, :8])
                expected.backward(gradient)
                output.backward(gradient[:, rank * local_length : (rank + 1) * local_length].contiguous())
                assert local_hidden.grad is not None
                _assert_fla_sp_parity(
                    local_hidden.grad,
                    full_hidden.grad[:, rank * local_length : (rank + 1) * local_length],
                    dtype,
                    f"{dtype} {lengths} width={conv_width} heads={num_heads} dim={head_dim} input gradient",
                )
                if lengths == [512] and rank < self.world_size - 1:
                    assert local_hidden.grad.abs().sum() > 0, "Missing backward gradient across SP segments"
                # Parameter replicas are synchronized by the trainer, not by attention.
                for name, parameter in module.named_parameters():
                    assert parameter.grad is not None, name
                    torch.distributed.all_reduce(parameter.grad, group=sp_mesh.get_group())
                    target = reference.get_parameter(name).grad
                    assert target is not None, name
                    _assert_fla_sp_parity(
                        parameter.grad, target, dtype, f"{dtype} {lengths} width={conv_width} {name}"
                    )

    @pytest.mark.gpu
    def test_checkpointed_forward_and_backward(self, device="cuda"):
        """Checkpoint replay preserves compiled boundary communication and all gradients."""
        self.create_pg(device)
        sp_mesh = init_data_mesh(device, self.world_size)["sp"]
        rank = sp_mesh.get_local_rank()
        ctx = SequenceContext.from_input_ids((torch.zeros(1, 256, dtype=torch.long),), device=device).split(sp_mesh)
        local_length = 256 // self.world_size
        for training in (True, False):
            for compile_module in (False, True):
                torch.manual_seed(1234)
                module = _build_xtuner_kda(sp_impl="fla").to(device).train(training)
                with torch.no_grad():
                    module.A_log.zero_()
                    module.dt_bias.fill_(-6)
                    for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
                        conv.weight.fill_(0.25)
                replay_module = deepcopy(module)
                forward_module = torch.compile(replay_module, fullgraph=True) if compile_module else replay_module
                checkpointed = apply_activation_checkpointing(forward_module)
                hidden = torch.randn(1, local_length, 64, device=device, requires_grad=True)
                replay_hidden = hidden.detach().clone().requires_grad_()
                expected = module(hidden, ctx)["projected_output"]
                actual = checkpointed(replay_hidden, ctx)["projected_output"]
                _assert_fla_sp_parity(actual, expected, hidden.dtype, "checkpointed output")
                # A loss on the last rank must still differentiate through earlier boundary states.
                expected_loss = expected.square().sum()
                actual_loss = actual.square().sum()
                if rank != self.world_size - 1:
                    expected_loss = expected_loss * 0
                    actual_loss = actual_loss * 0
                expected_loss.backward()
                actual_loss.backward()
                assert replay_hidden.grad is not None
                _assert_fla_sp_parity(replay_hidden.grad, hidden.grad, hidden.dtype, "checkpointed input gradient")
                assert replay_hidden.grad.abs().sum() > 0, "Missing checkpointed boundary gradient"
                for name, parameter in module.named_parameters():
                    gradient = replay_module.get_parameter(name).grad
                    assert gradient is not None and parameter.grad is not None, name
                    assert torch.isfinite(gradient).all(), name
                    _assert_fla_sp_parity(gradient, parameter.grad, hidden.dtype, "checkpointed " + name)

    @pytest.mark.gpu
    def test_compiled_forward_and_backward(self, device="cuda"):
        """Fullgraph compilation must keep boundary communication in the custom ops."""
        self.create_pg(device)
        sp_mesh = init_data_mesh(device, self.world_size)["sp"]
        torch.manual_seed(1234)
        module = _build_xtuner_kda(sp_impl="fla").to(device)
        compiled = torch.compile(deepcopy(module), fullgraph=True)
        ctx = SequenceContext.from_input_ids((torch.zeros(1, 256, dtype=torch.long),), device=device).split(sp_mesh)
        hidden = torch.randn(1, 256 // self.world_size, 64, device=device, requires_grad=True)
        replay_hidden = hidden.detach().clone().requires_grad_()
        expected = module(hidden, ctx)["projected_output"]
        actual = compiled(replay_hidden, ctx)["projected_output"]
        _assert_fla_sp_parity(actual, expected, hidden.dtype, "compiled output")
        expected.square().sum().backward()
        actual.square().sum().backward()
        _assert_fla_sp_parity(replay_hidden.grad, hidden.grad, hidden.dtype, "compiled input gradient")
        for name, parameter in module.named_parameters():
            _assert_fla_sp_parity(compiled.get_parameter("_orig_mod." + name).grad, parameter.grad, hidden.dtype, name)

    @property
    def world_size(self) -> int:
        return 2


class TestKDAFLASequenceParallel4(TestKDAFLASequenceParallel):
    # Target the production dtype here. The native-FLA diagnostic separately
    # measures accumulated all-FP32 transfer error over three rank boundaries.
    parity_dtypes = (torch.bfloat16,)

    @property
    def world_size(self) -> int:
        return 4


class TestKDAFLASequenceParallel8(TestKDAFLASequenceParallel4):
    @property
    def world_size(self) -> int:
        return 8


class TestKDASequenceSPConfig:
    @pytest.mark.parametrize("implementation", ["ulysses", "fla"])
    def test_config_round_trip_and_build(self, implementation):
        cfg = KDAConfig(num_heads=4, head_dim=16, sp_impl=implementation)
        rebuilt = KDAConfig.model_validate_json(cfg.model_dump_json())
        assert rebuilt.build(hidden_size=64).sp_impl == implementation

    def test_default_uses_fla(self):
        cfg = KDAConfig(num_heads=4, head_dim=16)
        assert cfg.sp_impl == "fla"
        assert cfg.build(hidden_size=64).sp_impl == "fla"
        assert KimiDeltaAttention(hidden_size=64, num_heads=4, head_dim=16).sp_impl == "fla"

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            KDAConfig(num_heads=4, head_dim=16, sp_impl="unknown")
