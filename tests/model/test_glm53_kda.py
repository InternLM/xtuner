"""GLM-5.3-Flash 的 Kimi Delta Attention，见 doc/xtuner_glm5p3flash_design.md F3。

需要 GPU：FLA 的 Triton kernel 没有 CPU 后端；SP 用例需要 2 卡。

TestKDAGate
    test_chunk_kda_signature_has_no_hf_style_gate_kwargs  钉住 fla 的签名，防上游漂移
    test_fused_kda_gate_matches_naive_reference           融合 gate 与朴素实现一致
TestKDAModuleParity
    test_kda_module_matches_hf_single_document            单文档下与 HF 实现一致
    test_kda_module_packed_multi_document_matches_concatenated_single_document_forwards
                                                          packed 多文档等价于逐文档前向
TestKDASequenceParallel
    test_forward_for_sp_matches_non_sp                    2 卡 SP 与非 SP 结果一致
"""

import inspect

import pytest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention.kda import KDAConfig, chunk_kda, fused_kda_gate
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
    return Glm5NextTextLinearAttention(config, layer_idx=0), config


def _build_xtuner_kda(hidden_size=64, num_heads=4, head_dim=16, conv_kernel_size=4):
    cfg = KDAConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        use_full_rank_gate=False,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    return cfg.build(hidden_size=hidden_size, layer_idx=0)


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
        params = set(inspect.signature(chunk_kda).parameters)
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
