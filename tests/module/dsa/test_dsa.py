# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention.dsa import DeepSeekSparseAttention, DSAConfig


def _make_dsa(
    compress_ratio: int,
    *,
    hidden_size: int = 128,
    num_attention_heads: int = 4,
    head_dim: int = 32,
    qk_rope_head_dim: int = 16,
    q_lora_rank: int = 64,
    o_lora_rank: int = 64,
    o_groups: int = 2,
    sliding_window: int = 16,
    index_head_dim: int = 32,
    index_n_heads: int = 4,
    index_topk: int = 8,
    pack_max_length: int | None = None,
    backend: str = "native",
    seed: int = 1234,
) -> DeepSeekSparseAttention:
    torch.manual_seed(seed)
    # ``DSAConfig.indexer_backend`` now defaults to ``"triton"`` so production
    # V4 (``index_n_heads=64``) picks up the fast tensor-core kernel without
    # any config edit. The small-dim test fixture here uses
    # ``index_n_heads=4``, which is below the triton kernel's tensor-core tile
    # floor (16); pin it back to ``"native"`` so the DSA correctness checks
    # exercise the pure-PyTorch reference path that handles arbitrary head
    # counts.
    cfg = DSAConfig(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        sliding_window=sliding_window,
        use_attn_sink=True,
        index_head_dim=index_head_dim,
        index_n_heads=index_n_heads,
        index_topk=index_topk,
        indexer_backend="native",
        pack_max_length=pack_max_length,
        backend=backend,  # type: ignore[arg-type]
    )
    module = cfg.build(hidden_size=hidden_size, layer_idx=0, compress_ratio=compress_ratio)
    # Non-trivial APE to keep the Compressor / Indexer non-degenerate; default
    # zero init would collapse the gate softmax to uniform.
    with torch.no_grad():
        if module.compressor is not None:
            torch.nn.init.normal_(module.compressor.ape, std=0.02)
        if module.indexer is not None:
            torch.nn.init.normal_(module.indexer.compressor.ape, std=0.02)
    module.eval()
    return module


def _make_position_embeddings(seq_len: int, rope_head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    # ``DualRotaryEmbedding`` emits D-dim ``(cos_full, sin_full_signed)``
    # pre-arranged for the per-layer ``x * cos_full + flip_pairs(x) * sin_full_signed``
    # kernel. Mirror the same layout here:
    #   ``cos_full[..., 2i] == cos_full[..., 2i+1] == cos[i]``
    #   ``sin_full_signed[..., 2i]   = -sin[i]``
    #   ``sin_full_signed[..., 2i+1] = +sin[i]``
    half = rope_head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [seq_len, half]
    cos_half = angles.cos()
    sin_half = angles.sin()
    cos_full = cos_half.repeat_interleave(2, dim=-1)
    sin_full_signed = torch.stack([-sin_half, sin_half], dim=-1).flatten(-2)
    return cos_full.unsqueeze(0), sin_full_signed.unsqueeze(0)


def _make_seq_ctx(cu: list[int]) -> SequenceContext:
    cu_t = torch.tensor(cu, dtype=torch.int32)
    seq_lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=cu_t,  # type: ignore[arg-type]
        cu_seq_lens_k=cu_t,  # type: ignore[arg-type]
        max_length_q=max(seq_lens),
        max_length_k=max(seq_lens),
        device="cpu",
    )


class TestDeepSeekSparseAttention:
    def test_sliding_only_forward(self) -> None:
        hidden_size = 128
        seq_len = 64
        dsa = _make_dsa(compress_ratio=0, hidden_size=hidden_size, sliding_window=16)

        torch.manual_seed(0)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, dsa.qk_rope_head_dim)
        seq_ctx = _make_seq_ctx([0, seq_len])

        out = dsa(hidden, (cos, sin), None, seq_ctx)
        projected = out["projected_output"]
        raw = out["raw_output"]
        assert projected.shape == (1, seq_len, hidden_size)
        assert raw.shape == (1, seq_len, dsa.num_attention_heads, dsa.head_dim)
        assert out["softmax_lse"] is None
        assert torch.isfinite(projected).all()

    def test_compressed_4_forward(self) -> None:
        hidden_size = 128
        seq_len = 64
        dsa = _make_dsa(compress_ratio=4, hidden_size=hidden_size, sliding_window=16)

        torch.manual_seed(1)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, dsa.qk_rope_head_dim, base=10000.0)
        # Compressed rope uses the V4 yarn'd theta 160000; we mirror that here
        # so the Indexer's positional bias is meaningfully distinct from the
        # window rope.
        cos_c, sin_c = _make_position_embeddings(seq_len, dsa.qk_rope_head_dim, base=160000.0)
        seq_ctx = _make_seq_ctx([0, seq_len])

        out = dsa(hidden, (cos, sin), (cos_c, sin_c), seq_ctx)
        projected = out["projected_output"]
        assert projected.shape == (1, seq_len, hidden_size)
        assert torch.isfinite(projected).all()

    def test_compressed_128_forward(self) -> None:
        # `compress_ratio == 128` builds a Compressor but no Indexer; top-k
        # is deterministic positional. Use S=256 so we actually get two
        # compressed positions per sample.
        hidden_size = 128
        seq_len = 256
        dsa = _make_dsa(compress_ratio=128, hidden_size=hidden_size, sliding_window=16)
        assert dsa.indexer is None
        # ``pack_max_length`` not set ⇒ padded path off, no buffer allocated,
        # forward runs the dynamic-width native build.
        assert dsa._hca_uses_padded_path is False
        assert "_compress_topk_pad" not in dict(dsa.named_buffers())

        torch.manual_seed(2)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, dsa.qk_rope_head_dim)
        # Ratio-128 layers own a Compressor (just no Indexer), and the Compressor rotates
        # its compressed KV against the compressed base — so the basis is required here
        # even though nothing consumes an Indexer top-k.
        cos_c, sin_c = _make_position_embeddings(seq_len, dsa.qk_rope_head_dim, base=160000.0)
        seq_ctx = _make_seq_ctx([0, seq_len])

        out = dsa(hidden, (cos, sin), (cos_c, sin_c), seq_ctx)
        projected = out["projected_output"]
        assert projected.shape == (1, seq_len, hidden_size)
        assert torch.isfinite(projected).all()

    def test_hca_pad_buffer_static_shape(self) -> None:
        # When ``pack_max_length`` is set AND backend resolves to flash_mla/
        # cudnn, ``_hca_uses_padded_path`` flips on at __init__ and the static
        # ``-1`` buffer is registered with the expected shape. CI box may not
        # have FlashMLA, so we tolerate the native-fallback case.
        sliding_window = 16
        pack_max_length = 256
        ratio = 128
        try:
            dsa = _make_dsa(
                compress_ratio=ratio,
                sliding_window=sliding_window,
                pack_max_length=pack_max_length,
                backend="cudnn",
            )
        except (RuntimeError, ImportError):
            pytest.skip("flash_mla / cudnn backend unavailable in this env")
        if not dsa._hca_uses_padded_path:
            pytest.skip("backend resolved to native; pad buffer intentionally not allocated")
        max_w = pack_max_length // ratio + 1
        assert dsa._hca_max_compress_w == max_w
        natural_topk = sliding_window + max_w
        padded_topk = ((natural_topk + 128 - 1) // 128) * 128
        assert dsa._hca_pad_cols == padded_topk - natural_topk
        buf = dsa._compress_topk_pad
        assert buf.shape == (1, pack_max_length, dsa._hca_pad_cols)
        assert buf.dtype == torch.int32
        assert (buf == -1).all()

    def test_hca_pack_max_length_without_flash_backend_falls_back(self) -> None:
        # Native backend doesn't need 128-alignment, so even with
        # ``pack_max_length`` set the padded path is intentionally OFF (saves
        # the 8 MB pad buffer when it'd be unused).
        dsa = _make_dsa(
            compress_ratio=128,
            sliding_window=16,
            pack_max_length=256,
            backend="native",
        )
        assert dsa._hca_uses_padded_path is False
        assert "_compress_topk_pad" not in dict(dsa.named_buffers())

    def test_two_samples_varlen(self) -> None:
        # Per-sample loop must produce bit-identical output for sample B
        # whether it's processed alone or packed after sample A.
        hidden_size = 128
        seq_len_a = 32
        seq_len_b = 64
        dsa = _make_dsa(compress_ratio=0, hidden_size=hidden_size, sliding_window=16)

        torch.manual_seed(42)
        hidden_a = torch.randn(1, seq_len_a, hidden_size, dtype=torch.float32)
        hidden_b = torch.randn(1, seq_len_b, hidden_size, dtype=torch.float32)
        packed = torch.cat([hidden_a, hidden_b], dim=1)

        # Each sample's rope starts at position 0, so build per-sample
        # embeddings and concat — same convention as the Indexer varlen test.
        cos_a, sin_a = _make_position_embeddings(seq_len_a, dsa.qk_rope_head_dim)
        cos_b, sin_b = _make_position_embeddings(seq_len_b, dsa.qk_rope_head_dim)
        cos_packed = torch.cat([cos_a, cos_b], dim=1)
        sin_packed = torch.cat([sin_a, sin_b], dim=1)

        seq_ctx_packed = _make_seq_ctx([0, seq_len_a, seq_len_a + seq_len_b])
        seq_ctx_b_only = _make_seq_ctx([0, seq_len_b])

        out_packed = dsa(packed, (cos_packed, sin_packed), None, seq_ctx_packed)
        out_b_only = dsa(hidden_b, (cos_b, sin_b), None, seq_ctx_b_only)

        # Sample B's slice of the packed output uses identical inputs and the
        # raw per-sample sparse_attn produces the same head outputs; only the
        # final O-LoRA stage runs over the whole packed tensor at once, but
        # that's pointwise per token and dtype-stable in fp32.
        torch.testing.assert_close(
            out_packed["raw_output"][:, seq_len_a:],
            out_b_only["raw_output"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            out_packed["projected_output"][:, seq_len_a:],
            out_b_only["projected_output"],
            rtol=0.0,
            atol=0.0,
        )

    def test_attn_sink_absorbs_mass(self) -> None:
        # If KV is exactly zero and the sink logit dominates, the softmax
        # routes all mass to the sink slot which is dropped from the output,
        # so `raw_output` is exactly zero.
        hidden_size = 128
        seq_len = 64
        dsa = _make_dsa(compress_ratio=0, hidden_size=hidden_size, sliding_window=16)
        with torch.no_grad():
            dsa.wkv.weight.zero_()
            # Force a large sink logit so it dominates over the q · K = 0 logits.
            dsa.attn_sink.fill_(100.0)
            # Make kv_norm a no-op so wkv's zero output stays zero after norm.
            dsa.kv_norm.weight.fill_(1.0)

        torch.manual_seed(7)
        hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
        cos, sin = _make_position_embeddings(seq_len, dsa.qk_rope_head_dim)
        seq_ctx = _make_seq_ctx([0, seq_len])

        out = dsa(hidden, (cos, sin), None, seq_ctx)
        # raw_output is the pre-O-LoRA per-head output: it's a weighted sum
        # over `kv` which is zero everywhere, so it must be exactly zero.
        assert torch.equal(out["raw_output"], torch.zeros_like(out["raw_output"]))
        # projected_output is just wo_b(wo_a · zeros) = wo_b(0) = 0.
        assert torch.equal(out["projected_output"], torch.zeros_like(out["projected_output"]))

    def test_grouped_o_lora_shapes(self) -> None:
        # Direct shape check on the O-LoRA reshape + einsum; protects the
        # contract between wo_a's flat Linear storage and the [o_groups,
        # o_lora_rank, head_dim_per_group] view used in forward.
        dsa = _make_dsa(compress_ratio=0)
        assert dsa.head_dim_per_group == dsa.num_attention_heads * dsa.head_dim // dsa.o_groups

        seq_len = 16
        raw = torch.randn(1, seq_len, dsa.num_attention_heads, dsa.head_dim)
        o_grouped = raw.reshape(1, seq_len, dsa.o_groups, dsa.head_dim_per_group)
        wo_a_view = dsa.wo_a.weight.view(dsa.o_groups, dsa.o_lora_rank, dsa.head_dim_per_group)
        o_proj = torch.einsum("bsgd,grd->bsgr", o_grouped, wo_a_view)
        assert o_proj.shape == (1, seq_len, dsa.o_groups, dsa.o_lora_rank)
        flat = o_proj.flatten(2)
        assert flat.shape == (1, seq_len, dsa.o_groups * dsa.o_lora_rank)
        out = dsa.wo_b(flat)
        assert out.shape == (1, seq_len, dsa.hidden_size)


class TestRopeSplitFused:
    """Tests for the fused Triton rope-split kernel used in DSA."""

    def _make_structured_cos_sin(
        self, T: int, rope_dim: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate cos/sin matching the DualRotaryEmbedding layout.

        DualRotaryEmbedding always outputs structured cos/sin (cast to bf16):
          cos[t, 2k] = cos[t, 2k+1] = cos_half[t, k]
          sin[t, 2k] = -sin_half[t, k]   (negative)
          sin[t, 2k+1] = +sin_half[t, k] (positive)
        This structure is required for the rotation to be orthogonal (c²+s²=1)
        and for the backward to be the transpose of the forward.
        """
        angles = torch.rand(T, rope_dim // 2, device=device)
        cos_half = torch.cos(angles).to(torch.bfloat16)
        sin_half = torch.sin(angles).to(torch.bfloat16)
        cos_full = cos_half.repeat_interleave(2, dim=-1)  # [T, rope_dim]
        # Interleave -sin and +sin per pair: [-s0, +s0, -s1, +s1, ...]
        sin_full = torch.stack([-sin_half, sin_half], dim=-1).flatten(-2)
        return cos_full.unsqueeze(0), sin_full.unsqueeze(0)  # [1, T, rope_dim]

    def _ref_forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
        """Reference implementation: slice → rotate → cat."""
        from xtuner.v1.module.attention.dsa._dsa_rope import _apply_rope

        nope = x[..., : x.size(-1) - rope_head_dim]
        tail = x[..., x.size(-1) - rope_head_dim :]
        return torch.cat([nope, _apply_rope(tail, cos, sin)], dim=-1)

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "shape,rope_head_dim",
        [
            ((1, 64, 8, 32), 16),   # q-like: [batch, T, H, D]
            ((1, 64, 32), 16),      # kv-like: [batch, T, D] (no head dim)
        ],
    )
    def test_forward_matches_reference(self, shape: tuple, rope_head_dim: int) -> None:
        from xtuner.v1.module.attention.dsa._dsa_rope import _apply_rope_split

        torch.manual_seed(0)
        T = shape[1]
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        # DualRotaryEmbedding returns structured cos/sin in bf16.
        cos, sin = self._make_structured_cos_sin(T, rope_head_dim, "cuda")

        ref = self._ref_forward(x, cos, sin, rope_head_dim)
        fused = _apply_rope_split(x, cos, sin, rope_head_dim)

        assert fused.shape == x.shape
        assert fused.dtype == x.dtype
        # Fused kernel accumulates in fp32; reference computes in bf16.  Allow
        # up to 2 bf16 ULPs (≈ 2 × 2^-7 for values ≈ 1) as tolerance.
        assert torch.allclose(fused.float(), ref.float(), atol=0.1), (
            f"max diff: {(fused.float() - ref.float()).abs().max().item():.4e}"
        )

    @pytest.mark.gpu
    def test_inverse_undoes_forward(self) -> None:
        from xtuner.v1.module.attention.dsa._dsa_rope import _apply_rope_inverse_split, _apply_rope_split

        torch.manual_seed(1)
        T, H, D, rope_dim = 32, 4, 48, 16
        x = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16)
        # Orthogonal rotation (c²+s²=1) is required for forward·inverse = I.
        cos, sin = self._make_structured_cos_sin(T, rope_dim, "cuda")

        rotated = _apply_rope_split(x, cos, sin, rope_dim)
        recovered = _apply_rope_inverse_split(rotated, cos, sin, rope_dim)

        # Round-trip must recover x within bf16 rounding (a few ULPs over 2 applications).
        assert torch.allclose(recovered.float(), x.float(), atol=0.1), (
            f"max roundtrip diff: {(recovered.float() - x.float()).abs().max().item():.4e}"
        )

    @pytest.mark.gpu
    def test_backward_matches_reference(self) -> None:
        from xtuner.v1.module.attention.dsa._dsa_rope import _apply_rope_split

        torch.manual_seed(2)
        T, H, D, rope_dim = 32, 4, 48, 16
        # Structured cos/sin: backward is the transpose = the inverse rotation,
        # which is what _rope_split_op uses internally (FORWARD=not ctx.forward).
        cos, sin = self._make_structured_cos_sin(T, rope_dim, "cuda")

        x_ref = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_fused = x_ref.detach().clone().requires_grad_(True)
        g = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16)

        ref_out = self._ref_forward(x_ref, cos, sin, rope_dim)
        ref_out.backward(g)

        fused_out = _apply_rope_split(x_fused, cos, sin, rope_dim)
        fused_out.backward(g)

        assert x_ref.grad is not None and x_fused.grad is not None
        denom = x_ref.grad.float().abs().max().item() + 1e-9
        rel = (x_ref.grad.float() - x_fused.grad.float()).abs().max().item() / denom
        assert rel < 2e-2, f"grad rel diff {rel:.3e} too large"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x"]))
