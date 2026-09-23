"""GLM-5.3-Flash 的 clamped SwiGLU 与 KPool indexer，见 doc/xtuner_glm5p3flash_design.md F5。

TestClampedSwiglu
    test_matches_hf_glm5next_text_mlp                        dense MLP 与 HF 逐值一致
    test_unlimited_mlp_still_honours_hidden_act              不限幅时仍按 hidden_act 取激活
    test_routed_experts_use_the_same_clamp_as_shared_experts routed experts 也走限幅
    test_clamped_swiglu_without_a_limit_is_rejected_at_config_time  缺 clip_limit 在配置期报错
    test_moe_mlp_matches_hf_apply_gate                       shared expert 与 HF 逐值一致
TestKpoolPoolLayout
    test_single_document_pool_layout                         单文档的池划分与 -1 补位
    test_pool_never_crosses_document_boundary                池不跨文档边界
    test_pads_up_to_alignment                                输出宽度按后端对齐补齐
TestKpoolSelection
    test_selected_token_sets_match_hf_single_document        单文档选择与 HF 一致
    test_packed_selection_matches_per_document_reference     packed 多文档互不串扰
    test_tilelang_matches_torch_reference                    生产 kernel 与 torch 参考一致
                                                             （含尾块未对齐、query 分块）
TestKpoolSequenceParallel
    test_sharded_queries_select_the_same_tokens_as_non_sp    分片后选择结果与非 SP 相同
    test_build_pools_rejects_key_features_covering_only_one_shard  建池要求全局 key
TestKpoolSequenceParallelParity
    test_kpool_topk_matches_non_sp                           2 卡生产路径与非 SP 一致
"""

import pytest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEMLP
from xtuner.v1.ops.sparse_mla.kpool import (
    build_pool_index,
    build_pools,
    kpool_output_width,
    kpool_topk_indices,
    torch_kpool_topk_indices,
)
from xtuner.v1.utils.test_utils import init_data_mesh


def _hf_glm53_mlp(hidden_size: int, intermediate_size: int, swiglu_limit: float):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.hidden_act = "silu"
    cfg.swiglu_limit = swiglu_limit
    return Glm5NextTextMLP(cfg)


class TestClampedSwiglu:
    def test_matches_hf_glm5next_text_mlp(self):
        # dense MLP 的限幅 SwiGLU 必须与 HF Glm5NextTextMLP 逐值一致（输入放大以触发限幅）。
        torch.manual_seed(0)
        hidden, inter, limit = 16, 32, 10.0
        hf_mlp = _hf_glm53_mlp(hidden, inter, limit)

        xtuner_mlp = DenseMLP(hidden_size=hidden, intermediate_size=inter, hidden_act="silu", swiglu_limit=limit)
        with torch.no_grad():
            xtuner_mlp.gate_proj.weight.copy_(hf_mlp.gate_proj.weight)
            xtuner_mlp.up_proj.weight.copy_(hf_mlp.up_proj.weight)
            xtuner_mlp.down_proj.weight.copy_(hf_mlp.down_proj.weight)

        # Use large-magnitude inputs so clamping actually engages both bounds.
        x = torch.randn(2, 5, hidden) * 20

        hf_out = hf_mlp(x)
        xtuner_out = xtuner_mlp(x)
        torch.testing.assert_close(xtuner_out, hf_out, atol=1e-5, rtol=1e-5)

    def test_unlimited_mlp_still_honours_hidden_act(self):
        """Clamping is GLM-5.3-Flash-specific; the elementwise activation is not. `DenseMLP`
        must keep deriving it from `hidden_act` (vision towers and InternVL use gelu), so a
        no-limit MLP is exactly `gelu(gate) * up` -- not a hardcoded silu."""
        # 不设 limit 时激活函数仍由 hidden_act 决定，限幅是 GLM-5.3 专属而非默认。
        torch.manual_seed(2)
        mlp = DenseMLP(hidden_size=4, intermediate_size=6, hidden_act="gelu")
        x = torch.randn(3, 4)
        expected = mlp.down_proj(torch.nn.functional.gelu(mlp.gate_proj(x)) * mlp.up_proj(x))
        torch.testing.assert_close(mlp(x), expected)

    def test_routed_experts_use_the_same_clamp_as_shared_experts(self):
        """The 288 routed experts run the fused gate_up path (``MoEBlock.moe_act``), not
        ``MoEMLP``. HF clamps there too (``Glm5NextTextExperts._apply_gate``), so the model
        config must select it -- plain SwiGLU diverges only once a pre-activation exceeds the
        limit, which short-sentence loss oracles rarely trigger."""
        # routed experts 走 fused gate_up 路径，配置必须把限幅接上去。
        from xtuner.v1.model.moe.glm53 import Glm53TextMoEConfig

        act = Glm53TextMoEConfig().moe_act_fn_cfg.build()
        fused = torch.tensor([[100.0, -100.0]])  # gate=100 -> clamp 10; up=-100 -> clamp -10
        expected = torch.nn.functional.silu(torch.tensor(10.0)) * torch.tensor(-10.0)
        torch.testing.assert_close(act(fused, split_dim=-1), expected.reshape(1, 1))

    def test_clamped_swiglu_without_a_limit_is_rejected_at_config_time(self):
        """Without `clip_limit` the activation reaches `gate.clamp(max=None)`, whose
        `RuntimeError: At least one of 'min' or 'max' must not be None` says nothing about the
        misconfiguration. Fail where the mistake is made instead."""
        # 缺 clip_limit 要在构造配置时报错，而不是到 forward 里 clamp(max=None)。
        from pydantic import ValidationError

        from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig

        with pytest.raises(ValidationError, match="clip_limit"):
            MoEActFnConfig(act_type="clamped_swiglu")

    def test_moe_mlp_matches_hf_apply_gate(self):
        # shared expert 走 MoEMLP，同样要与 HF 的 _apply_gate 逐值一致。
        torch.manual_seed(1)
        hidden, inter, limit = 16, 8, 10.0

        gate_up_proj = torch.randn(2 * inter, hidden) * 5
        down_proj = torch.randn(hidden, inter)

        def hf_apply_gate(x):
            gate_up = torch.nn.functional.linear(x, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            return torch.nn.functional.silu(gate) * up

        moe_mlp = MoEMLP(
            hidden_size=hidden, n_shared_experts=1, moe_intermediate_size=inter, hidden_act="silu", swiglu_limit=limit
        )
        with torch.no_grad():
            moe_mlp.gate_proj.weight.copy_(gate_up_proj[:inter])
            moe_mlp.up_proj.weight.copy_(gate_up_proj[inter:])
            moe_mlp.down_proj.weight.copy_(down_proj)

        x = torch.randn(3, hidden) * 5
        hf_gated = hf_apply_gate(x)
        hf_out = torch.nn.functional.linear(hf_gated, down_proj)
        xtuner_out = moe_mlp(x)
        torch.testing.assert_close(xtuner_out, hf_out, atol=1e-5, rtol=1e-5)


def _seq_ctx(doc_lens: list[int]) -> SequenceContext:
    return SequenceContext.from_input_ids(tuple(torch.zeros(1, n, dtype=torch.long) for n in doc_lens), device="cpu")


def _sp_seq_ctx(doc_lens: list[int], *, sp_size: int, sp_rank: int) -> SequenceContext:
    """A shard as ``SequenceContext.split()`` produces it: **global** ``cu_seq_lens_q``, local
    ``input_ids``, and ``shard_start``/``shard_size`` marking this rank's contiguous slice."""
    total = sum(doc_lens)
    assert total % sp_size == 0
    local = total // sp_size
    cu = torch.cumsum(torch.tensor([0] + list(doc_lens)), dim=0).int()
    return SequenceContext(
        input_ids=torch.zeros(1, local, dtype=torch.long),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(doc_lens),
        max_length_k=max(doc_lens),
        device="cpu",
        shard_start=sp_rank * local,
        shard_size=local,
    )


class TestKpoolPoolLayout:
    """池的划分与输出宽度：packed 多文档下的布局规则，HF 的单批参考覆盖不到。"""

    def test_single_document_pool_layout(self):
        # 10 tokens, kpool=4 -> pools [0,1,2,3], [4,5,6,7], [8,9,-1,-1].
        seq_ctx = _seq_ctx([10])
        pool_index = build_pool_index(seq_ctx, seq_len=10, index_kpool=4, device="cpu")
        expected = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, -1]], dtype=torch.int64)
        torch.testing.assert_close(pool_index, expected)

    def test_pool_never_crosses_document_boundary(self):
        # doc0 has 5 tokens (pools: [0..3], [4,-1,-1,-1]); doc1 starts fresh at token 5.
        seq_ctx = _seq_ctx([5, 6])
        pool_index = build_pool_index(seq_ctx, seq_len=11, index_kpool=4, device="cpu")
        # doc0: ceil(5/4)=2 pools; doc1: ceil(6/4)=2 pools -> 4 pools total.
        expected = torch.tensor([[0, 1, 2, 3], [4, -1, -1, -1], [5, 6, 7, 8], [9, 10, -1, -1]], dtype=torch.int64)
        torch.testing.assert_close(pool_index, expected)
        # No pool mixes tokens from doc0 (ids < 5) and doc1 (ids >= 5).
        for pool in pool_index:
            ids = pool[pool >= 0]
            assert (ids < 5).all() or (ids >= 5).all()

    def test_pads_up_to_alignment(self):
        # index_topk=2048, index_kpool=4 -> semantic width 2051.
        assert kpool_output_width(2048, 4, alignment=512) == 2560
        assert kpool_output_width(2048, 4, alignment=64) == 2112
        assert kpool_output_width(2048, 4, alignment=1) == 2051


class TestKpoolSelection:
    """公开 API 选出的 token 集合：与 HF 对齐、packed 文档隔离、后端一致。"""

    def _hf_indexer(self, *, hidden_size, index_n_heads, index_head_dim, q_lora_rank, index_topk, index_kpool):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.hidden_size = hidden_size
        cfg.index_n_heads = index_n_heads
        cfg.index_head_dim = index_head_dim
        cfg.qk_rope_head_dim = 0
        cfg.index_topk = index_topk
        cfg.q_lora_rank = q_lora_rank
        cfg.index_kpool = index_kpool
        cfg.index_kpool_always_select_tail = True
        indexer = Glm5NextTextIndexer(cfg, layer_idx=0)
        with torch.no_grad():
            for p in indexer.parameters():
                p.normal_(mean=0.0, std=0.02)
        return indexer

    def test_selected_token_sets_match_hf_single_document(self):
        # 单文档下选出的 token 集合必须与 HF Glm5NextTextIndexer 一致。
        torch.manual_seed(0)
        hidden_size, index_n_heads, index_head_dim = 16, 4, 8
        q_lora_rank, index_topk, index_kpool = 12, 8, 4
        seq_len = 19  # deliberately not a multiple of index_kpool

        indexer = self._hf_indexer(
            hidden_size=hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            q_lora_rank=q_lora_rank,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )

        hidden_states = torch.randn(1, seq_len, hidden_size)
        q_resid = torch.randn(1, seq_len, q_lora_rank)
        attention_mask = torch.ones(1, seq_len, dtype=torch.bool)
        hf_topk = indexer(hidden_states, q_resid, attention_mask, past_key_values=None)

        # Reproduce the same internal projections XTuner's KPoolIndexer will use, with HF's
        # own weights, then run XTuner's reference KPool implementation on top.
        with torch.no_grad():
            q = indexer.wq_b(q_resid).view(1, seq_len, index_n_heads, index_head_dim)
            k = indexer.k_norm(indexer.wk(hidden_states))
            gate_scores = torch.nn.functional.linear(hidden_states, indexer.index_kpool_compress_gate)
            weights = indexer.weights_proj(hidden_states)

        seq_ctx = _seq_ctx([seq_len])
        xtuner_topk = torch_kpool_topk_indices(
            q,
            k,
            gate_scores,
            weights,
            indexer.index_kpool_compress_ape,
            seq_ctx,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_kpool=index_kpool,
            always_select_tail=True,
            alignment=1,
        )

        assert xtuner_topk.shape[-1] == hf_topk.shape[-1]
        for row in range(seq_len):
            hf_set = set(hf_topk[0, row].tolist()) - {-1}
            xtuner_set = set(xtuner_topk[row, 0].tolist()) - {-1}
            assert xtuner_set == hf_set, f"row {row}: hf={hf_set} xtuner={xtuner_set}"

    def test_packed_selection_matches_per_document_reference(self):
        # packed 多文档时每个文档的选择必须与单独跑该文档一致，互不串扰。
        torch.manual_seed(3)
        index_head_dim, index_n_heads = 8, 4
        index_topk, index_kpool = 8, 4
        len0, len1 = 13, 17

        k_all = torch.randn(len0 + len1, index_head_dim)
        gate_all = torch.randn(len0 + len1, index_head_dim)
        q_all = torch.randn(1, len0 + len1, index_n_heads, index_head_dim)
        weights_all = torch.randn(1, len0 + len1, index_n_heads)
        kpool_ape = torch.randn(index_kpool, index_head_dim)

        packed_ctx = _seq_ctx([len0, len1])
        packed_out = torch_kpool_topk_indices(
            q_all,
            k_all.unsqueeze(0),
            gate_all.unsqueeze(0),
            weights_all,
            kpool_ape,
            packed_ctx,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_kpool=index_kpool,
            always_select_tail=True,
            alignment=1,
        )

        for offset, length in ((0, len0), (len0, len1)):
            solo_ctx = _seq_ctx([length])
            solo_out = torch_kpool_topk_indices(
                q_all[:, offset : offset + length],
                k_all[offset : offset + length].unsqueeze(0),
                gate_all[offset : offset + length].unsqueeze(0),
                weights_all[:, offset : offset + length],
                kpool_ape,
                solo_ctx,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                index_kpool=index_kpool,
                always_select_tail=True,
                alignment=1,
            )
            for local_row in range(length):
                packed_selected = set(packed_out[offset + local_row, 0].tolist()) - {-1}
                # Every selected id, remapped to this document's local frame, must fall inside
                # [0, length) -- i.e. no pool/tail ever reaches into the other document.
                assert all(offset <= idx < offset + length for idx in packed_selected)
                local_selected = {idx - offset for idx in packed_selected}
                solo_selected = set(solo_out[local_row, 0].tolist()) - {-1}
                assert local_selected == solo_selected, f"doc offset={offset} row={local_row}"

    @pytest.mark.gpu
    @pytest.mark.parametrize("len1, query_chunk_size", [(97, None), (98, None), (97, 64)])
    def test_tilelang_matches_torch_reference(self, len1, query_chunk_size):
        # 生产 kernel 与 torch 参考实现选出的 token 集合必须一致；len1=98 让总 query 数
        # 不能被 block_q(=128/32) 整除，覆盖尾块填充，query_chunk_size 覆盖分块路径。
        torch.manual_seed(5)
        index_head_dim, index_n_heads = 128, 32
        # Real selection pressure: index_topk // index_kpool (16) << pools available per doc
        # (~25-33), so this exercises genuine top-k, not "select everything".
        index_topk, index_kpool = 64, 4
        len0 = 131
        seq_len = len0 + len1
        device = "cuda"

        k_all = torch.randn(seq_len, index_head_dim, device=device, dtype=torch.bfloat16)
        gate_all = torch.randn(seq_len, index_head_dim, device=device, dtype=torch.bfloat16)
        q_all = torch.randn(1, seq_len, index_n_heads, index_head_dim, device=device, dtype=torch.bfloat16)
        weights_all = torch.randn(1, seq_len, index_n_heads, device=device)
        kpool_ape = torch.randn(index_kpool, index_head_dim, device=device, dtype=torch.bfloat16)

        seq_ctx = SequenceContext.from_input_ids(
            (torch.zeros(1, len0, dtype=torch.long), torch.zeros(1, len1, dtype=torch.long)), device=device
        )

        kwargs = dict(
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_kpool=index_kpool,
            always_select_tail=True,
            alignment=1,
        )
        torch_out = torch_kpool_topk_indices(
            q_all, k_all.unsqueeze(0), gate_all.unsqueeze(0), weights_all, kpool_ape, seq_ctx, **kwargs
        )
        tilelang_out = kpool_topk_indices(
            q_all,
            k_all.unsqueeze(0),
            gate_all.unsqueeze(0),
            weights_all,
            kpool_ape,
            seq_ctx,
            query_chunk_size=query_chunk_size,
            **kwargs,
        )

        assert torch_out.shape == tilelang_out.shape
        for row in range(seq_len):
            torch_set = set(torch_out[row, 0].tolist()) - {-1}
            tilelang_set = set(tilelang_out[row, 0].tolist()) - {-1}
            assert torch_set == tilelang_set, f"row {row}: torch={torch_set} tilelang={tilelang_set}"


class TestKpoolSequenceParallel:
    """KPool bookkeeping must live in the same *global* token coordinate system as
    ``cu_seq_lens_q`` (which ``SequenceContext.split`` leaves unsharded), exactly like
    ``SequenceContext.packed_causal_query_ranges``."""

    def test_sharded_queries_select_the_same_tokens_as_non_sp(self):
        """Full chain on a shard: pools are built over the whole sequence, only the queries are
        local. Equivalent to what each rank computes after ``kpool_topk_indices`` gathers the
        key-side features (with ``sp_mesh=None`` the gather is the identity, so passing the
        global key features here reproduces the post-gather state exactly)."""
        # SP 分片后每个 rank 对本地 query 的选择必须与非 SP 完全一致。
        torch.manual_seed(0)
        doc_lens, kpool, topk, n_heads, head_dim = [10, 6], 4, 8, 2, 8
        total, sp_size = sum(doc_lens), 2
        local = total // sp_size

        q = torch.randn(1, total, n_heads, head_dim)
        k = torch.randn(1, total, head_dim)
        gate = torch.randn(1, total, head_dim)
        weights = torch.randn(1, total, n_heads)
        ape = torch.randn(kpool, head_dim)

        kwargs = dict(index_head_dim=head_dim, index_topk=topk, index_kpool=kpool, alignment=1)
        ref = torch_kpool_topk_indices(q, k, gate, weights, ape, _seq_ctx(doc_lens), **kwargs)

        for rank in range(sp_size):
            sl = slice(rank * local, (rank + 1) * local)
            out = torch_kpool_topk_indices(
                q[:, sl], k, gate, weights[:, sl], ape, _sp_seq_ctx(doc_lens, sp_size=sp_size, sp_rank=rank), **kwargs
            )
            torch.testing.assert_close(out, ref[sl], msg=f"rank {rank} selected different tokens")

    def test_build_pools_rejects_key_features_covering_only_one_shard(self):
        """Pool construction is a whole-sequence operation: a pool can straddle the shard seam
        (documents start at arbitrary offsets), so no rank can build its own pools from local
        keys. Callers must gather first; this guard turns a silent mis-pooling into an error."""
        # 池构建是全序列操作，喂入分片 key 必须报错而不是静默建错池。
        rank1 = _sp_seq_ctx([16], sp_size=2, sp_rank=1)
        with pytest.raises(RuntimeError, match="whole sequence"):
            build_pools(torch.randn(8, 4), torch.randn(8, 4), torch.randn(4, 4), rank1, index_kpool=4)


class TestKpoolSequenceParallelParity(DistributedTestBase):
    """2-rank SP parity for the production KPool path: each rank's selected token sets must
    equal the non-SP reference's rows for the same global tokens."""

    @pytest.mark.gpu
    def test_kpool_topk_matches_non_sp(self, device="cuda"):
        # 2 卡下生产 kernel 的选择结果与非 SP 参考一致（文档长度让池跨接缝）。
        self.create_pg(device)
        torch.manual_seed(7)
        index_head_dim, index_n_heads, index_topk, index_kpool = 128, 32, 64, 4
        # Document lengths chosen so a pool straddles the shard seam: doc0 ends at 53, so
        # doc1's pools start at an offset that is not a multiple of index_kpool.
        len0, len1 = 53, 75
        seq_len, sp_size = len0 + len1, self.world_size
        assert seq_len % sp_size == 0
        local = seq_len // sp_size

        def _bcast(t):
            torch.distributed.broadcast(t, src=0)
            return t

        k = _bcast(torch.randn(1, seq_len, index_head_dim, device=device, dtype=torch.bfloat16))
        gate = _bcast(torch.randn(1, seq_len, index_head_dim, device=device, dtype=torch.bfloat16))
        q = _bcast(torch.randn(1, seq_len, index_n_heads, index_head_dim, device=device, dtype=torch.bfloat16))
        weights = _bcast(torch.randn(1, seq_len, index_n_heads, device=device))
        ape = _bcast(torch.randn(index_kpool, index_head_dim, device=device, dtype=torch.bfloat16))

        kwargs = dict(
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_kpool=index_kpool,
            always_select_tail=True,
            alignment=1,
        )
        input_ids = (torch.zeros(1, len0, dtype=torch.long), torch.zeros(1, len1, dtype=torch.long))
        ref = kpool_topk_indices(
            q, k, gate, weights, ape, SequenceContext.from_input_ids(input_ids, device=device), **kwargs
        )

        sp_mesh = init_data_mesh(device, sp_size)["sp"]
        rank = sp_mesh.get_local_rank()
        sp_ctx = SequenceContext.from_input_ids(input_ids, device=device).split(sequence_parallel_mesh=sp_mesh)
        sl = slice(rank * local, (rank + 1) * local)
        out = kpool_topk_indices(
            q[:, sl].contiguous(),
            k[:, sl].contiguous(),
            gate[:, sl].contiguous(),
            weights[:, sl].contiguous(),
            ape,
            sp_ctx,
            **kwargs,
        )

        assert out.shape == ref[sl].shape
        for row in range(local):
            assert set(out[row, 0].tolist()) - {-1} == set(ref[sl][row, 0].tolist()) - {-1}, f"rank={rank} row={row}"

    @property
    def world_size(self) -> int:
        return 2
