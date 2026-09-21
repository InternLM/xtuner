# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash 在 XTuner 中的实现伪代码。

配套设计文档：``doc/xtuner_glm5p3flash_design.md``。

本文件只用于评审数据流与接口契约，**不可运行**：省略了 import 细节、错误分支和大部分
类型标注之外的样板代码。每一段前的 ``# === file: ... ===`` 标出真实落点。

约定：
    B   batch（训练期恒为 1，packed）
    S   本地序列长度；S_g 为 SP gather 后的全局长度
    D   hidden_size = 4096
    H   hc_mult = 4
    N   num_attention_heads = 64
    Rq  q_lora_rank = 1536；Rkv kv_lora_rank = 512
    Dn  qk_nope_head_dim = 256；Dv v_head_dim = 256；Dr qk_rope_head_dim = 0
    P   pool 数 = ceil(S / index_kpool)
"""

# =============================================================================
# === file: xtuner/v1/ops/act_fn.py  (F5, 增量)
# =============================================================================


def native_clamped_swiglu(fused_x, split_dim=-1, limit=10.0):
    """GLM-5.3-Flash 的限幅 SwiGLU，用于 fused gate_up 的 routed expert 路径。

    与已有的 ``native_clipped_swiglu``（GPT-OSS 的 ``(up+1)*gate*sigmoid(alpha*gate)``）
    不是同一个函数，必须新增而不是复用。
    """
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


act_fn_type_map_cuda["clamped_swiglu"] = native_clamped_swiglu


class MoEActFnConfig(BaseModel):  # 增量：新增 act_type 取值与 build 分支
    act_type: Literal["clipped_swiglu", "clamped_swiglu", "swiglu"] = "swiglu"
    clip_alpha: float | None = None
    clip_limit: float | None = None

    def build(self) -> MoEActFnProtocol:
        act_fn = get_act_fn(self.act_type)
        if self.act_type == "clipped_swiglu":
            return partial(act_fn, alpha=self.clip_alpha, limit=self.clip_limit)
        if self.act_type == "clamped_swiglu":
            return partial(act_fn, limit=self.clip_limit)
        return act_fn


# ``DenseMLP`` / ``MoEMLP``（非 fused 的 dense 前 3 层与 shared expert）增加 swiglu_limit：
class DenseMLP(nn.Module):  # 增量
    def __init__(self, *, swiglu_limit: float | None = None, **kwargs):
        ...
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


# =============================================================================
# === file: xtuner/v1/module/attention/kda.py  (F3)
# === 直接移植 ~/github/xtuner-ncp-k3 的同名文件；GLM 以 use_full_rank_gate=False 构建。
# =============================================================================


class KDAConfig(BaseModel):
    """Kimi Delta Attention。参数名与发布版 checkpoint 一一对应。"""

    num_heads: int = 64
    head_dim: int = 128
    conv_kernel_size: int = 4
    # GLM-5.3-Flash 是 g_a_proj -> g_b_proj 的低秩输出门
    use_full_rank_gate: bool = False
    gate_lower_bound: float | None = -5.0
    rms_norm_eps: float = 1e-5

    def build(self, hidden_size, float8_cfg=None, layer_idx=0, **_) -> "KimiDeltaAttention":
        return KimiDeltaAttention(**self.model_dump(), hidden_size=hidden_size,
                                  float8_cfg=float8_cfg, layer_idx=layer_idx)


class KimiDeltaAttention(nn.Module):
    """参数：q/k/v_proj, q/k/v_conv1d, f_a_proj+f_b_proj+A_log+dt_bias, b_proj,
    g_a_proj+g_b_proj, o_norm, o_proj —— 与 ``model.language_model.layers.N.self_attn.*``
    的 34 个 KDA 层键名完全一致。"""

    def forward(self, hidden_states, seq_ctx, **kwargs):
        # decoder layer 仍会传 position_embeddings（NoPE，KDA 也不用），显式丢弃
        del kwargs
        if seq_ctx.sequence_parallel_mesh is not None and seq_ctx.sequence_parallel_mesh.size() > 1:
            return self.forward_for_sp(hidden_states, seq_ctx)

        cu_seqlens = seq_ctx.cu_seq_lens_q  # packed 文档边界，短卷积与 chunk kernel 都消费它
        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cu_seqlens=cu_seqlens)

        # 门控必须在 kernel **外**算：已安装的 fla 0.4.2 的 chunk_kda 签名里
        # 没有 A_log / dt_bias / use_beta_sigmoid_in_kernel，照抄 ncp-k3 的写法会让这三个
        # kwarg 落进 **kwargs 被静默丢弃 —— 门控不施加 A_log/dt_bias/下界、beta 不过 sigmoid，
        # 且不报错。见设计文档 3.5.1。
        #
        # fused_kda_gate 有独立的 fwd/bwd Triton kernel（KDAGateFunction）；
        # 其参考实现 naive_kda_lowerbound_gate 的源码是
        #     lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
        # 与 HF Glm5NextTextForgetGate 逐字一致，可直接当 oracle。
        g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).view(B, S, self.num_heads, self.head_dim)
        gate = fused_kda_gate(
            g_raw,
            dtensor_full_tensor(self.A_log),            # fp32 holder
            dt_bias=dtensor_full_tensor(self.dt_bias),  # fp32 holder
            lower_bound=self.gate_lower_bound,
        )
        beta = self.b_proj(hidden_states).float().sigmoid()   # sigmoid 也在外面

        # 分派同 Automodel：长序列 / CP 用 chunk，短序列用 fused_recurrent
        kernel = chunk_kda if (cp_context is not None or S > 64) else fused_recurrent_kda
        o, _ = kernel(
            q=q.view(B, S, self.num_heads, self.head_dim),
            k=k.view(B, S, self.num_heads, self.head_dim),
            v=v.view(B, S, self.num_heads, self.head_dim),
            g=gate, beta=beta,
            use_qk_l2norm_in_kernel=True, transpose_state_layout=True,
            safe_gate=(self.gate_lower_bound is not None),    # 只传给 chunk 路径
            cu_seqlens=cu_seqlens,
        )
        gate_out = self.g_b_proj(self.g_a_proj(hidden_states)).view(B, S, self.num_heads, self.head_dim)
        raw_output = self.o_norm(o, gate_out).reshape(B, S, -1)   # sigmoid-gated RMSNorm
        return {"raw_output": raw_output, "projected_output": self.o_proj(raw_output), "softmax_lse": None}

    def forward_for_sp(self, hidden_states, seq_ctx):
        """Ulysses：本地 seq / 全局 head -> 全局 seq / head 分片，再 all-to-all 回来。

        短卷积和 recurrent state 都有从左到右的生命周期。Ulysses 把它们变成 rank-local 问题：
        每个 rank 拿到**完整序列**的一部分 head，因此不需要跨 rank 传状态。
        A_log / dt_bias 同步按 head 切片。

        与 Automodel 的取舍（设计文档 3.6）：Automodel 走 contiguous CP，序列保持分片、
        靠 fla.ops.cp 的 send_recv 左→右传 recurrent state + 3 token 的卷积左 halo；
        通信量极小但**串行**，延迟 ∝ cp_size。Ulysses 通信量 ~S*C/sp 但全并行，
        约束是 num_heads % sp_size == 0（64 heads，宽松）。XTuner 全栈都是 Ulysses，
        不引入第二条并行轴。

        前提（由 SequenceContext.split 保证，必须断言而不是靠注释）：
        分片是**连续**的且等长非空。若将来引入 zigzag / load-balanced 切分，
        短卷积与 recurrent 状态都会静默出错。
        """
        ...


# =============================================================================
# === file: xtuner/v1/module/decoder_layer/mhc.py  (F4)
# =============================================================================


class MHCConfig(BaseModel):
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20


def hc_split_sinkhorn(mixes, scale, base, hc_mult, iters, eps):
    """把 ``[..., (2+H)*H]`` 的 mix logits 拆成 pre / post / comb。

    全程 fp32（20 轮迭代在 bf16 下会 NaN）。第一轮必须用 ``torch.softmax`` 而非手写
    ``x - amax + exp + /sum``：后者与 HF 有 ~6e-8 的 ULP 差，会在下游 bf16 cast 边界翻转。
    """
    assert mixes.dtype == torch.float32
    pre_w, post_w, comb_w = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    pre_b, post_b, comb_b = base.split([hc_mult, hc_mult, hc_mult * hc_mult])
    pre_s, post_s, comb_s = scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps            # [..., H]
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)        # [..., H]

    comb = comb_w.view(*comb_w.shape[:-1], hc_mult, hc_mult) * comb_s + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):                                   # Sinkhorn-Knopp 投影到双随机流形
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


@maybe_compile
def hc_pre(streams, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps):
    """``[B,S,H,D]`` -> 子层输入 ``[B,S,D]`` 以及 post / comb。

    ``input_norm`` 是 HF 的 ``Glm5NextTextUnweightedRMSNorm``（无权重 RMS rescale）。
    """
    flat = streams.flatten(2)                                     # [B,S,H*D]
    flat_normed = F.rms_norm(flat, (flat.size(-1),), weight=None, eps=norm_eps)
    mixes = F.linear(flat_normed, hc_fn.to(flat.dtype)).float()    # [B,S,(2+H)*H]
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    return collapsed, post, comb


def hc_post(sublayer_out, residual, post, comb):
    """``out[h,d] = post[h]*x[d] + sum_{h'} comb[h',h] * residual[h',d]``。

    ``comb`` 是双随机但**非对称**的，归约轴是第一个 hc 轴，所以必须 transpose。

    三级分派（从 dsv4 移植，见设计文档 3.5.3）。Automodel 只有下面的 eager matmul 分支，
    它在 K=hc_mult=4 时低于 Hopper wgmma tile 下限、cuBLAS 回落 CUDA-core 带宽受限；
    dsv4 实测 pack=16384 下约 3 ms/call，GLM-5.3 有 45x2=90 call/step。
    """
    if hf_parity_enabled():                       # bitwise 锚点，优先级最高
        return _hc_post_eager(sublayer_out, residual, post, comb)
    if _USE_MHC_KERNELS and residual.is_cuda and residual.dtype == torch.bfloat16:
        from xtuner.v1.ops.mhc import mhc_post    # TileKernels 后端
        return mhc_post(sublayer_out, residual, post, comb)
    if residual.is_cuda and residual.dtype == torch.bfloat16 and _hc_post_fused_available():
        from xtuner.v1.ops.hc_post import hc_post_fused   # 默认：Triton
        return hc_post_fused(sublayer_out, residual, post, comb)
    return _hc_post_eager(sublayer_out, residual, post, comb)


@maybe_compile
def _hc_post_eager(sublayer_out, residual, post, comb):
    """eager 回退。必须在 compile cfg 里：否则 ``[B,S,H,H,D]`` 会真实 materialize
    （pack=16384 / H=4 / D=4096 约 8 GB）。"""
    post = post.to(residual.dtype)
    mixed = torch.matmul(comb.to(residual.dtype).transpose(-1, -2), residual)
    return post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) + mixed


def unshard_hc_params(*params):
    """FSDP 下把 DTensor 提前 full_tensor()，避免在 compile 区内每层 graph break 三次。"""
    return tuple(p.full_tensor() if isinstance(p, DTensor) else p for p in params)


# =============================================================================
# === file: xtuner/v1/ops/sparse_mla/kpool.py  (F5)
# === 生产路径 = A(pool 构建) + B(复用现成 indexer kernel) + C(展开 + tail)
# =============================================================================


# KPool 输出 buffer 宽度：语义宽度 index_topk + kpool - 1 = 2051，
# 但各 SparseMLA fwd kernel 对宽度有各自的对齐要求，所以补到对齐宽度、尾部填 -1
# （kernel 内 `mask = Indices != -1` / `topk_length` 已处理）。
def kpool_output_width(index_topk: int, index_kpool: int, alignment: int) -> int:
    # 对齐值由所选 SparseMLA fwd 决定，不硬编码：
    #   FlashMLA -> _FLASH_MLA_TOPK_ALIGNMENT = 512 => 2051 补到 2560
    #   TileLang -> block_I = 64                   => 2051 补到 2112
    return ceil_div(index_topk + index_kpool - 1, alignment) * alignment


# ---------------------------------------------------------------------------
# A. pool 构建：O(S * Di)，纯 PyTorch，inductor 可全融合
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_pools(k, gate_scores, kpool_ape, seq_ctx, *, index_kpool):
    """按**文档**切 pool，返回 (pool_key [P, Di], pool_index [P, kpool], pool_complete [P])。

    前提：SP 分片连续（``split_for_sequence_parallel`` 是 ``narrow``，且 ``SequenceContext.split``
    先 pad 到 sp_size 倍数），否则"按文档分池"的下标推导不成立。

    与 HF ``Glm5NextTextIndexer.get_pooled_states`` 的差异只有 pool 起点：
    HF 假设 ``[B, S]`` 左 padding batch，从整行第一个有效 token 起分池；
    XTuner 是 bsz=1 的 packed 多文档，**每个文档各自从自己的起点重新分池**，
    否则 pool 会跨文档、造成样本间泄漏。

    代价：gather ``[P, kpool, Di]``（bf16，P=4096/kpool=4/Di=128 时 4 MB）+ softmax + 求和。
    相对 B 段的 GEMM 可忽略，不需要自定义 kernel。
    """
    doc_start, doc_end = seq_ctx.packed_causal_query_ranges(k.shape[0], k.device)
    # pool_index[p] = 该 pool 的 kpool 个全局 token id，越界或跨文档填 -1
    pool_index = build_pool_index(doc_start, doc_end, index_kpool)          # [P, kpool] int32
    pool_complete = (pool_index >= 0).all(-1)                               # [P]

    valid = pool_index >= 0
    safe = pool_index.clamp(min=0)
    grouped_k = k[safe]                                                     # [P, kpool, Di]
    grouped_gate = gate_scores[safe]                                        # [P, kpool, Di]
    # pool 内按 softmax(gate + ape) 加权平均；全无效 pool 的 softmax 会出 NaN，需 nan_to_num
    logits = grouped_gate.float() + kpool_ape.float()
    logits = logits.masked_fill(~valid[..., None], float("-inf"))
    probs = torch.nan_to_num(logits.softmax(dim=1)).to(grouped_k.dtype)
    pool_key = (probs * grouped_k).sum(dim=1)                               # [P, Di]
    return pool_key, pool_index, pool_complete


def pool_causal_ranges(seq_ctx, pool_index, pool_complete, query_len, device):
    """把每个 query 的因果区间从 **token 空间** 换算到 **pool 空间**。

    这是 B 段复用现成 kernel 的唯一适配点：kernel 只要求 starts/ends 是 key 索引空间的
    半开区间，对 key 序列本身是什么毫无假设。

    一个 pool 可被 query 选中的条件 = 池完整 且 池末 token 对该 query 可见
    （同文档 + 因果），与 HF 的 ``pool_visible & pool_valid`` 等价；
    因为 pool 在文档内是按 token 顺序单调排列的，该条件恰好是一个连续区间。
    """
    return starts, ends                                                     # 各 [S] int32


# ---------------------------------------------------------------------------
# C. pool 展开 + tail：单次 gather，带宽受限
# ---------------------------------------------------------------------------
@torch.no_grad()
def expand_pools_and_tail(selected_pool_ids, pool_index, seq_ctx, *,
                          index_topk, index_kpool, always_select_tail):
    """[S, 512] pool id -> [S, 2112] token id（尾部 -1 padding）。

    ``[S, 512, kpool] -> [S, 2048]`` 的 gather 在 S=16384 时写出 134 MB int32，
    与 GLM-5.2 已有的 ``[S, 1, 2048]`` 输出同量级，不构成新瓶颈。
    若 profile 显示这一段成为热点，可把它和 B 的 top-k 输出融进一个 Triton kernel，
    省掉中间 ``[S, 512, kpool]`` 的往返——属于可选优化，不进第一版。
    """
    width = kpool_output_width(index_topk, index_kpool)
    out = selected_pool_ids.new_full((selected_pool_ids.shape[0], width), -1, dtype=torch.int32)
    expanded = pool_index[selected_pool_ids].flatten(-2)                    # [S, 512*kpool]
    out[:, : expanded.shape[-1]] = expanded.masked_fill(~selected_valid_expanded, -1)
    if always_select_tail:
        # 当前文档可见前缀里不满一池的至多 kpool-1 个 token，原样追加
        out[:, index_topk : index_topk + index_kpool - 1] = build_visible_tail(seq_ctx, index_kpool)
    return out.unsqueeze(1)                                                 # [S, 1, 2112]


# ---------------------------------------------------------------------------
# 组装：生产路径
# ---------------------------------------------------------------------------
@torch.no_grad()
def kpool_topk_indices(
    q,                  # [1, S, Ni, Di]  index query，SP 下保持本地分片
    k,                  # [1, S, Di]      index key
    gate_scores,        # [1, S, Di]      index_kpool_compress_gate(hidden)
    weights,            # [1, S, Ni]      weights_proj(hidden)
    kpool_ape,          # [kpool, Di]     index_kpool_compress_ape
    seq_ctx,
    *,
    index_head_dim, index_topk, index_kpool=4, always_select_tail=True,
    backend: DSAIndexerBackend = "tilelang",
    query_chunk_size: int | None = None,
):
    """整个 KPool indexer 只有 B 段是重计算，而它是 XTuner 已有生产 kernel 的 4 倍小实例。

    规模（S=16384, Ni=32, Di=128, kpool=4 => P=4096）::

                          GLM-5.2 token-level        GLM-5.3 KPool
        打分 GEMM         S*S*Ni*Di  = 1.1e12 MAC    S*(S/4)*Ni*Di = 2.7e11 MAC
        logits 缓冲       [16384,16384] fp32 1.07GB  [16384,4096] fp32 268MB
        top-k             16384 选 2048              4096 选 512

    另外两处正好命中现成特化：
      - lmdeploy_sparse_index_topk._SUPPORTED_TOPK = (512, 2048)，KPool 的 512 有 byte-radix 特化；
      - DEEPGEMM_MQA_SUPPORTED_HEADS = (32, 64, 128) 且要求 index_head_dim==128，
        GLM-5.3 的 (32, 128) 命中，FP8 indexer 可直接启用。
    """
    # --- A ---
    pool_key, pool_index, pool_complete = build_pools(
        k.squeeze(0), gate_scores.squeeze(0), kpool_ape, seq_ctx, index_kpool=index_kpool)

    # SP：query 保持分片，只 all-gather 体积小的 pool 侧（pool 数是 token 数的 1/kpool，
    # 比 GLM-5.2 gather 全量 token key 更省）
    sp_mesh = seq_ctx.sequence_parallel_mesh
    pool_key = gather_for_sequence_parallel(pool_key, dim=0, sp_mesh=sp_mesh)
    pool_index = gather_for_sequence_parallel(pool_index, dim=0, sp_mesh=sp_mesh)
    pool_complete = gather_for_sequence_parallel(pool_complete, dim=0, sp_mesh=sp_mesh)

    # --- B：零改动复用现成 indexer kernel ---
    #   k          -> pool_key [P, Di]        （而不是 token key [S_k, Di]）
    #   starts/ends-> pool 空间的因果区间       （而不是 token 空间）
    #   index_topk -> 512 = index_topk // kpool
    # kernel 内部已包含 relu(score) * weights 的 head 加权求和与 -inf 掩码，
    # 语义与 HF Glm5NextTextIndexer 一致；weights 的 Ni^-0.5 / Di^-0.5 缩放由 wrapper 施加。
    starts, ends = pool_causal_ranges(seq_ctx, pool_index, pool_complete, q.shape[1], q.device)
    selected_pool_ids = dispatch_indexer_topk(
        backend,
        q.squeeze(0), pool_key, weights.squeeze(0),
        starts, ends,
        index_topk=index_topk // index_kpool,     # 512，命中 radix top-k 特化
        index_head_dim=index_head_dim,
        query_chunk_size=query_chunk_size,        # 把 logits 从 [S,P] 降到 [chunk,P]
    )

    # --- C ---
    return expand_pools_and_tail(selected_pool_ids, pool_index, seq_ctx,
                                 index_topk=index_topk, index_kpool=index_kpool,
                                 always_select_tail=always_select_tail)


@torch.no_grad()
def torch_kpool_topk_indices(q, k, gate_scores, weights, kpool_ape, seq_ctx, **kw):
    """**参考实现，禁止用于生产。**

    einsum 会 materialize ``[S, Ni, P]`` fp32——S=16384 时 8.6 GB。
    只用于单测、CPU 与小 shape，以及给 tilelang 路径做 backend parity 的对照。
    """
    pool_key, pool_index, pool_complete = build_pools(...)
    scores = torch.relu(torch.einsum("bshd,pd->bshp", q.float(), pool_key.float()) * index_head_dim**-0.5)
    index_scores = torch.einsum("bshp,bsh->bsp", scores, weights.float() * q.shape[2] ** -0.5)
    index_scores = index_scores.masked_fill(~(visible & pool_complete[None, :]), float("-inf"))
    selected = index_scores.topk(min(index_topk // index_kpool, index_scores.shape[-1]), dim=-1).indices
    return expand_pools_and_tail(selected, pool_index, seq_ctx, ...)


# =============================================================================
# === file: xtuner/v1/ops/sparse_mla/tilelang_sparse_mla_fwd.py  (F5, 改动)
# =============================================================================


# NoPE 需要的唯一 kernel 改动（fwd 与 bwd 各一处）：
#
#   现状                                            NoPE 下
#   assert dim_plus_tail_dim == 576                 512    -> 放开为白名单 {(576,512), (512,512)}
#   assert dim == next_power_of_2(dim)              512 OK  -> 不用改
#   assert tail_dim == next_power_of_2(tail_dim)    0 FAIL  -> next_power_of_2(0)==2，必须放开 0
#   assert topk % block_I == 0  (block_I=64)        2051 FAIL -> 输出 buffer 固定 2112
#
# tail_dim == 0 时必须整段跳过 K_tail_shared 的加载与 tail GEMM；
# sm_scale 由调用方显式传入 256^-0.5，不走 kernel 内 1/(dim+tail_dim) 的默认值。
# `-1` 语义无需改：kernel 内已有 `mask[bi] = Indices[...] != -1` 并把 masked 槽位置 -inf。


# =============================================================================
# === file: xtuner/v1/model/moe/glm53/nope_dsa_mla.py  (F5)
# =============================================================================


class NoPEDSAMLAConfig(MLAConfig):
    """NoPE + KPool 的 DSA。

    ``qk_rope_head_dim=0`` 使基类自动得到正确的 ``q_head_dim = 256`` 与
    ``softmax_scale = 256^-0.5``，``kv_a_proj_with_mqa`` 输出 512，无需改基类。
    """

    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    indexer_types: list[str] | None = None
    # NoPE 生产路径 = FlashMLA fwd(原生支持 512 head_dim) + cuDNN bwd(无维度硬编码)，
    # 等价于 Automodel 的 cudnn_sparse_attention；不需要改 TileLang kernel（设计文档 3.5.2）。
    # 现有的 flash_mla / cudnn_dsa 各带一半 TileLang，NoPE 下都用不了。
    sparse_mla_backend: Literal["torch", "flash_mla_cudnn", "tilelang"] = "flash_mla_cudnn"
    indexer_backend: Literal["torch", "tilelang", "deep_gemm_fp8"] | None = None
    indexer_topk_query_chunk_size: int | None = None
    freeze_dsa_indexer: bool = True

    @model_validator(mode="after")
    def _check(self):
        assert self.qk_rope_head_dim == 0, "GLM-5.3-Flash 主 attention 是 NoPE"
        assert self.index_topk % self.index_kpool == 0
        if self.indexer_types is not None and "shared" in self.indexer_types:
            raise ValueError("GLM-5.3-Flash 发布 config 的 indexer_types 全为 full，不支持 IndexShare")
        return self

    def build(self, hidden_size, layer_idx=0, **kwargs) -> "NoPEDSAMultiLatentAttention":
        return NoPEDSAMultiLatentAttention(**self.model_dump(), hidden_size=hidden_size,
                                           layer_idx=layer_idx, **kwargs)


class KPoolIndexer(nn.Module):
    """键名对齐 ``self_attn.indexer.*``：wq_b / wk / k_norm / weights_proj /
    index_kpool_compress_ape / index_kpool_compress_gate。"""

    def forward(self, hidden_states, q_resid, seq_ctx):
        q = self.wq_b(q_resid).view(B, S, self.index_n_heads, self.index_head_dim)
        k = self.k_norm(self.wk(hidden_states))                      # LayerNorm(eps=1e-6)，带 bias
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        weights = self.weights_proj(hidden_states).float()
        return kpool_topk_indices(                                  # 生产路径；torch_* 只作参考
            q, k, gate_scores, weights, self.index_kpool_compress_ape, seq_ctx,
            index_head_dim=self.index_head_dim, index_topk=self.index_topk,
            index_kpool=self.index_kpool,
            always_select_tail=self.index_kpool_always_select_tail,
        )


class NoPEDSAMultiLatentAttention(MultiLatentAttention):
    """Absorbed NoPE-MLA：query/key 都是 512 维 latent，scale 仍取 256^-0.5。

    数据流::

        hidden (1,S,4096)
           ├─ q_a_proj -> q_a_layernorm -> q_resid (1,S,1536)
           │     ├─ q_b_proj -> q (1,64,S,256) --absorb(w_kc)--> query (S,64,512)
           │     └─ KPoolIndexer(hidden, q_resid) --------------> topk (S,1,2051)
           ├─ kv_a_proj_with_mqa -> kv_a_layernorm -> key (S,1,512) --SP gather--> (S_g,1,512)
           └─ SparseMLA(query, key, topk) -> (S,64,512) --absorb^-1(w_vc)--> (1,S,16384) -> o_proj
    """

    def __init__(self, *, index_topk, index_head_dim, index_n_heads, index_kpool,
                 index_kpool_always_select_tail, indexer_types, sparse_mla_backend,
                 freeze_dsa_indexer, **kwargs):
        super().__init__(**kwargs)
        self.sparse_mla_func = get_sparse_mla(sparse_mla_backend)     # 生产默认 tilelang（NoPE 分支）
        self.indexer = KPoolIndexer(...)
        if freeze_dsa_indexer:
            self.indexer.requires_grad_(False)

    def forward(self, hidden_states, position_embeddings=None, seq_ctx=None, **kwargs):
        # NoPE：position_embeddings 由 MoE._forward 统一构造，这里显式忽略
        del position_embeddings, kwargs
        assert hidden_states.size(0) == 1, "packed 训练路径 bsz == 1"

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))                    # [1,S,1536]
        q_nope = self.q_b_proj(q_resid).view(1, S, N, self.qk_nope_head_dim).transpose(1, 2)
        kv_compressed = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))   # [1,S,512]

        wkv_b = to_local(self.kv_b_proj.weight).view(N, self.qk_nope_head_dim + self.v_head_dim, Rkv)
        w_kc, w_vc = torch.split(wkv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

        query_states = torch.einsum("bhsd,hdm->bhsm", q_nope, w_kc).squeeze(0).transpose(0, 1).contiguous()
        key_states = kv_compressed.squeeze(0).unsqueeze(1).contiguous()               # [S,1,512]
        # DSA 只有一个 KV group，top-k 与 head 无关：gather 小的 compressed KV 比 Ulysses 更省
        key_states = gather_for_sequence_parallel(key_states, dim=0, sp_mesh=seq_ctx.sequence_parallel_mesh)

        with torch.no_grad():
            topk_ids = reuse_during_recompute(self.indexer, hidden_states, q_resid, seq_ctx)

        out = self.sparse_mla_func(query_states, key_states, topk_ids,
                                   self.softmax_scale, value_dim=self.kv_lora_rank)
        raw = torch.einsum("shm,hdm->shd", out.raw_output, w_vc).reshape(1, S, N * self.v_head_dim)
        return {"raw_output": raw, "projected_output": self.o_proj(raw), "softmax_lse": out.softmax_lse}


# =============================================================================
# === file: xtuner/v1/model/moe/glm53/decoder_layer.py  (F4)
# =============================================================================


class Glm53MoEDecoderLayer(MoEDecoderLayer):
    """mHC 包裹的 MoE decoder layer。

    ``use_mhc=False`` 时退回普通 pre-norm 残差，供 MTP 层复用——checkpoint 的
    ``layers.45`` 没有任何 ``hc_*`` 参数。
    """

    def __init__(self, *, mhc_cfg: MHCConfig | None, **kwargs):
        super().__init__(**kwargs)
        self.use_mhc = mhc_cfg is not None
        if self.use_mhc:
            mix = (2 + mhc_cfg.hc_mult) * mhc_cfg.hc_mult
            hc_dim = mhc_cfg.hc_mult * self.hidden_size
            # fp32 存放：20 轮 Sinkhorn 对精度敏感，且要避开 FSDP mixed precision 降精度
            for site in ("attn", "ffn"):
                self.register_parameter(f"hc_{site}_fn", nn.Parameter(torch.zeros(mix, hc_dim, dtype=torch.float32)))
                self.register_parameter(f"hc_{site}_base", nn.Parameter(torch.zeros(mix, dtype=torch.float32)))
                self.register_parameter(f"hc_{site}_scale", nn.Parameter(torch.ones(3, dtype=torch.float32)))

    def _forward(self, hidden_states, *, seq_ctx, position_embeddings, **_):
        if not self.use_mhc:                       # MTP 路径
            return super()._forward(hidden_states, seq_ctx=seq_ctx,
                                    position_embeddings=position_embeddings)

        # ---- attention site：hidden_states 是 [B,S,H,D]
        fn, scale, base = unshard_hc_params(self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        residual = hidden_states
        x, post, comb = hc_pre(hidden_states, fn, scale, base, self.hc_mult,
                               self.hc_sinkhorn_iters, self.hc_eps, self.rms_norm_eps)
        attn_out = self.self_attn(self.input_layernorm(x), seq_ctx=seq_ctx,
                                  position_embeddings=position_embeddings)["projected_output"]
        hidden_states = hc_post(attn_out, residual, post, comb)

        # ---- ffn site：MoE / DenseMLP 收到的仍是普通 [B,S,D]
        fn, scale, base = unshard_hc_params(self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        residual = hidden_states
        x, post, comb = hc_pre(hidden_states, fn, scale, base)
        ffn_out, router_results = self._moe_forward(self.post_attention_layernorm(x), seq_ctx)
        hidden_states = hc_post(ffn_out, residual, post, comb)

        return {"hidden_states": hidden_states,
                "router_logits": router_results["logits"],
                "router_weights": router_results["router_weights"],
                "router_topk_ids": router_results["topk_ids"]}


class Glm53DenseDecoderLayer(DenseDecoderLayer):
    """前 3 层：KDA + 限幅 SwiGLU DenseMLP，同样被 mHC 包裹。"""


# =============================================================================
# === file: xtuner/v1/model/moe/glm53/glm53.py  (F4 / F6)
# =============================================================================


MOE_NON_EP_COMPILE_CFG = {
    # hc_post 在 eager 下会 materialize [B,S,H,H,D]，必须在 compile 区内
    "xtuner.v1.module.decoder_layer.mhc.hc_pre": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.mhc.hc_post": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.attention.kda.KimiDeltaAttention.forward": TorchCompileOption(fullgraph=True),
    "...nope_dsa_mla.NoPEDSAMultiLatentAttention.forward": TorchCompileOption(fullgraph=False),
    "...decoder_layer.Glm53MoEDecoderLayer.forward": TorchCompileOption(fullgraph=False),
    **DEFAULT_FLOAT8_CFG,
}
MOE_EP_COMPILE_CFG = {k: v for k, v in MOE_NON_EP_COMPILE_CFG.items()
                      if "Glm53MoEDecoderLayer.forward" not in k}


class Glm53TextMoE(MoE):
    dense_decoder_layer_cls = Glm53DenseDecoderLayer
    moe_decoder_layer_cls = Glm53MoEDecoderLayer
    mtp_layer_cls = Glm53MTPLayer
    mtp_block_cls = Glm53MTPBlock

    @override
    def _decoder_stack(self, *, hidden_states, **kwargs):
        """mHC 对公共栈的唯一侵入点：入口 expand、出口 unweighted mean。

        这样 ``MoE._forward`` 前后看到的仍是 ``[B,S,D]``：``self.norm`` / ``lm_head`` /
        MTP 消费的 ``layer_hidden_states`` 全都不用改。``.contiguous()`` 不能省，
        ``hc_pre`` 的 ``flatten(2)`` 与 ``hc_post`` 的 view 都假定稠密布局。
        """
        streams = hidden_states.unsqueeze(-2).expand(-1, -1, self.hc_mult, -1).contiguous()
        streams = super()._decoder_stack(hidden_states=streams, **kwargs)
        return streams.mean(dim=-2)          # HF Glm5NextTextHyperHead：无权重均值

    @override
    def _micro_batch_decoder_stack(self, *, hidden_states_list, **kwargs):
        ...  # 同构：逐 micro-batch expand / collapse

    @override
    def _call_decoder_layer(self, *, decoder_layer, layer_idx, hidden_states, **kwargs):
        """只做 activation offload 窗口；**不需要** GLM-5.2 那套跨层 dsa_topk_ids 传递
        —— 发布 config 的 indexer_types 45 层全是 full，没有 IndexShare。

        offload 对象是每层入口的 4 流张量 [B,S,4,D]，block_idx 计数口径与 GLM-5.2 一致。
        """
        offload_tensors = []
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1 and layer_idx >= self.config.first_k_dense_replace:
            offload_tensors = as_list(hidden_states)
        block_idx = sum(self.config.first_k_dense_replace <= int(i) < layer_idx for i in self.layers)
        with self._saved_tensors_offload_ctx(block_idx, offload_tensors):
            return decoder_layer(hidden_states, position_embeddings=..., seq_ctx=...)

    def to_hf_key_list(self, key: str) -> list[str]:
        """目标是**发布版布局**（真实 320B checkpoint 与 vLLM/SGLang 读取的格式）。"""
        if key.startswith("mtp_block."):
            # mtp_block.layers.0.decoder_layer.X -> layers.45.X
            key = re.sub(r"mtp_block\.layers\.(\d+)\.", f"layers.{self.config.num_hidden_layers}.", key)
            key = key.replace(".decoder_layer.", ".")
            key = re.sub(r"layers\.(\d+)\.final_layernorm\.", r"layers.\1.shared_head.norm.", key)
        if "layers" in key or "embed_tokens" in key or key.startswith("norm."):
            key = "model.language_model." + key                      # 注意是嵌套前缀
        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts)", r"layers.\1.mlp.\2", key)
        if "fused_w1w3.weight" in key:                               # 展开成 per-expert 键
            return [key.replace("fused_w1w3.weight", f"{i}.{p}_proj.weight")
                    for i in range(self.config.n_routed_experts) for p in ("gate", "up")]
        if "fused_w2.weight" in key:
            return [key.replace("fused_w2.weight", f"{i}.down_proj.weight")
                    for i in range(self.config.n_routed_experts)]
        if "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        return [key]


class Glm53TextMoEConfig(MoEConfig):
    model_type: str = "glm5_next_text"
    vocab_size: int = 154880
    max_position_embeddings: int = 1048576
    pad_token_id: int | None = 154820
    eos_token_id: int = 154820
    hf_eos_token_id: list[int] = [154820, 154827, 154829]
    num_hidden_layers: int = 45
    first_k_dense_replace: int = 3
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0

    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=1, topk_group=1, scoring_func="sigmoid",
        norm_topk_prob=True, router_scaling_factor=2.5)
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(act_type="clamped_swiglu", clip_limit=10.0)

    attention: NoPEDSAMLAConfig = NoPEDSAMLAConfig(
        q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256,
        num_attention_heads=64, head_dim=0,
        index_topk=2048, index_head_dim=128, index_n_heads=32, index_kpool=4)
    linear_attention: KDAConfig = KDAConfig(num_heads=64, head_dim=128,
                                            conv_kernel_size=4, use_full_rank_gate=False,
                                            gate_lower_bound=-5.0, rms_norm_eps=1e-5)

    mhc_cfg: MHCConfig = MHCConfig(hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20)
    mlp_layer_types: list[Literal["dense", "sparse"]] | None = None
    hf_layer_types: list[str] | None = None            # HF 的 layer_types 原样保存
    num_nextn_predict_layers: int | None = 1
    mtp_config: MTPConfig | None = None
    # A_log / dt_bias / hc_* 必须以 fp32 存回 HF
    hf_save_cfg: HFSaveCfg = HFSaveCfg(fp32_keys_pattern=[
        r"model\.language_model\.layers\.\d+\.self_attn\.(A_log|dt_bias)",
        r"model\.language_model\.layers\.\d+\.hc_(attn|ffn)_(fn|base|scale)",
    ])

    @computed_field
    def layers_type(self) -> list[str]:
        """HF ``deepseek_sparse_attention`` -> XTuner ``full_attention``；
        ``linear_attention`` 原样。发布 schedule 为 3,7,11,...,43 是 DSA，其余 KDA。"""
        return ["full_attention" if t == "deepseek_sparse_attention" else "linear_attention"
                for t in self.hf_layer_types]

    @classmethod
    def from_hf(cls, hf_path) -> Self:
        cfg = Glm5NextConfig.from_pretrained(hf_path).text_config
        assert cfg.mla_use_nope and cfg.qk_rope_head_dim == 0
        assert cfg.mhc and cfg.hc_mult == 4
        assert set(cfg.indexer_types) == {"full"}, "不支持 IndexShare"
        return cls(hf_layer_types=list(cfg.layer_types))  # 其余字段逐项从 cfg 取

    @property
    def hf_config(self) -> Glm5NextTextConfig:
        """回写 HF 嵌套 text_config；由 compose config 组装进顶层 Glm5NextConfig。"""

    def build(self) -> Glm53TextMoE:
        return Glm53TextMoE(self)


# =============================================================================
# === file: xtuner/v1/model/moe/glm53/mtp.py  (F6)
# =============================================================================


class Glm53MTPLayer(MTPLayer):
    """包裹一个 ``use_mhc=False`` 的 DSA + MoE decoder layer。

    checkpoint 键：``layers.45.{enorm,hnorm,eh_proj,shared_head.norm}`` 加完整的
    ``self_attn.*``（含 indexer）与 ``mlp.*``；没有 ``hc_*``。
    因为不存在 IndexShare，也不需要 GLM-5.2 ``GLM52MTPBlock`` 的 dsa_topk_ids 传递。
    """


class Glm53MTPBlock(MTPBlock):
    pass


# =============================================================================
# === file: xtuner/v1/model/compose/glm53/glm53_config.py  (F2)
# === 权威契约：duanyanhui《xtuner_glm53_flash_vl_vision_design.md》§5.1
# =============================================================================



class Glm53VisionConfig(XTunerBaseModelConfig):
    """字段必须对照官方 config.json 的 vision_config 逐项固化，不抄表格。"""

    model_config = ConfigDict(extra="forbid")

    in_channels: int = 3
    depth: int = 24
    hidden_size: int = 1024
    # checkpoint 里的字段名是 num_heads（HF 两个名字都认）；两个 config 都是 extra="forbid"，
    # 所以 XTuner 侧必须能直接吃 config.json 的 vision_config（设计文档 3.7.4）。
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    # checkpoint 的 vision_config **没有** rope_parameters 字段，HF 填这个默认值；
    # 若声明成必填会挡住从 config.json 的构造。
    rope_parameters: dict = {"rope_type": "axial", "rope_theta": 10000.0}
    # head_dim 不设显式字段：checkpoint 没有，按 hidden_size // num_heads = 64 推导
    attention_bias: bool = True    # GLM vision 的 qkv / proj / mlp 都带 bias
    attention_dropout: float = 0.0
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"
    fully_shard: bool = True

    @property
    def spatial_merge_unit(self) -> int:
        """全链路唯一的 merge 单元来源；禁止在任何地方硬编码 4（视觉设计文档 §16.1）。"""
        return self.spatial_merge_size**2

    def build(self) -> "Glm53VisionModel": ...


class Glm53ProjectorConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(extra="forbid")

    vision_hidden_size: int = 1024
    # checkpoint 字段名是 out_hidden_size（= 文本 hidden_size）；两个 config 都是 extra="forbid"，
    # 用 text_hidden_size 这种自造名会挡住从 config.json 构造（设计文档 3.7.4）。
    out_hidden_size: int = 4096
    projection_intermediate_size: int = 10240
    spatial_merge_size: int = 2
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    fully_shard: bool = True

    def build(self) -> "Glm53Projector": ...


# =============================================================================
# === file: xtuner/v1/model/compose/glm53/vision_utils.py  (F2)
# =============================================================================



def flatten_video_grid_thw(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """``[t, h, w]`` -> t 行 ``[1, h, w]``。

    GLM vision tower 的输入契约是**已展开**的行（HF ``get_video_features`` 在调用 ``self.visual``
    之前就展开）；qwen3_vl 则是在 vision 模块内部消化 ``t``。两种表示 patch 总数相同，
    混用不会触发任何 shape/计数报错，只会静默改变 attention 边界（视觉设计文档 §7.3 / §16.7）。

    ``video_grid_thw[:, 0]`` 是 **tubelet 数**（``padded_num_frames // temporal_patch_size``，
    奇数帧由 ``patchify`` 用尾帧复制补齐），不是原始帧数。
    """
    t, hw = video_grid_thw[:, 0], video_grid_thw[:, 1:]
    flattened_hw = torch.repeat_interleave(hw, t, dim=0)
    ones = torch.ones(flattened_hw.shape[0], 1, dtype=video_grid_thw.dtype, device=video_grid_thw.device)
    return torch.cat([ones, flattened_hw], dim=1)


# =============================================================================
# === file: xtuner/v1/model/compose/glm53/modeling_vision.py  (F2)
# =============================================================================



class Glm53VisionModel(BaseModel):
    """patch_embed + axial RoPE + 24 block + post_layernorm。

    与 Qwen3-VL 的复用边界：复用变长非因果 attention 的 cu_seqlens 组织与 Ulysses 通信写法，
    **不继承** ``Qwen3VLVisionModel``。GLM 必须自己实现的差异：
      - block norm 是 RMSNorm（非 LayerNorm），无 bias
      - attention 在 RoPE 前对 Q/K 各做 head-dim RMSNorm；qkv/proj 带 bias
      - 位置编码是纯 2D axial RoPE（无时间轴），频率布局 ``[h, w, h, w]`` 覆盖整个 head dim，
        位置按 merge-block-major 展开，每个 temporal slice 重复同一组 ``(h, w)``
      - MLP 是 gate/up/down + 限幅 SwiGLU（带 bias）
      - 无 deepstack —— **不要返回空 list 伪装 Qwen 接口**
    """

    def __init__(self, config: Glm53VisionConfig):
        self.patch_embed = nn.Conv3d(
            config.in_channels, config.hidden_size,
            kernel_size=[config.temporal_patch_size, config.patch_size, config.patch_size],
            stride=[config.temporal_patch_size, config.patch_size, config.patch_size],
        )                                                     # model.visual.patch_embed.proj.*
        self.rotary_pos_emb = Glm53VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([Glm53VisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._hf_prefix = "model.visual."
        self._init_load_spec()

    def forward(self, hidden_states, grid_thw, sequence_parallel_mesh=None) -> torch.Tensor:
        """``grid_thw`` 必须是已展开的 ``[1, h, w]`` 行；视频由调用方先过 flatten_video_grid_thw。

        这是**约定检查**，不是数值护栏：实测两种表示产生的 cu_seqlens 与 position_ids 完全相同
        （下面两个 helper 本身就是 t-aware），所以混用不会改变 attention 边界。
        真正受影响的是下游 feature 分组（per-video vs per-frame）与 num_img_tokens 语义 ——
        回归重心在 F1，不在这里（设计文档 3.7.1，修订了视觉设计文档 §16.7 的机制描述）。
        """
        assert grid_thw[:, 0].eq(1).all(), "vision tower 约定只接受展开后的 [1, h, w] 行"

        # 位置、cu_seqlens 都从**全局** grid_thw 构造；SP 只在之后按同一 offset 切分
        # (N, 2) block-major —— 注意 HF Glm5NextVisionRotaryEmbedding.forward 的注释写的是
        # "(2, N) row 0 = h coords"，与 recomposition_frequencies 的 freq[:, 0] 取法矛盾；
        # get_vision_position_ids 实际返回 (total_tokens, 2)，按注释写会 shape 报错（3.7.2）。
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        # merge_temporal=False：每个 temporal slice 是独立 attention sequence（5.17.0 默认值，已核实）
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, merge_temporal=False)

        hidden = self.patch_embed(hidden_states)
        cos, sin = self.rotary_pos_emb(hidden, position_ids)   # recomposition -> [h, w, h, w]

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            hidden, cos, sin, num_valid_patches = self._sp_shard(hidden, cos, sin, grid_thw,
                                                                 sequence_parallel_mesh)
        for blk in self.blocks:
            hidden = blk(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                         position_embeddings=(cos, sin),
                         sequence_parallel_mesh=sequence_parallel_mesh)
        return self.post_layernorm(hidden)

    def _sp_shard(self, hidden, cos, sin, grid_thw, sp_mesh):
        """merge-aligned padding + 切分（视觉设计文档 §9.3）。

        pixel patch / RoPE position / 其他 patch 对齐张量必须用**同一个** offset 切分；
        ``cu_seqlens`` 保持全局语义，不按 rank 截断。
        """
        merge_unit = self.spatial_merge_size**2                # 禁止硬编码 4
        multiple = sp_mesh.size() * merge_unit
        num_valid_patches = hidden.shape[0]
        padded_n = ceil_div(num_valid_patches, multiple) * multiple
        # 尾部补零 + 追加合法 fake grid（H/W 都能被 spatial_merge_size 整除）
        hidden = pad_tail_zeros(hidden, padded_n)
        local_n = padded_n // sp_mesh.size()
        assert local_n % merge_unit == 0
        ...
        return hidden_local, cos_local, sin_local, num_valid_patches

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]


class Glm53VisionBlock(nn.Module):
    """``x + attn(norm1(x))`` -> ``x + mlp(norm2(x))``，norm 均为 RMSNorm。"""

    def forward(self, hidden_states, *, cu_seqlens, max_seqlen, position_embeddings, sequence_parallel_mesh):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            position_embeddings=position_embeddings, sequence_parallel_mesh=sequence_parallel_mesh)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm53VisionAttention(nn.Module):
    """非因果变长 attention；Q/K 在 RoPE 前各做 head-dim RMSNorm。"""

    def forward(self, hidden_states, *, cu_seqlens, max_seqlen, position_embeddings, sequence_parallel_mesh):
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)     # qkv 带 bias
        q, k = self.q_norm(q_heads), self.k_norm(k_heads)      # GLM 特有，Qwen 没有
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            # Ulysses：[all heads, local seq] -> [local heads, global seq]
            # 每个 head 仍看到整张图的完整 patch sequence；
            # **绝不能**让每个 rank 只对本地 patch 做 attention（§16.4）
            q, k, v = (ulysses_all_to_all(t, scatter_dim=1, gather_dim=0, mesh=sequence_parallel_mesh)
                       for t in (q, k, v))

        out = varlen_attn(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, causal=False)

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            out = ulysses_all_to_all(out, scatter_dim=0, gather_dim=1, mesh=sequence_parallel_mesh)
        return self.proj(out)                                  # proj 带 bias


# =============================================================================
# === file: xtuner/v1/model/compose/glm53/modeling_projector.py  (F2)
# =============================================================================



class Glm53Projector(BaseModel):
    """Conv2d downsample (1024 -> 4096, kernel=stride=spatial_merge_size) + PatchMerger。

    merger：``proj -> LayerNorm -> GELU -> 限幅 SwiGLU(gate/up/down)``，全部无 bias。

    SP 下由反向 all-to-all 之后**本地**执行：``local_n % merge_unit == 0`` 已由 padding 保证，
    所以不需要先 gather patch hidden states（视觉设计文档 §9.6）。
    """

    def forward(self, hidden_states):                          # [N_patch, 1024]
        m = self.spatial_merge_size
        x = hidden_states.view(-1, m, m, self.vision_hidden_size).permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)  # [N_patch/merge_unit, 4096]
        x = self.merger.proj(x)
        x = F.gelu(self.merger.post_projection_norm(x))         # LayerNorm，带 bias
        gate = self.merger.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.merger.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.merger.down_proj(F.silu(gate) * up)         # [N_patch/merge_unit, 4096]

    def to_hf_key_list(self, key: str) -> list[str]:
        return ["model.visual." + key]                          # downsample.* / merger.*


# =============================================================================
# === file: xtuner/v1/model/compose/glm53/modeling_glm53.py  (F6，不属于 F2)
# =============================================================================



class Glm53Config(BaseComposeConfig):
    """Compose config（F6）。F2 阶段只有 vision / projector 两个 config，不含本类。"""

    vision_config: Glm53VisionConfig = Glm53VisionConfig()
    projector_config: Glm53ProjectorConfig = Glm53ProjectorConfig()
    text_config: Glm53TextMoEConfig = Glm53TextMoEConfig()

    image_token_id: int = 154854
    video_token_id: int = 154855          # 仅 chat template 展开**前**的 marker，展开后不出现
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    freeze_vision: bool = True            # 与 AutoModel recipe 一致：冻结视觉塔，训语言塔
    only_llm_forward: bool = False

    @classmethod
    def from_hf(cls, hf_path) -> Self:
        """拆解 Glm5NextConfig 的嵌套 text_config / vision_config。"""

    def build(self) -> "Glm53ForConditionalGeneration":
        return Glm53ForConditionalGeneration(self)


class Glm53ForConditionalGeneration(BaseComposeModel):
    """Compose 层。F2 只交出 vision_tower / projector / flatten_video_grid_thw；
    三段挂载、feature splice、dummy 图、完整 HF index 合并都在这里（视觉设计文档 §16.5）。"""

    config: Glm53Config

    def _visual_features(self, seq_ctx: SequenceContext):
        """image 与 video **分两次** vision forward（视觉设计文档 §9.7）。

        好处：分别与 HF get_image_features / get_video_features 逐项对齐；
        padding 与有效 token 数分开追踪；视频需要先展开 tubelet。
        """
        image_feats = video_feats = None
        if seq_ctx.pixel_values is not None:
            image_feats = self.multi_modal_projector(self.vision_tower(
                seq_ctx.pixel_values, seq_ctx.image_grid_thw, seq_ctx.sequence_parallel_mesh))
        if seq_ctx.pixel_values_videos is not None:
            flat_grid = flatten_video_grid_thw(seq_ctx.video_grid_thw)   # [t,h,w] -> t 行 [1,h,w]
            video_feats = self.multi_modal_projector(self.vision_tower(
                seq_ctx.pixel_values_videos, flat_grid, seq_ctx.sequence_parallel_mesh))
        return image_feats, video_feats

    def _prepare_llm_inputs(self, seq_ctx: SequenceContext):
        inputs_embeds = self.language_model.embed_tokens(seq_ctx.input_ids)
        image_feats, video_feats = self._visual_features(seq_ctx)

        if image_feats is None and video_feats is None:
            if not self.config.only_llm_forward:
                # 纯文本 batch 也要让视觉参数进 autograd 图，否则 FSDP 下各 rank 不对称
                inputs_embeds = inputs_embeds + self._dummy_visual_features(inputs_embeds).sum() * 0.0
            return inputs_embeds

        # SP 下必须用**全局** mm_token_type_ids（或 raw_input_ids）算完 mask 再切，
        # 不能按本 rank 的 begin/end token 局部推导 modality（设计文档 3.6.4 / §8.3 / §16.3）。
        #
        # modality 只能从 mm_token_type_ids 判定：
        #   展开后的 input_ids 里 **不存在** video_token_id（154855），
        #   视频帧用的也是 <|image|>(154854)，靠 <|begin_of_video|>..<|end_of_video|> span 区分。
        #   任何 `input_ids == video_token_id` 的写法都会失效（§9.8 / §16.3）。
        #   Automodel 的 `mask = input_ids == image_token_id` 在它的 image-only 范围内正确，
        #   但有 video 时会把视频帧一起命中，不能照抄（设计文档 3.7.3）。
        mm_types = seq_ctx.mm_token_type_ids
        for feats, type_id in ((image_feats, 1), (video_feats, 2)):
            if feats is None:
                continue
            mask = mm_types == type_id
            # 数量不符立即抛错；不要 broad except 后继续训练（§16.2）
            if mask.sum() != feats.shape[0]:
                raise ValueError(f"visual token/feature mismatch: tokens={mask.sum()}, feats={feats.shape[0]}")
            inputs_embeds[mask] = inputs_embeds[mask] * 0.0 + feats
        return inputs_embeds

    def forward(self, seq_ctx, loss_ctx=None):
        lang_ctx = seq_ctx.copy(input_ids=None, inputs_embeds=self._prepare_llm_inputs(seq_ctx))
        return self.language_model(lang_ctx, loss_ctx)


# =============================================================================
# === file: xtuner/v1/data_proto/messages/glm53_chat.py  (F1)
# =============================================================================



# chat template 输出的是**未展开**的单个占位 token；展开由 Glm5NextProcessor 完成。
_MEDIA_PLACEHOLDER = {
    "image": "<|begin_of_image|><|image|><|end_of_image|>",
    "video": "<|begin_of_video|><|video|><|end_of_video|>",
    "audio": "<|begin_of_audio|><|end_of_audio|>",
}


def render_glm53_chat(messages, tools=None, add_generation_prompt=False,
                      enable_thinking=True, reasoning_effort="max", clear_thinking=False):
    """与 ``render_glm52_chat`` 同构，只有两处差异：

    1. ``reasoning_effort`` 取值域 ``{low, high, max}``（GLM-5.2 只有 ``{high, max}``）；
    2. 媒体不再渲染成 ``<reminder>`` 文本，而是 ``_MEDIA_PLACEHOLDER`` 里的占位 token。

    loss mask 口径沿用 GLM-5.2：``<|user|>`` / ``<|observation|>`` 作为上一轮 assistant 的
    停止目标计 loss；``<think>`` 开标签是模板脚手架不计 loss，思维链正文与 ``</think>`` 计 loss；
    最后一轮没有角色边界时补 ``<|endoftext|>`` 并计 loss。
    """
    ...
    return text, loss_mask


class Glm53ChatMessages(BaseModel):
    """在 ``OpenaiTokenizeFunction`` 里注册为 ``chat_template="glm5.3"``。"""

    def tokenize(self, tokenizer, chat_template): ...


# =============================================================================
# === file: xtuner/v1/datasets/data_item.py  (F1, 增量)
# =============================================================================



class Glm53VLDataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor          # CPU，由模型视觉入口搬到目标 device
    image_grid_thw: torch.Tensor        # [num_images, 3]，不在 DataItem 里展开
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor        # [num_videos, 3]，第 0 列是 tubelet 数
    mm_token_type_ids: torch.Tensor     # 0=text/pad, 1=image, 2=video


# =============================================================================
# === file: xtuner/v1/datasets/mllm_tokenize_fn/glm53_vl_tokenize_fn.py  (F1)
# === 权威契约：视觉设计文档 §8.2；阶段划分见阶段文档 §3
# =============================================================================



class Glm53VLTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    processor_path: str
    min_pixels: int | None = None
    max_pixels: int | None = None
    fps: float | None = None
    max_frames: int | None = None
    # 基类默认 0.0；这里改成 1.0 是有意的行为变更（视觉 sample 的 packing 权重与文本对齐），
    # review 时需显式确认。
    visual_pack_weight: float = 1.0
    llm_pack_weight: float = 1.0

    def build(self) -> "Glm53VLTokenizeFunction": ...


class Glm53VLTokenizeFunction(BaseMLLMTokenizeFunction):
    """cache / runtime 双路径。

    cache  : 只用 metadata + 官方**纯几何** API 预测尺寸与 token 数，不解码媒体；
    runtime: 调官方 processor 产生真实 tensor。

    已核实（pinned transformers 5.17.0）：
      图片 Glm5NextImageProcessor.get_number_of_image_patches(h, w, images_kwargs)  -> 存在
      视频 Glm5NextVideoProcessor.get_number_of_video_patches(...)                  -> **不存在**
    后者在整个 transformers 包里没有任何 `def`，却被 10+ 个 processor（含 glm5_next 的
    _get_num_multimodal_tokens）调用，是上游 bug；走到视频分支会 AttributeError。
    所以视频侧必须用视觉设计文档 §7.1 的退路：组合 Glm5NextVideoProcessor.sample_frames
    + 模块级 smart_resize（真实预处理路径内部用的同两个公开调用），并用单测断言
    复现出的 video_grid_thw 与真实 processor 输出逐元素一致。
    resize 由 min_image_tokens=16 / max_image_tokens=8000 动态约束，与 Qwen smart_resize
    不是同一个函数，禁止移植（§16.1）。

    cache key 必须含：processor revision、resize 上下限、fps/max_frames、merge size、
    flash block size、packing weights —— 任一变更使旧 cache 失效。
    """

    def _expand_placeholders(self, text: str, grids, modality: str) -> tuple[str, list[int]]:
        """placeholder 展开 + mm_token_type_ids 生成，cache 与 runtime **共用这一份代码**。

        GLM 的两段式协议（本次在 5.17.0 上核实）：

            image: <|image|> * (prod(image_grid_thw) / merge_unit)

            video: 每个 tubelet 一段 —— replace_frame_token_id():
                   <|begin_of_image|> + <|image|> * (H*W / merge_unit)
                   + <|end_of_image|> + f"{ts:.1f} seconds"
                   重复 grid_t 次；时间戳取 metadata.timestamps[::2]（步长 = temporal_patch_size）

        三条后果：
          1. 展开后 **不存在 video_token_id(154855)**，视频帧也用 <|image|>(154854)；
             区分 image/video 只能靠 <|begin_of_video|>..<|end_of_video|> span，
             即 mm_token_type_ids。任何 `ids == video_token_id` 的写法都会失效。
          2. tubelet 之间夹着真实文本 token "{ts:.1f} seconds"，cache 预测 num_tokens 必须算进去；
             "placeholder 数 == sum(prod(grid))/merge_unit" 对视频只在 placeholder 维度成立。
          3. mm_token_type_ids 由本函数在插入 placeholder 的位置确定性生成（生产），
             官方 processor 的同名输出只作 golden 做交叉校验（校验），二者职责分离。
        """

    def _cache_predict(self, metadata) -> CacheItem:
        merge_unit = self.spatial_merge_size**2
        # image: grid = [1, gh, gw]；video: grid = [grid_t, gh, gw]，grid_t 是 tubelet 数
        #   num_img_tokens 语义 = 每条 ViT attention sequence 的 raw patch 长度：
        #     图片 -> 每张一个 gh*gw
        #     视频 -> [gh*gw] * grid_t      （**不得照抄 qwen3_vl 的 per-video 单条 t*h*w**）
        llm_num_patch = round(num_tokens / flash_attn_block_size) ** 2
        img_num_patch = sum((n / flash_attn_block_size) ** 2 for n in num_img_tokens)
        proxy_attn_flops = self.llm_pack_weight * llm_num_patch + self.visual_pack_weight * img_num_patch
        return {"num_tokens": num_tokens, "num_img_tokens": num_img_tokens,
                "proxy_attn_flops": proxy_attn_flops}

    def _runtime(self, data_item, media_root="") -> Glm53VLDataItem:
        outputs = self.processor(text=..., images=..., videos=...)
        # fail-fast 交叉校验（§8.2）：缺字段或值错位立即抛错，错误信息带 processor revision
        # 与 transformers 版本。双向值级契约：
        #   shape == input_ids；processor 值为 1/2 的位置 ⇔ XTuner 同位置是 visual placeholder；
        #   type==1 计数 == sum(prod(image_grid_thw)) / merge_unit；
        #   type==2 计数 == 展开后 video patch 总数 / merge_unit；其余为 0。
        self._assert_mm_token_type_ids_contract(outputs, own_mm_token_type_ids, own_input_ids)
        # runtime 重算的 num_tokens / num_img_tokens 必须与 cache 预测完全一致
        ...

    def __call__(self, item, media_root="", **kwargs):
        if has_image(item) and has_video(item):
            # 不能因为复用基类的 image 优先分支而静默丢 video
            raise NotImplementedError("GLM-5.3-Flash 初版不支持单样本内 image+video 混合")
        # 截断若切进一个 visual span，整样本 drop 并记录原因，不产生半个 span
        ...


# =============================================================================
# === file: xtuner/v1/data_proto/sequence_context.py  (F1, 增量)
# =============================================================================


# SequenceContext 新增三个可选字段，并同步 __init__ / split / cat / to / copy：
#   pixel_values_videos: torch.FloatTensor | None
#   video_grid_thw:      torch.Tensor | None
#   mm_token_type_ids:   torch.Tensor | None
# mm_token_type_ids 必须与 input_ids 使用**同一个** pad/truncate/split 结果；
# 不能按本 rank 的 begin/end token 局部推导 modality（§8.3 / §16.3）。


# =============================================================================
# === file: xtuner/v1/datasets/collator.py  (F1, 增量)
# =============================================================================



def glm53_vl_sft_collator(instances, **kwargs):
    """1) 公共 text collator -> 2) 视觉 tensor/grid 按保留顺序拼接 ->
    3) mm_token_type_ids 与 input_ids 同规则 padding（text/pad=0）->
    4) 拼 num_img_tokens 并重算 batch proxy -> 5) 挂到 SequenceContext。

    拼接后必须重新断言：每样本 placeholder 数、grid 数、mm_token_type_ids 长度、num_img_tokens 总量。
    """


def build_text_ctx_labels(instances, **kwargs):
    """**共享改动**：改为额外返回保留的 instance 索引。

    现状是 drop-from-end 截断且不剪媒体，会出现"文本被丢弃、图片仍留在 batch"的错配。
    该改动波及 qwen3_vl / intern_s1 两条现有路径，必须为它们补回归测试，不得只修 GLM 路径（§8.4）。
    """
    return text_ctx, labels, kept_indices


# =============================================================================
# === file: xtuner/tools/model_converters/make_glm53_25b_hf.py  (F0)
# =============================================================================

# 减层 profile：3 dense(KDA) + layer3(DSA,sparse) + layer4(KDA,sparse) + 原 layer45(MTP)
# 约 24.9B —— embed/lm_head 1.27B + 视觉 0.57B + dense 0.87B + 3 个 sparse 层约 22.2B
PROFILE_25B = CropProfile(num_main_layers=5, include_mtp=True)


def dequantize_fp8_block(weight_fp8, scale_inv, block=128):
    """native FP8 (e4m3) + 128x128 block scale -> BF16。

    ``scale_inv`` 形状是 ``ceil(out/128) x ceil(in/128)``；按 block 展开相乘后截断回原形状。
    """
    scale = scale_inv.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (weight_fp8.float() * scale[: weight_fp8.shape[0], : weight_fp8.shape[1]]).bfloat16()


def rewrite_config(config: dict, profile: CropProfile) -> dict:
    """裁剪嵌套 text_config 的 4 个 schedule 列表并重写 KDA/DSA 层号。"""
    tc = config["text_config"]
    n = profile.num_main_layers
    tc["num_hidden_layers"] = n
    tc["first_k_dense_replace"] = min(tc["first_k_dense_replace"], n)
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        tc[key] = tc[key][:n]
    tc["linear_attn_config"]["kda_layers"] = [i for i, t in enumerate(tc["layer_types"]) if t == "linear_attention"]
    tc["linear_attn_config"]["full_attn_layers"] = [i for i, t in enumerate(tc["layer_types"]) if t != "linear_attention"]
    tc["num_nextn_predict_layers"] = 1 if profile.include_mtp else 0
    config.pop("quantization_config", None)          # 输出是 BF16
    return config


def target_name(name: str, *, num_main_layers, original_main_layers, include_mtp) -> str | None:
    """``model.visual.*`` / ``lm_head`` / ``embed_tokens`` 原样保留；
    主栈只留前 ``num_main_layers`` 层；原 ``layers.45`` 重映射到 ``layers.{num_main_layers}``。"""


def main():
    # 1. 按白名单挑 tensor  2. FP8 反量化  3. 重命名  4. 重写 config  5. 分片写出
    # 最后打印实际参数量，偏离 25B ±10% 时提示调整保留层数
    ...


# =============================================================================
# === file: examples/v1/config/sft_glm53.py  (F6)
# =============================================================================

GLM5_3_FLASH_PATH = os.environ["GLM5_3_FLASH_PATH"]      # ~/model/GLM-5.3-Flash-25B

model_cfg = get_model_config_from_hf(GLM5_3_FLASH_PATH)  # 新增 glm5_next 分支
model_cfg.text_config.ep_size = int(os.environ.get("EP_SIZE", "4"))   # 288 % ep_size == 0
model_cfg.text_config.dispatcher = os.environ.get("DISPATCHER", "all2all")
model_cfg.text_config.compile_cfg = _get_bool_env("MODEL_COMPILE", True)
model_cfg.freeze_vision = _get_bool_env("FREEZE_VISION", True)        # 与 AutoModel recipe 对齐
# 生产默认：indexer 复用现成 TileLang/DeepGEMM kernel；
# SparseMLA 走 flash_mla_cudnn（FlashMLA fwd 原生支持 512 + cuDNN bwd 无维度硬编码）。
# torch 只作参考实现；tilelang 需 tail_dim=0 改造，是无 FlashMLA/cuDNN 硬件的备选。
model_cfg.text_config.attention.sparse_mla_backend = os.environ.get("SPARSE_MLA_BACKEND", "flash_mla_cudnn")
model_cfg.text_config.attention.indexer_backend = os.environ.get("INDEXER_BACKEND")  # deep_gemm_fp8 可选
model_cfg.text_config.attention.indexer_topk_query_chunk_size = resolve_indexer_topk_query_chunk_size(
    os.environ.get("INDEXER_TOPK_QUERY_CHUNK_SIZE"), model_cfg.text_config.attention.sparse_mla_backend)

dataset_config = [{
    "dataset": DatasetConfig(name="alpaca", anno_path=os.environ["ALPACA_PATH"]),
    "tokenize_fn": OpenaiTokenizeFunctionConfig(
        chat_template="glm5.3",
        max_length=int(os.environ.get("SAMPLE_MAX_LENGTH", "4096"))),
}]

trainer = TrainerConfig(
    model_cfg=model_cfg, load_from=GLM5_3_FLASH_PATH, tokenizer_path=GLM5_3_FLASH_PATH,
    dataloader_cfg=DataloaderConfig(dataset_config_list=dataset_config, pack_level="soft",
                                    pack_max_length=int(os.environ.get("PACK_MAX_LENGTH", "16384"))),
    fsdp_cfg=FSDPConfig(ep_size=..., torch_compile=_get_bool_env("TORCH_COMPILE", True)),
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    total_step=int(os.environ.get("TOTAL_STEP", "20")),
)


# =============================================================================
# === file: tests/model/test_glm53.py  (验收 1)
# =============================================================================


class TestGlm53(DistributedTestBase):
    @pytest.mark.gpu
    @parametrize("ep_size", [1, 4])
    def test_fsdp_accuracy(self, ep_size):
        """25B 减层模型上 XTuner 与 transformers 的前向 / loss 一致。

        参照模型直接 from_pretrained 即可：transformers 5.17 内建了 glm5_next 的
        conversion mapping（conversion_mapping.py 的 "glm5_next" 条目），发布版布局的
        hc_attn_* / q,k,v_conv1d / 扁平 forget gate / per-expert 权重都会自动转换。
        """
        hf_model = Glm5NextForConditionalGeneration.from_pretrained(
            GLM53_25B_PATH, dtype=torch.bfloat16, device_map="cuda")
        expected = [hf_model(input_ids=ids, labels=ids.clone()).loss for ids in inputs]
        del hf_model

        cfg = get_model_config_from_hf(GLM53_25B_PATH)
        cfg.text_config.ep_size = ep_size
        cfg.text_config.compile_cfg = False
        model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.fully_shard(FSDPConfig(ep_size=ep_size))
        model.from_hf(GLM53_25B_PATH)

        for ids, ref in zip(inputs, expected):
            got = model(SequenceContext.from_input_ids(...), loss_ctx).loss
            assert_close(got, ref, rtol=1e-2, atol=1e-2)
