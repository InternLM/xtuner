# 设计文档：DeepSeek-V4-Flash 支持

> Phase 0 — 仅设计，不落代码。本文档枚举接入 DeepSeek-V4-Flash 所需的全部原语，
> 给出每个原语的模块边界、配置字段差异、forward 契约和单元测试计划，并把它们
> 排成一条**单一职责、单 PR 单原语**的落地序列。最后一步才是把这些原语在
> `xtuner/v1/model/moe/deepseek_v4.py` 里粘合起来。
>
> 参考来源：
> - HF 配置 `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json`
> - HF 推理参考 `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py`

---

## 1. 背景

DeepSeek-V4-Flash 是 DeepSeek 在 V3.2（DSA）/ V3-NoAux 之后的下一代 MoE。它把多
个独立研究线索（Sparse Attention、Hyper-Connections、Hash Routing、grouped
O-LoRA、FP4 expert）**同时**塞进了一个模型族。配置层面与 V3 重叠的部分有限：

```jsonc
// 节选自 config.json
"model_type": "deepseek_v4",
"num_hidden_layers": 43,
"hidden_size": 4096, "head_dim": 512, "num_attention_heads": 64, "num_key_value_heads": 1,
"q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8, "qk_rope_head_dim": 64,
"index_head_dim": 128, "index_n_heads": 64, "index_topk": 512,
"n_routed_experts": 256, "n_shared_experts": 1, "num_experts_per_tok": 6,
"scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
"num_hash_layers": 3, "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20,
"sliding_window": 128, "swiglu_limit": 10.0,
"compress_rope_theta": 160000, "rope_theta": 10000,
"compress_ratios": [0,0,4,128,4,128,...,4,128,4,0],   // 长度 = num_hidden_layers + 1
"rope_scaling": {"type":"yarn","factor":16,"original_max_position_embeddings":65536,...},
"quantization_config": {"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]},
"expert_dtype": "fp4",
"num_nextn_predict_layers": 1
```

对照 XTuner 现有能力，缺口如下（行号对应 `dsv4` 分支当前 HEAD）：

| V4-Flash 原语 | XTuner 现状 | 关键文件:行 |
|---|---|---|
| **DSA / Indexer / KV Compressor** | **缺** | `xtuner/v1/module/attention/` 下无 indexer/compressor |
| **Q-LoRA + GQA + grouped O-LoRA + sliding window** 的注意力 | **不兼容** — `MLAConfig` 把 `kv_lora_rank/qk_nope_head_dim/v_head_dim` 设为必填，V4-Flash 没有 KV 低秩，K/V 直接从 hidden 投到 `head_dim` | `xtuner/v1/module/attention/mla.py:56-60`、`mla.py:243-262`（`o_proj` 是单 Linear，无 `o_groups`/`o_lora_rank`） |
| **`sqrtsoftplus` scoring** | **缺** — `Literal["sigmoid","softmax"]`，`softmax` 分支甚至直接 `raise NotImplementedError` | `xtuner/v1/module/router/noaux_router.py:16`，`:82-83` |
| **Hash routing**（前 `num_hash_layers` 层用 `tid2eid` 查表，与 `input_ids` 绑定） | **缺** — 现有 router 均按 score top-k | `xtuner/v1/module/router/` |
| **Dual RoPE**（per-layer 在 `rope_theta=10000` 的滑窗与 `compress_rope_theta=160000` 的压缩之间切换） | **缺** — `RotaryEmbedding` 全模型共享一份 `inv_freq` | `xtuner/v1/module/rope/rope.py:293-326` |
| **Hyper-Connections (HC)** 装饰 attention/FFN 块（保留 `hc_mult` 份 hidden state 副本，sinkhorn 归一化的混合） | **缺** — `MoEDecoderLayer/DenseDecoderLayer` 是普通残差块 | `xtuner/v1/module/decoder_layer/` |
| **Attention sink** (`layers.N.attn.attn_sink`，每层一个 learnable sink 向量，与滑窗结合) | **缺** — flash_attn 当前调用未注入 sink | `xtuner/v1/module/attention/mla.py:323-336`、`mha.py` |
| **FP4 expert + FP8 block-scaled 加载** | **部分** — `Float8Config` 有 e4m3，存在 `per_block_quant_torch(block_size=128)`；FP4 解量化与 `expert_dtype` 字段缺失。**注**：本地 BF16 reference 已存在（见 §6），FP4 加载对 parity 测试**非阻塞** | `xtuner/v1/float8/config.py:20`，`xtuner/v1/model/base.py:932`，`:1672-1686`（`_load_fp8`） |
| **MTP** | **有** — `MTPConfig.num_layers` 直接对应 `num_nextn_predict_layers` | `xtuner/v1/module/mtp/`、commit `7e30e7e0` |
| **NoAux 框架（grouped score, e_score_correction_bias）、shared expert、`first_k_dense_replace`、yarn rope** | **有** | `xtuner/v1/model/moe/moe.py:140`、`deepseek_v3.py:54-101`、`rope.py:63-241` |

---

## 2. 不在本设计范围内（Non-goals）

- **推理 / generation 路径**。HF `inference/model.py` 中的 `kv_cache`、`block_table`、
  `decoding()` 优化与训练正交，本设计只覆盖 `forward_training` 一支。
- **从零训练 hash routing**。`tid2eid` 是预训练产物（V4-Flash 已发布），本设计
  只保证「加载并 forward」，不实现 hash table 的训练时更新。
- **FP4/FP8 native 训练**。本设计只保证「以 FP8/FP4 形式加载预训练权重 → 解量化
  到 bf16 → 训练」；FP4 反向传播超出范围。
- **CUDA 内核**。`hc_split_sinkhorn` 在 HF 参考里是外部 kernel；本设计提供
  纯 PyTorch 参考实现，CUDA 化作为可选后续 PR。

---

## 3. 设计原则

1. **不在基类塞 `if model_type == "deepseek_v4"`**。每个原语要么是新模块，
   要么是对现有模块在已有扩展点（`scoring_func` 的 `Literal`、`layer_type`、
   `rope_type`）上做枚举扩张。
2. **保留 V3 工作流不变**。`MLAConfig` 不改字段（它正确描述 V3 的 KV 低秩），
   V4 的注意力是一个 sibling config (`DSAConfig`) — 因为 V4-Flash 的 K/V
   投影根本没有 latent compression（`wkv: Linear(dim, head_dim)`），强行复用
   会让 `MLAConfig` 的 invariant 出现「`kv_lora_rank` 可以为 None」这种例外。
3. **每个 PR 落 1 个原语 + 测试**。符合用户的 `feedback_refactor_small_steps`：
   接口签名变更要单独 PR。
4. **测试分两档**：原语单测（小 tensor、纯数值验证）+ 整模型 decoder-layer
   parity（用 BF16 reference）。整模型 forward 不做（>600B 参数，CI 跑不动）。

---

## 4. 模块边界

下面 7 个模块按依赖顺序排列。每个模块包含：路径、新增/修改的配置字段、forward
契约、与现有代码的接缝、单元测试要点。

### 4.1 `sqrtsoftplus` scoring（最小 PR）

**路径**：`xtuner/v1/module/router/noaux_router.py`（修改）。

**变更**：
- `NoAuxRouterConfig.scoring_func` 的 `Literal` 增加 `"sqrtsoftplus"`：
  ```python
  scoring_func: Annotated[Literal["sigmoid", "softmax", "sqrtsoftplus"], Parameter(group="router")]
  ```
- `NoAuxRouter.forward` 的 `scoring_func` 分支用 `match` 重写（同时把现有
  `softmax` 的 `NotImplementedError` 也实掉，或保留 `softmax` 的 TODO 但单
  独 PR 处理）：
  ```python
  match self.scoring_func:
      case "sigmoid":      scores = logits.sigmoid()
      case "softmax":      scores = logits.softmax(dim=-1)
      case "sqrtsoftplus": scores = F.softplus(logits).sqrt()  # √(log(1 + e^x))
  ```
- `NoAuxGroupedRouter.forward` 同步。

**为什么是扩展不是新类**：V4 的 router 拓扑与 V3 完全一致（grouped top-k +
`e_score_correction_bias`），只换了 score 函数。新增一个 `SqrtSoftplusRouter`
会重复 200 行的 grouping/correction 代码——典型的「同一 router，不同
score」，符合扩展场景。

**单测**：`tests/module/router/test_noaux_router.py::test_sqrtsoftplus_scoring`。
对一组随机 logits，验证：
1. `scores == sqrt(softplus(logits))`（element-wise，绝对误差 < 1e-6）。
2. 与 HF 参考 `Gate.forward` 在 `score_func="sqrtsoftplus"` 分支输出一致
   （只比较 score，不比较 top-k 索引——top-k 与 grouping 是 router 本身职责）。

**预估**：~50 行代码 + 30 行测试。

---

### 4.2 Dual RoPE（`compress_rope_theta` + `compress_ratios`）

**路径**：
- `xtuner/v1/module/rope/rope.py`（扩展 `RopeParametersConfig`、新增
  `DualRotaryEmbedding`）。
- `xtuner/v1/model/base.py`（`build_rotary_embedding` 调度）。

**HF 行为**：
- `compress_ratios[layer_idx] == 0` → 该层是 **pure sliding window**，用
  `rope_theta=10000` 的 freqs。
- `compress_ratios[layer_idx] in {4, 128}` → 该层使用 DSA（indexer 选 top-k
  压缩 token），用 `compress_rope_theta=160000` 的 freqs。
- `yarn` scaling（`factor=16`、`original_max_position_embeddings=65536`）作用
  在「实际用的那条 freqs」上。

**变更**：
- `RopeParametersConfig` 新增（仅在 V4 路径生效）：
  ```python
  compress_rope_theta: float | None = None  # 当存在 compress_ratios 时必填
  compress_ratios: list[int] | None = None  # 长度 = num_hidden_layers (+ 可选 MTP)
  ```
- 新增 `DualRotaryEmbedding(RotaryEmbedding)`：
  - 构造时分别用 `rope_theta` 与 `compress_rope_theta` 调用 `rope_init_fn`，
    注册两组 `inv_freq_dense` / `inv_freq_compressed`。
  - `forward(x, position_ids, *, use_compressed: bool)` 返回两组里被选中的
    那一组 `(cos, sin)`。
- `get_rope_embedding` 在 `compress_rope_theta is not None` 时返回
  `DualRotaryEmbedding`。
- 调用侧（attention 层）持有 `self.use_compressed = compress_ratios[layer_idx] > 0`
  并在 forward 里把它传给 rope。

**关键不变量**：rope module 仍然是模型级单例（只构造一份），per-layer 的
选择**由 attention 模块自己决定**，rope 不感知 `layer_idx`。这避免在 rope
里塞 layer-aware 的分支逻辑。

**单测**：`tests/module/rope/test_dual_rope.py`：
1. 单独构造 `DualRotaryEmbedding`，验证 `use_compressed=False` 时与
   `RotaryEmbedding(rope_theta=10000)` 输出 bit-identical。
2. 同上验证 `use_compressed=True` 与 `RotaryEmbedding(rope_theta=160000)`。
3. yarn scaling 在两条 freqs 上都正确生效。

**预估**：~150 行代码 + 80 行测试。

---

### 4.3 Grouped O-LoRA + V4-Flash 注意力配置（`DSAConfig`）

**路径**：`xtuner/v1/module/attention/dsa.py`（新增）。

**为什么不复用 `MLAConfig`**：

| 字段 | V3 MLA | V4-Flash DSA |
|---|---|---|
| `kv_lora_rank` | 512 | — (无 latent KV) |
| `qk_nope_head_dim` | 128 | — |
| `v_head_dim` | 128 | — |
| Q 投影 | `q_a_proj → q_a_layernorm → q_b_proj` | `wq_a → q_norm → wq_b` (一致) |
| K/V 投影 | `kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj`（先低秩，再展） | `wkv: Linear(dim, head_dim)`（直接到 `head_dim=512`，单 KV head） |
| O 投影 | 单 `Linear(n_heads * v_head_dim, dim)` | grouped LoRA: `wo_a: Linear(n_heads * head_dim // o_groups, o_groups * o_lora_rank)` + `wo_b: Linear(o_groups * o_lora_rank, dim)` |
| Indexer / 稀疏选 token | 无 | 有（`compress_ratios > 0` 的层） |
| Sliding window | 共用 `sliding_window` 字段 | 共用，但**所有 V4-Flash 层都启用**滑窗，叠加在 DSA 之上 |

把 `kv_lora_rank` 改成 `Optional` 会让 V3 路径每个地方都得判空，是典型的
spaghetti。新增 `DSAConfig` + `DeepSeekSparseAttention` 反而保持每个 config
描述「一种稳定的概念」（CLAUDE.md 设计原则 1）。

**`DSAConfig` 字段**（仅列与 `MLAConfig` 不同的）：
```python
class DSAConfig(BaseModel):
    num_attention_heads: int          # 64
    num_key_value_heads: int          # 1  ← GQA, 单 KV head
    head_dim: int                     # 512
    q_lora_rank: int                  # 1024
    o_lora_rank: int                  # 1024
    o_groups: int                     # 8
    qk_rope_head_dim: int             # 64 (rope 维度，从 head_dim 里切)
    sliding_window: int               # 128
    use_attn_sink: bool = True        # V4-Flash 每层带 learnable sink 向量
    # Indexer 子配置：
    index_head_dim: int               # 128
    index_n_heads: int                # 64
    index_topk: int                   # 512
    # compress_ratios 不入 DSAConfig，留在 RopeParametersConfig（rope 层是它的
    # 主要消费者）；DSA 通过构造参数 attn_mode 接收 per-layer 模式。
```

**模块拓扑**（从本地 BF16 checkpoint 的 safetensors index 反推确认）：每层 DSA
持有 **两个** Compressor 实例和一个 Indexer：

```
layers.N.attn/
├── wq_a, q_norm, wq_b              # Q-LoRA
├── wkv, kv_norm                    # K/V (no latent compression)
├── wo_a, wo_b                      # grouped O-LoRA (o_groups=8)
├── attn_sink                       # learnable per-layer sink, shape [num_attention_heads]
├── compressor/                     # 仅 compress_ratios[N] > 0 的层存在
│   ├── wkv, wgate, norm, ape       #   主 KV 压缩，被 sparse attention 消费
└── indexer/                        # 仅 compress_ratios[N] > 0 的层存在
    ├── wq_b, weights_proj
    └── compressor/                 #   indexer 内部的独立压缩，带 Hadamard rotate
        ├── wkv, wgate, norm, ape
```

**`DeepSeekSparseAttention.__init__` 参数**：除 `DSAConfig` 字段外，新增
```python
attn_mode: Literal["sliding", "compressed_4", "compressed_128"]
rope_compress_ratio: int  # 0 / 4 / 128，来自 RopeParametersConfig.compress_ratios[layer_idx]
```
由 `MoE.build_layers` 在装配每层时计算后传入。

**forward 契约**（训练路径）：
```python
def forward_training(
    self,
    hidden_states: Tensor,                                  # [B, S, hidden_size]
    position_embeddings_dense: tuple[Tensor, Tensor],       # 两套 cos/sin
    position_embeddings_compressed: tuple[Tensor, Tensor],
    seq_ctx: SequenceContext,
) -> Tensor:
    # 1. Q: wq_a → q_norm → wq_b → split nope/rope → 选 cos/sin（按 attn_mode） → 拼回
    # 2. K/V: wkv(x) → kv_norm → split nope/rope → 选 cos/sin → 拼回
    # 3a. 若 attn_mode == "sliding"：
    #     flash_attn_varlen_func(window_size=(sliding_window, 0), causal=True)
    # 3b. 若 attn_mode in ("compressed_4", "compressed_128")：
    #     topk_idx = self.indexer(hidden_states, q_lowrank_residual, ...)  # [B, S, index_topk]
    #     gathered_kv = compressor.gather(kv_compressed, topk_idx)
    #     sparse_attn(q, gathered_kv, softmax_scale, causal_mask_for_sparse)
    # 4. O: reshape 为 [B, S, o_groups, head_dim * num_heads / o_groups]
    #       → einsum 与 wo_a → 与 wo_b → 还原到 [B, S, hidden_size]
```

调用侧把两套 `position_embeddings` 同时传入，attention 内部按 `attn_mode`
选择——这样 rope 模块**不感知 layer**，attention 也**不感知 rope 计算细节**。

**单测**：
1. `tests/module/attention/test_dsa_olora.py`：构造单层 DSA + 单层 HF
   `Attention`（同一份权重），随机 `[B=1, S=128, D=4096]` 输入，比较输出
   `max_abs_diff < 1e-3`（bf16）。`attn_mode="sliding"` 与 `attn_mode="compressed_4"`
   分别测一次。
2. `tests/module/attention/test_olora_shapes.py`：纯静态检查 `wo_a`/`wo_b` 的
   shape 与 `o_groups` × `o_lora_rank` 关系。

**依赖**：4.1（不依赖）、4.2（依赖 dual rope 的输出契约）、4.4（Indexer 单独
PR）、4.5（Compressor 单独 PR）。
**预估**：~400 行代码 + 200 行测试，但**不含** Indexer/Compressor 本体。

---

### 4.4 KV Compressor

**路径**：`xtuner/v1/module/attention/kv_compressor.py`（新增）。

**作用**：把长 KV 序列以 `compress_ratio`(=4 或 128) 为粒度做学习池化，输出
压缩后的 KV cache 供 Indexer 评分和后续 sparse attention 取用。

**`__init__` 关键参数**：
```python
def __init__(
    self,
    hidden_size: int,
    head_dim: int,
    compress_ratio: int,        # 4 or 128
    overlap: bool = False,      # 仅 compress_ratio == 4 时为 True
    rotate: bool = False,       # Indexer 内部使用时 True（带 Hadamard）
):
    self.wkv   = Linear(hidden_size, (1 + int(overlap)) * head_dim)
    self.wgate = Linear(hidden_size, (1 + int(overlap)) * head_dim)
    self.norm  = RMSNorm(head_dim)
    self.ape   = nn.Parameter(torch.zeros(compress_ratio, (1 + int(overlap)) * head_dim))
```

**forward 契约**（仅训练路径，无 cache）：
```python
def forward(self, x: Tensor) -> Tensor:
    # x: [B, S, hidden_size]
    # 先 reshape 成 [B, S//ratio, ratio, hidden_size]
    # gate = softmax(wgate(x) + ape, dim=ratio-axis)
    # kv   = norm(sum(gate * wkv(x), dim=ratio-axis))
    # 返回 [B, S//ratio, head_dim]
```

注：`S` 在训练时是 packed 序列长度。Compressor 内部按 `cu_seq_lens` 分段处理，
避免跨样本污染——这一段在 V3 的 attention 里已有先例（`flash_attn_varlen_func`
就靠 `cu_seq_lens` 隔离）。压缩边界对 `cu_seq_lens` 的处理细节需要在该 PR
的设计 commit 里单独说明（建议方案：每条样本 padding 到 ratio 的整数倍，
padding 位置 mask 掉）。

**单测**：
1. 与 HF `Compressor` 的输出 bit-identical（同权重、同输入）。
2. `cu_seq_lens` 边界：两条样本拼成的 packed 序列，验证第二条样本的输出**不
   依赖**第一条样本的最后 `ratio-1` 个 token。

**预估**：~150 行 + 100 行测试。

---

### 4.5 Indexer

**路径**：`xtuner/v1/module/attention/indexer.py`（新增）。

**作用**：基于压缩后的 KV，给当前 query 评分，返回 top-k 压缩 KV 位置索引。

**`__init__`**：
```python
def __init__(self, dsa_cfg: DSAConfig, compress_ratio: int):
    self.wq_b        = Linear(dsa_cfg.q_lora_rank, dsa_cfg.index_n_heads * dsa_cfg.index_head_dim)
    self.weights_proj = Linear(dsa_cfg.hidden_size, dsa_cfg.index_n_heads)
    self.compressor   = Compressor(
        hidden_size=dsa_cfg.hidden_size,
        head_dim=dsa_cfg.index_head_dim,
        compress_ratio=compress_ratio,
        rotate=True,                       # Hadamard rotation
    )
    self.softmax_scale = dsa_cfg.index_head_dim ** -0.5
```

**forward 契约**：
```python
def forward(
    self,
    hidden_states: Tensor,                  # [B, S, hidden_size]
    q_lowrank: Tensor,                      # [B, S, q_lora_rank]  ← 来自 DSA 的 q_norm 输出
    position_embeddings_compressed: tuple[Tensor, Tensor],
    cu_seq_lens: Tensor,
) -> Tensor:                                # topk_idxs: [B, S, index_topk]
    # 1. q = rope(wq_b(q_lowrank))，FP4-quant 模拟（训练时跳过 quant，保留接口）
    # 2. kv_compressed = compressor(hidden_states)   # [B, S//ratio, index_head_dim]
    # 3. scores = einsum("bshd,btd->bsht", q, kv_compressed) * weights_proj(x)
    # 4. apply causal-masked top-k → 返回 index_topk 个位置
```

**Hadamard rotation**：HF 参考用它来抑制激活的方差差异，提升 FP4 量化质量。
训练路径用纯 BF16，**rotation 仍要做**（数值上有差异），但 quant 跳过。

**单测**：
1. 与 HF `Indexer.forward` 的 top-k 索引在 95% 位置上一致（剩余 5% 允许因
   tied scores 顺序不同）。
2. causal mask：第 i 个 query 选出的索引 `t / ratio <= i / ratio`。

**预估**：~250 行 + 150 行测试。

---

### 4.6 Hash routing

**路径**：`xtuner/v1/module/router/hash_router.py`（新增）。

**作用**：前 `num_hash_layers=3` 层的 MoE gate 不做 score，直接按 `input_ids`
查 `tid2eid` 决定专家。

**`HashRouterConfig`**：
```python
class HashRouterConfig(BaseModel):
    vocab_size: int
    n_routed_experts: int
    num_experts_per_tok: int          # 6
```

**`HashRouter.__init__`**：
```python
self.register_buffer(
    "tid2eid",
    torch.zeros((vocab_size, num_experts_per_tok), dtype=torch.int32),
    persistent=True,  # 从 checkpoint 加载
)
```

**forward 契约**：
```python
def forward(
    self,
    hidden_states: Tensor,            # 仅为接口对齐，不读
    input_ids: Tensor,                # [B, S] 或 packed [total_tokens]
) -> RouterResults:
    topk_ids = self.tid2eid[input_ids.long()]               # [..., num_experts_per_tok]
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32) / num_experts_per_tok
    return {"topk_ids": topk_ids, "topk_weights": topk_weights, "logits": None, ...}
```

**与现有 router 的关系**：
- 共用 `RouterProtocol`（`xtuner/v1/module/router/protocol.py`）。
- MoE 装配层在 `MoEDecoderLayer.__init__` 里按 `layer_idx < num_hash_layers`
  选择 `HashRouter` 或 `NoAuxRouter`。这条选择逻辑**留在 MoE 装配层**而不是
  放进基类的 `if model_type` 分支里——`DeepSeekV4Config.build_router(layer_idx)`
  做这个选择。

**input_ids 注入**：现有 `MoEDecoderLayer.forward` 不收 `input_ids`，需要把
`SequenceContext` 沿调用栈传到 router。这是签名变更，**单 PR 处理**，且要
对所有 router 协议保持向后兼容（其它 router 接收但不使用）。

**单测**：
1. `tid2eid` 加载后，相同 `input_ids` 永远得到相同 expert 索引。
2. `topk_weights` 归一化求和等于 1。

**预估**：~120 行 + 80 行测试，外加 `RouterProtocol` 的签名扩展。

---

### 4.7 Hyper-Connections (HC) decoder block

**路径**：`xtuner/v1/module/decoder_layer/hc_block.py`（新增）。

**作用**：把一层 attention/FFN 包装成「保留 `hc_mult` 份 hidden state 副本，
forward 时通过 sinkhorn 归一化的 `hc_pre` 把多份合成一份给 inner block，再
通过 `hc_post` 把单份输出扩成多份」的结构。

**`HCWrapperConfig`**：
```python
class HCWrapperConfig(BaseModel):
    hc_mult: int          # 4
    hc_eps: float         # 1e-6
    hc_sinkhorn_iters: int  # 20
```

**模块结构**（伪代码）：
```python
class HCDecoderLayer(nn.Module):
    def __init__(self, inner: MoEDecoderLayer | DenseDecoderLayer, hc_cfg: HCWrapperConfig, hidden_size: int):
        self.inner = inner
        mix_dim = (2 + hc_cfg.hc_mult) * hc_cfg.hc_mult
        self.hc_attn_fn   = nn.Parameter(torch.zeros(mix_dim, hc_cfg.hc_mult * hidden_size))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_dim))
        self.hc_attn_scale = nn.Parameter(torch.zeros(3))
        self.hc_ffn_fn    = nn.Parameter(torch.zeros(mix_dim, hc_cfg.hc_mult * hidden_size))
        self.hc_ffn_base  = nn.Parameter(torch.zeros(mix_dim))
        self.hc_ffn_scale = nn.Parameter(torch.zeros(3))

    def forward(self, x: Tensor, ...) -> Tensor:
        # x: [B, S, hc_mult, hidden_size]
        # attention path：
        x_in,   post_a, comb_a = hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        attn_out = self.inner.self_attn(x_in, ...)
        x = hc_post(attn_out, x, post_a, comb_a)
        # ffn path：
        x_in,   post_f, comb_f = hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        ffn_out = self.inner.mlp(x_in, ...)
        x = hc_post(ffn_out, x, post_f, comb_f)
        return x
```

**`hc_split_sinkhorn` 实现**：纯 PyTorch 版本（`xtuner/v1/module/decoder_layer/hc_sinkhorn.py`）
作为 baseline；同 PR **不**包含 CUDA kernel——把 kernel 化作为后续可选 PR。

**为什么是 wrapper 而不是修改 inner block**：CLAUDE.md 设计原则 2「把复杂度放
在拥有上下文的模块里」。inner block 的契约（输入 [B,S,D]，输出 [B,S,D]）不
变；HC 只在 wrapper 层维护「hc_mult 份副本」的概念。这样：
- V3、Qwen3 等模型继续用裸 `MoEDecoderLayer`，零改动。
- V4 在 `build_layers` 时把每层 `MoEDecoderLayer` / `DenseDecoderLayer` 包一层
  `HCDecoderLayer`。
- 残差与 norm **仍由 inner block 自己负责**——HC 在 inner block 之外做混合，
  inner block 看到的还是普通 hidden state。

**embedding 与 head 的 HC 注入**：
- `MoE.embed_tokens` 之后立刻把 `[B, S, D]` 扩成 `[B, S, hc_mult, D]`（复制
  `hc_mult` 份）。
- LM head 入口处一个**独立**的 HC reduce（HF 在顶层存了 `hc_head_fn` /
  `hc_head_base` / `hc_head_scale`，结构与 `hc_attn_*` 一致但参数独立）。
- `MoE.norm` 之前先把 `[B, S, hc_mult, D]` 通过 `hc_head_*` reduce 回 `[B, S, D]`。
- 这两个 hook 由 `DeepSeekV4.build_embeddings` / `build_norm`（小覆写）注入；
  `hc_head_*` 三个参数作为 `DeepSeekV4` 模型级 `nn.Parameter`（不属于任何 layer）。
- MTP block 也带自己独立的 `hc_attn_*` / `hc_ffn_*` / `hc_head_*`（HF key 前缀
  `mtp.0.*`），意味着 MTPBlock 是一个完整的 `HCDecoderLayer` + 自己的 HC head
  reduce，**不**复用主干 layers 的 HC 参数。

**单测**：
1. 单步 HC wrapper forward：`hc_mult=1` 时应该等价于裸 inner block（验证
   wrapper 在 degenerate case 下不引入数值偏差）。
2. 与 HF `Block.forward` 在同一份权重下做 single-token forward，比较
   `[B=1, S=4, hc_mult=4, D=4096]` 输出 `max_abs_diff < 5e-3`（bf16）。
3. `hc_split_sinkhorn` 单独测：固定输入、固定 iters，输出与 HF 一致。

**预估**：~350 行代码 + 200 行测试，是 7 个原语里最重的一个。

---

### 4.8 FP4 expert 权重加载（FP8 block-scaled + FP4 dequant）

**路径**：
- `xtuner/v1/float8/`（扩展）或新增 `xtuner/v1/fp4/`。
- `xtuner/v1/model/base.py:1672-1686`（扩展 `_is_loaded_param_fp8` / `_load_fp8`
  到 fp4）。
- `xtuner/v1/utils/load_spec.py`（`LoadSpec` 字段——可能需要 `dequant_dtype`）。

**HF 存储**：
- `config.quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128,128]}`
- `config.expert_dtype = "fp4"` → 只有 expert 权重以 fp4 存储；其它（attention、
  embed、norm）按 fp8 e4m3 + per-block scale 存储。

**变更**：
1. `Float8Config` 增加（或新建 `QuantConfig`）：
   ```python
   expert_dtype: Literal["bf16", "fp8_e4m3", "fp4"] | None = None
   weight_block_size: tuple[int, int] = (128, 128)
   ```
2. `BaseModel._load_fp8` 扩展：检测到 `quant_method == "fp4"` 时走 fp4 dequant
   路径（block-scaled fp4 → bf16）。
3. `LoadSpec` 增加 `dequant_dtype` 字段；fp8/fp4 路径在 LoadSpec 里只标
   `_ori_shape`（按 memory `project_fp8_load_spec_coupling`）。

**关键约束**：memory `project_fp8_load_spec_coupling` 明确「`LoadSpec` schema
不承担 fp8 感知，用 `_ori_shape` 在路径内现场处理」。fp4 复用同一约束：
**不要**在 `LoadSpec` 顶层加 `is_fp4`。

**单测**：
1. 构造一份小 fp4 weight + scale，验证 dequant 后与 ground-truth bf16
   `max_abs_diff < 1/128`（fp4 量化粒度）。
2. 加载一个 V4-Flash 小型子集（仅 attention + 1 个 expert）的 safetensor 切片，
   验证 forward 数值与 HF reference 一致。

**预估**：~200 行 + 150 行测试。**风险点**：FP4 e2m1/e3m0 等具体 format 在
HF config 里没写明，需要从 `inference/model.py` 的 dequant kernel 里读出。
设计 commit 阶段先确认。

---

### 4.9 把它们粘合起来：`DeepSeekV4Config` / `DeepSeekV4`

**路径**：`xtuner/v1/model/moe/deepseek_v4.py`（新增）+ `xtuner/v1/model/__init__.py`
（4 处修改：import、`model_mapping`、`get_model_config_from_hf` 分支、`__all__`）。

**`DeepSeekV4Config`** 关键字段（仅列与 `DeepSeekV3Config` 不同的）：
```python
attention: DSAConfig                              # 4.3
router: NoAuxRouterConfig                         # 4.1 (sqrtsoftplus)
hash_router: HashRouterConfig | None              # 4.6
num_hash_layers: int                              # 3
hc_cfg: HCWrapperConfig                           # 4.7
rope_parameters_cfg: RopeParametersConfig         # 4.2 (compress_*)
mtp_config: MTPConfig | None                      # 复用，num_layers = num_nextn_predict_layers
quant_cfg: QuantConfig | None                     # 4.8
sliding_window: int = 128
swiglu_limit: float = 10.0
```

**`DeepSeekV4` 类的覆写**：
- `to_hf_key_list`（必须）：与 V3 不同，V4-Flash 的 HF key **不带 `model.` 前缀，
  也不带 MoE 的 `mlp.` 前缀**（从本地 safetensors index 反推确认）。规则比 V3
  简单但映射数量更多：
  - `embed_tokens.weight` → `embed.weight`，`norm.weight` → `norm.weight`，
    `lm_head.weight` → `head.weight`，`layers.N.X` → `layers.N.X`（无前缀）。
  - 注意力命名：`q_a_proj`→`wq_a`，`q_b_proj`→`wq_b`，`q_a_layernorm`→`q_norm`，
    K/V 合并到 `wkv`（XTuner 侧需要把 `kv_a_proj_with_mqa`/`kv_b_proj` 折叠为
    单个 `wkv`——这是 DSAConfig 的天然结构，**不**像 MLA 那样有低秩 KV），
    `o_proj` → `wo_a`+`wo_b`（grouped LoRA 两段）。
  - MoE 命名：`experts.E.gate_proj`→`ffn.experts.E.w1`，`up_proj`→`w3`，
    `down_proj`→`w2`，`shared_experts.X`→`ffn.shared_experts.X`，
    `router.weight`→`ffn.gate.weight`，`router.e_score_correction_bias`→`ffn.gate.bias`，
    `hash_router.tid2eid`→`ffn.gate.tid2eid`（hash 层和 score 层共用 `gate` 前缀，
    通过有无 `tid2eid` / `bias` 区分模式）。
  - HC 参数：`hc_attn_fn` → `layers.N.hc_attn_fn`（不嵌套在 wrapper.inner 路径下，
    `to_hf_key_list` 需把 XTuner 的 `layers.N.hc_attn_fn` 直接映射到同名 HF key）。
  - 顶层 HC head：`hc_head_fn/base/scale` 直接保留为顶层 key。
  - MTP：`mtp_layer.0.X` → `mtp.0.X`（XTuner 侧 `MTPBlock` 子模块命名需对齐）。
  - Fused expert 展开：`layers.N.ffn.experts.fused_w1w3.weight` → N×(`w1` + `w3`) 对；
    `fused_w2.weight` → N×`w2`（沿用 V3 模式）。
- `build_layers`（**只在这里**做 per-layer 装配选择）：
  ```python
  for layer_idx in range(num_hidden_layers):
      attn_mode = _attn_mode_from_compress_ratio(config.rope_parameters_cfg.compress_ratios[layer_idx])
      router    = HashRouter(...) if layer_idx < config.num_hash_layers else NoAuxRouter(...)
      inner     = MoEDecoderLayer(..., self_attn=DeepSeekSparseAttention(..., attn_mode=attn_mode), router=router)
      layers[str(layer_idx)] = HCDecoderLayer(inner, config.hc_cfg, hidden_size=config.hidden_size)
  ```
- `build_embeddings`（小覆写）：embed 之后扩 `hc_mult` 维度。
- `build_norm`（小覆写）：最终 `hc_pre` reduce。
- `_init_weights`：HC 参数全 0 初始化，`hc_attn_scale` 后两位（post / comb）也
  全 0，第一位（pre）置 1——这是 HC 等价于纯残差的初始化（degenerate-safe）。
- `safetensors_to_params` / `param_to_safetensor`：**不需要**。所有差异都能
  通过命名映射（`to_hf_key_list`）+ fp4 dequant 路径覆盖。

**单测**：
1. `tests/model/test_deepseek_v4_moe.py::test_decoder_layer_parity` —— 取 V4-Flash
   `compress_ratios = [4, 128, 0, ...]` 的前三层各跑一次：
   - 加载对应 layer 的 BF16 reference 权重（用户提供 `DEEPSEEK_V4_BF16_PATH`）。
   - 构造对应模式的 `HCDecoderLayer + DeepSeekSparseAttention + NoAuxRouter/HashRouter`。
   - 输入 `[B=1, S=64, hidden_size=4096]`，比较 `forward` 输出与 HF `Block.forward`
     的 `max_abs_diff < 5e-3`、`cos_sim > 0.999`。
2. `test_save_hf_roundtrip` —— V4 没有内置 HF config 类（`hf_config` 返回 `None`），
   走 `save_hf` 的「拷贝原 `*.py`」分支，确认 `safetensors.index.json` 与
   `model.py` / `config.json` 都被复制到目标目录。
3. `test_entry_point` —— `get_model_config_from_hf(<v4_bf16_path>)` 返回
   `DeepSeekV4Config`，所有字段与 HF config 一致。

**预估**：~600 行代码 + 400 行测试。

---

## 5. PR 序列与依赖图

```
                       ┌─ PR1: sqrtsoftplus scoring (4.1)
                       │
                       ├─ PR2: dual rope (4.2)
                       │
                       ├─ PR3: KV Compressor (4.4)
                       │
                       ├─ PR4: Indexer (4.5)         ← 依赖 PR3
                       │
                       ├─ PR5: DSA attention + O-LoRA (4.3)
                       │        ← 依赖 PR2, PR4
                       │
                       ├─ PR6: Hash Router + RouterProtocol 扩展 (4.6)
                       │
                       ├─ PR7: Hyper-Connections wrapper (4.7)
                       │
                       ├─ PR9: DeepSeekV4 model + config + 入口注册 + parity test (4.9)
                       │        ← 依赖 PR1..PR7
                       │
                       └─ PR8 (optional, post-merge): FP4 expert load path (4.8)
```

**说明**：
- PR1、PR2、PR3、PR6、PR7 之间**互相独立**，可以并行评审与落地。
- PR4（Indexer）依赖 PR3（Compressor 是 Indexer 的内部组件）。
- PR5（DSA attention）依赖 PR2（取双 rope）和 PR4（用 Indexer），但**不**依赖
  PR1/PR6/PR7——这意味着 PR5 落地后已经可以单独跑「DSA 注意力 vs HF」的
  parity 测试，不用等 HC/Hash。
- PR9 是粘合层，**必须**在 PR1–PR7 都 merge 后落，但**不依赖 PR8**——
  parity 测试从本地 BF16 reference（见 §6）加载，FP4 路径仅在用户直接加载
  HF 原版 release 时才需要。PR8 作为 follow-up，可以在 V4 上线后再做。

按用户的 `feedback_refactor_small_steps`：每个 PR 都是单一职责，签名变更
（`scoring_func` 加 literal、`RouterProtocol` 加 `input_ids`、`RopeParametersConfig`
加字段、`Float8Config` 加 `expert_dtype`）都在各自原语的 PR 里完成，不和 PR9
绑定。

---

## 6. Parity 测试方案

**Reference 权重**：本地 BF16 reference 路径
`/mnt/shared-storage-user/llmrazor-share/yehaochen/model/DeepSeek-V4-Flash`
（109 个 safetensor 分片，542 GB，由 HF 上 46-shard FP4/FP8 release 离线
dequant 而来；见仓内 `convert.log`）。测试通过环境变量
`DEEPSEEK_V4_BF16_PATH` 注入此路径；CI 与本地都从此读取，**不**直接读 HF Hub
上的 FP8/FP4 release。
- 该 checkpoint 的 `config.json` 已剥离 `quantization_config`/`expert_dtype`
  字段，是纯 BF16，可以直接喂给 `from_hf` 路径。
- `to_hf_key_list` 的命名映射以此 checkpoint 的 `model.safetensors.index.json`
  为权威来源（见 §4.9 详细映射）。

**测试金字塔**：

| 层级 | 测试 | 输入规模 | 容差 |
|---|---|---|---|
| L1 原语数值 | sqrtsoftplus、Compressor、Indexer top-k、hc_sinkhorn 收敛、fp4 dequant | 单 tensor，B=1, S=16 | `atol=1e-3` (bf16) / `1e-6` (fp32) |
| L2 模块 forward | `DeepSeekSparseAttention(attn_mode=...)`、`HCDecoderLayer(inner=...)`、`HashRouter` | B=1, S=64, D=4096 | `max_abs_diff < 5e-3`、`cos_sim > 0.999` |
| L3 单层 decoder parity | 完整 `HC(MoE(DSA, Router))` vs HF `Block` | B=1, S=64 | 同上 |
| L4 多层 parity | 前 3 层（覆盖 hash + score、sliding + compressed_4 + compressed_128） | B=1, S=64 | `max_abs_diff < 1e-2` |

**整模型 forward parity 不做**——参数量 >600B、HF reference 在单卡跑不动。
`tests/model/test_qwen3_moe.py` 的 `_check_loss_curve` 模式不适用。

**FSDP / EP 矩阵**：PR9 在 L3 测试里跑
`{(dispatcher=None, ep_size=1), (dispatcher="all2all", ep_size=8)}`，覆盖
EP 路径下的 expert 通信不变性。`"deepep"` 由于 V4 expert 数（256）较 V3
未变，预期可直接复用。

---

## 7. 风险与未解决问题

1. ~~**FP4 具体 format**~~ — **已解除**：本地 BF16 reference 已存在，FP4 dequant
   只影响「直接加载 HF release」场景，可作为 follow-up（PR8）独立处理，不
   阻塞主线。若未来要做，从 HF `inference/model.py` 的 `Linear.dequant` 路径
   抽出格式即可。
2. ~~**`compress_ratios` 长度**~~ — **已确认**：长度 `= num_hidden_layers + 1 = 44`，
   索引 0..42 对应 layers 0..42，索引 43 对应 MTP layer。本地 config 中
   `compress_ratios[43] = 0` 表示 MTP 层用 sliding-window-only attention。
3. **HC sinkhorn 的数值稳定性**。`hc_sinkhorn_iters=20` 的纯 PyTorch 实现
   在 bf16 下可能出现 `NaN`（softmax 在低精度下的常见问题）。PR7 在 fp32
   下做 sinkhorn 迭代，最后 cast 回 bf16——这是 HF 参考的做法，需在测试
   里固定 random seed 验证。
4. **Hash router 在 SP / EP 下的行为**。`tid2eid` 是 vocab-size × top_k 的
   buffer（V4-Flash 是 `129280 × 6 = ~770K int32 = 3MB`），可以全 rank 复制
   不切分。`HashRouter.forward` 接收 SP 切分后的 `input_ids` shard 即可，
   不需要跨 rank 通信。PR6 在测试里覆盖 SP=2 的情形。
5. **MTP `num_nextn_predict_layers=1` 的接入**。XTuner 现有 `MTPConfig.num_layers`
   语义一致，PR9 直接复用；但**HC wrapper 是否包住 MTP 层**需要从 HF
   `MTPBlock` 的 forward 里再次确认（它继承 `Block`，所以 yes，但 embed/head
   的 HC 处理略有差异——预留为 PR9 内子问题）。
6. **`swiglu_limit=10.0`**。HF Expert 在 `silu` 后做 clamp(min=-10, max=10)。
   XTuner `xtuner/v1/module/decoder_layer/moe_decoder_layer.py` 的 expert
   是否支持？需在 PR9 的设计 commit 阶段验证；若不支持，加一个
   `MoEActFnConfig.swiglu_limit` 字段，单独子 PR 处理。
7. **`attn_sink` 的具体语义**。每层 `attn_sink` 是一个 learnable 向量
   （shape 推测为 `[num_attention_heads]`），与 sliding window 配合使用。
   HF `inference/model.py` 的 `sparse_attn` / FA 调用方式与 XTuner 现用的
   `flash_attn_varlen_func` 兼容性需在 PR5 设计 commit 阶段确认；若 sink 注入
   方式不被 FA varlen 支持，可能需要 fallback 到 eager attention 或扩展
   `xtuner/v1/ops/`。

---

## 8. 变更清单（概要）

| 文件 | 操作 | PR |
|---|---|---|
| `xtuner/v1/module/router/noaux_router.py` | `Literal` 加 `sqrtsoftplus`，分支实现 | PR1 |
| `tests/module/router/test_noaux_router.py` | sqrtsoftplus 数值测试 | PR1 |
| `xtuner/v1/module/rope/rope.py` | `RopeParametersConfig` 加 `compress_rope_theta`/`compress_ratios`；新增 `DualRotaryEmbedding`；`get_rope_embedding` 分发 | PR2 |
| `tests/module/rope/test_dual_rope.py` | 双 freqs / yarn / per-layer 选择测试 | PR2 |
| `xtuner/v1/module/attention/kv_compressor.py` | 新增 `KVCompressor` | PR3 |
| `tests/module/attention/test_kv_compressor.py` | 单点测试 + cu_seq_lens 边界 | PR3 |
| `xtuner/v1/module/attention/indexer.py` | 新增 `Indexer`（依赖 `KVCompressor`） | PR4 |
| `tests/module/attention/test_indexer.py` | top-k vs HF parity、causal mask | PR4 |
| `xtuner/v1/module/attention/dsa.py` | 新增 `DSAConfig` + `DeepSeekSparseAttention` | PR5 |
| `xtuner/v1/module/attention/__init__.py` | 导出 `DSAConfig` | PR5 |
| `tests/module/attention/test_dsa.py` | 三种 attn_mode 各做一次 parity | PR5 |
| `xtuner/v1/module/router/hash_router.py` | 新增 `HashRouterConfig` + `HashRouter` | PR6 |
| `xtuner/v1/module/router/protocol.py` | `RouterProtocol.forward` 增加可选 `input_ids` 参数 | PR6 |
| `xtuner/v1/module/decoder_layer/moe_decoder_layer.py` | 把 `input_ids` 透传给 router | PR6 |
| `tests/module/router/test_hash_router.py` | tid2eid 加载、确定性、SP 切分 | PR6 |
| `xtuner/v1/module/decoder_layer/hc_block.py` | 新增 `HCWrapperConfig` + `HCDecoderLayer` + 纯 PyTorch sinkhorn | PR7 |
| `tests/module/decoder_layer/test_hc_block.py` | degenerate case + 与 HF Block parity + sinkhorn 单测 | PR7 |
| `xtuner/v1/model/moe/deepseek_v4.py` | 新增 `DeepSeekV4Config` / `DeepSeekV4` + `build_layers` 覆写 + `to_hf_key_list` | PR9 |
| `xtuner/v1/model/__init__.py` | import、`model_mapping`、`get_model_config_from_hf` 分支、`__all__` | PR9 |
| `tests/model/test_deepseek_v4_moe.py` | decoder-layer parity（hash + score、sliding + compressed_4 + compressed_128） | PR9 |
| `xtuner/v1/float8/config.py` | `Float8Config` 加 `expert_dtype` / `weight_block_size` | PR8（optional, post-PR9） |
| `xtuner/v1/model/base.py` | `_load_fp8` 路径扩展到 fp4 | PR8（optional） |
| `tests/float8/test_fp4_load.py` | fp4 dequant 数值 + 小 safetensor 加载 | PR8（optional） |

---

## 9. 开源参考与复用策略

> 调研结果（2026-05-19）：每个原语已知最近上游 + 落地建议。本地缓存：
> `.dev_scripts/deepseek_v4_reference/{model.py,kernel.py}`（V4-Flash 官方 inference 代码，MIT）。

| PR | 原语 | 最近上游 | License | 落地策略 |
|---|---|---|---|---|
| PR1 | sqrtsoftplus | V4 `inference/model.py:Gate.forward`（一行 `F.softplus(x).sqrt()`） | MIT | **从零写**（1 行公式无需 vendor） |
| PR2 | Dual RoPE | V4 `inference/model.py:precompute_freqs_cis`（带 `@lru_cache(2)`，对 `rope_theta` 与 `compress_rope_theta` 各调一次 HF `_compute_yarn_parameters`） | MIT | **adapt** — yarn 数学直接复用 XTuner 现有 `ROPE_INIT_FUNCTIONS["yarn"]`，仅在 `RopeParametersConfig` 加两个字段 + 新增 `DualRotaryEmbedding`（持两组 `inv_freq`） |
| PR3 | KV Compressor | V4 `inference/model.py:Compressor`（class L279-379，~100 行） | MIT | **port** — 无第三方 OSS 实现，从 V4 inference 直接移植；XTuner 侧改为 varlen + `cu_seq_lens` 边界处理 |
| PR4 | Indexer | (1) `lmdeploy/lmdeploy/pytorch/models/deepseek_v32.py:43-118`（~80 行精简版）；(2) `sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py:136-322`（含 FP8 / paged） | Apache 2.0 | **lift from lmdeploy** + 改 varlen。注意去掉 paged KV-cache 逻辑，scoring 第一版用 bf16 GEMM 不上 DeepGEMM FP8 |
| PR4 子项 | sparse-attn kernel（top-k → attention） | sglang TileLang kernel **仅前向，无 backward** | Apache 2.0 | **不可直接 lift 训练用**。第一版用 PyTorch reference（`gather` + varlen FA），性能优化作为 follow-up |
| PR5 | DSA glue（Q/K/V/O 投影 + Indexer 联调） | V4 `inference/model.py:Attention` (L436-545) | MIT | **port** — `wq_a/wq_b/wkv/wo_a/wo_b` 全部从 V4 inference 移植，因为 grouped O-LoRA + `attn_sink` 在 sglang/lmdeploy/transformers 都没有 |
| PR5 子项 | Grouped O-LoRA (`o_groups=8`) | 无 OSS — V4 全新 | — | **从 V4 inference 移植**（class `Attention` 的 `wo_a`/`wo_b` 段，约 30 行） |
| PR5 子项 | `attn_sink` + sliding-window varlen | sglang `flashattention_backend.py:832` 用 patched FA3 的 `sinks` kwarg。本机 wheel（`flash_attn==2.8.3`、`flash_attn_3==3.0.0b1`）**都没有 sinks 参数** | Apache 2.0 | **fallback 路径** — 第一版 PyTorch 把 sink 拼到 query mask 上（性能差但能跑端到端测试），FA3-with-sinks 作为后续 wheel 升级 |
| PR6 | Hash Router | (1) HF transformers v5.8.1 `transformers/models/deepseek_v4/modeling_deepseek_v4.py:DeepseekV4HashGate`；(2) V4 `inference/model.py:Gate` hash 分支 | Apache 2.0 / MIT | **port from HF transformers**（含已 vendor 的 `tid2eid` 加载路径，Apache-2.0 友好） |
| PR7 | Hyper-Connections | (1) **`lucidrains/hyper-connections` PyPI 0.4.11**（纯 PyTorch，bf16-safe，有 `manifold_constrained_hyper_connections.py` / `mHCv2.py` / `triton_sinkhorn.py` with PyTorch fallback）；(2) V4 `inference/kernel.py:hc_split_sinkhorn`（TileLang JIT，**不能直接训练用**） | MIT / MIT | **lucidrains 或 vendor**（决策见 §10） |
| PR7 子项 | `hc_split_sinkhorn` | lucidrains `mHCv2.sinkhorn_knopps`（fp32 内部、`-amax(dim=-2).detach()` 数值稳定）vs V4 TileLang kernel | MIT | **lucidrains** — 纯 PyTorch，bf16 安全，无新依赖（除 PyPI 包本身） |
| PR8 | FP4 expert | V4 `inference/kernel.py` 的 Linear FP4 dispatch | MIT | **defer** — 本地已有 BF16 reference，不阻塞 parity；如未来要直接加载 HF release 再做 |

**关键 takeaway**：
1. **半数 PR 没有 OSS 可直接 lift**（PR1/PR3/PR5/PR7 子项），唯一权威实现是 V4-Flash 的 `inference/model.py` —— 已缓存到 `.dev_scripts/deepseek_v4_reference/`。
2. **PR4 (Indexer) 有 lmdeploy 80 行精简版可参考 adapt**；但 sparse-attn kernel 训练 backward 需要自己写 PyTorch fallback（sglang 的 TileLang 只有 forward）。
3. **PR6 (Hash Router) HF transformers v5.8.1 已有完整实现**（`DeepseekV4HashGate`），直接 port。
4. **PR7 (HC) 推荐用 lucidrains/hyper-connections**（PyPI，MIT，bf16-safe）作为 baseline；V4 官方的 TileLang kernel 留作可选性能优化。
5. **`attn_sink` 是个新风险** —— 本机 FA wheel 都没有 `sinks` 参数，第一版只能 PyTorch fallback，性能差但能验证正确性。

---

## 10. 与项目规则的对照

- **CLAUDE.md 设计原则 1**（稳定概念建模）：每个新 config (`DSAConfig`、
  `HashRouterConfig`、`HCWrapperConfig`) 描述**一种**注意力 / 路由 / 块结构，
  没有「`if model_type == "deepseek_v4"`」分支泄漏到基类。
- **设计原则 2**（复杂度内聚）：HC 在 wrapper 层；Indexer 在 attention 模块
  内；FP4 dequant 在 `_load_fp8` 同一路径——每个原语的细节都不外泄。
- **设计原则 5**（共享逻辑放在合适层级）：sliding window、yarn rope、
  shared expert、`first_k_dense_replace` 全部复用 V3 已有实现，不抽新基类。
- **设计原则 8**（重构后留一份模型）：每个 PR merge 后，主线代码里**没有**
  「旧 API + 新 API」并存，直到 PR9 落地，V4 路径才在 `__init__.py` 上线。
- **memory `feedback_refactor_small_steps`**：9 个 PR，每个 PR 单一原语，
  签名变更（`scoring_func` literal、`RouterProtocol.forward(input_ids=...)`、
  `RopeParametersConfig` 新字段、`Float8Config` 新字段）全部在自己的 PR 里
  独立完成，PR9 不携带任何签名变更。
- **memory `project_fp8_load_spec_coupling`**：fp4 复用 `_ori_shape` 现场处理
  约定，不改 `LoadSpec` 顶层 schema。
- **OSS 复用与归属**：所有 lift/port 自 OSS 的代码必须在文件头保留原始 license
  与归属注释（V4-Flash MIT、lmdeploy Apache-2.0、HF transformers Apache-2.0、
  lucidrains/hyper-connections MIT），并在 PR 描述里列明来源 URL + commit SHA。
