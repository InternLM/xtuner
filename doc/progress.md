# GLM-5.3-Flash SFT 开发进度

> 设计文档：`doc/xtuner_glm5p3flash_design.md`（伪代码：`doc/xtuner_glm5p3flash_design.py`）
> 原则：每个 feature（F0~F6）单独开发、单独测试、单独提交；GPU 任务优先用本地
> `~/github/xtuner/zdev/gpu_lock.sh`，长时间拿不到锁再走 `yd_clusterx.sh` 提交集群任务。

## F0 25B 减层参照模型

**目标**：`xtuner/tools/model_converters/make_glm53_25b_hf.py`，从发布版 checkpoint（FP8
62-shard 或官方原生 BF16 120-shard，两者均可作为 `--source`）裁出 `~/model/GLM-5.3-Flash-25B`
（发布版布局，BF16，约 24.9B）。

**状态**：已完成，已提交。

**交付**：
- `xtuner/tools/model_converters/make_glm53_25b_hf.py`
- `tests/model/test_glm53_25b_crop.py`（合成小 checkpoint，覆盖裁剪键集合 + FP8 反量化数值 +
  原生 BF16 输入直通不误触发反量化）

- 已核实源 checkpoint 实际字段（与设计文档一致）：
  - `text_config.layer_types[0:5] = [linear_attention, linear_attention, linear_attention,
    deepseek_sparse_attention, linear_attention]`，`mlp_layer_types[0:5] = [dense, dense, dense,
    sparse, sparse]`。
  - 层键前缀是 `model.language_model.layers.N.`（不是 GLM-5.2 的 `model.layers.N.`），
    因此裁剪脚本不能直接复用 `make_glm52_30b_hf.py` 的 `LAYER_KEY_RE`，需要改前缀。
  - FP8 量化是 128×128 block scale：`weight.shape=[out,in]` fp8 e4m3，
    `weight_scale_inv.shape=[ceil(out/128), ceil(in/128)]` fp32；`modules_to_not_convert`
    覆盖 `hc_*`、`A_log`/`dt_bias`、attn 的门控/norm 类小张量，这些本来就没有 `weight_scale_inv`。
  - 顶层 `quantization_config` 需要在裁剪后整体删除（输出是纯 BF16）。
  - MTP 层键是 `model.language_model.layers.45.*`（`eh_proj`/`enorm`/`hnorm`/`shared_head.norm`
    + 完整 `self_attn`（含 indexer）+ `mlp`，无 `hc_*`），裁剪后重映射到 `layers.5.*`。
  - MoE 专家权重是**按专家拆分存储**的官方原版布局：`mlp.experts.{i}.gate_proj/up_proj/
    down_proj.weight`（i=0..287），不是融合成单个 `[288, ...]` 张量。两个发布版
    checkpoint（FP8 62-shard 与官方原生 BF16 120-shard）在这一点上完全一致，也与
    xtuner 自己的加载/保存路径一致——`Glm53TextMoE.to_hf_key_list()`（`glm53.py:246-255`）
    本来就是把内部融合的 `fused_w1w3.weight`/`fused_w2.weight` 展开成 288 个按专家的 HF
    键在读写，`hf_tensor_to_canonical()` 负责把读回的按专家张量在专家轴上拼回融合形状；
    这条路径对两个 checkpoint 源天然通用，不需要额外的"按专家 tensor 合并"转换步骤。
- 参考实现：GLM-5.2 的 `make_glm52_30b_hf.py`（`xtuner` 主仓 commit `35eae5cc7`，无 FP8 反量化，
  层前缀是 `model.layers.`）。

**官方原生 BF16 发布版交叉验证**（`/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/
zskj-hub/models--zai-org--GLM-5.3-Flash-BF16/`，45 层完整模型，120 shards，纯 BF16 无
`quantization_config`，~313B 参数）：

- 结构逐项核对与 FP8 发布版一致：`model_type=glm5_next`、`text_config` 字段集合
  （`layer_types`/`mlp_layer_types`/`linear_attn_config`/`mhc`/`indexer_types` 等）、层键前缀
  `model.language_model.layers.N.`、专家权重按专家拆分存储，均逐项吻合；裁剪脚本对着这个源
  `--dry-run` 选中的 tensor 集合（3100 个，14 source shards）与 FP8 源一致，无需改代码即可
  直接裁剪。
- **用这份原生 BF16 交叉验证了 F0 反量化数值的正确性**：分别从两个源各裁一份 25B crop，逐
  tensor 比较——`modules_to_not_convert`（未量化，如 embed_tokens/layernorm/`hc_*`）的张量
  **完全逐 bit 相同**；FP8 量化过的张量（`gate_proj`/专家权重/`q_a_proj` 等）92%~96% 逐 bit
  相同，其余差异 `max_abs_diff` ≈ 2.3e-5、`mean_abs_diff` ≈ 1e-7~1e-6 量级——与 FP8 e4m3 的
  量化粒度完全吻合，不是反量化 bug。
- 据此**把 `~/model/GLM-5.3-Flash-25B`（所有代码/测试默认读取的路径）换成了从这份原生 BF16
  裁出的版本**（不再经过 FP8 反量化），原 FP8 反量化版本保留在
  `~/model/GLM-5.3-Flash-25B-fp8dequant` 作为交叉验证的历史留档，不再是默认路径。换源后
  `TestGlm53TextMoEWeightMapping::test_real_checkpoint_weight_coverage` 与
  `TestGlm53TextMoEAccuracy::test_fsdp_accuracy`（全部 3 组 ep_size）重新跑过，结果不变
  （详见 F6 一节）。

**实测结果**（`~/model/GLM-5.3-Flash-25B`，当前为原生 BF16 裁剪版本；早期 FP8 反量化版本的
等价结果见上方交叉验证一节）：

- 裁出 3100 tensors / 14 shards，`total_params = 24.95B`，与设计文档 §3.2 表格的 24.9B 目标
  一致（偏差 < 1%）。
- `AutoConfig.from_pretrained` 在 pinned `transformers==5.17.0` 下正确加载嵌套 `text_config`，
  `num_hidden_layers=5`、`layer_types`/`mlp_layer_types` 与裁剪后配置一致。
- `Glm5NextForConditionalGeneration.from_pretrained(..., dtype=bf16, device_map="cuda")` 能
  bitwise 加载并跑通一次真实 forward（`gpu_lock.sh` 单卡），loss 有限值。
  - **重要发现**：本机安装的 transformers 5.17.0 的 `Glm5NextForConditionalGeneration`
    **没有实现 MTP 前向**——`modeling_glm5_next.py` 里完全不出现 `eh_proj`/`enorm`/`hnorm`/`mtp`，
    `self.layers`就是 `num_hidden_layers` 个普通 decoder layer，没有额外的 MTP 模块。
    因此裁剪后 checkpoint 里 `layers.5.*`（即原始 `layers.45`，MTP 层）在 `from_pretrained`
    时全部显示为 `UNEXPECTED`（非报错，只是被忽略），模型实际加载参数量是 17.51B
    （= 24.95B 总量 − MTP 层 7.43B，与设计文档 §3.2 表格逐项吻合）。
    这不是转换脚本的 bug：MTP 权重原样保留在 checkpoint 里，供 F6 里 XTuner 自己的 MTP
    实现读取；`test_fsdp_accuracy`（验收 1）用 HF 做 oracle 时，双方都只跑主栈的 5 层，
    这条本来就是"设计doc §5 一致"的比较范围，不涉及 MTP。F6 验证 MTP 数值时不能用
    `Glm5NextForConditionalGeneration.forward` 做 oracle，需要另外单独对 `layers.45` 的
    权重做等价的 eager 计算（或等 transformers 后续版本补上 MTP 前向）。

## F1 数据侧前处理闭环

**状态**：未开始。权威来源见设计文档 F1（duanyanhui 的 VL 文档 + 本次核实的 F1.b/F1.c）。

## F2 视觉塔与 projector

**状态**：未开始。

## F3 KDA 线性注意力

**状态**：已完成，已提交。

**交付**：
- `xtuner/v1/module/attention/kda.py`：`KDAConfig` / `KimiDeltaAttention`（非 SP 与
  `forward_for_sp` 两条路径）。
- `TransformerConfig.linear_attention` 类型放宽为 `GatedDeltaNetConfig | KDAConfig | None`；
  同步放宽了 `dense_decoder_layer.py` / `moe_decoder_layer.py` / `moe.py`(×2) / `dense.py`
  里的 `attention_config` 局部类型标注（mypy strict 需要，公共栈改动仅限类型层面）。
- `tests/model/test_glm53_kda.py`：5 个用例，覆盖设计文档 F3 的 4 类单测。

**实现要点**：
- 参考 `~/github/xtuner-ncp-k3` 的 `kda.py` 模块结构（独立 `q/k/v_conv1d`、低秩
  `f_a_proj->f_b_proj` forget gate、fp32 `A_log`/`dt_bias`、低秩 `g_a_proj->g_b_proj` 输出门），
  但按设计文档 3.5.1 改写了 `chunk_kda` 调用约定：门控用 `fla.ops.kda.gate.fused_kda_gate`
  在 kernel 外算好，`beta` 显式 sigmoid；**不**向 `chunk_kda`/`fused_recurrent_kda` 传
  `A_log`/`dt_bias`/`use_beta_sigmoid_in_kernel`（已安装 fla 0.4.2 的 `chunk_kda` 签名里
  根本没有这些参数，会静默吞进 `**kwargs`）。加了反向护栏单测断言这三个参数不在签名里。
- 短卷积没有直接复用 ncp-k3 的写法或本仓库 `gated_deltanet.py` 的 `nn.Conv1d`+`seq_idx`
  写法，而是自定义了 `KDAShortConvolution(fla.modules.ShortConvolution)` 子类，加
  `materialize_weight_bias()` + 显式 `weight`/`bias` override 的 `forward`——因为 SP 路径需要
  用同一个 conv 入口跑"外部传入的按 channel 切片后的权重"，而 FLA 原生 `ShortConvolution.forward`
  不接受外部 weight/bias（会导致 `causal_conv1d(..., weight=..., **kwargs)` 里 `weight` 重复传参报错）。
- `o_norm` 用了本地定义的 `FusedRMSNormGated(fla.modules.FusedRMSNormGated)` 子类（未复用
  `gated_deltanet.py` 里的同名类，避免 kda.py 对另一个 attention 类型模块产生不必要的横向依赖，
  两边各自 20 行左右的 try/except 导入块可接受的重复）。

**HF 数值 oracle**（用于 parity 测试）：`transformers.models.glm5_next.modeling_glm5_next.
Glm5NextTextLinearAttention`。**发现**：HF 内部把 XTuner/checkpoint 的三个独立
`q/k/v_conv1d.weight` 融合成一个 `self_attn.conv1d.weight`（沿 dim0 concat，顺序 q,k,v），
把 `f_a_proj`/`f_b_proj`/`A_log`/`dt_bias` 收进 `self.forget_gate` 子模块——这正是设计文档
3.1 说的"发布版 checkpoint 布局 vs HF 内部模块布局"两套命名，XTuner 侧按发布版布局实现是对的，
不需要额外桥接模块。HF 的 `forget_gate.forward` 直接返回`safe_gate_lower_bound *
sigmoid(decay_rate*(f_b(f_a(x))+dt_bias))`，与 `chunk_kimi_delta_attention` 调用处
（`g=self.forget_gate(hidden_states)`，不传 `A_log`/`dt_bias` 给 kernel）逐字印证了设计文档
3.5.1 的结论。另外 `Glm5NextTextLinearAttention.forward` 返回单个 tensor 而非 tuple
（不是 `(output, ...)`），第一次写 parity 测试时按 tuple 索引 `[0]` 误取了 batch 维，已修正。

**测试结果**：`tests/model/test_glm53_kda.py` 5/5 通过（单卡 GPU + `TestKDASequenceParallel`
2 卡 SP，经 `gpu_lock.sh` 跑）。其中 packed-multi-doc 测试第一次跑全量 5 个用例时偶发一次
NaN 失败；按最小复现原则排查：单独重跑该测试类、原始失败组合、以及完整套件各跑了 3 次，
全部稳定通过（详见排查记录），判定为外部 `fla` Triton kernel 首次编译/autotune 的一次性
GPU flake，不是 XTuner 侧实现问题，未做任何生产代码改动。

## F4 mHC 四流残差

**状态**：已完成，已提交。

**交付**：
- `xtuner/v1/module/decoder_layer/mhc.py`：`MHCConfig` / `hc_split_sinkhorn` / `hc_pre` /
  `hc_post` / `unshard_hc_params`。
- `xtuner/v1/ops/hc_post.py`：Triton `hc_post_fused`（前向+反向，`torch.library.custom_op`）。
- `xtuner/v1/model/moe/glm53/decoder_layer.py`：`Glm53DenseDecoderLayer` /
  `Glm53MoEDecoderLayer`，`mhc_cfg=None` 时纯透传基类逻辑（供 F6 的 MTP 层复用）。
- `tests/model/test_glm53_mhc.py`、`tests/ops/test_hc_post.py`、`tests/model/test_glm53_decoder_layer.py`。

**来源与改编**：核心数学移植自 `xtuner` 主仓 `dsv4` 分支的 DeepSeek-V4 实现
（commit `01c31a833`，未合入本分支，通过 `git show 01c31a833:<path>` 读取），但按设计文档重新
落到公共路径 `module/decoder_layer/mhc.py`（dsv4 原实现在 `module/decoder_layer/deepseek_v4/`
下，是模型私有的）。有意**不**移植的部分：
- dsv4 的 `XTUNER_V4_HF_PARITY` 全局开关与 `hf_parity.py`（V4 专用命名，且本分支不存在该模块）；
  改为直接让默认路径（bf16 Linear + fp32 Sinkhorn）在 fp32 输入下自然退化为与 HF 一致的计算，
  作为测试用的 bitwise 锚点（见 `test_hc_pre_hc_post_matches_hf_hyper_connection`），不新增
  一套开关基础设施。
- dsv4 的 `xtuner/v1/ops/mhc.py`（TileKernels/Hopper TileLang 后端，721 行）**未移植**：
  没有对应硬件/依赖可验证正确性，盲移植风险大于收益。`XTUNER_USE_MHC_KERNELS=1` 时
  `hc_post` 显式抛 `NotImplementedError`（而不是静默指向不存在的模块），已记录为已知缺口，
  待真正需要 TileKernels 加速时再补。

**HF 数值 oracle**：`transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextHyperConnection`
+ `Glm5NextTextDecoderLayer.forward` 里 `hc_post` 的展开表达式。核对后确认与 dsv4 的数学完全一致
（GLM-5.3 的 mHC 与 DeepSeek-V4 用同一套 split-sinkhorn）。`test_glm53_mhc.py` 里 fp32 输入下
`hc_pre`/`hc_post` 与 HF 逐元素一致（atol=1e-4/1e-5），bf16 下在设计doc预期的 ULP 级误差内。

**`Glm53MoEDecoderLayer` 的设计要点**：只重写 `MoEDecoderLayer._pre_moe_forward` /
`_post_moe_forward` 这两个基类已经预留的接缝（attention+gate 阶段 / combine+residual 阶段），
完全不碰中间约 400 行的 EP/dispatcher/domino micro-batch 流水线。`residual` 在 mHC 模式下的类型从
基类的 `Tensor` 放宽成 `Tensor | _MHCResidual`（一个 `NamedTuple`，携带 4 流残差 + FFN 位点的
`post`/`comb`）——基类的 `_forward`/`_micro_batch_forward` 只是把 `_pre_moe_forward` 的返回值原样
透传给 `_post_moe_forward`，从不检查其类型，所以这个类型加宽是安全的协变扩展，用
`# type: ignore[override]` 显式标注而非静默绕过 mypy。

**排查记录（真实根因，非推测）**：为验证 `Glm53MoEDecoderLayer` 的 mHC 包裹逻辑，测试用 F3 的
真实 KDA 作为占位 attention（F5 的 NoPE-DSA 还没做，但 mHC 包裹逻辑与具体 attention 类型无关）。
首轮测试在 GPU 上出现间歇性 NaN，一度怀疑是 pytest 的 `typeguard` 插件干扰（`-p no:typeguard`
确实让单个失败用例转为通过），但用 `-p no:typeguard` 跑**完整**测试文件仍然失败，证明那只是
巧合（typeguard 改变了内存分配时序，偶然avoid 到了问题，不是真正的 fix）。最终用逐步插桩定位到
真实根因：**测试自己构造的 `MoEGate.weight` 与 `GroupedLinear.weight`（专家权重）是
`torch.empty` 未初始化**——真实模型通过顶层 `MoE.init_weights()` 才初始化这些参数，这条路径
在独立构造单个 decoder layer 的单测里从未被调用到。`torch.empty` 不消耗随机数流，所以两个独立
构造的 layer（即使用同一个 seed）在这些参数上拿到的是两块不同的未初始化显存内容——在
`test_mhc_cfg_none_matches_plain_moe_decoder_layer` 里表现为两个理论上应该完全一致的输出
100% 不匹配且含 NaN。修复：测试构造 layer 后显式对 `gate.weight` / `experts.fused_w1w3.weight`
/ `experts.fused_w2.weight` 做 `normal_(std=0.02)`。这是测试 fixture 的问题，不是 mHC 生产代码
的 bug——`hc_post_fused`（连同其 D=4096 的专项测试）、`hc_pre`/`hc_split_sinkhorn`（CPU 逐元素
对齐 HF）全程独立验证正确，未做任何生产代码改动。

## F5 NoPE DSA + KPool indexer + 限幅 SwiGLU

**状态**：未开始。

## F6 端到端训练与 MTP

**状态**：未开始。依赖 F1~F5 全部完成。
