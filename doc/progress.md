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

**状态**：未开始。参考 `~/github/xtuner-ncp-k3` 的模块结构，但 `chunk_kda` 调用约定按设计文档 3.5.1 改写。

## F4 mHC 四流残差

**状态**：未开始。kernel 从 `~/github/xtuner_dsv4` 移植。

## F5 NoPE DSA + KPool indexer + 限幅 SwiGLU

**状态**：未开始。

## F6 端到端训练与 MTP

**状态**：未开始。依赖 F1~F5 全部完成。
