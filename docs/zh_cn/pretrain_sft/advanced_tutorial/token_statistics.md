# Token 长度与截断统计

统计各数据子集的 token 长度，以及 tokenize、packing/collator 带来的损失和数据占比变化。

## 生成统计（推荐）

在已安装 XTuner 的环境中，将 `train_config.py` 替换为你的训练配置路径，执行以下命令即可。配置文件需定义 `trainer = TrainerConfig(...)`，无需修改配置或先启动训练：

```shell
python -m xtuner.tools.token_stats \
  --config train_config.py \
  --packing \
  --output-dir work_dirs/token_stats
```

例如 GLM5.2 SFT 使用 `--config examples/v1/config/sft_glm5p2.py`，沿用训练时的模型、数据路径及 `SAMPLE_MAX_LENGTH`、`PACK_MAX_LENGTH` 等环境变量。结果保存在 `work_dirs/token_stats/token_stats_<时间>_<随机后缀>/`。

工具自动包装 tokenize 配置，沿用训练的数据、过滤、采样、packing 参数及 `trainer.seed`，不启动模型训练。`--packing` 会额外构建并遍历打包数据；去掉该参数，仅统计首次 tokenize 截断。命令以单进程启动；Python 配置顶层代码仍会执行，需要准备其依赖的环境和路径。

支持标准文本 `DataloaderConfig` 和 `JsonlDataset`，以及现有 OpenAI（含 GLM5.2/Qwen 文本）、FTDP、标准预训练 tokenize 实现；不支持自定义 tokenize 实现、多模态、RL 或 LongText chunk。

也支持自定义 `build_stats_inputs(work_dir)` 入口，返回 `datasets`；统计 packing 时还需返回 `dataloader_config`，可同时提供已有的 `packed_dataset`。
提供已有的 `packed_dataset` 时，必须复用 `datasets` 中的同一组数据集实例，且每个实例的出现次数一致。

## 训练时记录长度（可选）

若要在训练的数据预处理阶段记录原始长度，可在数据集的原有 `tokenize_fn` 配置外加一层 `TokenStatsConfig`。将下例中的配置赋给对应数据集的 `tokenize_fn`，模板和长度上限沿用原值：

```python
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.token_stats import TokenStatsConfig

tokenize_fn = TokenStatsConfig(
    OpenaiTokenizeFunctionConfig(chat_template="glm5.2", max_length=32768)
)
```

训练不会自动输出 CSV，仍需运行统计命令。匹配的统计缓存可直接复用；普通旧缓存缺少原始长度，首次统计需重新采集。

## 读取已有缓存（可选）

创建 `cache_manifest.json`，同一子集的多个缓存目录放入同一个 `meta_dirs`：

```json
[
  {"name": "coding-swe", "meta_dirs": ["cache/file_hash/tokenize_hash/jsonl_meta"]}
]
```

```shell
python -m xtuner.tools.token_stats \
  --cache-manifest cache_manifest.json \
  --output-dir work_dirs/token_stats
```

路径相对于 manifest 所在目录。此模式只读缓存，不加载 tokenizer，也不能统计 packing。`--workers` 控制缓存读取并行度；tokenize 并行度由 `XTUNER_TOKENIZE_WORKERS` 控制。

## 查看结果

每次生成新的结果目录，不覆盖已有报告：

| 文件 | 内容 |
| --- | --- |
| `subset_token_stats.csv` | 每个子集一行：样本数、原始长度分布、首次截断数量及比例；启用 `--packing` 后增加各子集后续损失、输入 token 占比变化及监督 token 数量和份额。 |
| `packing_token_stats.csv` | 启用 `--packing` 时生成：打包后有效长度分布、裁切、整条丢弃、label shift、padding 和全局监督指标。 |

| 模式 | 统计对象 |
| --- | --- |
| `--config` | 过滤、采样后的实例，重复采样重复计数；packing 指标再按训练配置打包后统计。 |
| `--cache-manifest` | 过滤、采样前的源记录。 |

seed 影响随机采样和打包顺序；`sample_ratio=1` 时首次截断统计不受影响，但 packing 结果仍可能变化。

| 指标 | 含义 |
| --- | --- |
| `original_num_tokens` | 套用模板并 tokenize 后、首次截断前的完整长度。 |
| `num_tokens` | 首次截断后保留的长度，即 `len(input_ids)`。 |
| 单条截断损失 | `original_num_tokens - num_tokens`。 |
| 截断样本比例 | 损失大于零的样本数 ÷ 原始长度已知的样本数。 |
| 截断 token 比例 | 损失 token 总数 ÷ 对应原始 token 总数。 |
| `packing_collator_loss_ratio` | packing/collator 裁切及丢弃的 token 数 ÷ 首次截断后、packing 前的 token 数。 |
| `token_share_before_packing` / `token_share_after_packing` | 子集在 packing 前、packing/collator 处理后的 token 占比，后者尚未扣除 label shift。 |
| `token_share_change` | 后占比减前占比；例如 `-0.05` 表示下降 5 个百分点。 |
| `effective_token_share` | 扣除 label shift 后的有效输入 token 占比，不含 padding。 |
| `supervised_tokens_after_collator` | packing 切片、collator 裁切及 label shift 后，非忽略目标 label 的数量；分别给出子集和全局总量。 |
| `supervised_token_share` | 子集监督 token 数 ÷ 全局监督 token 数。 |
| `supervision_ratio` | 全局监督 token 数 ÷ 全局有效输入 token 数，不含 padding；仅在 packing CSV 中提供。 |
| `zero_supervision_pack_count` | 所有目标 labels 都被忽略的 pack 数；仅统计数量，不改变训练过滤规则。 |
| `padding_ratio` | padding token 数 ÷（有效输入 token 数 + padding token 数），表示实际 padding 比例。 |

恰好达到长度上限不算截断。原始长度缺失或为 `-1` 表示未知，不计入首次截断指标，也不能视为零损失；无已知样本或分母为零时，相应指标留空。

监督统计复用 packing/collator 汇总时取得的 labels，仅统计非忽略训练目标，不代表加权 loss 或梯度贡献。子集归属按目标 label 的来源计算；label shift 后，目标来源和有效输入来源在边界上可能不同，因此监督比例只在全局提供。有效输入非零但全部 labels 被忽略时，监督比例为 0，子集监督份额留空；cache-only 和仅统计首次截断时不输出监督指标。

Packing 支持文本 `none/soft/hard/__legacy` 和 `sft_llm_collator`。首次截断、后续裁切、整条丢弃分别记录，label shift 和 padding 不算截断损失。每个 pack 统计一次，位于分布式 sampler 之前；报告不代表某个训练 step 或多轮训练的累计消费。
