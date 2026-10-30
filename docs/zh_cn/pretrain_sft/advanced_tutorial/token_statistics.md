# Token 长度与截断统计

按数据集名称分别统计 token 长度、tokenize 截断损失，以及 packing 和 collator 处理造成的损失与数据占比变化。

## 运行统计

在已安装 XTuner 的环境中，将 `train_config.py` 替换为你的训练配置路径：

```shell
python -m xtuner.tools.token_stats \
  --config train_config.py \
  --packing \
  --output-dir work_dirs/token_stats
```

配置需定义 `trainer = TrainerConfig(...)`，无需修改配置或先启动训练。工具沿用训练的数据、过滤、采样及 packing 参数，自动记录截断前后的长度。`--packing` 会额外构建并遍历打包数据；只关注 tokenize 截断时，去掉该参数即可。

tokenize 的 `max_length` 控制单条样本截断，`pack_max_length` 控制 packing 和 collator 的长度处理；模型上下文上限不会自动调整这两项配置。一个 pack 可以包含多条上下文独立的样本，其长度不代表单条对话长度。

例如 GLM5.2 SFT 使用 `--config examples/v1/config/sft_glm5p2.py`，沿用训练时的模型、数据路径及 `SAMPLE_MAX_LENGTH`、`PACK_MAX_LENGTH` 等环境变量。

## 查看结果

结果保存在 `work_dirs/token_stats/token_stats_<时间>_<随机后缀>/`。输出目录不存在时自动创建，每次运行生成新目录，不覆盖已有报告。

| 文件 | 内容 |
| --- | --- |
| `subset_token_stats.csv` | 每个子集一行：样本数、长度分布、截断损失；启用 `--packing` 后增加后续损失、token 占比和监督指标。 |
| `packing_token_stats.csv` | 启用 `--packing` 时生成：打包后有效长度、裁切、丢弃、label shift、padding 和全局监督指标。 |

| 指标 | 含义 |
| --- | --- |
| `count` | 配置模式（`--config`）：过滤、采样后的实例数，重复采样重复计数；缓存模式（`--cache-manifest`）：过滤、采样前的源记录数。 |
| `known_original_count` / `unknown_original_count` | 原始长度已知 / 未知的实例数，用于判断截断统计的覆盖范围。已知不代表没有截断。 |
| `original_mean_tokens` / `original_p99_tokens` | 原始长度已知的实例的平均长度 / P99。 |
| `truncated_sample_ratio` | tokenize 截断的实例数 ÷ 原始长度已知的实例数。 |
| `truncated_token_ratio` | tokenize 损失的 token 数 ÷ 已知原始长度的 token 总数。 |
| `packing_collator_loss_ratio` | packing/collator 裁切及丢弃的 token 数 ÷ packing 前的 token 数。 |
| `token_share_change` | 子集在 packing/collator 处理后的 token 占比减去处理前占比；`-0.05` 表示下降 5 个百分点。 |

单条 tokenize 截断损失是 `original_num_tokens - num_tokens`：前者为套用模板并 tokenize 后、首次截断前的长度，后者为首次截断后保留的长度。恰好达到长度上限不算截断。

原始长度缺失或为 `-1` 表示未知，不计入首次截断指标，也不能视为零损失；无已知样本或分母为零时，相应指标留空。`original_total_tokens` 只汇总已知原始长度；`retained_tokens_all_samples` 汇总所有样本首次截断后保留的 token。

## 支持范围与统计口径

支持标准文本 `DataloaderConfig` 和 `JsonlDataset`，以及现有 OpenAI（含 GLM5.2/Qwen 文本）、FTDP、标准预训练 tokenize 实现；不支持自定义 tokenize 实现、多模态、RL 或 LongText chunk。

Packing 支持文本 `none/soft/hard/__legacy` 和 `sft_llm_collator`。首次截断、后续裁切、整条丢弃分别记录，label shift 和 padding 不算截断损失。每个 pack 统计一次，位于分布式 sampler 之前；报告不代表某个训练 step 或多轮训练的累计消费。

CSV 的 `seed` 记录本次采样和构建 packs 所用的种子，标准训练配置使用 `trainer.seed`。seed 影响随机采样和打包顺序；`sample_ratio=1` 时首次截断统计不受影响，但 packing 结果仍可能变化。

## 高级用法

### 读取已有缓存

创建 `cache_manifest.json`，同一数据集 name 下的多个缓存目录放入同一个 `meta_dirs`：

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

路径相对于 manifest 所在目录。此模式只读缓存，不加载 tokenizer，也不能统计 packing，CSV 的 `seed` 留空。

| 模式 | 统计对象 |
| --- | --- |
| `--config` | 过滤、采样后的实例，重复采样重复计数。 |
| `--cache-manifest` | 过滤、采样前的源记录，`count` 表示源记录数。 |

`num_shards` 在配置模式下记录同名数据集实例数，在缓存模式下记录缓存目录数；分块大小不影响该值。

`--workers` 控制缓存读取并行度；配置模式的 tokenize 并行度由 `XTUNER_TOKENIZE_WORKERS` 控制。缓存按固定大小分块读取并累计，中位数和 P99 保持 NumPy 默认线性插值的精确口径。常见长度使用 NumPy 频次数组，单张数组至多 8 MiB；超长样本使用稀疏频次表，其内存随不同超长长度的数量增长。多个 worker 各自持有正在统计子集的频次表。

### 训练时记录长度

若要在训练的数据预处理阶段记录原始长度，可在原有 `tokenize_fn` 配置外加一层 `TokenStatsConfig`，模板和长度上限沿用原值：

```python
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.token_stats import TokenStatsConfig

tokenize_fn = TokenStatsConfig(
    OpenaiTokenizeFunctionConfig(chat_template="glm5.2", max_length=32768)
)
```

训练不会自动输出 CSV，仍需运行统计命令。匹配的统计缓存可直接复用；普通旧缓存缺少原始长度，首次使用配置模式统计时需重新采集。

### 自定义数据集入口

配置可定义 `build_stats_inputs(work_dir)`，返回 `{"datasets": datasets}`。启用 `--packing` 时还需返回 `dataloader_config`，工具会按该配置统一构建 packs。返回 `dataloader_config` 本身不会开启 packing。

自定义入口应提供已完成过滤、采样的文本数据集；如需截断统计，构建时使用 `TokenStatsConfig` 记录原始长度。该入口使用配置顶层的 `seed`，默认值为 42。

### 进程内调用

Python 配置顶层代码会执行，需要准备其依赖的环境和路径。统计入口临时设置 CPU 采样种子，正常返回或异常退出后恢复调用方的 Python、NumPy 和 PyTorch CPU 随机状态。工具不会重设 CUDA 随机种子、重配训练日志或修改进度条环境变量；摘要输出到标准错误流。自定义配置自身产生的其他副作用仍由配置负责。

`summarize_samples` 消费已切分的数据块，文件数由读取层记录。`summarize_packing` 返回 `(全局统计, 各子集 packing 统计)`，各子集按名称排序；调用方按 `name` 合并样本统计。`save_reports` 写入调用方提供的最终行，`seed` 等元数据需预先放入行中。

### 迁移已有脚本

自定义入口不再接收 `packed_dataset`。删除该返回项，提供 `datasets` 和 `dataloader_config`，并用 `--packing` 开启打包统计。旧入口会报出迁移提示，避免默默使用不同的 packs。

CSV 移除重复字段，读取脚本可按下表调整：

| 原字段 | 替代字段 |
| --- | --- |
| `complete_count` | `known_original_count` |
| `incomplete_count` | `unknown_original_count` |
| `complete_original_tokens` | `original_total_tokens`；没有已知原始长度时留空，不再以 0 表示。 |

直接调用 Python 函数的代码需接收 `summarize_packing` 返回的两项结果，不再传入 `subset_rows`；`save_reports` 不再接收独立的 `seed` 参数。

## Packing 指标详解

| 指标 | 含义 |
| --- | --- |
| `token_share_before_packing` / `token_share_after_packing` | 子集在 packing 前、packing/collator 处理后的 token 占比，后者尚未扣除 label shift。 |
| `effective_token_share` | 扣除 label shift 后的有效输入 token 占比，不含 padding。 |
| `supervised_tokens_after_collator` | packing 切片、collator 裁切及 label shift 后，非忽略目标 label 的数量；分别给出子集和全局总量。 |
| `supervised_token_share` | 子集监督 token 数 ÷ 全局监督 token 数。 |
| `supervision_ratio` | 全局监督 token 数 ÷ 全局有效输入 token 数，不含 padding；仅在 packing CSV 中提供。 |
| `zero_supervision_pack_count` | 所有目标 labels 都被忽略的 pack 数；仅统计数量，不改变训练过滤规则。 |
| `padding_ratio` | padding token 数 ÷（有效输入 token 数 + padding token 数）。 |

监督统计复用 packing/collator 汇总时取得的 labels，仅统计非忽略训练目标。子集归属按目标 label 的来源计算；label shift 后，目标来源和有效输入来源在边界上可能不同，因此监督比例只在全局提供。有效输入非零但全部 labels 被忽略时，监督比例为 0，子集监督份额留空；缓存模式和仅统计首次截断时不输出监督指标。
