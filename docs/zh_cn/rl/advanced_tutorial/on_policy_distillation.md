# On-Policy Distillation

本文介绍 XTuner 当前的 On-Policy Distillation（OPD）能力，包括支持的蒸馏目标、如何与 policy gradient（PG）组合、如何配置训练目标权重和多 Teacher 路由，以及如何选择 Rollout Teacher 或 Training-side Teacher。

OPD 使用当前 Student 策略生成的序列作为训练数据，再让 Teacher 对这些序列提供 token-level 目标：

```text
Student rollout
  -> Teacher 计算 sampled-token 或 Top-K 目标
  -> 目标与训练 labels 对齐
  -> PG-OPD 或直接蒸馏
  -> 更新 Student
```

本文中的 MOPD 指 Multi-Teacher OPD：不同数据源可以路由到不同 Teacher。它不表示一定混合 task reward；是否使用 task reward 由 `task_adv_weight` 决定。

## 快速选择

先选择蒸馏目标，再选择 Teacher 的执行位置。这两个维度可以独立组合。

### 蒸馏目标

| 目标 | `loss_mode` | `use_policy_gradient` | `top_k` | 优化方式 |
| --- | --- | ---: | ---: | --- |
| sampled-token | `k1` | `True` | 不设置 | 蒸馏信号合入 advantage，再计算 PG loss |
| Teacher Top-K | `forward` | `False` | 必填 | Top-K forward KL（含 tail mass）直接反传 |
| Teacher Top-K | `reverse` | `False` | 必填 | Top-K reverse KL（含 tail mass）直接反传 |
| Teacher Top-K | `forward_kl_topk` | `False` | 必填 | Teacher Top-K 上的截断 forward KL 直接反传 |

sampled-token 当前只支持 `k1 + PG`。`k1` 不能走直接反传；三个 Top-K mode 不能走 PG。

### Teacher 执行位置

| Teacher runtime | sampled-token | Top-K | Teacher 如何运行 |
| --- | ---: | ---: | --- |
| Rollout Teacher + LMDeploy | 支持 | 支持 | 独立 HTTP 服务 |
| Training-side Teacher | 支持 | 支持 | Training Worker 内的 frozen model |

使用 Rollout Teacher 时，可以让 XTuner 的启动脚本同时启动 Teacher 服务和 Student 训练，也可以直接传入已经运行的 endpoint。使用 Training-side Teacher 时不需要 Teacher endpoint，Teacher checkpoint 由训练引擎加载。

```{note}
同一个 `DistillationConfig` 中的所有 Teacher 必须使用同一种 runtime：只能全部是 `RolloutTeacherConfig`，或者全部是 `TrainTeacherConfig`，不能混用。
```

## 核心配置

OPD 由三个配置层组成：

- `DistillationLossConfig`：定义蒸馏目标、PG loss 和目标权重。
- `RolloutTeacherConfig` 或 `TrainTeacherConfig`：定义 Teacher 在哪里执行。
- `DistillationConfig`：组合 loss、Teacher 列表和数据源路由。

同一个 `DistillationLossConfig` 需要同时传给 Training Worker 和 `DistillationConfig`：

```python
from xtuner.v1.rl.distillation import DistillationConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.trainer import WorkerConfig

loss_cfg = DistillationLossConfig(...)

train_worker_cfg = WorkerConfig(
    ...,
    loss_cfg=loss_cfg,
)

distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[...],
    data_source_teacher_map={...},
)
```

Trainer 会检查 `train_worker_cfg.loss_cfg == distillation_config.loss_config`。最后把 `distillation_config` 传给 `RLColocateTrainerConfig` 或相应的 RL Trainer 配置：

```python
trainer = RLColocateTrainerConfig(
    ...,
    train_worker_cfg=train_worker_cfg,
    distillation_config=distillation_config,
)
```

`policy_loss_cfg`、`use_kl_loss`、rollout importance sampling、`mode` 和 `chunk_size` 等公共 RL loss 配置仍然有效。公共字段可参考[损失函数](loss.md)。其中：

- `loss_mode` 控制 OPD 目标。
- `use_kl_loss`、`kl_loss_coef` 和 `kl_loss_type` 控制相对于 Reference Model 的普通 KL penalty。
- OPD Teacher 和 Reference Model 是两个不同角色；开启 OPD 不要求开启 Reference KL。

## Sampled-token PG-OPD

sampled-token OPD 不让 Teacher 重新选 token。Student 先生成 token，Teacher 只计算这些 token 的 logprob。

```text
Student rollout
  -> Student sampled tokens
  -> Teacher 对相同 token 计算 logprob
  -> 对齐到 shifted labels
  -> 构造 k1 蒸馏信号
  -> 合入 task advantage
  -> Policy gradient loss
```

它的训练目标契约是：

```text
teacher_logprobs: [B, T]
target_token_ids: None
```

令每个有效 token 上的旧 Student logprob 为 $\ell_{old}$，Teacher logprob 为 $\ell_T$，则 `k1` 蒸馏信号为：

$$
d_{OPD} = \ell_{old} - \ell_T
$$

XTuner 将它 detach 后与 task advantage 组合：

$$
A_{combined}
= w_{task} A_{task}
- w_{distill} d_{OPD}
$$

其中：

- $w_{task}$ 对应 `task_adv_weight`。
- $w_{distill}$ 对应 `distillation_loss_weight`。
- 最终梯度由 `policy_loss_cfg` 指定的 policy loss 产生。

最小 loss 配置如下：

```python
from xtuner.v1.rl.loss import DistillationLossConfig

loss_cfg = DistillationLossConfig(
    policy_loss_cfg={
        "loss_type": "vanilla",
        "cliprange_low": 0.2,
        "cliprange_high": 0.28,
        "clip_ratio_c": 10.0,
        "log_prob_diff_min": -20.0,
        "log_prob_diff_max": 20.0,
    },
    loss_mode="k1",
    use_policy_gradient=True,
    task_adv_weight=0.0,
    distillation_loss_weight=1.0,
    use_kl_loss=False,
    mode="chunk",
    chunk_size=512,
)
```

上例是纯 OPD：`task_adv_weight=0.0`，因此训练样本可以没有 task reward。如果要同时使用 task reward，将它设为正数，例如 `1.0`。

## Teacher Top-K OPD

Top-K OPD 由 Teacher 在每个训练位置选择 K 个 token，并同时提供这些 token 的 id 和 logprob。Student 在同一组 token id 上计算自己的 logprob，再计算可微的蒸馏 loss。

```text
Teacher forward
  -> Teacher Top-K token ids + logprobs
  -> Student 在相同 token ids 上 gather logprobs
  -> Top-K KL
  -> 直接反向传播
```

它的训练目标契约是：

```text
target_token_ids: [B, T, K]
teacher_logprobs: [B, T, K]
```

设 Teacher 选择的 Top-K 集合为 $S_T$，Teacher 和 Student 在 token $i$ 上的概率分别为 $q_i$ 和 $p_i$。

### `forward_kl_topk`

`forward_kl_topk` 只对 Teacher 选出的 K 个 token 求截断 forward KL 项：

$$
L_{topk} = \sum_{i \in S_T} q_i(\log q_i - \log p_i)
$$

它不把 Top-K 之外的概率质量合并成额外 tail bin。`log_prob_min_clamp` 可以在计算前限制 Student/Teacher logprob 下界，`loss_max_clamp` 可以限制逐 token loss 的范围。

### `forward`

`forward` 在 Teacher Top-K 项之外，把剩余概率质量作为一个 tail bin：

$$
q_{tail}=1-\sum_{i\in S_T}q_i,\qquad
p_{tail}=1-\sum_{i\in S_T}p_i
$$

$$
L_{forward}
= \sum_{i\in S_T}q_i(\log q_i-\log p_i)
+q_{tail}(\log q_{tail}-\log p_{tail})
$$

### `reverse`

`reverse` 使用 Student 概率对相反方向加权，并同样保留 tail bin：

$$
L_{reverse}
= \sum_{i\in S_T}p_i(\log p_i-\log q_i)
+p_{tail}(\log p_{tail}-\log q_{tail})
$$

推荐从与常见 Teacher Top-K 训练语义一致的 `forward_kl_topk` 开始：

```python
loss_cfg = DistillationLossConfig(
    policy_loss_cfg={
        "loss_type": "vanilla",
        "cliprange_low": 0.2,
        "cliprange_high": 0.28,
    },
    loss_mode="forward_kl_topk",
    use_policy_gradient=False,
    top_k=64,
    log_prob_min_clamp=-10.0,
    loss_max_clamp=10.0,
    task_adv_weight=0.0,
    distillation_loss_weight=1.0,
    use_kl_loss=False,
    mode="chunk",
    chunk_size=512,
)
```

Top-K 模式不把蒸馏项放进 advantage，而是计算：

$$
L_{total}
= L_{PG}(w_{task}A_{task})
+w_{distill}L_{topk}
$$

因此，即使 `use_policy_gradient=False`，`policy_loss_cfg` 仍用于可选的 task loss；它只表示 OPD 项本身使用直接反传。

## 配置 Teacher

### 让 XTuner 同时启动 Rollout Teacher

使用 `RolloutTeacherConfig.launch_config` 描述需要启动的 Teacher 服务：

```python
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherConfig,
    RolloutTeacherLaunchConfig,
)

distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(
            name="teacher",
            num_replicas=1,
            enable_prefix_caching=True,
            launch_config=RolloutTeacherLaunchConfig(
                model_path=teacher_model_path,
                num_workers=1,
                server_port=13141,
                dtype="bfloat16",
                tensor_parallel_size=1,
                expert_parallel_size=1,
                context_length=max_prompt_length + max_response_length,
                max_batch_size=2048,
                gpu_memory_utilization=0.8,
            ),
        )
    ],
    data_source_teacher_map={"math": "teacher"},
)
```

仓库中的 OPD 启动工具会：

1. 读取 `launch_config`。
2. 为各个 Teacher replica 分配 GPU 和端口。
3. 保留剩余 GPU 给 Student workers。
4. 启动 LMDeploy Teacher 服务并等待健康检查。
5. 通过 `XTUNER_OPD_TEACHER_ENDPOINTS_JSON` 把实际 endpoint map 传给 Trainer。
6. 启动 Student 训练，并在退出时关闭由脚本启动的 Teacher。

每个 Teacher replica 必须完整放在一个节点上。启动工具按 `num_workers` 为 replica 分配卡，并拒绝“Teacher 占满所有 GPU、没有 Student GPU”的配置。`num_workers`、TP、EP 和后端的并行拓扑应保持一致。

使用 LMDeploy Rollout Teacher 时设置：

```bash
export XTUNER_USE_LMDEPLOY=1
```

sampled-token 和 Top-K 都支持 LMDeploy Rollout Teacher。

### 使用已经启动好的 endpoint

如果 Teacher 服务由外部系统管理，不设置 `launch_config`，直接填写 `endpoints`：

```python
distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(
            name="teacher",
            num_replicas=2,
            endpoints=[
                "http://teacher-0:8000",
                "http://teacher-1:8000",
            ],
            api_key=None,
            request_timeout_s=1200.0,
            max_retry_per_sample=2,
            max_concurrency=128,
        )
    ],
    data_source_teacher_map={"math": "teacher"},
)
```

此时 XTuner 不启动或关闭外部服务，只执行健康检查、请求路由、并发限制、失败重试和响应解析。`endpoints` 数量应与 `num_replicas` 一致，每个 endpoint 填写服务根地址，不包含 `/generate`。

同一个 Teacher 的多个 replica 会根据 Teacher 名称、数据源和 rollout group 稳定路由；某个 replica 请求失败时，客户端会按照 `max_retry_per_sample` 尝试后续 replica。

### 由训练引擎加载 Teacher

Training-side Teacher 使用 `TrainTeacherConfig`：

```python
from pathlib import Path

from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.distillation import DistillationConfig, TrainTeacherConfig

teacher_model_cfg = get_model_config_from_hf(Path(teacher_model_path))
teacher_model_cfg.compile_cfg = False

distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        TrainTeacherConfig(
            name="teacher",
            model_path=teacher_model_path,
            model_cfg=teacher_model_cfg,
            # 不传 fsdp_cfg 时，XTuner 会创建 requires_grad=False 的默认配置。
            fsdp_cfg=None,
        )
    ],
    data_source_teacher_map={"math": "teacher"},
)
```

`model_cfg` 必须与 Teacher checkpoint 匹配，并建议使用独立于 Student 的配置对象。Teacher 与 Student 可以有不同层数或 hidden size，但需要使用兼容的 token id 语义；当前配置校验要求二者 vocabulary size 相同。

Training Worker 初始化时会构建 frozen Teacher 并将其留在 CPU。每个训练阶段的设备切换顺序是：

```text
Actor 和 optimizer 在 GPU
  -> Actor 和 optimizer offload 到 CPU
  -> Teacher onload 到训练设备
  -> Teacher 计算 targets
  -> Teacher offload 到 CPU
  -> 恢复 Actor 和 optimizer
  -> 计算 old logprobs 和训练 loss
```

Training-side Teacher 不创建 HTTP 服务，也不需要 `XTUNER_OPD_TEACHER_ENDPOINTS_JSON`。当前支持 CPU offload，不支持 Teacher disk offload。

## MOPD 与数据源路由

`data_source_teacher_map` 根据训练样本的 `origin_data_source` 选择 Teacher：

```python
distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(name="math_teacher", ...),
        RolloutTeacherConfig(name="code_teacher", ...),
    ],
    data_source_teacher_map={
        "math": "math_teacher",
        "code": "code_teacher",
    },
)
```

Teacher 名称必须唯一，映射中的 Teacher 必须存在。每个参与训练的数据源都应有对应映射，否则 Teacher 路由阶段无法生成该数据所需的蒸馏目标。

### Rollout Teacher 的 MOPD 行为

Rollout 路径在发起 HTTP 请求前完成路由：

```text
origin_data_source
  -> teacher_name
  -> 对应 RolloutTeacherClient
  -> 只请求这个 Teacher
```

一条数据不会请求所有 Rollout Teacher。

### Training-side Teacher 的 MOPD 行为

训练引擎首先把每条数据的 `origin_data_source` 转换为 `teacher_indices`。当前实现采用 Teacher-major 的 FSDP 调度：

```text
for each Teacher:
  onload Teacher
  for each local pack:
    Teacher 对整个 pack 执行 forward
    根据 teacher_indices 选择属于当前 Teacher 的 token
    只把这些位置写入最终 targets
  offload Teacher
```

因此，Training-side MOPD 当前是“所有 Teacher 依次对所有本地 pack 做完整 forward，再按 token 选择结果”，而不是“每条数据只执行对应 Teacher”。sampled-token 和 Top-K 都遵循这一调度。

这样做是为了保证所有 FSDP rank 按相同的 Teacher、pack 顺序进入 collective；即使某个 rank 没有属于某个 Teacher 的 token，也不能单独跳过该 Teacher forward，否则各 rank 可能进入不同 collective 序列。

它的直接影响是：Teacher 数量增加时，Teacher forward 计算量也会增加；但 Teacher 会逐个 onload/offload，不会同时驻留训练 GPU。

## 训练目标权重

OPD 使用 `task_adv_weight` 和 `distillation_loss_weight` 控制 task reward 与蒸馏目标的相对贡献。

### `task_adv_weight`

它缩放 task reward 产生的 advantage：

- `0.0`：纯蒸馏，训练不依赖 task reward。
- `1.0`：完整使用 task advantage。
- 其他非负数：按比例调整 task 目标的贡献。

### `distillation_loss_weight`

它缩放 OPD 目标：

- sampled-token：缩放合入 advantage 的 `k1` 蒸馏信号。
- Top-K：缩放直接反传的 Top-K distillation loss。

常见组合如下：

| 训练目标 | `task_adv_weight` | `distillation_loss_weight` |
| --- | ---: | ---: |
| 纯 OPD | `0.0` | `1.0` |
| Task reward 与 OPD 等权 | `1.0` | `1.0` |
| Task reward 为主、OPD 为辅助 | `1.0` | `0.1` |

这些数值不是通用最优值。调整时应同时观察 reward、蒸馏指标、policy ratio/clip 指标和训练稳定性。

## 启动示例

仓库提供 sampled-token OPD 和 MOPD recipe：

- `recipe/on_policy_distillation/config/rl_dapo_math_opd.py`
- `recipe/on_policy_distillation/config/rl_dapo_math_mopd.py`
- `recipe/on_policy_distillation/scripts/run_sampled_token_opd.sh`
- `recipe/on_policy_distillation/scripts/run_sampled_token_mopd.sh`

### 单 Teacher sampled-token OPD

```bash
export XTUNER_USE_LMDEPLOY=1
export WORK_DIR=work_dirs/dapo_math_opd

bash recipe/on_policy_distillation/scripts/run_sampled_token_opd.sh \
    /path/to/student \
    /path/to/teacher \
    /path/to/train.jsonl
```

### 多 Teacher sampled-token MOPD

```bash
export XTUNER_USE_LMDEPLOY=1
export GSM8K_TEACHER_MODEL_PATH=/path/to/gsm8k_teacher
export GEO3K_TEACHER_MODEL_PATH=/path/to/geo3k_teacher
export WORK_DIR=work_dirs/dapo_math_mopd

bash recipe/on_policy_distillation/scripts/run_sampled_token_mopd.sh \
    /path/to/student \
    /path/to/dataset_meta.json
```

如果配置使用外部 endpoints，上述启动工具不会创建对应 Teacher 进程，但仍会等待 endpoint 健康检查并把 endpoint map 传给 Trainer。

Training-side Teacher 不需要 `start_teacher_servers`。在普通 RL 配置中加入 `TrainTeacherConfig` 后，使用项目标准 RL 启动方式即可；Teacher 会随 Training Worker 初始化。

## 指标

sampled-token 常用指标包括：

- `opd_reverse_kl`：有效 token 上的平均 `old_student_logprob - teacher_logprob`。
- `opd_abs_logprob_loss`：两者 logprob 差值的平均绝对值。
- `reduced_distillation_kl`、`reduced_distillation_abs_loss`：统一蒸馏指标。
- `teacher_score_time_s`：Rollout Teacher 对单条状态评分的耗时，保存在 rollout 状态的额外字段中。

Top-K 常用指标包括：

- `reduced_topk_opd_loss`：当前 batch 的加权 Top-K 蒸馏 loss。
- `reduced_topk_opd_kl`：有效 token 上的平均 Top-K KL。
- `reduced_topk_opd_student_selected_mass`：Student 在 Teacher Top-K support 上的概率质量。
- `reduced_topk_opd_teacher_selected_mass`：Teacher Top-K 概率质量。
- `reduced_topk_opd_overlap_fraction`：Student Top-K 与 Teacher Top-K 的平均重叠比例。

Training-side Teacher 还会记录：

- `time/train_teacher_compute`
- `time/train_teacher_onload`
- `time/train_teacher_offload`

最终蒸馏指标会写入 `distillation/` 前缀下；不同 worker 的原始 step 平均值也会写入 `train_metrics/worker_<rank>/`。

## 当前限制

- sampled-token 只支持 `k1 + Policy Gradient`。
- Rollout Top-K 只支持 LMDeploy。
- 一个 `DistillationConfig` 不能混用 Rollout Teacher 和 Training-side Teacher。
- Training-side MOPD 会让所有 Teacher 依次 forward 所有本地 pack。
- Training-side Teacher 支持 CPU offload，但不支持 disk offload。
- 当前 Top-K 是 Teacher-selected Top-K，不是 full-vocabulary distillation。
- Reference Model 和 OPD Teacher 是不同角色；即使 checkpoint 相同，配置语义也不相同。
