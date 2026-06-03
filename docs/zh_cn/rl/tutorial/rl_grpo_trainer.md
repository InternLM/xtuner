```{important}
XTuner 的 RL（强化学习）功能目前为 Beta 版本，RL 功能特性持续完善中，欢迎试用并反馈问题。
```

# [Beta] 使用 Python 配置自定义 GRPO 训练

在之前的[快速开始](../../get_started/grpo.md)中，我们已经通过脚本启动了一次 GRPO 训练。本教程只介绍**共卡模式**下如何通过 Python 配置文件自定义一个 RL 训练。

共卡模式使用 {class}`~xtuner.v1.train.rl_trainer.RLColocateTrainerConfig`：训练 worker 和 rollout worker 使用同一组加速卡，训练流程会在 rollout 生成和模型训练之间切换资源，并在训练后把权重同步回 rollout worker。

和快速开始中的配置逻辑一致，一个 GRPO 训练可以分成两部分：

- **Generation Config**：定义如何采样 prompt、调用推理后端生成 response、用 judger 打分，并把完成的 rollout 组写入 replay buffer。
- **Trainer Config**：定义模型训练 worker、优化器、loss、资源、训练步数、评估和保存策略，并把 generation 侧的配置组装进 `RLColocateTrainerConfig`。

CLI 的入口是 `xtuner/v1/train/cli/rl.py`。它会读取配置文件中的 `trainer`，然后执行：

```python
trainer = cfg.trainer.build()
trainer.fit()
```

## 1. Generation Config

Generation 侧负责生产 RL 训练数据。对单任务 GRPO 来说，主要包含 `RolloutConfig`、`JudgerConfig`、`SamplerConfig`、`SingleTurnAgentLoopConfig`、`AgentLoopManagerConfig`、`SyncReplayBufferConfig` 和 `EvaluatorConfig`。

### 1.1 RolloutConfig

{class}`~xtuner.v1.rl.rollout.worker.RolloutConfig` 描述 rollout worker 如何启动推理后端。具体使用 SGLang、LMDeploy 还是 vLLM，由启动脚本设置 `XTUNER_USE_SGLANG`、`XTUNER_USE_LMDEPLOY` 或 `XTUNER_USE_VLLM`。

```{code-block} python
:caption: 配置 rollout worker
from xtuner.v1.rl.rollout.worker import RolloutConfig

experiment_name = "grpo_gsm8k"
model_path = "/path/to/Qwen3-8B"
max_prompt_length = 512
max_response_length = 1024

rollout_config = RolloutConfig(
    env=experiment_name,
    device="GPU",
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=1,
    expert_parallel_size=1,
    gpu_memory_utilization=0.8,
    context_length=max_prompt_length + max_response_length,
)
```

常用自定义项：

- `model_path`：rollout 使用的 HF 模型路径。
- `tensor_parallel_size` / `expert_parallel_size`：推理并行度。
- `gpu_memory_utilization`：推理后端可使用的显存比例。
- `context_length`：prompt 和 response 的最大总长度。

### 1.2 Dataset、TokenizeFn 和 SamplerConfig

训练数据通过 `DatasetConfig` 和 `RLTextTokenizeFnConfig` 读入。`RLTextTokenizeFnConfig` 会把 JSONL 中的 `prompt` 转为 `RolloutState.message` 和 `RolloutState.prompt_ids`。

`SamplerConfig(prompt_repeat_k=...)` 会控制每个 prompt 采样多少条 response。GRPO 需要同一个 prompt 的多条 response 组成一组来计算优势，因此训练时通常让 `prompt_repeat_k > 1`。

```{code-block} python
:caption: 配置训练数据采样
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.rl.agent_loop_manager import SamplerConfig

data_path = "/path/to/train.jsonl"
pack_max_length = 32 * 1024
prompt_repeat_k = 5

tokenize_fn = RLTextTokenizeFnConfig(max_length=max_prompt_length)
train_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[
        {
            "dataset": DatasetConfig(name=experiment_name, anno_path=data_path),
            "tokenize_fn": tokenize_fn,
        }
    ],
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
train_sampler_config = SamplerConfig(
    dataloader_cfg=train_dataloader_cfg,
    prompt_repeat_k=prompt_repeat_k,
)
```

文本 RL JSONL 一般需要包含：

- `prompt`：对话消息列表。
- `data_source`：样本来源，用于选择或映射 judger。
- `reward_model`：judger 需要的参考答案或打分信息。

### 1.3 JudgerConfig

Judger 负责给 rollout response 写入 reward。GSM8K 可以直接使用 {class}`~xtuner.v1.rl.judger.GSM8KJudgerConfig`。

```{code-block} python
:caption: 配置 GSM8K judger
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.utils import CPUResourcesConfig

judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(
        num_workers=1,
        num_cpus_per_worker=1,
    ),
)
```

`judger_name` 是该 Judger 的逻辑名称。使用单个 `JudgerConfig` 时，样本会直接交给这个 Judger 打分；使用 `ComposedJudgerConfig` 时，`RolloutState.data_source` 会用于选择 branch，字符串值或字典 key 需要和 `branches` 中的名称对应。`cpu_resources` 表示该 Judger 使用 PG 外 Ray CPU actor 执行打分；如果不配置 `cpu_resources`，Judger 会在本地执行。打分后，训练流程会从 `RolloutState.reward["score"]` 读取分数并计算 advantage。

### 1.4 AgentLoopConfig 和 AgentLoopManagerConfig

`SingleTurnAgentLoopConfig` 定义单轮生成行为，`SampleParams` 控制采样参数。`AgentLoopManagerConfig` 把 sampler、agent loop、judger 和 produce strategy 组装成一个训练任务。

```{code-block} python
:caption: 配置训练 rollout 任务
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)

train_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=max_response_length,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    ),
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=train_agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=train_sampler_config,
    )
)
```

在共卡同步 GRPO 中，常用 `SyncProduceStrategyConfig`。每个训练 step 会先生成足够的 rollout group，再切换到训练 worker 消费这些数据。

### 1.5 Eval AgentLoop 和 EvaluatorConfig

验证侧也使用 `AgentLoopManagerConfig`，但通常设置 `prompt_repeat_k=1` 和 `temperature=0.0`，让评估结果更稳定。

```{code-block} python
:caption: 配置验证 rollout 和 evaluator
from xtuner.v1.rl.evaluator import EvaluatorConfig

eval_data_path = "/path/to/eval.jsonl"

eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[
        {
            "dataset": DatasetConfig(name=experiment_name, anno_path=eval_data_path, sample_ratio=1.0),
            "tokenize_fn": tokenize_fn,
        }
    ],
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
eval_sampler_config = SamplerConfig(
    dataloader_cfg=eval_dataloader_cfg,
    prompt_repeat_k=1,
)
eval_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=max_response_length,
        top_k=1,
        top_p=1.0,
        temperature=0.0,
        min_tokens=0,
    ),
)
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="eval_task",
        agent_loop_config=eval_agent_loop_config,
        judger_config=judger_config,
        sampler_config=eval_sampler_config,
    )
)

evaluator_config = EvaluatorConfig(compute_metric_func=None)
```

`EvaluatorConfig(compute_metric_func=None)` 会使用默认 accuracy 计算逻辑：统计 `reward["score"] > 0` 的样本比例。

### 1.6 ReplayBufferConfig

共卡同步 GRPO 使用 {class}`~xtuner.v1.rl.replay_buffer.SyncReplayBufferConfig` 即可。它会按 FIFO 策略提供已完成的 rollout group。

```{code-block} python
:caption: 配置 replay buffer
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig

replay_buffer_config = SyncReplayBufferConfig()
```

## 2. Trainer Config

Trainer 侧负责模型训练和资源编排。共卡模式下，核心是 `AcceleratorResourcesConfig`、`WorkerConfig`、`GRPOLossConfig`、`GRPOAdvantageConfig` 和 `RLColocateTrainerConfig`。

### 2.1 AcceleratorResourcesConfig

资源配置会用于创建共卡 placement group。`num_workers` 通常等于参与训练的加速卡数量。

```{code-block} python
:caption: 配置共卡资源
import os

from xtuner.v1.rl.utils import AcceleratorResourcesConfig

nnodes = int(os.environ.get("WORLD_SIZE", "1"))

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * nnodes,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)
```

### 2.2 WorkerConfig

{class}`~xtuner.v1.rl.trainer.WorkerConfig` 定义训练 worker：模型结构、优化器、学习率、FSDP、GRPO loss 和序列长度都在这里配置。

```{code-block} python
:caption: 配置训练 worker
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.trainer import WorkerConfig

train_optimizer_steps = 1

model_cfg = get_model_config_from_hf(Path(model_path))
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None

train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1),
    loss_cfg=GRPOLossConfig(
        policy_loss_cfg=dict(
            cliprange_high=0.28,
            cliprange_low=0.2,
            loss_type="vanilla",
            clip_ratio_c=10.0,
            log_prob_diff_min=-20.0,
            log_prob_diff_max=20.0,
        ),
        ignore_idx=-100,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        mode="chunk",
        chunk_size=512,
    ),
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
    fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1),
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)
```

常用自定义项：

- `model_cfg`：从 HF checkpoint 推断，也可以替换为手写模型配置。
- `optim_cfg` / `lr_cfg`：优化器和学习率策略。
- `loss_cfg`：GRPO loss 的裁剪、KL、chunk 计算等参数。
- `fsdp_cfg` / `sp_size`：分布式训练策略。
- `optimizer_steps`：每个 rollout batch 上训练多少个 optimizer step。
- `pack_max_length`：训练时的最大 pack 长度。

### 2.3 RLColocateTrainerConfig

最后把 generation 和 trainer 两侧配置组装到 {class}`~xtuner.v1.train.rl_trainer.RLColocateTrainerConfig`。

```{code-block} python
:caption: 组装共卡 RL trainer
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig

work_dir = os.environ["WORK_DIR"]
train_batch_size = 64 * train_optimizer_steps
total_train_steps = 45
evaluate_step = 45

trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=replay_buffer_config,
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    total_train_steps=total_train_steps,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    sync_weights_interval=1,
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
)
```

需要注意：

- `total_train_steps` 和 `total_epochs` 至少设置一个。
- `train_batch_size` 表示每个训练 step 消费多少个 rollout group。
- `sync_weights_interval` 控制多少训练步同步一次权重到 rollout worker。
- 开启评估时，`evaluate_step` 必须是 `sync_weights_interval` 的倍数。
- `checkpoint_interval` 和 `hf_interval` 如果启用，也必须是 `sync_weights_interval` 的倍数。

## 3. 启动训练

把上文配置保存为 `my_grpo.py` 后，推荐继续使用快速开始中的启动脚本。它会启动 Ray、设置推理后端环境变量、准备 `WORK_DIR`，最后调用 RL CLI。

```bash
bash examples/v1/scripts/run_rl.sh my_grpo.py lmdeploy /path/to/Qwen3-8B /path/to/train.jsonl /path/to/eval.jsonl
```

第二个参数是推理后端，可选 `sglang`、`lmdeploy` 或 `vllm`。

如果不用 `run_rl.sh`，需要自己先启动 Ray，并设置 `MODEL_PATH`、`DATA_PATH`、`EVAL_DATA_PATH`、`WORK_DIR` 等环境变量，然后执行：

```bash
python xtuner/v1/train/cli/rl.py --config my_grpo.py
```

完整共卡 GRPO 示例可以参考 `examples/v1/config/rl_grpo_gsm8k_judge.py`。
