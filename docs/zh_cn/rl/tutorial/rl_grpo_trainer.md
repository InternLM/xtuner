```{important}
XTuner 的 RL（强化学习）功能目前为 Beta 版本，RL功能特性持续完善中，欢迎试用并反馈问题。
```



# [Beta] 使用 Python 代码自定义 GRPO 训练




在之前的[教程](../../get_started/grpo.md)中，我们已经通过命令行体验了快速启动 GRPO 强化学习训练。本教程将介绍如何通过 Python 代码自定义 GRPO 训练配置，让您能够更灵活地控制训练参数。

GRPO 训练主要包含两大配置模块：**Generation Config（生成配置）** 和 **Trainer Config（训练配置）**。

## 1. Generation Config（生成配置）

在强化学习训练中，数据生成是一个关键环节，通常包含**采样 → 推理 → 过滤**三个步骤。在推理阶段，我们使用高效的推理引擎（如 LMDeploy）来生成模型响应。本节将介绍数据生成相关的各项配置，帮助您掌控整个生成流程。

### 1.1 DataFlowConfig

`DataFlow` 是训练数据生成的核心控制器，负责协调整个生成流程。

对于GRPO算法来说，在`DataFlowConfig`中，您需要修改的关键参数如下：
- `prompt_repeat_k`：每个 prompt 的重复采样次数
- `global_batch_size`：每轮 Rollout 的全局批次大小  

```{tip}
:class: margin

更多配置参数请参考API文档：{class}`~xtuner.v1.ray.dataflow.DataFlowConfig`
```

```{code-block} python
:caption: 配置数据流
from xtuner.v1.ray.dataflow import DataFlowConfig

dataflow_config = DataFlowConfig(
    prompt_repeat_k=5,
    global_batch_size=1024
)
```


### 1.2 ReplayBufferConfig

经验回放池（`Replay Buffer`）就像一个"数据仓库"，它的工作很简单：**采样数据、存储数据、按照一定规则提供数据**。在强化学习中，模型生成的样本会先存到这个"仓库"里，然后训练时再从这里取出数据来训练模型。

**对于大多数用户来说，您只需要修改`ReplayBufferConfig`中四个关键参数就能正常使用**：
- `model_path`：模型路径
- `train_data_path`：训练数据路径  
- `max_prompt_length`：输入文本的最大长度
- `pack_max_length`：训练数据打包的最大长度

```{code-block} python
:caption: 配置经验回放池
from transformers import AutoTokenizer
from xtuner.v1.config import DatasetConfig, DataloaderConfig
from xtuner.v1.ray.dataflow import ReplayBufferConfig
from xtuner.v1.datasets import RLTextTokenizeFnConfig

train_data_path = "./gsm8k/train.jsonl"    # 训练数据路径
model_path = "/path/to/qwen3-8B"           # 模型路径
max_prompt_length = 512                    # 输入最大长度
pack_max_length = 32768                    # 打包最大长度

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=[{
        "dataset": DatasetConfig(name="gsm8k", anno_path=train_data_path),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=max_prompt_length),
    }],
    dataloader_cfg=DataloaderConfig(
        pack_max_length=pack_max_length,             
        collator='fake_collator',           
        pack_level='none',                 
    ),
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
)
```

### 1.3 RolloutConfig

`RolloutConfig` 负责配置模型推理环境，它决定了如何使用模型来生成训练所需的样本数据。可以把它理解为"推理引擎的配置文件"。

在本示例中，您只需要指定模型路径即可开始使用。其他使用默认配置。

```{tip}
:class: margin

如果您需要更精细的控制（如分布式推理、推理优化选项等），可以参考API文档：{class}`~xtuner.v1.ray.config.worker.RolloutConfig`
```

```{code-block} python
:caption: 配置推理环境
from xtuner.v1.ray.config.worker import RolloutConfig

model_path = "/path/to/qwen3-8B"  # 替换为您的模型路径

rollout_config = RolloutConfig(
    model_path=model_path,           # 推理模型路径
    model_name="qwen3-8B",           # 模型名称
    tokenizer_path=model_path,       # 分词器路径
)
```


### 1.4 JudgerConfig

XTuner 为GSM8K提供了现成的判断器。您可以直接使用示例代码。

```{code-block} python
:caption: 配置奖励模型
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig

judger_cfg = JudgerConfig(
    reward_judger_configs={
        "openai/gsm8k": GSM8KJudgerConfig()  # GSM8K数学题判断器
    }
)
```

**使用说明**：
- `"openai/gsm8k"`：数据集标识符，需要与您数据集中的 `data_source` 字段匹配
- `GSM8KJudgerConfig()`：专门用于 GSM8K 数学题的判断器，会检查答案的数值是否正确

💡 **扩展功能**：XTuner 还支持多种判断方式（函数式、API服务式）和自定义Judger，相关教程即将推出。

## 2. Trainer Config（训练配置）

### 2.1 WorkerConfig

`WorkerConfig` 是训练阶段的核心，它控制着模型如何学习和优化。这里包含了模型结构、优化器、损失函数等所有训练相关的核心配置。

对于 Qwen3-8B 模型，我们已经为您准备了最佳实践配置。大多数情况下，您只需要指定模型路径、训练优化步数、训练数据打包长度等基本参数：

```{tip}
:class: margin

更多配置参数请参考API文档：{class}`~xtuner.v1.rl.base.worker.WorkerConfig`
```

```{code-block} python
:caption: 配置训练策略
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.grpo import GRPOLossConfig

model_path = "/path/to/qwen3-8B"        # 填入您的模型路径
train_optimizer_steps = 4               # 训练优化步数
pack_max_length = 32768                 # 数据打包最大长度

train_worker_cfg = WorkerConfig(
    model_cfg=Qwen3Dense8BConfig(),                    # 使用预设的 Qwen3-8B 配置
    optim_cfg=AdamWConfig(lr=1e-6, foreach=False),    # 优化器：学习率 1e-6
    loss_cfg=GRPOLossConfig(                          # GRPO 损失函数配置
        policy_loss_cfg=dict(
            cliprange_high=0.2,     # 策略梯度裁剪上限
            cliprange_low=0.2,      # 策略梯度裁剪下限
            loss_type="vanilla",    # 损失类型
        ),
        ignore_idx=-100,            # 忽略的 token 索引
        use_kl_loss=True,           # 启用 KL 散度损失
        kl_loss_coef=0.001,         # KL 损失系数
        kl_loss_type="low_var_kl",  # KL 损失类型
        mode="chunk",               # 计算模式
        chunk_size=512              # 分块大小
    ),
    lr_cfg=LRConfig(warmup_ratio=0),       # 学习率策略：无预热
    fsdp_cfg=FSDPConfig(),                 # 分布式训练配置
    load_from=model_path,                  # 加载模型路径
    optimizer_steps=train_optimizer_steps, # 优化步数
    pack_max_length=pack_max_length,       # 序列最大长度
)
```


### 2.2 EvaluatorConfig [可选]

如果您需要在训练过程中进行验证，可以配置 `EvaluatorConfig`。它定义了验证数据集、验证频率等。
在本示例中，您仅需要修改eval_data_path和evaluate_step间隔即可。

```{code-block} python
:caption: 配置验证流程
from xtuner.v1.ray.evaluator import EvaluatorConfig

eval_data_path = "./gsm8k/test.jsonl"
eval_dataset_cfg = [{"dataset": DatasetConfig(name="gsm8k", anno_path=eval_data_path)}]
evaluator_cfg = EvaluatorConfig(
    dataset_cfg=eval_dataset_cfg,
    tokenizer=tokenizer,
    evaluate_step=10, # 每训练10个epoch验证一次
)
```

## 3、构建并启动 RLTrainer

### 3.1 AcceleratorResourcesConfig

除以上的生成和训练配置外，我们需要配置系统所需资源（如GPU、CPU、内存）等，此处我们使用默认的资源配置，示例如下。

```{code-block} python
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_accelerators_per_worker=1,
    num_cpus_per_worker=12,
    num_workers=8,
    cpu_memory_per_worker=16 * 1024**3, 
)
```

### 3.2 组装 RLTrainer
完成所有组件的配置后，我们就可以将它们组装进 `RLTrainer`，并启动训练流程。

```{code-block} python
:caption: 构建并启动 RLTrainer
import ray
from xtuner.v1.train.rl_trainer import RLTrainer

# 初始化 Ray
ray.init(num_cpus=128, ignore_reinit_error=True)

# 修改路径
model_path = "/path/to/qwen3-8B"
train_data_path = "./gsm8k/train.jsonl"
eval_data_path = "./gsm8k/test.jsonl"
work_dir = "work_dirs/grpo_py_train"

# 配置参数
prompt_repeat_k = 5
global_batch_size = 1024
max_prompt_length = 512
pack_max_length = 32768
train_optimizer_steps = 4

# 声明上述所有config
# ...

# 组装RLTrainer
trainer = RLTrainer(
    resources=resources,
    rollout_config=rollout_config,
    dataflow_config=dataflow_config,
    judger_config=judger_cfg,
    replay_buffer_config=replay_buffer_cfg,
    evaluator_config=evaluator_cfg,
    train_worker_cfg=train_worker_cfg,
    tokenizer_path=model_path,
    work_dir=work_dir,
    total_epochs=15,
    enable_evaluate=False
)
# 开始训练
trainer.fit()
```

## 4、结语

将以上所有配置组合并保存为 Python 文件（例如 `train_grpo.py`），即可通过以下命令启动训练：

```bash
XTUNER_USE_FA3=1 XTUNER_USE_LMDEPLOY=1 python train_grpo.py
```

恭喜你！现在你已经掌握了通过 Python 代码自定义 `RLTrainer` 的方法，可以更灵活地进行强化学习实验了。