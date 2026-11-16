(trainer-sft)=
# 使用 Trainer 进行大模型微调

在之前的[教程](../../get_started/sft.md)中我们通过命令行,用最简单的方式启动了一次微调训练，而在这快速启动的背后，则是 XTuner 的核心组件 `Trainer` 在发挥作用。这一节我们将初识 Trainer，用更加细力度的方式控制训练的各个环节。


(model-cfg)=
## 选择模型：

Trainer 通过配置文件的方式来构建模型，我们以 XTuner 内置支持的 `Qwen3 8B` 为例，来快速获取一个模型配置实例

```{code-block} python
:caption: 构建模型配置

from xtuner.v1.model import Qwen3Dense8BConfig

model_cfg = Qwen3Dense8BConfig()
```

如果我们想修改模型的某些参数，例如减少模型层数，可以这样：

```{tip}
:class: margin

不妨跳转到 Qwen3Dense8BConfig 的源码处，看看它都有哪些参数可以配置
```

```{code-block} python
:caption: 修改模型层数

model_cfg = Qwen3Dense8BConfig(num_hidden_layers=16)
```


## 选择数据集：

Trainer 也通过配置文件的方式来构建数据集，我们以之前教程中使用的 jsonl 格式数据为例，来快速获取一个数据集配置实例

数据集格式参考[文档](../../get_started/sft.md#sft-dataset)

```{tip}
:class: margin

每次启动数据加载太慢？不如设置一下 `cache_dir`?
```

```{code-block} python
:caption: 构建数据配置


dataset_cfg = []
dataloader_cfg = 

```


## 选择优化器和学习率调度器：

```{code-block} python
:caption: Optimizer & LR Scheduler

from xtuner.v1.config import LRConfig, AdamWConfig


optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
```

## 构建 Trainer 配置


执行完上述步骤，构建完 Trainer 的核心组件后，我们就可以构建 Trainer 实例了：

```{code-block} python
:caption: 构建 Trainer

from xtuner.v1.train import Trainer


load_from = "<模型路径>" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "<tokenizer 路径，通常和模型路径一致>"

trainer = Trainer(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
)

```


## 实验日志

`Trainer` 和 `TrainerConfig` 支持 `exp_tracker="jsonl"`（默认）、
`"tensorboard"` 和 `"swanlab"`。已有配置可以这样启用 SwanLab：

```python
from xtuner.v1._writer import SwanLabConfig
from xtuner.v1.train import Trainer

trainer_cfg.exp_tracker = "swanlab"
trainer_cfg.swanlab_config = SwanLabConfig(
    project="xtuner",
    experiment_name="sft",
    mode="online",
)
trainer = Trainer.from_config(trainer_cfg)
trainer.fit()
```

通过 `pip install 'xtuner[swanlab]'` 安装可选 SDK，并用 `swanlab login` 登录。
`mode="offline"` 仅记录本地日志。每个 rank 创建独立的 SwanLab run，透传该 rank
原本交给 TensorBoard 或 JSONL 的指标名称、数值和 `cur_step`。本地指标和训练代码
已经归约的指标都会按原有调用记录。

各 run 默认使用共同的 group：`<工作目录名>-<实验目录名>`，也可以通过
`swanlab_config.group` 显式指定。run 名称为 `experiment_name` 加 `-rank<N>`；
未指定名称时使用 group 加该后缀。显式设置 `run_id` 时，rank 0 使用该 ID，
其他 rank 使用 `<run_id>-rank<N>`。

每个 rank 的实验标识和 group 分别保存在
`logs/exp_tracking/rank<N>/swanlab_run.json` 和原生 checkpoint 的
`swanlab/rank<N>.json` 中。自动续训和显式指定 checkpoint 都会恢复各 rank 自己的
标识和原 group，也支持更换日志目录。旧 checkpoint 如果只有 rank 0 的实验标识，
则由 rank 0 复用该标识，其他 rank 创建各自的 run。`fit()` 正常结束或抛出异常时，
各 rank 都会关闭自己的 writer。
SDK 配置和续训选项见 [RL 实验日志](../../rl/advanced_tutorial/rl_trainer.md#实验日志)。

## 启动训练

完整代码如下：

````{toggle}
```python
from xtuner.v1.model import Qwen3Dense8BConfig
from xtuner.v1.config import LRConfig, AdamWConfig
from xtuner.v1.train import Trainer


model_cfg = Qwen3Dense8BConfig()
dataset_cfg = []
dataloader_cfg = 
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)

load_from = "<模型路径>" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "<tokenizer 路径，通常和模型路径一致>"

trainer = Trainer(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
)
trainer.fit()
```
````

写完上述 python 脚本后，命名为 `toy_train.py`，我们就能通过 `torchrun` 启动分布式训练了：

```{code-block} bash
:caption: 启动训练

torchrun --nproc_per_node=8 toy_train.py
```

恭喜你，已经自己实现了一个 XTuner 的训练入口！你可以在这个脚本里尽情地发挥，定制化自己的训练参数。
