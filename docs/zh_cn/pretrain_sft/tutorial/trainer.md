# 使用 Trainer 进行微调

在之前的[教程](../../get_started/sft.md)中我们通过命令行,用最简单的方式启动了一次微调训练，而在这快速启动的背后，则是 XTuner 的核心组件 `Trainer` 在发挥作用。这一节我们将初识 Trainer，用更加细力度的方式控制训练的各个环节。


## 选择模型：

Trainer 通过配置文件的方式来构建模型，我们以 XTuner 内置支持的 `Qwen3 8B` 为例，来快速获取一个模型配置实例

```{code-block} python
:caption: 构建模型配置

from xtuner.v1.model import Qwen3_8BConfig

model_cfg = Qwen3_8BConfig()
```

如果我们想修改模型的某些参数，例如减少模型层数，可以这样：

```{tip}
:class: margin

不妨跳转到 Qwen3_8BConfig 的源码处，看看它都有哪些参数可以配置
```

```{code-block} python
:caption: 修改模型层数

model_cfg = Qwen3_8BConfig(num_hidden_layers=16)
```


## 选择数据集：

Trainer 也通过配置文件的方式来构建数据集，我们以之前教程中使用的 jsonl 格式数据为例，来快速获取一个数据集配置实例

数据集格式参考[文档](../../get_started/sft.md#sft-dataset)

[TODO](@gaojianfei，使用 xtuner dataset 的配置方式)

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

[TODO](@yehaochen，修改 config 的 import 路径)

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

from xtuner.v1.train.trainer import Trainer


load_from = "<模型路径>" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "<tokenizer 路径，通常和模型路径一致>"

trainer = Trainer(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_cfg,
    lr_cfg=lr_cfg,
)

```


## 启动训练

完整代码如下：

````{toggle}
```python
from xtuner.v1.model import Qwen3_8BConfig
from xtuner.v1.config import LRConfig, AdamWConfig
from xtuner.v1.train.trainer import Trainer


model_cfg = Qwen3_8BConfig()
dataset_cfg = []
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)

load_from = "<模型路径>" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "<tokenizer 路径，通常和模型路径一致>"

trainer = Trainer(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_cfg,
    lr_cfg=lr_cfg,
)
trainer.fit()
```
````


```{code-block} python
:caption: 启动训练

trainer.fit()
```

恭喜你，已经自己实现了一个 XTuner 的训练入口！你可以在这个脚本里尽情地发挥，定制化自己的训练参数。
