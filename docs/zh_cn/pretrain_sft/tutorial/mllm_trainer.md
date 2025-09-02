# 使用 Trainer 进行多模态大模型微调

在之前的[教程](../../get_started/mllm_sft.md)中我们通过命令行,用最简单的方式启动了一次微调训练，而在这快速启动的背后，则是 XTuner 的核心组件 `Trainer` 在发挥作用。这一节我们将初识 Trainer，用更加细力度的方式控制训练的各个环节。

在阅读本教程前，请先阅读 [使用 Trainer 进行大模型微调](./llm_trainer.md)。

## 选择模型：

Trainer 通过配置文件的方式来构建模型，我们以 XTuner 内置支持的 `Intern-S1-mini` 为例，来快速获取一个模型配置实例

```{code-block} python
:caption: 构建模型配置

from xtuner.v1.model import InternS1MiniConfig

model_cfg = InternS1MiniConfig()
```

如果我们想修改模型的某些参数，例如减少模型层数，可以这样：

```{tip}
:class: margin

不妨跳转到 InternS1MiniConfig 的源码处，看看它都有哪些参数可以配置
```

```{code-block} python
:caption: 修改模型层数

from xtuner.v1.model import Qwen3Dense8BConfig
text_cfg = Qwen3Dense8BConfig(num_hidden_layers=16)
model_cfg = InternS1MiniConfig(text_config=text_cfg)
```

注意： 如果修改了层数，则权重无法完全加载。

## 选择数据集：

Trainer 也通过配置文件的方式来构建数据集，我们以之前教程中使用的 jsonl 格式数据为例，来快速获取一个数据集配置实例

数据集格式参考[文档](../../get_started/mllm_sft.md#sft-dataset)


```{tip}
:class: margin

每次启动数据加载太慢？不如设置一下 `cache_dir`?
```

```{code-block} python
:caption: 构建数据配置

from xtuner.v1.config import (
    DataloaderConfig,
    DatasetConfig,
)
from xtuner.v1.datasets import InternS1TokenizeFnConfig

sample_max_length = 8192 # 单条样本的最大长度，超过会被截断，并且会有警告输出
pack_max_length = 16384 # 训练一次 iter 所能包含的最大长度，pack 机制会尽可能将多条样本拼接在一起，减少 padding
# 如果你的显存不够，可以适当调小上述两个参数，但是请确保 sample_max_length <= pack_max_length

dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text', # 数据别名
                                 # 标注文件路径，可以是单个 jsonl 也可以是文件夹，会自动遍历当前文件夹下所有 jsonl 文件
                                 anno_path='tests/resource/mllm_sft_text_example_data.jsonl', # 纯文本数据
                                 sample_ratio=5.0, # 数据采样比例，这里是重复 5 遍，可以是小数
                                 class_name='VLMJsonlDataset'), # 对应的 dataset 类名
        # 一个 dataset 要配一个对应的 tokenizer fun 函数用于处理 dataset 输出的单条 item 数据
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(name='media', # 数据别名
                                 anno_path='tests/resource/mllm_sft_media_example_data.jsonl', # 多模态数据
                                 media_root='tests/',
                                 sample_ratio=20.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
# dataloader 配置
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, 
                                     num_workers=8,
                                     pack_level="expand_soft", # pack 样本有 2 种策略，默认选择更高效的 expand_soft 策略
                                     collator='sft_vllm_collator')

```

上述构建了一个比较通用的数据集例子，其中 `dataset_config` 定义了 2 个数据集，分别是纯文本数据和多模态数据，并且指定了各自的采样比例，而 `dataloader_config` 则定义了数据加载器的相关参数。

通过上述灵活的配置组合方式，用户可以轻松配置各类数据集，并且控制各自的采样比例。


## 选择优化器和学习率调度器：

[TODO](@yehaochen，修改 config 的 import 路径)

```{code-block} python
:caption: Optimizer & LR Scheduler

from xtuner.v1.config import LRConfig, AdamWConfig


optim_cfg = AdamWConfig(lr=1e-6, foreach=False) # 不同模块的 device mesh 有差别，foreach 必须是 False
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)
```

## 构建 Trainer 配置


执行完上述步骤，构建完 Trainer 的核心组件后，我们就可以构建 Trainer 实例了：

```{code-block} python
:caption: 构建 Trainer

from xtuner.v1.train.trainer import Trainer
from xtuner.v1.loss import CELossConfig

load_from = "<模型路径>" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "<tokenizer 路径，通常和模型路径一致>"

trainer = TrainerConfig(
    load_from=load_from, # 如果是微调模式，必须指定，否则会重头训练
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    # 全局 batch size
    # 假设是 8 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 1
    # 假设是 4 卡训练，那么每张卡的 forward shape 是 (1, pack_max_length)，梯度累加次数是 2 (自动折算)
    global_batch_size=8, 
    epoch_num=2,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024), # 可以显著减少显存占用，推荐总是开启
    work_dir='work_dirs'
)

```


## 启动训练

完整代码如下：

````{toggle}
```python
from xtuner.v1.model.interns1 import InternS1MiniConfig
from xtuner.v1.config import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    LRConfig,
)
from xtuner.v1.datasets import InternS1TokenizeFnConfig
from xtuner.v1.loss import CELossConfig

# model config
model_cfg = InternS1MiniConfig()

# dataset and dataloader config
sample_max_length = 8192
pack_max_length = 16384

dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text',
                                 anno_path='tests/resource/mllm_sft_text_example_data.jsonl',
                                 sample_ratio=5.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(name='media',
                                 anno_path='tests/resource/mllm_sft_media_example_data.jsonl',
                                 media_root='tests/',
                                 sample_ratio=20.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1TokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length,
                                     num_workers=8,
                                     pack_level="expand_soft",
                                     collator='sft_vllm_collator')

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-6, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)

load_from = "<模型路径>" # 如果是微调模式，必须指定，否则会重头训练
tokenizer = "<tokenizer 路径，通常和模型路径一致>"

# trainer config
trainer = TrainerConfig(
    load_from=load_from,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    global_batch_size=8,
    epoch_num=2,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    work_dir='work_dirs'
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
