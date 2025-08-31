# 训练配置

在之前的[教程](../../get_started/sft.md)中，我们通过命令行用最简单的方式启动了一次微调训练。不过在实际使用过程中，我们往往需要对训练过程进行更精细的控制，比如设置学习率、并行策略、序列长度等参数。这时候，用一个配置文件来管理实验配置会更方便哦！看完这个教程，相信你会有所收获。

## 命令行启动

在正式介绍配置文件之前，咱们先来回顾一下之前的命令行入口：

```{code-block} bash
python xtuner/v1/train/cli/sft.py --help
```

```{hint}
:class: margin
如果觉得训练入口不合心意？也可以自己动手 DIY 哦。
```

参数有点多对吧？让我们来梳理一下。目前训练入口支持两种启动方式：传入配置文件或者直接传命令行参数，**这两种方式是互斥的**。也就是说，如果你用了配置文件，就不能再用命令行参数啦，反之亦然。简单总结一下：

- **命令行传参**
  - 😊 简单快捷，适合需求简单的快速实验。比如只是换个数据集、模型路径、并行策略或训练步数啥的
  - 😅 扩展性有限，用不了自定义模块，配置粒度比较粗

- **配置文件**
  - 😊 配置粒度超细，扩展性强，支持自定义模块，版本管理也很方便
  - 😅 得自己写配置文件，有点小门槛

```{note}
:class: margin
其实命令行传参功能也很丰富哒，建议看看 `--help` 输出，说不定有惊喜。
```

看到这里，相信你已经心里有数了，接下来咱们重点聊聊配置文件怎么用。


## 配置文件

XTuner 采用 Python 风格的配置文件，让你可以充分利用 Python 的语法特性来导入核心组件，在支持语法提示的编辑器中享受更流畅的配置体验。

```{tip}
:class: margin
如果你已经看过[trainer教程](./trainer.md)，可以直接跳转到[构建 TrainerConfig](TrainerConfig)部分哦。
```

简单来说，XTuner 的训练配置就是围绕 `Trainer` 展开的。写一个训练配置，本质上就是构建一个 `TrainerConfig` 实例。接下来咱们一步步搞定它！

### 构建模型配置

以 XTuner 内置的 `Qwen3 8B` 为例，获取模型配置就是这么简单：

```{code-block} python
:caption: 构建模型配置

from xtuner.v1.model import Qwen3_8BConfig

model_cfg = Qwen3_8BConfig()
```

### 构建数据配置

[TODO](@gaojianfei，使用 xtuner dataset 的配置方式)
数据集格式参考[文档](../../get_started/sft.md#sft-dataset)

```{tip}
:class: margin

每次启动数据加载太慢？不如设置一下 `cache_dir`?
```

```{code-block} python
:caption: 构建数据配置

# 数据集配置
dataset_cfg = []
# 数据加载器配置  
dataloader_cfg = 
```

### 构建优化器和学习率调度器

[TODO](@yehaochen，修改 config 的 import 路径)

```{code-block} python
:caption: 优化器 & 学习率调度器配置

from xtuner.v1.config import LRConfig, AdamWConfig

# 优化器配置
optim_cfg = AdamWConfig(lr=6e-05)
# 学习率调度器配置
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
```

(TrainerConfig)=
### 构建 TrainerConfig

[TODO](@yehaochen，修改 config 的 import 路径)

终于要整合所有配置啦！看这里：

```{code-block} python
:caption: 构建完整的 TrainerConfig

from xtuner.v1.config import TrainerConfig

# 基础路径配置
load_from = "<模型路径>"  # 微调模式下必须指定，否则会从零开始训练
tokenizer_path = "<tokenizer路径，通常和模型路径一致>"

# 整合所有配置
trainer = TrainerConfig(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer_path,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
    work_dir="<目标工作目录>",
)
```

## 启动训练

配置文件搞定后，启动训练就是小菜一碟啦！只需要把配置文件路径传给训练入口就行：

```{code-block} bash
:caption: 启动训练

python xtuner/v1/train/cli/sft.py --config <配置文件路径>
```
