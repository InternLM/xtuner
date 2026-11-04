# 训练配置

在之前的[教程](../../get_started/sft.md)中，我们通过命令行用最简单的方式启动了一次微调训练。不过在实际使用过程中，我们往往需要对训练过程进行更精细的控制，比如设置学习率、并行策略、序列长度等参数。这时候，用一个配置文件来管理实验配置会更方便哦！看完这个教程，相信你会有所收获。

## 命令行启动

在正式介绍配置文件之前，咱们先来回顾一下之前的命令行入口：

```{code-block} bash
python xtuner/v1/train/cli/sft.py --help
```

如果训练使用 `torch.compile` 或 JIT 内核，可以在启动前设置共享编译缓存目录。后续使用相同环境和配置的任务可以复用 PyTorch Inductor、Triton、TileLang、CuTe DSL 和 CUDA 驱动的缓存；第一次运行仍需生成缓存。

```{code-block} bash
export XTUNER_COMPILE_CACHE_DIR=/path/to/shared/compile_cache
torchrun --nproc-per-node 8 xtuner/v1/train/cli/sft.py --config config.py
```

请在导入 XTuner 之前设置该变量。若已单独设置 `TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`TILELANG_CACHE_DIR`、`CUTE_DSL_CACHE_DIR` 或 `CUDA_CACHE_PATH`，各自的设置优先。升级编译器或 GPU 环境后，建议使用新的缓存目录。

当共享缓存位于远端文件系统时，本地模式会让训练期间编译器只访问节点本地目录；设置了 `XTUNER_COMPILE_CACHE_DIR` 就会自动启用。每个节点的 local rank 0 会在导入 XTuner 之前把共享缓存预热到 `/tmp/xtuner-compile-cache`，训练结束后由 global rank 0 把各节点新增的缓存产物合并回共享缓存。共享缓存会按编译器版本和 GPU 架构存放在一个命名空间子目录中。显式设置在节点本地目录之外的编译器缓存变量会被直接报错，以避免个别编译器继续写远端。共享同一个节点或缓存根目录的任务不应并发运行。

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

from xtuner.v1.model import Qwen3Dense8BConfig

model_cfg = Qwen3Dense8BConfig()
```

### 构建数据配置

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

终于要整合所有配置啦！看这里：

```{code-block} python
:caption: 构建完整的 TrainerConfig

from xtuner.v1.train import TrainerConfig

# 基础路径配置
load_from = "<模型路径>"  # 微调模式下必须指定，否则会从零开始训练
tokenizer_path = "<tokenizer路径，通常和模型路径一致>"

# 整合所有配置
trainer = TrainerConfig(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer_path,
    load_from=load_from,
    optim_cfg=optim_cfg,
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
