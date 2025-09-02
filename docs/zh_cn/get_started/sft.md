# 语言模型微调

安装完 XTuner 后，让我们通过语言模型微调来小试牛刀，体验最简单的训练启动方式。

(sft-dataset)=
## 准备数据集

微调前需先准备数据集。XTuner 默认支持 OpenAI 格式的数据，只需将数据整理为 `jsonl` 格式即可使用：

```{code-block} json
:caption: jsonl 格式数据示例

[{"content": "Give three tips for staying healthy.\n", "role": "user"}, {"content": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.", "role": "assistant"}]
[{"content": "What are the three primary colors?\n", "role": "user"}, {"content": "The three primary colors are red, blue, and yellow.", "role": "assistant"}]

```


## 准备模型

XTuner 支持直接使用 Hugging Face 上的模型进行微调。我们以 `Qwen3 8B` 为例，先从 Hugging Face 下载预训练模型：


```{code-block} bash
:caption: 下载 Qwen3 8B 模型

# 国内用户可以使用 huggingface 镜像站点，在执行命令之前设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-8B --local-dir </path/qwen3-8B>

```

````{note}

注意：模型路径需具体到模型文件所在目录

```{code-block} bash
:caption: <span class="x-strong">合法模型路径</span>

model-path/
├── config.json
├── model-00001-of-00005.safetensors
├── ...
```

而不是这样带多个版本的路径：

```{code-block} bash
:caption: <del>非法模型路径<del>

models--Qwen--Qwen3-8B
├── blobs
├── refs
└── snapshots
```

如果是上述这种路径结构，需要指定到 snapshots 下面的某个版本号目录，例如：

`models--Qwen--Qwen3-8B/snapshots/版本号`
````

## 启动微调

准备好数据集和模型后，即可启动微调。XTuner 提供了简洁的命令行接口，只需指定模型路径、数据集路径和训练参数：

```{tip}
:class: margin

OOM 怎么办？试试 `--fsdp-config.cpu-offload` 吧！

```
```{code-block} bash
:caption: 启动微调训练
torchrun --nproc-per-node 8  xtuner/v1/train/cli/sft.py  --load-from <模型路径>  --dataset <数据集路径>  --total-step 100 --work-dir <目标工作目录>
```

执行命令后，可以看到以下日志：

```{code-block} bash
:class: toggle
[XTuner][RANK 2][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0578 lr: 0.000020 time: 4.9770 text_tokens: 4008.0 total_loss: 1.722 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 805.3 e2e_tgs: 796.1 
[XTuner][RANK 5][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0641 lr: 0.000020 time: 4.9716 text_tokens: 4010.0 total_loss: 1.506 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 806.6 e2e_tgs: 796.3 
[XTuner][RANK 6][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0617 lr: 0.000020 time: 4.9783 text_tokens: 4069.0 total_loss: 1.802 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 817.3 e2e_tgs: 807.3 
[XTuner][RANK 7][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0614 lr: 0.000020 time: 4.9796 text_tokens: 4058.0 total_loss: 1.589 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 814.9 e2e_tgs: 805.0 
[XTuner][RANK 1][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0571 lr: 0.000020 time: 4.9848 text_tokens: 3929.0 total_loss: 1.623 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 788.2 e2e_tgs: 779.3 
[XTuner][RANK 3][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0600 lr: 0.000020 time: 4.9837 text_tokens: 4077.0 total_loss: 1.686 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 818.1 e2e_tgs: 808.3 
[XTuner][RANK 4][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0542 lr: 0.000020 time: 4.9981 text_tokens: 3931.0 total_loss: 1.779 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 786.5 e2e_tgs: 778.1 
[XTuner][RANK 0][2025-08-29 09:17:51][INFO] Step 1/100 data_time: 0.0674 lr: 0.000020 time: 4.9857 text_tokens: 4044.0 total_loss: 1.764 reduced_llm_loss: 1.684 max_memory: 15.90 GB reserved_memory: 17.87 GB grad_norm: 13.948 tgs: 811.1 e2e_tgs: 800.3 
[XTuner][RANK 2][2025-08-29 09:17:52][INFO] Step 2/100 data_time: 0.0516 lr: 0.000040 time: 0.8883 text_tokens: 4037.0 total_loss: 1.592 reduced_llm_loss: 1.606 max_memory: 18.02 GB reserved_memory: 22.20 GB grad_norm: 12.398 tgs: 4544.6 e2e_tgs: 1346.2 
[XTuner][RANK 5][2025-08-29 09:17:52][INFO] Step 2/100 data_time: 0.0442 lr: 0.000040 time: 0.8948 text_tokens: 4049.0 total_loss: 1.620 reduced_llm_loss: 1.606 max_memory: 18.02 GB reserved_memory: 22.20 GB grad_norm: 12.398 tgs: 4524.8 e2e_tgs: 1348.5 
[XTuner][RANK 1][2025-08-29 09:17:52][INFO] Step 2/100 data_time: 0.0438 lr: 0.000040 time: 0.8899 text_tokens: 4031.0 total_loss: 1.367 reduced_llm_loss: 1.606 max_memory: 18.02 GB reserved_memory: 22.20 GB grad_norm: 12.398 tgs: 4529.9 e2e_tgs: 1331.8 
```

```{tip}
:class: margin

工作目录下还有一个 `.xtuner` 文件，快去看看里面写了啥？
```

与[快速开始](./installation.md)中的验证日志相比，本次初始 loss 明显更低，这是因为我们加载了预训练模型权重和真实的 tokenizer。训练完成后就可以看到工作目录下保存了相应的模型权重。


```{hint}
想进一步了解更多训练参数和配置选项？不妨来看看这些教程:
- [配置文件启动训练](../pretrain_sft/tutorial/config.md)
- [Python 代码启动训练]。(../pretrain_sft/tutorial/llm_trainer.md)。
```
