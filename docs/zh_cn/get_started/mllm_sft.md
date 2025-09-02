# 多模态大模型微调

在阅读本教程前，请先阅读 [语言模型微调](./sft.md)。

本小结将以 [Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini) 科学多模态大模型为例，，体验最简单的微调训练启动方式。

(sft-dataset)=
## 准备数据集

微调前需先准备数据集。XTuner 多模态部分采用定制化数据格式，只需将数据整理为 `jsonl` 格式即可使用：


```{code-block} json
:caption: jsonl 格式数据示例

{"id": 0, "conversations": [{"from": "human", "value": "The distance between two stars is 6.52 \u00d7 10^5 light years. What is the distance between the two stars in parsecs? (1 parsec = 3.26 light years)\nAnswer Choices: (A) 2 \u00d7 10^5 (B) 4 \u00d7 10^6 (C) 5 \u00d7 10^7 (D) 7 \u00d7 10^7 (E) 9 \u00d7 10^8"}, {"from": "gpt", "value": "Let's think about the multi-choice question.\n6.52 \u00d7 10^5 ly / (3.26 ly/parsec) = 2 x 10^5 persec\nThe answer is A."}]}
{"id": 1, "conversations": [{"from": "human", "value": "How many ways can the letters in the word COMMON be arranged?\nAnswer Choices: (A) 6 (B) 30 (C) 90 (D) 120 (E) 180"}, {"from": "gpt", "value": "Let's solve the multi-choice question step by step.\nAccording to the above the # of permutations of 6 letters COMMON out of which 2 O's and 2 M's are identical is 6!2!\u22172!=180\nThe answer is E."}]}
{"id": 2, "image": "resource/mscoco_dog_000000319154.jpg", "conversations": [{"from": "human", "value": "<image>\n图片中的狗是什么颜色？"}, {"from": "gpt", "value": "图片中的狗是棕色的。"}, {"from": "human", "value": "这只狗有什么类型的项圈？"}, {"from": "gpt", "value": "这只狗有一条红色的项圈。"}, {"from": "human", "value": "那只狗在窗后的位置是什么？"}, {"from": "gpt", "value": "那只狗正坐在窗户后面。"}, {"from": "human", "value": "那只狗望着窗外时，它的表情是怎样的？"}, {"from": "gpt", "value": "当狗凝视窗外的时候，它看起来是在渴望着什么东西。 这也许暗示了它想要到外面去或者与某人或某物进行互动。"}, {"from": "human", "value": "你能描述一下狗的毛吗？"}, {"from": "gpt", "value": "这只狗有蓬松、毛茸茸的棕色皮毛，使它看起来既可爱又温顺。"}], "image_wh": [[375, 500]]}
{"id": 3, "image": "resource/mscoco_dog_000000319154.jpg", "conversations": [{"from": "human", "value": "<image>\n图片中的狗是什么颜色？"}, {"from": "gpt", "value": "图片中的狗是棕色的。"}], "image_wh": [[375, 500]]}

```

- id 字段只是用于方便索引数据而已，可选
- conversations 字段是存放对话内容，可以包括 system、human、gpt 三种角色
- 如果是图片多模态数据，则以 image 字段存放图片路径，image_wh 字段存放图片的宽高，并且 conversations 中必须要存在 `<image>` 标签
- 如果是视频多模态数据，则以 video 字段存放视频路径，image_wh 字段存放视频帧的宽高，并且 conversations 中必须要存在 `<video>` 标签


## 准备模型

XTuner 支持直接使用 Hugging Face 上的模型进行微调。我们以 Intern-S1-mini` 为例，先从 Hugging Face 下载预训练模型：


```{code-block} bash
:caption: 下载 Intern-S1-mini 模型

# 国内用户可以使用 huggingface 镜像站点，在执行命令之前设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download internlm/Intern-S1-mini --local-dir </path/Intern-S1-min>

```

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

成功运行上述命令后，你应该可以看到打印的 loss 大致是 1~2 以内，证明预训练权重正确加载了。
