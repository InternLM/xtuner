# 多模态大模型微调

在阅读本教程前，请先阅读 [语言模型微调](./sft.md)。

本小结将以 [Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini) 科学多模态大模型为例，体验最简单的微调训练启动方式。如果想微调 InternVL 模型则只需要把模型路径换掉就可以。

(sft-dataset)=
## 准备数据集

微调前需先准备数据集。XTuner 多模态部分和 SFT 一样，采用标准的 OpenAI 格式，只需将数据整理为 `jsonl` 格式即可使用：


```{code-block} json
:caption: jsonl 格式数据示例

{"id": 0, "messages": [{"role": "user", "content": "The distance between two stars is 6.52 \u00d7 10^5 light years. What is the distance between the two stars in parsecs? (1 parsec = 3.26 light years)\nAnswer Choices: (A) 2 \u00d7 10^5 (B) 4 \u00d7 10^6 (C) 5 \u00d7 10^7 (D) 7 \u00d7 10^7 (E) 9 \u00d7 10^8"}, {"from": "assistant", "value": "Let's think about the multi-choice question.\n6.52 \u00d7 10^5 ly / (3.26 ly/parsec) = 2 x 10^5 persec\nThe answer is A."}]}
{"id": 1, "messages": [{"role": "user", "content": "How many ways can the letters in the word COMMON be arranged?\nAnswer Choices: (A) 6 (B) 30 (C) 90 (D) 120 (E) 180"}, {"from": "assistant", "value": "Let's solve the multi-choice question step by step.\nAccording to the above the # of permutations of 6 letters COMMON out of which 2 O's and 2 M's are identical is 6!2!\u22172!=180\nThe answer is E."}]}
{"id": 2, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {'url':"resource/mscoco_dog_000000319154.jpg","image_wh": [375, 500]}}, {"type": "text", "text": "<IMG_CONTEXT>\n图片中的狗是什么颜色？"}]}, {"role": "assistant", "content": "图片中的狗是棕色的。"}, {"role": "user", "content": "这只狗有什么类型的项圈？"}, {"role": "assistant", "content": "这只狗有一条红色的项圈。"}, {"role": "user", "content": "那只狗在窗后的位置是什么？"}, {"role": "assistant", "content": "那只狗正坐在窗户后面。"}, {"role": "user", "content": "那只狗望着窗外时，它的表情是怎样的？"}, {"role": "assistant", "content": "当狗凝视窗外的时候，它看起来是在渴望着什么东西。 这也许暗示了它想要到外面去或者与某人或某物进行互动。"}, {"role": "user", "content": "你能描述一下狗的毛吗？"}, {"role": "assistant", "content": "这只狗有蓬松、毛茸茸的棕色皮毛，使它看起来既可爱又温顺。"}]}
{"id": 3, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {'url':"resource/mscoco_dog_000000319154.jpg","image_wh": [375, 500]}}, {"type": "text", "text": "<IMG_CONTEXT>\n图片中的狗是什么颜色？"}]}, {"role": "assistant", "content": "图片中的狗是棕色的。"}]}
```

- id 字段只是用于方便索引数据而已，可选
- messages 字段是存放对话内容，可以包括 system、user、assistant 等角色
- 如果是图片多模态数据，则以 image_wh 字段存放图片的宽高

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
