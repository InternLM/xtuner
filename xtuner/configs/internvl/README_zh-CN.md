# InterVL 全流程

[English](./README.md) | 简体中文

## InterVL 2

> [InternVL-2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)

我们引入了 InternVL-2,目前最强大的开源多模态大语言模型(MLLM)。InternVL-2 系列包括从适合于边缘设备的 2B 模型到强大的 108B 模型等多种规模的模型。借助更大规模的语言模型,InternVL-2-Pro 展现出了出色的多模态理解能力,在各种基准测试中的性能与商业闭源模型相匹配。

InternVL-2 系列基于以下设计:

- 渐进式的大型语言模型:我们引入了一种渐进式对齐训练策略,实现了首个与大型语言模型对齐的视觉基础模型。通过采用从小到大模型扩展、从粗到细数据优化的渐进式训练策略,我们以较低的成本完成了大模型的训练。这种方法已经展示了出色的性能,资源有限的情况下也能取得良好的结果。
- 多模态输入:使用一套参数,我们的模型支持文本、图像、视频、音频和 3D 点云等多种输入模态。
- 多任务输出:我们的模型支持图像、边界框和掩码等各种输出格式,展现出广泛的多功能性。通过将 MLLM 与多个下游任务解码器相连接,InternVL-2 可以泛化到数百个视觉语言任务,并取得与专家模型相当的性能。

<div align="center">
<img src="https://github.com/OpenGVLab/InternVL/assets/17425982/07845268-8b2c-4dc7-88dd-d10a173bdafe" alt="Image" />
</div>

### 基本说明

- `./v2/` 包含着 InterVL 2 训练配置的配置文件
- 支持了 InternVL 2B/4B/8B/26B 模型全量/LoRA/QLoRA 单图模式的微调，会尽快支持多图和视频的微调。
- 在训练完成后，可以使用 `./v1_5/convert_to_official.py` 脚本将 XTuner 训练的模型转换为官方格式，从而复用官方所支持的所有工具链
- 目前所有配置都是以 8xA100 80G 显卡为基准，2B/4B 可以使用 ZERO1 训练，8B 模型要 ZERO2 运行，26B 模型必须要 ZERO3，并且没有对参数进行过多的调整，你可以按照你自己的需求进行修改
- 目前是以 LLaVA SFT 数据进行验证，无法充分反应微调性能，你可以根据自己的需求进行数据自定义，后续我们会提供一个相对公平的微调数据集

### 数据准备

如果你也想使用 LLaVA SFT 数据集进行训练，请参考[文档](../../../docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset) 准备数据。

对于自定义数据，支持多种 json 和 jsonl 格式，内部数据组织可以参考 LLaVA SFT 格式，且支持数据采样操作。

**(1) 支持多个 json 或者 jsonl 数据**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    template=prompt_template,
    max_length=max_length)
```

**(2) 支持自定义采样**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    repeat_times=[2,0.5,3.5],
    template=prompt_template,
    max_length=max_length)
```

### 训练流程

所提供的配置主要用于基于官方权重继续微调。在准备好数据后，你可以使用以下命令进行训练：

```bash
NPROC_PER_NODE=8 xtuner train internvl_v2_internlm2_5_8b_lora_finetune --deepspeed deepspeed_zero2
```

默认保存在 `./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/`。

### 模型转换

训练后，我们将获得一组权重即 `./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/iter_xxx.pth`，为了方便评测和对话，可以将其转换为官方权重。

```bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v2/internvl_v2_internlm2_5_8b_lora_finetune.py ./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/iter_xxx.pth ./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/convert_model/
```

此时，会在 `./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/convert_model` 下生成一组包括配置的完整官方权重，你可以使用[官方工具链](https://huggingface.co/OpenGVLab/InternVL2-8B)进行评测和对话。

如果你在使用中碰到任何问题，欢迎联系我们！！！

## InterVL 1.5

> [How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](https://arxiv.org/abs/2404.16821)

在本报告中,我们介绍了开源多模态大语言模型 InternVL 1.5,以弥补开源模型与商业专有模型在多模态理解能力上的差距。我们引入了三项简单的改进:(1) 强大的视觉编码器:我们探索了大规模视觉基础模型 InternViT-6B 的连续学习策略,提升了其视觉理解能力,并使其可以在不同的大语言模型中进行迁移和重复利用。(2) 动态高分辨率:我们根据输入图像的长宽比和分辨率,将图像划分为从1到40个448×448像素的瓦片,支持高达4K分辨率的输入。(3) 高质量双语数据集:我们精心收集了一个高质量的双语数据集,涵盖了常见场景、文档图像,并用英语和中文问答对进行了注释,显著提升了在OCR和中文相关任务中的性能。我们通过一系列基准测试和对比研究评估了 InternVL 1.5。与开源和专有模型相比,InternVL 1.5 表现出了竞争力,在18个基准中的8个中取得了最先进的结果。

<div align="center">
<img src="https://github.com/InternLM/xtuner/assets/17425982/6dbe6a46-f01a-4c9d-ba44-0d857e5c0373" alt="Image" width="700" />
</div>

### 基本说明

- `./v1_5/` 包含着 InterVL 1.5 训练配置的配置文件
- 支持 InternVL 2B/4B/26B 模型全量/LoRA/QLoRA 微调，综合考虑效率性能，建议你优先选择 4B 模型
- 在训练完成后，可以使用 `./v1_5/convert_to_official.py` 脚本将 XTuner 训练的模型转换为官方格式，从而复用官方所支持的所有工具链
- 目前所有配置都是以 8xA100 80G 显卡为基准，2B/4B 可以使用 ZERO1 训练，26B 模型必须要 ZERO3 运行，并且没有对参数进行过多的调整，你可以按照你自己的需求进行修改
- 目前是以 LLaVA SFT 数据进行验证，无法充分反应微调性能，你可以根据自己的需求进行数据自定义，后续我们会提供一个相对公平的微调数据集

### 数据准备

如果你也想使用 LLaVA SFT 数据集进行训练，请参考[文档](../../../docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset) 准备数据。

对于自定义数据，支持多种 json 和 jsonl 格式，内部数据组织可以参考 LLaVA SFT 格式，且支持数据采样操作。

**(1) 支持多个 json 或者 jsonl 数据**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    template=prompt_template,
    max_length=max_length)
```

**(2) 支持自定义采样**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    repeat_times=[2,0.5,3.5],
    template=prompt_template,
    max_length=max_length)
```

### 训练流程

所提供的配置主要用于基于官方权重继续微调。在准备好数据后，你可以使用以下命令进行训练：

```bash
NPROC_PER_NODE=8 xtuner train internvl_v1_5_phi3_4b_lora_finetune --deepspeed deepspeed_zero1
# NPROC_PER_NODE=8 xtuner train internvl_v1_5_internlm2_26b_lora_finetune.py --deepspeed deepspeed_zero3
```

默认保存在 `./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/`。

### 模型转换

训练后，我们将获得一组权重即 `./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/iter_xxx.pth`，为了方便评测和对话，可以将其转换为官方权重。

```bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v1_5/internvl_v1_5_phi3_4b_lora_finetune.py ./work_dirs/iter_xxx.pth ./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/internvl_v1_5_phi3_4b/
```

此时，会在 `./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/internvl_v1_5_phi3_4b/` 下生成一组包括配置的完整官方权重，你可以使用[官方工具链](https://github.com/OpenGVLab/InternVL)进行评测和对话。

如果你在使用中碰到任何问题，欢迎联系我们！！！
