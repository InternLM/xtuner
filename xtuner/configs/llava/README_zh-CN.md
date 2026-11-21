# LLaVA 全流程

[English](./README.md) | 简体中文

## 配置文件

- `./${LLM}_${ViT}/` 包含着与 LLaVA-InternLM 训练配置对齐的配置文件（即使用 LoRA / QLoRA）。
- `./official/` 包含着与 LLaVA-v1.5 官方训练配置对齐的配置文件。

## 结果

XTuner 推荐使用基于 LLM-QLoRA / ViT-LoRA 的 LLaVA 架构，其在各个数据集的评测结果如下：

| 模型                         | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev | MME  | SEEDBench_IMG | MMVet | MMMU Dev | MathVista MiniTest | HallusionBench aAcc |                                                                                                                                        配置文件                                                                                                                                         | 预训练 Projector 权重                                                                                                                                                |                                                                  微调 LLaVA 权重                                                                   |
| :--------------------------- | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :--: | :-----------: | :---: | :------: | :----------------: | :-----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B (XTuner)       |       67.7        |       69.2       |       61.0        |       59.7       |    28.4     | 1716 |     66.4      | 32.2  |   33.7   |        24.2        |        46.2         |           [Pretrain](./vicuna_7b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)           | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner-pretrain)   |  🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner)  |
| LLaVA-v1.5-13B (XTuner)      |       68.8        |       69.5       |       64.7        |       63.1       |    32.9     | 1766 |     67.9      | 35.9  |   35.2   |        26.2        |        46.9         |         [Pretrain](./vicuna_13b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_13b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./vicuna_13b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_13b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)         | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-13b-xtuner-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-13b-xtuner-pretrain) | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-13b-xtuner) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-13b-xtuner) |
| LLaVA-InternLM-7B (XTuner)   |       69.0        |       68.5       |       66.7        |       63.8       |    37.3     | 1637 |     65.7      | 32.4  |   36.9   |        26.3        |        49.1         |     [Pretrain](./internlm_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)     | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b-pretrain)         |     🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b)     |
| LLaVA-InternLM2-7B (XTuner)  |       73.3        |       74.6       |       71.7        |       72.0       |    42.5     | 1700 |     71.2      | 35.9  |   40.1   |        25.5        |        46.8         |   [Pretrain](./internlm2_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm2_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)   | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-7b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-7b-pretrain)       |    🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-7b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-7b)    |
| LLaVA-InternLM2-20B (XTuner) |       75.1        |       73.5       |       73.7        |       72.8       |    46.3     | 1868 |     70.2      | 37.2  |   39.4   |        24.6        |        47.7         | [Pretrain](./internlm2_chat_20b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm2_chat_20b_clip_vit_large_p14_336/finetune/llava_internlm2_chat_20b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py) | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-20b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-20b-pretrain)     |   🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-20b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-20b)   |

当与 LLaVA 官方训练架构对齐时，其评测结果如下：

| 模型          |   框架   | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev | MME  | SEEDBench_IMG | MMVet |                                                           配置文件                                                           |
| :------------ | :------: | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :--: | :-----------: | :---: | :--------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B | Official |       65.2        |       63.0       |       57.3        |       57.4       |    25.2     | 1775 |     65.6      | 32.7  |                                                              -                                                               |
| LLaVA-v1.5-7B |  XTuner  |       68.6        |       68.0       |       61.5        |       61.4       |    26.5     | 1786 |     65.8      | 31.4  | [Pretrain](./official/llava_v15_7b/llava_v15_7b_pretrain.py) / [Fine-tune](./official/llava_v15_7b/llava_v15_7b_finetune.py) |

## 数据准备

请参考[文档](../../../docs/zh_cn/legacy/user_guides/dataset_prepare.md#llava-dataset)。

## 训练流程

LLaVA 训练一共分为两步：对齐模块预训练、指令跟随微调（本指南以 8 卡训练 LLaVA-InternLM2-7B 为例，实际使用时如遇到显卡数量不足、显存不足等情况可以适当调低 batchsize 来降低显存开销）

预训练的 Projector 默认保存在 `./work_dirs/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain`，并且指令微调阶段将默认在此路径载入 Projector 权重 （`iter_2181.pth`）。

1. 对齐模块训练（默认保存在 `./work_dirs/`）

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
```

2. 指令跟随微调（默认保存在 `./work_dirs/`）

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
```

## 模型转换（和合并）

训练后，我们将获得一组权重（即，`iter_xxx.pth`，但它并不是通用的 HuggingFace 格式。我们需要对其进行转换。

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., xtuner convert pth_to_hf llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune ./iter_5198.pth ./iter_5198_hf
```

此时，我们将获得所需要的模型（LLM或对应的 LoRA）。

之后，如果想要合并 LoRA 至 LLM 或 CLIP-ViT 中，请使用下列命令：

```bash
(LLM) xtuner convert merge $LLM $LLM_ADAPTER $SAVE_PATH
(CLIP) xtuner convert merge $CLIP $CLIP_ADAPTER $SAVE_PATH --is-clip
```

## 对话测试

开源的 LLaVA-InternLM2-7B 模型在 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-7b) 和 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-7b) 都可以下载，您可以利用下列命令实现图文问答！

```bash
xtuner chat internlm/internlm2-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava xtuner/llava-internlm2-7b \
  --prompt-template internlm2_chat \
  --image $IMAGE_PATH
```

此处， `--llava` 请传入模型转换阶段所获得的权重（示例中为 `./iter_5198_hf`）。

## 评测

XTuner 的 LLaVA 模型可以利用 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 进行评测。

同时，为了方便使用，XTuner 内也集成了 MMBench 评测，您可以通过下列命令下载 MMBench 评测数据集：

```
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv
```

之后，您可以利用下列命令实现评测：

```bash
xtuner mmbench internlm/internlm2-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava xtuner/llava-internlm2-7b \
  --prompt-template internlm2_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```

其中，`$DATA_PATH` 指上一步骤所下载的某一个 tsv 文件，如 `MMBench_DEV_EN.tsv`。

评测完成后，若为开发集则会直接打印出结果；若为测试集，则需将 mmbench_result.xlsx 提交至 MMBench 官方完成评测取得精度结果！

### Refcoco

若您想要评测 Refcoco 数据集，您需要下载评测数据文件 [链接](https://github.com/Vision-CAIR/MiniGPT-4/tree/main/eval_scripts/eval_data). 之后，您可以利用下列命令实现评测：

```bash
xtuner eval_refcoco $LLM \
  --visual-encoder $VISUAL_ENCODER \
  --llava $LLAVA_PATH \
  --prompt-template $PROMPT_TEMPLATE \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```
