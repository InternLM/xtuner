# 使用多模态数据集训练 (VLM)

XTuner 支持 LLaVA 图文模型的微调，本文将以 [xtuner/llava-internlm2-7b](https://huggingface.co/xtuner/llava-internlm2-7b) 为例，讲解如何利用 XTuner 快速上手多模态数据集训练，及后续的对话、评测。

## 数据准备

XTuner 支持 LLaVA 格式数据集的多模态图文预训练、微调。本节将从「LLaVA 开源数据集准备」和「自定义数据集准备」两部分展开介绍。

### LLaVA 开源数据集准备

#### 数据文件结构

```
./data/llava_data
├── LLaVA-Pretrain
│   ├── blip_laion_cc_sbu_558k.json
│   ├── blip_laion_cc_sbu_558k_meta.json
│   └── images
├── LLaVA-Instruct-150K
│   └── llava_v1_5_mix665k.json
└── llava_images
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
```

#### 预训练数据下载

LLaVA-Pretrain

```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain --depth=1
```

#### 指令微调数据下载

1. 文本数据

   1. LLaVA-Instruct-150K

      ```shell
      # Make sure you have git-lfs installed (https://git-lfs.com)
      git lfs install
      git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K --depth=1
      ```

2. 图片数据

   1. COCO (coco): [train2017](http://images.cocodataset.org/zips/train2017.zip)

   2. GQA (gqa): [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)

   3. OCR-VQA (ocr_vqa): [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)

      1. ⚠️ OCR-VQA 所下载的图片命名需要利用如下脚本进行处理，以确保所有图片后缀为 `.jpg`！

         ```shell
         #!/bin/bash
         ocr_vqa_path="<your-directory-path>"

         find "$target_dir" -type f | while read file; do
             extension="${file##*.}"
             if [ "$extension" != "jpg" ]
             then
                 cp -- "$file" "${file%.*}.jpg"
             fi
         done
         ```

   4. TextVQA (textvqa): [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)

   5. VisualGenome (VG): [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### 自定义数据集准备

如果用户期望使用自定义数据集进行图文训练，可以参照 LLaVA 开源数据集格式进行准备，具体格式如下：

```json
[
  {
    "image": "xxx/xxx",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nHello! What's this?"
      },
      {
        "from": "gpt",
        "value": "This is a dog!"
      },
      {
        "from": "human",
        "value": "Is it cute?"
      },
      {
        "from": "gpt",
        "value": "Yes."
      }
    ]
  },
  ...
]
```

目前针对自定义数据有一些约束：

1. `image` 字段表示图片路径，且仅能有一张图片
2. `conversations` 字段第 0 条的 `value` 需要包括 `<image>` ，以确保图片被正确嵌入。

## 训练

多模态图文训练一般分为两步：预训练（pretrain）、指令跟随微调（finetune）。`xtuner/llava-internlm2-7b` 对应的配置文件：[预训练](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/internlm2_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [指令跟随微调](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/internlm2_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)，用户可以对其中的模型路径、数据路径进行自定义修改。

### 预训练

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
```

训得模型将默认保存在 `./work_dirs/`，用户可以通过命令  `xtuner train --work-dir ${SAVE_PATH}` 指定保存路径。

### 指令跟随微调

指令跟随微调时，需要载入预训练阶段所得到的 `.pth` 模型，以提供良好的初始化，这一通过在配置文件中的 `pretrained_pth` 指定，用户可以自行修改。

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
```

### 模型转换

模型训练后会自动保存成 PTH 模型（例如 `iter_5198.pth`），我们需要利用 `xtuner convert pth_to_hf` 将其转换为 HuggingFace 模型，以便于后续使用。具体命令为：

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# 例如：xtuner convert pth_to_hf llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune ./iter_5198.pth ./iter_5198_hf
```

此时，我们将获得所需要的模型。如果使用默认的微调范式，文件结构应与 [这里](https://huggingface.co/xtuner/llava-internlm2-7b/tree/main) 一致。

### 模型合并（可选）

如果您使用了 LoRA / QLoRA 微调，则模型转换后将得到 adapter 参数，而并不包含原 LLM 参数。如果您期望获得合并后的模型权重，那么可以利用 `xtuner convert merge` ：

```bash
(LLM) xtuner convert merge $LLM $LLM_ADAPTER $SAVE_PATH
(CLIP) xtuner convert merge $CLIP $CLIP_ADAPTER $SAVE_PATH --is-clip
```

## 对话

用户可以利用 `xtuner chat` 实现与微调后的多模态图文模型对话。假设模型转换阶段获得的模型路径为 `./iter_5198_hf`，则我们可以利用下列命令实现对话：

```bash
xtuner chat internlm/internlm2-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava ./iter_5198_hf \
  --prompt-template internlm2_chat \
  --image $IMAGE_PATH
```

注意事项

1. 如果 LLM 为全参数微调，请将 `internlm/internlm2-chat-7b` 替换为转换后所得到的模型，示例中为 `./iter_5198_hf`。

## 评测

XTuner 的 LLaVA 模型可以利用 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 进行评测，请参考 [这里](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) 快速上手。

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
  --llava ./iter_5198_hf \
  --prompt-template internlm2_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```

其中，`$DATA_PATH` 指上一步骤所下载的某一个 tsv 文件，如 `MMBench_DEV_EN.tsv`。

评测完成后，若为开发集则会直接打印出结果；若为测试集，则需将 `mmbench_result.xlsx` 提交至 [MMBench 官方](https://mmbench.opencompass.org.cn/home) 完成评测取得精度结果。
