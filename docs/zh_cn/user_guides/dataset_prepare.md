# 数据集准备

- [数据集准备](#数据集准备)
  - [HuggingFace 数据集](#huggingface-数据集)
  - [其他](#其他)
    - [Arxiv Gentitle 生成题目](#arxiv-gentitle-生成题目)
    - [MOSS-003-SFT](#moss-003-sft)
    - [Chinese Lawyer](#chinese-lawyer)
    - [LLaVA dataset](#llava-dataset)
      - [文件结构](#文件结构)
      - [预训练 Pretrain](#预训练-pretrain)
      - [微调 Finetune](#微调-finetune)
    - [RefCOCO dataset](#refcoco-dataset)
      - [文件结构](#文件结构-1)

## HuggingFace 数据集

针对 HuggingFace Hub 中的数据集，比如 [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)，用户可以快速使用它们。更多使用指南请参照[单轮对话文档](./single_turn_conversation.md)和[多轮对话文档](./multi_turn_conversation.md)。

## 其他

### Arxiv Gentitle 生成题目

Arxiv 数据集并未在 HuggingFace Hub上发布，但是可以在 Kaggle 上下载。

**步骤 0**，从 https://kaggle.com/datasets/Cornell-University/arxiv 下载原始数据。

**步骤 1**，使用 `xtuner preprocess arxiv ${DOWNLOADED_DATA} ${SAVE_DATA_PATH} [optional arguments]` 命令处理数据。

例如，提取从 `2020-01-01` 起的所有 `cs.AI`、`cs.CL`、`cs.CV` 论文：

```shell
xtuner preprocess arxiv ${DOWNLOADED_DATA} ${SAVE_DATA_PATH} --categories cs.AI cs.CL cs.CV --start-date 2020-01-01
```

**步骤 2**，所有的 Arixv Gentitle 配置文件都假设数据集路径为 `./data/arxiv_data.json`。用户可以移动并重命名数据，或者在配置文件中重新设置数据路径。

### MOSS-003-SFT

MOSS-003-SFT 数据集可以在 https://huggingface.co/datasets/fnlp/moss-003-sft-data 下载。

**步骤 0**，下载数据。

```shell
# 确保已经安装 git-lfs (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/fnlp/moss-003-sft-data
```

**步骤 1**，解压缩。

```shell
cd moss-003-sft-data
unzip moss-003-sft-no-tools.jsonl.zip
unzip moss-003-sft-with-tools-no-text2image.zip
```

**步骤 2**, 所有的 moss-003-sft 配置文件都假设数据集路径为 `./data/moss-003-sft-no-tools.jsonl` 和 `./data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl`。用户可以移动并重命名数据，或者在配置文件中重新设置数据路径。

### Chinese Lawyer

Chinese Lawyer 数据集有两个子数据集，它们可以在 https://github.com/LiuHC0428/LAW-GPT 下载。

所有的 Chinese Lawyer 配置文件都假设数据集路径为 `./data/CrimeKgAssitant清洗后_52k.json` 和 `./data/训练数据_带法律依据_92k.json`。用户可以移动并重命名数据，或者在配置文件中重新设置数据路径。

### LLaVA dataset

#### 文件结构

```
./data/llava_data
├── LLaVA-Pretrain
│   ├── blip_laion_cc_sbu_558k.json
│   ├── blip_laion_cc_sbu_558k_meta.json
│   └── images
├── LLaVA-Instruct-150K
│   └── llava_v1_5_mix665k.json
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

#### 预训练 Pretrain

LLaVA-Pretrain

```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain --depth=1
```

#### 微调 Finetune

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

      1. ⚠️ OCR-VQA 所下载的图片命名需要进行修改，以确保所有图片后缀为 `.jpg`！

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

### RefCOCO dataset

#### 文件结构

```

./data
├── refcoco_annotations
│   ├── refcoco
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(unc).p
│   ├── refcoco+
│   │   ├── instances.json
│   │   └── refs(unc).p
│   └── refcocog
│       ├── instances.json
│       ├── refs(google).p
│       └─── refs(und).p
├── coco_images
|    ├── *.jpg
...
```

下载以下链接中的 RefCOCO、RefCOCO+、RefCOCOg文件。
Coco 2017 与 Coco 2014 都可以作为coco的图片数据。

| Image source |                                        Download path                                         |
| ------------ | :------------------------------------------------------------------------------------------: |
| RefCOCO      | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"> annotations </a>  |
| RefCOCO+     | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"> annotations </a> |
| RefCOCOg     | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"> annotations </a> |

在下载完refcoco相关数据文件后，解压文件并将它们放在 `./data/refcoco_annotations` 目录中。
然后，我们使用以下命令将注释转换为json格式。此命令将转换后的json文件保存在 `./data/llava_data/RefCOCOJson/` 目录中。

```shell
xtuner preprocess refcoco --ann-path $RefCOCO_ANN_PATH --image-path $COCO_IMAGE_PATH \
--save-path $SAVE_PATH # ./data/llava_data/RefCOCOJson/
```
