# Dataset Prepare

- [Dataset Prepare](#dataset-prepare)
  - [HuggingFace datasets](#huggingface-datasets)
  - [Others](#others)
    - [Arxiv Gentitle](#arxiv-gentitle)
    - [MOSS-003-SFT](#moss-003-sft)
    - [Chinese Lawyer](#chinese-lawyer)
    - [LLaVA dataset](#llava-dataset)
      - [File structure](#file-structure)
      - [Pretrain](#pretrain)
      - [Finetune](#finetune)
    - [RefCOCO dataset](#refcoco-dataset)
      - [File structure](#file-structure-1)

## HuggingFace datasets

For datasets on HuggingFace Hub, such as [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), you can quickly utilize them. For more details, please refer to [single_turn_conversation.md](./single_turn_conversation.md) and [multi_turn_conversation.md](./multi_turn_conversation.md).

## Others

### Arxiv Gentitle

Arxiv dataset is not released on HuggingFace Hub, but you can download it from Kaggle.

**Step 0**, download raw data from https://kaggle.com/datasets/Cornell-University/arxiv.

**Step 1**, process data by `xtuner preprocess arxiv ${DOWNLOADED_DATA} ${SAVE_DATA_PATH} [optional arguments]`.

For example, get all `cs.AI`, `cs.CL`, `cs.CV` papers from `2020-01-01`:

```shell
xtuner preprocess arxiv ${DOWNLOADED_DATA} ${SAVE_DATA_PATH} --categories cs.AI cs.CL cs.CV --start-date 2020-01-01
```

**Step 2**, all Arixv Gentitle configs assume the dataset path to be `./data/arxiv_data.json`. You can move and rename your data, or make changes to these configs.

### MOSS-003-SFT

MOSS-003-SFT dataset can be downloaded from https://huggingface.co/datasets/fnlp/moss-003-sft-data.

**Step 0**, download data.

```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/fnlp/moss-003-sft-data
```

**Step 1**, unzip.

```shell
cd moss-003-sft-data
unzip moss-003-sft-no-tools.jsonl.zip
unzip moss-003-sft-with-tools-no-text2image.zip
```

**Step 2**, all moss-003-sft configs assume the dataset path to be `./data/moss-003-sft-no-tools.jsonl` and `./data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl`. You can move and rename your data, or make changes to these configs.

### Chinese Lawyer

Chinese Lawyer dataset has two sub-dataset, and can be downloaded form https://github.com/LiuHC0428/LAW-GPT.

All lawyer configs assume the dataset path to be `./data/CrimeKgAssitant清洗后_52k.json` and `./data/训练数据_带法律依据_92k.json`. You can move and rename your data, or make changes to these configs.

### LLaVA dataset

#### File structure

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

#### Pretrain

LLaVA-Pretrain

```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain --depth=1
```

#### Finetune

1. Text data

   1. LLaVA-Instruct-150K

      ```shell
      # Make sure you have git-lfs installed (https://git-lfs.com)
      git lfs install
      git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K --depth=1
      ```

2. Image data

   1. COCO (coco): [train2017](http://images.cocodataset.org/zips/train2017.zip)

   2. GQA (gqa): [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)

   3. OCR-VQA (ocr_vqa): [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)

      1. ⚠️ Modify the name of OCR-VQA's images to keep the extension as `.jpg`!

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

#### File structure

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

Download the RefCOCO, RefCOCO+, RefCOCOg annotation files using below links.
Both of coco train 2017 and 2014 are valid for coco_images.

| Image source |                                        Download path                                         |
| ------------ | :------------------------------------------------------------------------------------------: |
| RefCOCO      | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"> annotations </a>  |
| RefCOCO+     | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"> annotations </a> |
| RefCOCOg     | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"> annotations </a> |

After downloading the annotations, unzip the files and place them in the `./data/refcoco_annotations` directory.
Then, we convert the annotations to json format using the below command. This command saves the converted json files in the `./data/llava_data/RefCOCOJson/` directory.

```shell
xtuner preprocess refcoco --ann-path $RefCOCO_ANN_PATH --image-path $COCO_IMAGE_PATH \
--save-path $SAVE_PATH # ./data/llava_data/RefCOCOJson/
```
