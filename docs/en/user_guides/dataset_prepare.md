# Dataset Prepare

- [HuggingFace datasets](#huggingface-datasets)
- [Others](#others)
  - [Arxiv Gentitle](#arxiv-gentitle)
  - [MOSS-003-SFT](#moss-003-sft)
  - [Chinese Lawyer](#chinese-lawyer)

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
