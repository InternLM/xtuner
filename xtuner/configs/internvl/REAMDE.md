# InterVL Full Pipeline

English | [简体中文](./README_zh-CN.md)

## InterVL 1.5

### Basic Introduction

- `./v1_5/` contains the configuration files for training InterVL 1.5
- Support InterVL 2B/4B/26B model full/LoRA/QLoRA fine-tuning, considering efficiency and performance, it is recommended to choose the 4B model first
- After training, you can use the `./v1_5/convert_to_official.py` script to convert the model trained by XTuner to the official format, so as to reuse all the official supported toolchains
- All configurations are based on 8xA100 80G graphics cards, 2B/4B can use ZERO1 training, 26B models must run ZERO3, and there is no excessive adjustment of parameters, you can modify them according to your own needs
- It is verified with LLaVA SFT data, which cannot fully reflect the fine-tuning performance. You can customize the data according to your own needs. We will provide a relatively fair fine-tuning dataset later

## Data preparation

If you also want to use the LLaVA SFT dataset for training, please refer to the [document](../../../docs/en/user_guides/dataset_prepare.md#llava-dataset) to prepare the data.

For custom data, support multiple json and jsonl formats, the data organization can refer to the LLaVA SFT format, and support data sampling operations.

**(1) Support multiple json or jsonl data**

```text
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=['a.json','b.jsonl','c.json'],
    image_folders=['a',None,'c'],
    template=prompt_template,
    max_length=max_length)
```

**(2) Support custom sampling**

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

## Training

The provided configuration is mainly used for fine-tuning based on the official weights. After preparing the data, you can use the following command to train:

```bash
NPROC_PER_NODE=8 xtuner train internvl_v1_5_phi3_4b_lora_finetune --deepspeed deepspeed_zero1
# NPROC_PER_NODE=8 xtuner train internvl_v1_5_internlm2_26b_lora_finetune.py --deepspeed deepspeed_zero3
```

Default saved in `./work_dirs/`.

## Model Conversion

After training, we will get a set of weights, that is `./work_dirs/iter_xxx.pth`, in order to facilitate evaluation and dialogue, we can convert it to official weights.

```bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v1_5/internvl_v1_5_phi3_4b_lora_finetune.py ./work_dirs/iter_xxx.pth ./work_dirs/internvl_v1_5_phi3_4b/
```

Here, a complete set of official weights including configuration will be generated under `./work_dirs/internvl_v1_5_phi3_4b/`, you can use the official toolchain for evaluation and dialogue.

If you encounter any problems during use, please feel free to contact us!!!
