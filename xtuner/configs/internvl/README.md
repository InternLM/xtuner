# InterVL Full Pipeline

English | [简体中文](./README_zh-CN.md)

## InterVL 2

> [InternVL-2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)

We introduce InternVL-2, currently the most powerful open-source Multimodal Large Language Model (MLLM). The InternVL-2 family includes models ranging from a 2B model, suitable for edge devices, to a 108B model, which is significantly more powerful. With larger-scale language models, InternVL-2-Pro demonstrates outstanding multimodal understanding capabilities, matching the performance of commercial closed-source models across various benchmarks.

InternVL-2 family is built upon the following designs:

- Progressive with larger language models: We introduce a progressive alignment training strategy, resulting in the first vision foundation model aligned with large language models. By employing the progressive training strategy where the model scales from small to large while the data refines from coarse to fine, we have completed the training of large models at a relatively low cost. This approach has demonstrated excellent performance with limited resources.
- Multimodal input: With one set of parameters, our model supports multiple modalities of input, including text, images, video, audio, and 3D point clouds.
- Multitask output: Our model supports various output formats, such as images, bounding boxes, and masks, demonstrating extensive versatility. By connecting the MLLM with multiple downstream task decoders, InternVL-2 can be generalized to hundreds of vision-language tasks while achieving performance comparable to expert models.

<div align="center">
<img src="https://github.com/OpenGVLab/InternVL/assets/17425982/07845268-8b2c-4dc7-88dd-d10a173bdafe" alt="Image" />
</div>

### Basic Introduction

- `./v2/` contains the configuration files for training InterVL 2
- Supported fine-tuning of the InternVL 2B/4B/8B/26B model in full/LoRA/QLoRA single-image mode for now. We will support fine-tuning on multiple images and videos as soon as possible.
- After training, you can use the `./v1_5/convert_to_official.py` script to convert the model trained by XTuner to the official format, so as to reuse all the official supported toolchains
- All configurations are based on 8xA100 80G graphics cards, 2B/4B can use ZERO1 training, 8B models can use ZERO2, 26B models must run ZERO3, and there is no excessive adjustment of parameters, you can modify them according to your own needs
- It is verified with LLaVA SFT data, which cannot fully reflect the fine-tuning performance. You can customize the data according to your own needs. We will provide a relatively fair fine-tuning dataset later

### Data preparation

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

### Training

The provided configuration is mainly used for fine-tuning based on the official weights. After preparing the data, you can use the following command to train:

```bash
NPROC_PER_NODE=8 xtuner train internvl_v2_internlm2_5_8b_lora_finetune --deepspeed deepspeed_zero2
```

Default saved in `./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/`.

### Model Conversion

After training, we will get a set of weights, that is `./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/iter_xxx.pth`, in order to facilitate evaluation and dialogue, we can convert it to official weights.

```bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v2/internvl_v2_internlm2_5_8b_lora_finetune.py ./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/iter_xxx.pth ./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/convert_model/
```

Here, a complete set of official weights including configuration will be generated under `./work_dirs/internvl_v2_internlm2_5_8b_lora_finetune/convert_model`, you can use the [official toolchain](https://huggingface.co/OpenGVLab/InternVL2-8B) for evaluation and dialogue.

If you encounter any problems during use, please feel free to contact us!!!

## InterVL 1.5

> [How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](https://arxiv.org/abs/2404.16821)

In this report, we introduce InternVL 1.5, an open-source multimodal large language model (MLLM) to bridge the capability gap between open-source and proprietary commercial models in multimodal understanding. We introduce three simple improvements: (1) Strong Vision Encoder: we explored a continuous learning strategy for the large-scale vision foundation model -- InternViT-6B, boosting its visual understanding capabilities, and making it can be transferred and reused in different LLMs. (2) Dynamic High-Resolution: we divide images into tiles ranging from 1 to 40 of 448×448 pixels according to the aspect ratio and resolution of the input images, which supports up to 4K resolution input. (3) High-Quality Bilingual Dataset: we carefully collected a high-quality bilingual dataset that covers common scenes, document images, and annotated them with English and Chinese question-answer pairs, significantly enhancing performance in OCR- and Chinese-related tasks. We evaluate InternVL 1.5 through a series of benchmarks and comparative studies. Compared to both open-source and proprietary models, InternVL 1.5 shows competitive performance, achieving state-of-the-art results in 8 of 18 benchmarks.

<div align="center">
<img src="https://github.com/InternLM/xtuner/assets/17425982/6dbe6a46-f01a-4c9d-ba44-0d857e5c0373" alt="Image" width="700" />
</div>

### Basic Introduction

- `./v1_5/` contains the configuration files for training InterVL 1.5
- Support InternVL 2B/4B/26B model full/LoRA/Qing efficiency and performance, it is recommended to choose the 4B model first
- After training, you can use the `./v1_5/convert_to_official.py` script to convert the model trained by XTuner to the official format, so as to reuse all the official supported toolchains
- All configurations are based on 8xA100 80G graphics cards, 2B/4B can use ZERO1 training, 8B models must run ZERO2, and there is no excessive adjustment of parameters, you can modify them according to your own needs
- It is verified with LLaVA SFT data, which cannot fully reflect the fine-tuning performance. You can customize the data according to your own needs. We will provide a relatively fair fine-tuning dataset later

### Data preparation

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

### Training

The provided configuration is mainly used for fine-tuning based on the official weights. After preparing the data, you can use the following command to train:

```bash
NPROC_PER_NODE=8 xtuner train internvl_v1_5_phi3_4b_lora_finetune --deepspeed deepspeed_zero1
# NPROC_PER_NODE=8 xtuner train internvl_v1_5_internlm2_26b_lora_finetune.py --deepspeed deepspeed_zero3
```

Default saved in `./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/`.

### Model Conversion

After training, we will get a set of weights, that is `./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/iter_xxx.pth`, in order to facilitate evaluation and dialogue, we can convert it to official weights.

```bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v1_5/internvl_v1_5_phi3_4b_lora_finetune.py ./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/iter_xxx.pth ./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/internvl_v1_5_phi3_4b/
```

Here, a complete set of official weights including configuration will be generated under `./work_dirs/internvl_v1_5_phi3_4b_lora_finetune/internvl_v1_5_phi3_4b/`, you can use the [official toolchain](https://github.com/OpenGVLab/InternVL) for evaluation and dialogue.

If you encounter any problems during use, please feel free to contact us!!!
