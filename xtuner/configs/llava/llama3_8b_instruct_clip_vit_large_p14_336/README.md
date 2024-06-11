# LLaVA-Llama-3-8B

## Results

<div  align="center">
<img src="https://github.com/InternLM/xtuner/assets/36994684/a157638c-3500-44ed-bfab-d8d8249f91bb" alt="Image" width=500" />
</div>

| Model                 | MMBench Test (EN) | MMBench Test (CN) | CCBench Dev | MMMU  Val | SEED-IMG | AI2D Test | ScienceQA Test | HallusionBench aAcc | POPE | GQA  | TextVQA |   MME    | MMStar |                                                                                                        Configs                                                                                                         |
| :-------------------- | :---------------: | :---------------: | :---------: | :-------: | :------: | :-------: | :------------: | :-----------------: | :--: | :--: | :-----: | :------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B         |       66.5        |       59.0        |    27.5     |   35.3    |   60.5   |   54.8    |      70.4      |        44.9         | 85.9 | 62.0 |  58.2   | 1511/348 |  30.3  |                                                                                                           -                                                                                                            |
| LLaVA-Llama-3-8B      |       68.9        |       61.6        |    30.4     |   36.8    |   69.8   |   60.9    |      73.3      |        47.3         | 87.2 | 63.5 |  58.0   | 1506/295 |  38.2  |           [Pretrain](./pretrain/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./finetune/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)           |
| LLaVA-Llama-3-8B-v1.1 |       72.3        |       66.4        |    31.6     |   36.8    |   70.1   |   70.0    |      72.9      |        47.7         | 86.4 | 62.6 |  59.0   | 1469/349 |  45.1  | [Pretrain](./pretrain/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain.py) / [Fine-tune](./finetune/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune.py) |

## Resources

- LLaVA-Llama-3-8B-v1.1

  - Official LLaVA format model (`xtuner/llava-llama-3-8b-v1_1-hf`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-v1_1-hf)
  - HuggingFace LLaVA format model (`xtuner/llava-llama-3-8b-v1_1-transformers`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-v1_1-transformers)
  - XTuner LLaVA format model (`xtuner/llava-llama-3-8b-v1_1`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-v1_1)
  - GGUF model (`xtuner/llava-llama-3-8b-v1_1-gguf`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-v1_1-gguf)
  - Pretrained projector weights: 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-v1_1-pretrain)

- LLaVA-Llama-3-8B

  - Official LLaVA format model (`xtuner/llava-llama-3-8b-hf`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-hf) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-hf)
  - HuggingFace LLaVA format model (`xtuner/llava-llama-3-8b-transformers`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-transformers) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-transformers)
  - XTuner LLaVA format model (`xtuner/llava-llama-3-8b`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b)
  - Pretrained projector weights: 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-llama-3-8b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-llama-3-8b-pretrain)

## Data Preparation

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

   1. COCO (coco): [download url](http://images.cocodataset.org/zips/train2017.zip)

   2. GQA (gqa): [download url](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)

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

   4. TextVQA (textvqa): [download url](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)

   5. VisualGenome (VG): [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### ShareGPT4V dataset

> Reference: https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md

#### File structure

```
./data/sharegpt4v
├── share-captioner_coco_lcs_sam_1246k_1107.json
├── sharegpt4v_instruct_gpt4-vision_cap100k.json
├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
└── data
    ├── sam
    │   └── images
    ├── share_textvqa
    │   └── images
    ├── web-celebrity
    │   └── images
    ├── web-landmark
    │   └── images
    ├── wikiart
    │   └── images
    ├── llava
    │   └── llava_pretrain
    │       └── images -> ../../../../llava_data/LLaVA-Pretrain/images
    ├── coco -> ../../llava_data/llava_images/coco
    ├── gqa -> ../../llava_data/llava_images/gqa
    ├── ocr_vqa -> ../../llava_data/llava_images/ocr_vqa
    ├── textvqa -> ../../llava_data/llava_images/textvqa
    └── vg -> ../../llava_data/llava_images/vg
```

#### Download

1. Text data

   ```shell
   wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json
   wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json
   wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
   ```

2. Image data

   1. SAM (sam): [download url](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link)

   2. ShareTextVQA (share_textvqa): [download url](https://drive.google.com/file/d/1f4v_3e1OJtyYqam1CEp6RenCNTU5_mG2/view?usp=share_link)

   3. Web-Celebrity (web-celebrity): [download url](https://drive.google.com/file/d/1-SB71C3j1mVg0kDDXwj2IWGEoBoRUD-J/view?usp=share_link)

   4. Web-Landmark (web-landmark): [download url](https://drive.google.com/file/d/1JpJkN7ZMA50xAhMx9O-rVb5yLhfGm3_o/view?usp=share_link)

   5. WikiArt (wikiart): [download url](https://drive.google.com/file/d/1FxB2Nw-vWUcTUSI_dBpPIykb-uGYoEqV/view?usp=share_link)

   6. llava, coco , gqa, ocr_vqa, textvqa, vg: Please refer to the preparation of LLaVA dataset.

### InternVL-SFT

> Reference: https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets

#### File structure

```
./data/internvl_sft
├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
├── llava_instruct_150k_zh.jsonl
├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl
├── dvqa_train_200k.jsonl
├── chartqa_train_18k.jsonl
├── ai2d_train_12k.jsonl
├── docvqa_train_10k.jsonl
├── geoqa+.jsonl
├── synthdog_en.jsonl
└── data
    ├── ai2d
    │   ├── abc_images
    │   └── images
    ├── chartqa
    │   ├── test
    │   ├── train
    │   └── val
    ├── docvqa
    │   ├── test
    │   ├── train
    │   └── val
    ├── dvqa
    │   └── images
    ├── synthdog-en
    │   └── images
    ├── geoqa+
    │   └── images
    ├── llava
    │   └── llava_pretrain
    │       └── images -> ../../../../llava_data/LLaVA-Pretrain/images
    ├── coco -> ../../llava_data/llava_images/coco
    ├── gqa -> ../../llava_data/llava_images/gqa
    ├── ocr_vqa -> ../../llava_data/llava_images/ocr_vqa
    ├── textvqa -> ../../llava_data/llava_images/textvqa
    ├── vg -> ../../llava_data/llava_images/vg
    ├── sam -> ../../sharegpt4v/data/sam
    ├── share_textvqa -> ../../sharegpt4v/data/share_textvqa
    ├── web-celebrity -> ../../sharegpt4v/data/web-celebrity
    ├── web-landmark -> ../../sharegpt4v/data/web-landmark
    └── wikiart -> ../../sharegpt4v/data/wikiart
```

#### Download

1. Text data

   ```shell
   wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/playground.zip
   unzip ./playground.zip
   ```

2. Image data

   1. AI2D (ai2d): [download url](https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing)

   2. ChartQA (chartqa): [download url](https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip)

   3. DocVQA (docvqa): [train](https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz), [val](https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz), [test](https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz)

   4. DVQA (dvqa): [download url](https://drive.google.com/file/d/1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ/view)

   5. SynthDoG-EN (synthdog-en): [download url](https://huggingface.co/OpenGVLab/InternVL/resolve/main/synthdog-en-images.zip)

   6. GeoQA+ (geoqa+): [download url](https://huggingface.co/OpenGVLab/InternVL/resolve/main/geoqa%2B_images.zip)

   7. llava, coco, gqa, ocr_vqa, textvqa, vg: Please refer to the preparation of LLaVA dataset.

   8. sam, share_textvqa, web-celebrity, web-landmark, wikiart: Please refer to the preparation of ShareGPT4V dataset.

## Training

### LLaVA-LLama-3-8B

1. Pretrain (saved by default in `./work_dirs/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_pretrain/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

2. Fine-tune (saved by default in `./work_dirs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_finetune/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2 --seed 1024
```

### LLaVA-LLama-3-8B-v1.1 (Recommended)

1. Pretrain (saved by default in `./work_dirs/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

2. Fine-tune (saved by default in `./work_dirs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune --deepspeed deepspeed_zero2 --seed 1024
```

### Singlg card?

XTuner also supports single-card training for LLaVA-Llama-3-8B (Youth Edition), requiring only a single card with 20GB to complete the entire process of multi-modal training.

1. Pretrain (saved by default in `./work_dirs/llava_llama3_8b_instruct_quant_clip_vit_large_p14_336_e1_gpu1_pretrain/`)

```bash
xtuner train llava_llama3_8b_instruct_quant_clip_vit_large_p14_336_e1_gpu1_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

2. Fine-tune (saved by default in `./work_dirs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e1_gpu1_finetune/`)

```bash
xtuner train llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e1_gpu1_finetune --deepspeed deepspeed_zero2 --seed 1024
```

## Model Conversion

After training, we will obtain a set of weights (*i.e.*, `iter_xxx.pth`), which are not in the universal HuggingFace format. We first need to convert them to the LLaVA model.

### Convert `.pth` file to LLaVA model in xtuner format ([xtuner/llava-llama-3-8b-v1_1](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1))

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., xtuner convert pth_to_hf llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune ./iter_39620.pth ./iter_39620_xtuner
```

At this point, we have obtained the relevant model (LLM or the corresponding LoRA).
If you use the default configuration of LLaVA-Llama-3-8B, you will obtain the following file structure after converting.
It includes the full-finetuned LLM weights, projector weights, and LoRA weights of the visual encoder.

```
./iter_39620_xtuner
├── config.json
├── generation_config.json
├── model-00001-of-00009.safetensors
├── model-00002-of-00009.safetensors
├── model-00003-of-00009.safetensors
├── model-00004-of-00009.safetensors
├── model-00005-of-00009.safetensors
├── model-00006-of-00009.safetensors
├── model-00007-of-00009.safetensors
├── model-00008-of-00009.safetensors
├── model-00009-of-00009.safetensors
├── model.safetensors.index.json
├── projector
│   ├── config.json
│   ├── configuration_projector.py
│   ├── modeling_projector.py
│   └── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── visual_encoder_adapter
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── README.md
```

LLaVA model in xtuner format can engage in conversation using xtuner chat, by

```bash
xtuner chat ./iter_39620_xtuner \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava ./iter_39620_xtuner \
  --prompt-template llama3_chat \
  --image $IMAGE_PATH
```

and in MMBench evaluation, by

```bash
xtuner mmbench ./iter_39620_xtuner \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava ./iter_39620_xtuner \
  --prompt-template llama3_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```

Here, `$DATA_PATH` refers to one of the mmbench datasets. You can download the expected data by

```bash
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv
```

### Convert `.pth` file to LLaVA model in official format ([xtuner/llava-llama-3-8b-v1_1-hf](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf))

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH --save-format official
# e.g., xtuner convert pth_to_hf llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune ./iter_39620.pth ./iter_39620_official --save-format official
```

Here, the converted LLaVA model in official LLaVA format is saved to `./iter_39620_official`.

```
./iter_39620_official
├── config.json
├── generation_config.json
├── model-00001-of-00009.safetensors
├── model-00002-of-00009.safetensors
├── model-00003-of-00009.safetensors
├── model-00004-of-00009.safetensors
├── model-00005-of-00009.safetensors
├── model-00006-of-00009.safetensors
├── model-00007-of-00009.safetensors
├── model-00008-of-00009.safetensors
├── model-00009-of-00009.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

### Convert `.pth` file to LLaVA model in HuggingFace format ([xtuner/llava-llama-3-8b-v1_1-transformers](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers))

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH --save-format huggingface
# e.g., xtuner convert pth_to_hf llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune ./iter_39620.pth ./iter_39620_huggingface --save-format huggingface
```

Here, the converted LLaVA model in HuggingFace LLaVA format is saved to `./iter_39620_huggingface`.

```
./iter_39620_huggingface
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

## Chat

- XTuner LLaVA format [docs](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1#quickstart)
- Official LLaVA format [docs](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf#quickstart)
- HuggingFace LLaVA format [docs](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers#quickstart)
- GGUF format [docs](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf#quickstart)

## Deployment

[LMDeploy](https://github.com/InternLM/lmdeploy) now supports the deployment of official LLaVA format models (e.g.,[xtuner/llava-llama-3-8b-v1_1-hf](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf)). For specifics, please refer to [here](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf#chat-by-lmdeploy).
