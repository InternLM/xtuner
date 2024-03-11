# LLaVA Full Pipeline

English | [简体中文](./README_zh-CN.md)

## Results

XTuner primarily promotes the LLM-QLoRA / ViT-LoRA LLaVA architecture, and the evaluation results on various datasets are as follows:

| Model                        | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev | MME  | SEEDBench_IMG | MMVet | MMMU Dev | MathVista MiniTest | HallusionBench aAcc |                                                                                                                                         Configs                                                                                                                                         | Pretrained Projector Checkpoints                                                                                                                                     |                                                            Fine-tuned LLaVA Checkpoints                                                            |
| :--------------------------- | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :--: | :-----------: | :---: | :------: | :----------------: | :-----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B (XTuner)       |       67.7        |       69.2       |       61.0        |       59.7       |    28.4     | 1716 |     66.4      | 32.2  |   33.7   |        24.2        |        46.2         |           [Pretrain](./vicuna_7b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)           | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner-pretrain)   |  🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner)  |
| LLaVA-v1.5-13B (XTuner)      |       68.8        |       69.5       |       64.7        |       63.1       |    32.9     | 1766 |     67.9      | 35.9  |   35.2   |        26.2        |        46.9         |         [Pretrain](./vicuna_13b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_13b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./vicuna_13b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_13b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)         | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-13b-xtuner-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-13b-xtuner-pretrain) | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-13b-xtuner) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-13b-xtuner) |
| LLaVA-InternLM-7B (XTuner)   |       69.0        |       68.5       |       66.7        |       63.8       |    37.3     | 1637 |     65.7      | 32.4  |   36.9   |        26.3        |        49.1         |     [Pretrain](./internlm_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)     | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b-pretrain)         |     🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b)     |
| LLaVA-InternLM2-7B (XTuner)  |       73.3        |       74.6       |       71.7        |       72.0       |    42.5     | 1700 |     71.2      | 35.9  |   40.1   |        25.5        |        46.8         |   [Pretrain](./internlm2_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm2_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)   | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-7b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-7b-pretrain)       |    🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-7b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-7b)    |
| LLaVA-InternLM2-20B (XTuner) |       75.1        |       73.5       |       73.7        |       72.8       |    46.3     | 1868 |     70.2      | 37.2  |   39.4   |        24.6        |        47.7         | [Pretrain](./internlm2_chat_20b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm2_chat_20b_clip_vit_large_p14_336/finetune/llava_internlm2_chat_20b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py) | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-20b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-20b-pretrain)     |   🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm2-20b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm2-20b)   |

When aligned completely with the official training settings, the results are as follows:

| Model         | Framework | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev | MME  | SEEDBench_IMG | MMVet |                                                                                                                         Configs                                                                                                                          |
| :------------ | :-------: | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :--: | :-----------: | :---: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B | Official  |       65.2        |       63.0       |       57.3        |       57.4       |    25.2     | 1775 |     65.6      | 32.7  |                                                                                                                            -                                                                                                                             |
| LLaVA-v1.5-7B |  XTuner   |       68.6        |       68.0       |       61.5        |       61.4       |    26.5     | 1786 |     65.8      | 31.4  | [Pretrain](./vicuna_7b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_finetune.py) |

## Data Preparation

Please refer to the [docs](../../../docs/en/user_guides/dataset_prepare.md#llava-dataset).

## Training

The training of LLaVA consists of two steps: alignment module (i.e., MLP) pretraining and instruction following fine-tuning

Note: this guide takes 8-card training LLaVA-InternLM as an example, if there are insufficient GPU resources or memory during actual use, you can reduce the batchsize appropriately to decrease memory consumption. The Pretrained projector is saved and re-loaded by default in `./work_dirs/llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain/iter_2181.pth`.

1. Alignment module pretraining (saved by default in `./work_dirs/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
```

2. Instruction following fine-tuning (saved by default in `./work_dirs/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
```

## Model Convert (and Merge)

After training, we will obtain a set of weights (*i.e.*, `iter_xxx.pth`), which are not in the universal HuggingFace format. We first need to convert them.

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., xtuner convert pth_to_hf llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune ./iter_5198.pth ./iter_5198_hf
```

At this point, we have obtained the relevant model (LLM or the corresponding LoRA).

Afterwards, if you want to merge LoRA into LLM or CLIP-ViT, please use the following command:

```bash
(LLM) xtuner convert merge $LLM $LLM_ADAPTER $SAVE_PATH
(CLIP) xtuner convert merge $CLIP $CLIP_ADAPTER $SAVE_PATH --is-clip
```

## Chat

You can download the released LLaVA-InternLM-7B model from 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) and 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b), and achieve image-text question answering with the following command!

```bash
xtuner chat internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava xtuner/llava-internlm-7b \
  --prompt-template internlm_chat \
  --image $IMAGE_PATH
```

Here, `--llava` is the converted weight from the above step (in our example, it is `./epoch_1_hf` ).

## Evaluation

XTuner's LLaVA models can be evaluated using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

For convenience, XTuner also integrates the [MMBench](https://mmbench.opencompass.org.cn/home) evaluation.

User can download the MMBench dataset with

```
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv
```

After that, the evaluations can be run with

```bash
xtuner mmbench internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava xtuner/llava-internlm-7b \
  --prompt-template internlm_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```

Here, `$DATA_PATH` refers to one of the datasets downloaded as mentioned above, such as `MMBench_DEV_EN.tsv`.

After the evaluation is completed, if it's a development set, it will directly print out the results; If it's a test set, you need to submit `mmbench_result.xlsx` to the official MMBench for final evaluation to obtain precision results!

### Refcoco

To evaluate your model with refcoco, you need download the evaluation data files in [link](https://github.com/Vision-CAIR/MiniGPT-4/tree/main/eval_scripts/eval_data). Second, you can use following command to evaluate your model.

```bash
xtuner eval_refcoco lmsys/vicuna-7b-v1.5 \
  --visual-encoder openai/clip-vit-large-patch14-336 \
  --llava $LLAVA_PATH \
  --prompt-template internlm_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```
