# LLaVA Full Pipeline

## Data Preparation

Please refer to the [docs](../../../docs/en/user_guides/dataset_prepare.md#llava-dataset).

## Training

The training of LLaVA consists of two steps: alignment module (i.e., MLP) pretraining and instruction following fine-tuning

Note: this guide takes 8-card training LLaVA-InternLM as an example, if there are insufficient GPU resources or memory during actual use, you can reduce the batchsize appropriately to decrease memory consumption.

1. Alignment module pretraining (saved by default in `./work_dirs/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
```

2. Instruction following fine-tuning (saved by default in `./work_dirs/`)

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
```

## Model Convert (and Merge)

After training, we will obtain a set of weights (i.e., `epoch_1.pth`), which are not in the universal HuggingFace format. We first need to convert them.

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., xtuner convert pth_to_hf llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune ./epoch_1.pth ./epoch_1_hf
```

At this point, we have obtained the relevant model (LLM or the corresponding LoRA).

Afterwards, if you want to merge LoRA into LLM or CLIP-ViT, please use the following command:

```bash
(For LLM) xtuner convert merge $LLM $LLM_ADAPTER $SAVE_PATH
(For CLIP) xtuner convert merge $CLIP $CLIP_ADAPTER $SAVE_PATH --is-clip
```

## Chat

You can download the released LLaVA-InternLM-7B model from 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) and 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b), and achieve image-text question answering with the following command!

```bash
xtuner chat internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14 \
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
  --visual-encoder openai/clip-vit-large-patch14 \
  --llava xtuner/llava-internlm-7b \
  --prompt-template internlm_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```

Here, `$DATA_PATH` refers to one of the datasets downloaded as mentioned above, such as `MMBench_DEV_EN.tsv`.

After the evaluation is completed, if it's a development set, it will directly print out the results; If it's a test set, you need to submit `mmbench_result.xlsx` to the official MMBench for final evaluation to obtain precision results!

| Model                      | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev |                                                                                                                                     Configs                                                                                                                                     |                                                                  Pretrained Projector Checkpoints                                                                  | Fine-tuned LLaVA Checkpoints                                                                                                                     |
| -------------------------- | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| LLaVA-v1.5-7B (XTuner)     |       67.9        |       68.0       |       60.1        |       58.2       |    27.0     |       [Pretrain](./vicuna_7b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)       | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner-pretrain) | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner) |
| LLaVA-InternLM-7B (XTuner) |       68.8        |       68.4       |       67.4        |       64.3       |    34.6     | [Pretrain](./internlm_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](./internlm_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py) |    🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b-pretrain)    | 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b)       |
