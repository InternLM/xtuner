# LLaVA Full Pipeline

## Data Preparation

Please refer to the [docs](../../../../docs/en/user_guides/dataset_prepare.md#llava-dataset).

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

## Chat

You can download the released LLaVA-InternLM-7B model from [HuggingFace](https://huggingface.co/xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336) and [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336), and achieve image-text question answering with the following command!

```bash
xtuner chat internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14 \
  --llava xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336 \
  --prompt-template internlm_chat \
  --image $IMAGE_PATH
```

## MMBench Evaluation

XTuner integrates the MMBench evaluation, and you can perform evaluations with the following command!

```bash
xtuner mmbench internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14 \
  --llava xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336 \
  --prompt-template internlm_chat \
  --data-path $MMBENCH_DATA_PATH \
  --language en \
  --work-dir $RESULT_PATH
```

After the evaluation is completed, if it's a development set, it will directly print out the results; If it's a test set, you need to submit `mmbench_result.xlsx` to the official MMBench for final evaluation to obtain precision results!

| Model                      | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev |                                                                                                                                    Configs                                                                                                                                     |                                                                                                                                                          Checkpoints                                                                                                                                                          |
| -------------------------- | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B (XTuner)     |       67.9        |       68.0       |       60.1        |       58.2       |    27.0     |       [Pretrain](./vicuna_7b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py), [Fine-tune](./vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)       | Pretrain: [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner-pretrain) / [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner-pretrain), Fine-tune: [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner) / [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner) |
| LLaVA-InternLM-7B (XTuner) |       68.8        |       68.4       |       67.4        |       64.3       |    34.6     | [Pretrain](./internlm_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py), [Fine-tune](./internlm_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py) |       Pretrain: [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b-pretrain) / [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b-pretrain), Fine-tune: [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) / [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b)       |
