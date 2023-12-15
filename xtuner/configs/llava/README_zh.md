# LLaVA 全流程

## 数据准备

请参考[文档](../../../../docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset)。

## 训练流程

LLaVA 训练一共分为两步：对齐模块预训练、指令跟随微调（本指南以 8 卡训练 LLaVA-InternLM 为例，实际使用时如遇到显卡数量不足、显存不足等情况可以适当调低 batchsize 来降低显存开销）

1. 对齐模块训练（默认保存在 `./work_dirs/`）

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
```

2. 指令跟随微调（默认保存在 `./work_dirs/`）

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
```

## 对话测试

开源的 LLaVA-InternLM-7B 模型在 [HuggingFace](https://huggingface.co/xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336) 和 [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336) 都可以下载，您可以利用下列命令实现图文问答！

```bash
xtuner chat internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14 \
  --llava xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336 \
  --prompt-template internlm_chat \
  --image $IMAGE_PATH
```

## MMBench 评测

XTuner 内集成了 MMBench 评测，您可以利用下列命令实现评测！

```bash
xtuner mmbench internlm/internlm-chat-7b \
  --visual-encoder openai/clip-vit-large-patch14 \
  --llava xtuner/llava-internlm-chat-7b-clip-vit-large-p14-336 \
  --prompt-template internlm_chat \
  --data-path $MMBENCH_DATA_PATH \
  --language en \
  --work-dir $RESULT_PATH
```

评测完成后，若为开发集则会直接打印出结果；若为测试集，则需将 mmbench_result.xlsx 提交至 MMBench 官方完成评测取得精度结果！

| Model                      | MMBench Test (EN) | MMBench Dev (EN) | MMBench Test (CN) | MMBench Dev (CN) | CCBench Dev |                                                                                                                                    Configs                                                                                                                                     |                                                                                                                                                          Checkpoints                                                                                                                                                          |
| -------------------------- | :---------------: | :--------------: | :---------------: | :--------------: | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B (XTuner)     |       67.9        |       68.0       |       60.1        |       58.2       |    27.0     |       [Pretrain](./vicuna_7b_v15_clip_vit_large_p14_336/pretrain/llava_vicuna_7b_v15_clip_vit_large_p14_336_e1_gpu8_pretrain.py), [Fine-tune](./vicuna_7b_v15_clip_vit_large_p14_336/finetune/llava_vicuna_7b_v15_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)       | Pretrain: [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner-pretrain) / [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner-pretrain), Fine-tune: [HuggingFace](https://huggingface.co/xtuner/llava-v1.5-7b-xtuner) / [ModelScope](https://modelscope.cn/models/xtuner/llava-v1.5-7b-xtuner) |
| LLaVA-InternLM-7B (XTuner) |       68.8        |       68.4       |       67.4        |       64.3       |    34.6     | [Pretrain](./internlm_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py), [Fine-tune](./internlm_chat_7b_clip_vit_large_p14_336/finetune/llava_internlm_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py) |       Pretrain: [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b-pretrain) / [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b-pretrain), Fine-tune: [HuggingFace](https://huggingface.co/xtuner/llava-internlm-7b) / [ModelScope](https://modelscope.cn/models/xtuner/llava-internlm-7b)       |
