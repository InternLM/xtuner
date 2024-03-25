# 使用 DeepSpeed 加速训练

## 使用 XTuner 提供的 DeepSpeed 配置

### 如何选择 ZeRO 策略

ZeRO 策略是一种显存优化方案，包含 ZeRO-1、ZeRO-2 和 ZeRO-3 三种策略，从 ZeRO-1 到  ZeRO-3 优化等级越来越高。

由于不同的优化方案不会影响模型训练结果，因此在不会导致 OOM 的前提下，建议使用优化等级较低的 ZeRO 策略。

### 使用 ZeRO 策略训练

[DeepSpeed](https://github.com/microsoft/DeepSpeed) 是一个开源的深度学习优化库，旨在简化并加速大规模模型的训练。

XTuner 支持一键启动 DeepSpeed 进行训练，只需在启动命令后插入 `--deepspeed deepspeed_zero2(deepspeed_zero1 or deepspeed_zero3)` 即可：

```shell
xtuner train xxx --deepspeed deepspeed_zero2
```

例如若想使用 DeepSpeed ZeRO2 显存优化算法运行 QLoRA 算法在 oasst1 数据集上微调 InternLM2-Chat-7B，可使用以下命令：

```shell
# 单卡
xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
# 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
```

> \[!IMPORTANT\]
> 由于 DeepSpeed ZeRO 策略尚未适配 QLoRA 算法需要用到的 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 中的量化模块，因此 QLoRA 算法不能与 DeepSpeed ZeRO3 同时使用。

## 使用自定义的 DeepSpeed 配置

XTuner 支持使用自定义 DeepSpeed 配置进行训练。但需要注意，以下配置 DeepSpeed config 会被 XTuner config 中对应字段覆盖：

1. `gradient_accumulation_steps` 会被 XTuner config 中的 `accumulative_counts` 设置覆盖；
2. `train_micro_batch_size_per_gpu` 会被 XTuner config 中的 `train_dataloader.batch_size` 设置覆盖；
3. `gradient_clipping` 会被 XTuner config 中的 `optim_wrapper.clip_grad.max_norm` 设置覆盖；
4. XTuner 会根据所使用的 GPU 架构自动选择 `fp16` 或 `bf16` 训练
