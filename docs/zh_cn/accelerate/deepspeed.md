# 使用 DeepSpeed 加速训练

[DeepSpeed](https://github.com/microsoft/DeepSpeed) 是一个开源的深度学习优化库，旨在简化并加速大规模模型的训练。

XTuner 支持一键启动 DeepSpeed 进行训练，只需在启动命令后插入 `--deepspeed deepspeed_zero2(deepspeed_zero1 or deepspeed_zero3)` 即可：

```shell
xtuner train xxx --deepspeed deepspeed_zero2
```

例如若想使用 DeepSpeed Zero3 显存优化算法运行 QLoRA 算法在 oasst1 数据集上微调 InternLM2-Chat-7B，可使用以下命令：

```shell
# 单卡
xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero3
# 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero3
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero3
```
