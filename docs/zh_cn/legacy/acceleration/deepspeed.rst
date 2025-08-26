============================
DeepSpeed
============================

借助 DeepSpeed 中的 ZeRO 技术（零冗余优化器），可以大幅降低 LLM 训练所消耗的显存

如何选择 ZeRO 策略
====================

模型训练阶段，每张卡中显存占用可以分为两类：

模型状态
    模型参数（fp16）、模型梯度（fp16）和 Adam 优化器状态（fp32 的模型参数备份，fp32 的 momentum 和 fp32 的 variance ）。
    假设模型参数量 :math:`x` ，则共需要 :math:`2x + 2x + (4x + 4x + 4x) = 16x` 字节存储。

.. tip::
    全量微调时，每增加 **1B** 参数，需要增加 **16GB** 的显存来存储模型状态

剩余状态
    除了模型状态之外的显存占用，包括激活值、各种临时缓冲区以及无法使用的显存碎片。

**ZeRO 策略只优化模型状态显存占用，** 从 ZeRO-1 到 ZeRO-3 优化等级越来越高。

- ZeRO-1 策略针对优化器状态进行分片，模型参数和梯度仍旧是每张卡保持一份，此时，每张卡的模型状态所需显存是 :math:`4x + \frac{12x}{N}` （ N 为 GPU 数目）
- ZeRO-2 策略针对模型梯度进行分片，模型参数仍旧是每张卡保持一份，此时，每张卡的模型状态所需显存是 :math:`2x + \frac{14x}{N}` （ N 为 GPU 数目）
- ZeRO-3 策略针对模型参数进行分片，此时每张卡的模型状态所需显存是 :math:`\frac{16x}{N}` （ N 为 GPU 数目）


.. tip::
    以 7B 模型 + 8 GPUs 全量微调为例:

    - ZeRO-1 模式下，每张卡上模型状态显存占用约为 :math:`2*7 + 2*7 + \frac{4*7 + 4*7 + 4*7}{8} = 38.5` GB
    - ZeRO-2 模式下，每张卡上模型状态显存占用约为 :math:`2*7 + \frac{2*7 + 4*7 + 4*7 + 4*7}{8} = 26.25` GB
    - ZeRO-3 模式下，每张卡上模型状态显存占用约为 :math:`\frac{2*7 + 2*7 + 4*7 + 4*7 + 4*7}{8} = 14` GB

.. tip::
    由于不同的优化方案不会影响模型训练结果，因此在不会导致 OOM 的前提下，建议使用优化等级较低的 ZeRO 策略。


使用 ZeRO 策略训练
===================

XTuner 内置 ZeRO 配置
---------------------

XTuner 内置了五种 DeepSpeed ZeRO 配置：

- deepspeed_zero1
- deepspeed_zero2
- deepspeed_zero2_offload
- deepspeed_zero3
- deepspeed_zero3_offload

可一键启动 DeepSpeed 进行训练，通过 ``--deepspeed`` 来选择不同的 ZeRO 配置：

.. code-block:: console

    $ # 以下命令根据需要任选其一
    $ xtuner train xxx --deepspeed deepspeed_zero1
    $ xtuner train xxx --deepspeed deepspeed_zero2
    $ xtuner train xxx --deepspeed deepspeed_zero2_offload
    $ xtuner train xxx --deepspeed deepspeed_zero3
    $ xtuner train xxx --deepspeed deepspeed_zero3_offload

例如若想使用 DeepSpeed ZeRO2 显存优化算法运行 QLoRA 算法在 oasst1 数据集上微调 InternLM2-Chat-7B，可使用以下命令：

.. code-block:: console

    $ # single gpu
    $ xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
    $ # multi gpus(torchrun)
    $ NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
    $ # multi gpus(slurm)
    $ srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2


自定义 ZeRO 配置
------------------------------------


可使用以下命令使用自定义 DeepSpeed 配置文件（需要是一个 json 文件）：

.. code-block:: console

    $ # single gpu
    $ xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed ${PATH_TO_DEEPSPEED_CONFIG}
    $ # multi gpus(torchrun)
    $ NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed ${PATH_TO_DEEPSPEED_CONFIG}
    $ # multi gpus(slurm)
    $ srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed ${PATH_TO_DEEPSPEED_CONFIG}


.. warning::
    DeepSpeed Config 中的 ``gradient_accumulation_steps`` 会被 XTuner config 中的 ``accumulative_counts`` 设置覆盖

.. warning::
    DeepSpeed Config 中的 ``train_micro_batch_size_per_gpu`` 会被 XTuner config 中的 ``train_dataloader.batch_size`` 设置覆盖

.. warning::
    DeepSpeed Config 中的 ``gradient_clipping`` 会被 XTuner config 中的 ``optim_wrapper.clip_grad.max_norm`` 设置覆盖

.. warning::
    XTuner 会根据所使用的 GPU 架构自动选择 ``fp16`` 或 ``bf16`` 训练，不受
