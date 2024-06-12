.. _triton_kernels:

使用 Triton Kernel 加速训练
===============================

`Triton <https://github.com/openai/triton>`_ 是一种用于编写高效深度学习算子的编程语言。XTuner 支持使用 Triton Kernel 加速训练。

可通过在环境变量中加入 `USE_TRITON_KERNEL=1` 来启动 Triton Kernel：

.. code:: bash

    # 单卡
    xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero1
    # 多卡
    (DIST) USE_TRITON_KERNEL=1 NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero1
    (SLURM) USE_TRITON_KERNEL=1 srun ${SRUN_ARGS} xtuner train ${CONFIG_NAME_OR_PATH} --launcher slurm --deepspeed deepspeed_zero1


RMS Norm Triton Kernel
------------------------

RMS Norm Triton Kernel 支持情况
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :widths: 25 50
  :header-rows: 1

  * - 模型
    - 支持情况
  * - internlm 1/2
    - ✅
  * - llama 2
    - ✅
  * - mistral
    - ✅
  * - mixtral
    - ✅
  * - qwen 1/1.5
    - ✅
  * - starcoder
    - ✅
  * - yi
    - ✅
  * - zephyr
    - ✅
  * - deepseek
    - ✅
  * - deepseek v2
    - ❌
  * - baichuan 1/2
    - ❌
  * - chatglm 2/3
    - ❌
  * - gemma
    - ❌
  * - cohere (no RMS Norm)
    - ❌


加速情况
~~~~~~~~~~~

下表展示了 Qwen 1.5 0.5B 在 Alpaca 数据集上的训练效率对比：

.. list-table::
  :widths: 50 25 50 50 25
  :header-rows: 1

  * - 模型
    - 序列长度
    - 是否使用 Triton Kernel
    - 是否使用 Varlen Attention
    - Tokens per Second
  * - Qwen 1.5 0.5B
    - 32k
    - ❌
    - ❌
    - 13712.7
  * - Qwen 1.5 0.5B
    - 32k
    - ✅
    - ❌
    - 14690.3
  * - Qwen 1.5 0.5B
    - 32k
    - ❌
    - ✅
    - 28451.7
  * - Qwen 1.5 0.5B
    - 32k
    - ✅
    - ✅
    - 33027.5

XTuner 未来会支持更多的 Triton Kernel 以进一步加速训练。
