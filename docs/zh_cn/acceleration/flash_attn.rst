.. _flash_attn:

Flash Attention
==================================================

Flash Attention (Flash Attention 2) 是一种用于加速 Transformer 模型中 Attention 计算，并减少其显存消耗的算法。XTuner 中 Flash Attention (Flash Attention 2) 的支持情况如下表所示：

.. list-table::
  :widths: 25 50
  :header-rows: 1

  * - 模型
    - Flash Attention 支持情况
  * - baichuan 1/2
    - ❌
  * - chatglm 2/3
    - ❌
  * - deepseek
    - ✅
  * - gemma
    - ❌
  * - internlm 1/2
    - ✅
  * - llama 2
    - ✅
  * - mistral
    - ✅
  * - qwen 1/1.5
    - ✅
  * - starcoder
    - ✅
  * - yi
    - ✅
  * - zephyr
    - ✅

.. note::
    XTuner 会根据运行环境自动控制 Flash Attention 的使用情况 (见 `dispatch_modules <https://github.com/InternLM/xtuner/blob/59834032c82d39994c13252aea9b00011d1b2457/xtuner/model/sft.py#L90>`_)：

    .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - 环境
        - Flash Attention 使用情况
      * - 安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_
        - Flash Attention 2
      * - 未安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_ 且 PyTorch Version <= 1.13
        - No Flash Attention
      * - 未安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_ 且 2.0 <= PyTorch Version <= 2.1
        - Flash Attention 1
      * - 未安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_ 且 PyTorch Version >= 2.2
        - Flash Attention 2

.. note::
    使用 XTuner 训练 QWen1/1.5 时若想使用 Flash Attention 加速，需要先安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_ （参考 `flash attn 安装 <https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features>`_，需要 cuda ）
