.. _training_extreme_long_sequence:

序列并行：训练极长序列大模型的系统优化
===============================

.. raw:: html

    <p align="center">
        <img src="https://github.com/InternLM/xtuner/assets/41630003/b67da9f0-4cdf-49b0-bc45-7624ab3bb24e" alt="XTuner"/>
    </p>

XTuner 中的序列并行设计思路参考了 DeepSpeed 的工作
`DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>`_，并加以优化，以达到直接基于 transformers 算法库或 Huggingface Hub 上的开源模型训练 1M 以上超长序列的目标。
