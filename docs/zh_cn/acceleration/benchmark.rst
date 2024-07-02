速度基准
========

我们在训练速度方面与
`LLaMA-Factory <https://github.com/hiyouga/LLaMA-Factory>`__
进行了对比。对比所使用的 LLaMA-Factory commit id 为
`8e04794 <https://github.com/hiyouga/LLaMA-Factory/tree/8e04794b2da067a4123b9d7091a54c5647f44244>`__\ 。使用
`Alpaca <https://huggingface.co/datasets/tatsu-lab/alpaca>`__
作为训练数据集测试速度。

硬件
----

-  NVIDIA A100-SXM4-80GB GPUs

-  Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz

软件环境
--------

-  Python 3.10

-  PyTorch 1.13

-  CUDA 11.7

-  CUDNN 8.5

-  NCCL 2.14.3

速度
----

|image1|

|image2|

|image3|

.. tip::
  TGS 全称是 Tokens per GPU per Second，每张 GPU 每秒训练的 Token 数

.. raw:: html

   <html xmlns="http://www.w3.org/1999/xhtml"><head></head><body><div align="center"></div></body></html>

.. list-table::
  :widths: 30 15 20 20 20 50
  :header-rows: 1

  * - 模型
    - GPUs
    - 序列长度
    - TGS
    - TFLOPs
    - Config
  * - Llama2-7B
    - 8
    - 8k
    - 3028.3
    - 185.3
    - `llama2_70b_full_alpaca_enzh_8k_sp1.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_8k_sp1.py>`_
  * - Llama2-7B
    - 8
    - 32k
    - 2234.2
    - 193.0
    - `llama2_7b_full_alpaca_enzh_32k_sp1.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_32k_sp1.py>`_
  * - Llama2-7B
    - 8
    - 128k
    - 948.6
    - 180.3
    - `llama2_7b_full_alpaca_enzh_128k_sp8.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_128k_sp8.py>`_
  * - Llama2-7B
    - 8
    - 256k
    - 540.1
    - 176.9
    - `llama2_7b_full_alpaca_enzh_256k_sp8.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_256k_sp8.py>`_
  * - Llama2-7B
    - 32
    - 1M
    - 133.6
    - 153.9
    - `llama2_7b_full_alpaca_enzh_1M_sp16.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_7b/llama2_7b_full_alpaca_enzh_1M_sp16.py>`_

.. list-table::
  :widths: 30 15 20 20 20 50
  :header-rows: 1

  * - 模型
    - GPUs
    - 序列长度
    - TGS
    - TFLOPs
    - Config
  * - Yi-34B-200K
    - 32
    - 8k
    - 485.1
    - 165.6
    - `yi_34b_200k_full_alpaca_enzh_8k_sp1.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_8k_sp1.py>`_
  * - Yi-34B-200K
    - 32
    - 32k
    - 491.5
    - 209.1
    - `yi_34b_200k_full_alpaca_enzh_32k_sp2.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_32k_sp2.py>`_
  * - Yi-34B-200K
    - 32
    - 128k
    - 251.1
    - 191.8
    - `yi_34b_200k_full_alpaca_enzh_128k_sp8.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_128k_sp8.py>`_
  * - Yi-34B-200K
    - 32
    - 256k
    - 119.7
    - 145.3
    - `yi_34b_200k_full_alpaca_enzh_256k_sp8.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/yi_34b/yi_34b_200k_full_alpaca_enzh_256k_sp8.py>`_

.. list-table::
  :widths: 30 15 20 20 20 50
  :header-rows: 1

  * - 模型
    - GPUs
    - 序列长度
    - TGS
    - TFLOPs
    - Config
  * - Llama2-70B
    - 32
    - 8k
    - 216.8
    - 144.7
    - `llama2_70b_full_alpaca_enzh_8k_sp1.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_8k_sp1.py>`_
  * - Llama2-70B
    - 32
    - 32k
    - 300.9
    - 239.6
    - `llama2_70b_full_alpaca_enzh_32k_sp4.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_32k_sp4.py>`_
  * - Llama2-70B
    - 32
    - 128k
    - 144.7
    - 189.7
    - `llama2_70b_full_alpaca_enzh_128k_sp8.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_128k_sp8.py>`_
  * - Llama2-70B
    - 32
    - 256k
    - 63.8
    - 127.6
    - `llama2_70b_full_alpaca_enzh_256k_sp16.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_256k_sp16.py>`_
  * - Llama2-70B
    - 64
    - 1M
    - 21.8
    - 133.5
    - `llama2_70b_full_alpaca_enzh_1M_sp64.py <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llama_speed_benchmark/llama2_70b/llama2_70b_full_alpaca_enzh_1M_sp64.py>`_

.. note::
  所有实验都会将 Alpaca 数据集拼接为最大长度。由于 Alpaca 数据集所含
  token 数较少，无法拼接成超长序列（如 1M
  长度），因此当序列长度较长时，会对 XTuner 代码进行如下修改：

  .. code:: diff

    # xtuner/dataset/huggingface.py
    def build_origin_dataset(dataset, split):
        ...
    +   # 6 times larger dataset (for speed testing purposes only)
    +   dataset = concatenate_datasets([dataset for _ in range(6)])
        return dataset

    def pack_dataset(dataset, max_length, use_varlen_attn, shuffle_before_pack,
                      map_num_proc):
        dataset = dataset.map(
            Packer(max_length, use_varlen_attn=use_varlen_attn),
            batched=True,
    -       num_proc=map_num_proc
    +       batch_size=25000,
    +       num_proc=1
        )
        return dataset


.. note::
  由于 Alpaca 数据量较小，因此做了第一处修改将数据集大小扩大了 6
  倍，以保证拥有足够的训练 iter 数（保证速度测试的稳定性）。另外，由于
  Alpaca
  数据集每条数据的长度较短，因此在数据拼接的时候做了第二处修改以保证拥有足够多的数据，足以拼接为
  ``max_length`` 最大长度。

.. |image1| image:: https://github.com/InternLM/xtuner/assets/41630003/c9c05dbd-0806-4fb2-9da9-62f04b150f7c
.. |image2| image:: https://github.com/InternLM/xtuner/assets/41630003/3ef6308c-595b-4624-b56d-a8737a1f2261
.. |image3| image:: https://github.com/InternLM/xtuner/assets/41630003/ba16368e-e5f7-41eb-89ed-1140a8633134
