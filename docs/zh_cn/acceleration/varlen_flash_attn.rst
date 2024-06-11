===============================================
Varlen Attention
===============================================

\ :ref:`数据集拼接 <pack_to_max_length>` \  一节中，我们讨论了“数据集拼接”策略对模型训练效率的显著提升。
理论上，数据集拼接可能会对注意力（Attention）机制的计算过程产生影响。这是因为，在未采用数据拼接策略的情况下，
每条数据在计算注意力时仅与自身相关联。然而，当采用数据拼接策略后，由多条短数据拼接成的长数据在计算注意力时会相互关联。
以一个由若干短数据拼接成长度为 4096 的数据为例，如果不采用变长注意力机制，在注意力计算阶段，每个 token 将会关注全部 4096 个 tokens ，如图左侧所示。

相反，在使用变长注意力机制的情况下，每个 token 在注意力计算阶段仅会关注其所在短数据中的所有 tokens ，如图右侧所示。因此， **变长注意力机制确保了无论是否采用“数据集拼接”策略，模型训练的行为保持一致性。**

.. raw:: html

    <p align="center">
        <img src="https://github.com/InternLM/InternLM/assets/41630003/7e0c6a02-a970-4bd3-a10b-8341720bf654" alt="XTuner" width="600"/>
        <br />变长注意力计算原理（拷贝自 https://github.com/InternLM/InternEvo/blob/develop/doc/usage.md）<br />
    </p>

支持列表
=====================

.. note::

    使用变长注意力需要首先安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_ （
    参考 `flash attn 安装 <https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features>`_ ）

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
    - ❌
  * - starcoder
    - ❌
  * - yi
    - ✅
  * - zephyr
    - ✅

使用变长注意力机制训练
=========================

步骤 1：安装 flash_attn
--------------------------

XTuner 中实现的变长注意力需要依赖 Flash Attention 2，可通过以下命令安装（需要 cuda）：

.. code:: console

  $ MAX_JOBS=4 pip install flash-attn --no-build-isolation

.. tip::
  更多安装方式请参考 `flash attn 安装 <https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features>`_

步骤 2：查找模板 config
---------------------------

XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

.. code-block:: console

    $ xtuner list-cfg -p internlm

.. tip::
  ``-p`` 为模糊查找，若想训练其他模型，可以修改 ``internlm`` 为 XTuner 支持的其他模型名称。

步骤 3：复制 config 文件
-----------------------------

导出需要使用的 config ：

.. code-block:: bash

    xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}

例如通过下列命令将名为 ``internlm_7b_full_oasst1_e3`` 的 config 导出至当前目录下：

.. code-block:: console

    $ xtuner copy-cfg internlm_7b_full_oasst1_e3 .

.. note::

   当前目录下会存在一个新 config
   ``internlm_7b_full_oasst1_e3_copy.py`` 。

步骤 4：修改 config 文件
-------------------------------

将步骤 3 复制得到的 config 文件中的 ``use_varlen_attn`` 属性由 False 改为 True 即可激活变长注意力训练机制：

.. code-block:: diff

    ...
    #######################################################################
    #                          PART 1  Settings                           #
    #######################################################################
    # Model
    pretrained_model_name_or_path = 'internlm/internlm-7b'
    - use_varlen_attn = False
    + use_varlen_attn = True
    ...

.. warning::

    当设置 ``use_varlen_attn = True`` 后， ``batch_size = 2, max_length = 2k`` 的配置与 ``batch_size = 1, max_length = 4k`` 的配置训练行为是近似的，
    因此 XTuner 目前只支持了 ``batch_size = 1`` 的情况。另外， ``use_varlen_attn = True`` 时 ``pack_to_max_length`` 也需设置为 True。

步骤 5：开始训练
-----------------------

.. code-block:: bash

    xtuner train ${CONFIG_NAME_OR_PATH}

例如，我们可以基于步骤 4 中修改得到的 `internlm_7b_full_oasst1_e3_copy.py` 进行训练：

.. code-block:: console

    $ # On a single GPU
    $ xtuner train internlm_7b_full_oasst1_e3_copy.py --deepspeed deepspeed_zero1
    $ # On multiple GPUs(torchrun)
    $ NPROC_PER_NODE=${GPU_NUM} xtuner train internlm_7b_full_oasst1_e3_copy.py --deepspeed deepspeed_zero1
    $ # On multiple GPUs(slurm)
    $ srun ${SRUN_ARGS} xtuner train internlm_7b_full_oasst1_e3_copy.py --launcher slurm --deepspeed deepspeed_zero1

.. tip::
  ``--deepspeed`` 表示使用 `DeepSpeed <https://github.com/microsoft/DeepSpeed>`_ 🚀 来优化训练过程。若未安装 DeepSpeed ，可通过 ``pip install deepspeed>=0.12.3`` 进行安装。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

步骤 6：模型转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^

将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型：

.. code-block:: bash

    xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}

对应上面的例子，模型转换脚本为：

.. code-block:: bash

    xtuner convert pth_to_hf internlm_7b_full_oasst1_e3_copy.py ${PTH} ${SAVE_PATH}

.. note::
  其中 ``${PTH}`` 为训练权重保存的路径，若训练时未指定，默认保存在 ``./work_dirs/internlm_7b_full_oasst1_e3_copy`` 路径下。
