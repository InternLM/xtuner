================
超大规模数据集
================

在线数据处理
===============

XTuner
默认采用在线数据预处理的策略，这样可以降低用户使用门槛，以达到“开箱即用”的要求。然而，在线数据处理的弊端在于，当数据集过大时，数据处理过程耗时相对较多，可能会触发
``nccl timeout`` 报错。

为什么会出现 ``nccl timeout``?
------------------------------------

使用 XTuner 训练模型时，在训练开始前会首先通过
`process_hf_dataset <https://github.com/InternLM/xtuner/blob/32e3e5f0581998fd84f30f8a1847554a287c161a/xtuner/dataset/huggingface.py#L222>`__
函数对整个训练集进行数据预处理，得到模型训练所需要的 ``input_ids``,
``labels`` 等数据。

由于数据预处理操作是一个 CPU 任务，因此在分布式训练过程中，如果多个 rank
各自执行预处理任务，会造成 CPU 资源抢占，拖慢数据处理速度。因此 XTuner
中采用的策略是统一由 rank0 处理，完成后通过
``torch.distributed.broadcast_object_list`` 接口广播至其他
rank。这样，不同 rank 就会得到一份完全一样的数据集。

然而，当使用 ``nccl``
通信策略时，\ ``torch.distributed.broadcast_object_list``
广播操作的超时时间与 ``nccl`` 通信超时时间相同（默认为 30
分钟）。当训练数据集较大时，rank0 可能无法在 30
分钟内处理完全部数据，这样就会导致 ``nccl timeout`` 报错。若修改
``nccl`` 通信超时时间，则除数据预处理外的其他涉及 ``nccl``
通信的超时时间设置都会被修改。

解决方案
-----------

为解决上述问题，可以在训练开始前设置环境变量 ``XTUNER_DATASET_TIMEOUT``
为一个更大的数（默认为 30 分钟超时，可以酌情将其调大，如：120）：

.. code:: console

   $ # On multiple GPUs(torchrun)
   $ XTUNER_DATASET_TIMEOUT=120 NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero1
   $ # On multiple GPUs(slurm)
   $ XTUNER_DATASET_TIMEOUT=120 srun ${SRUN_ARGS} xtuner train ${CONFIG_NAME_OR_PATH} --launcher slurm --deepspeed deepspeed_zero1

.. note::
   该超时设置只针对数据预处理阶段的广播操作生效。

离线数据处理
===============

当训练数据量非常大时，每次训练的时候都先在线处理数据可能会极为耗时。我们可以先对原始数据进行离线处理并保存至本地，随后的多次训练可以读入本地离线处理好的数据后直接开始训练。

第一小节介绍如何针对纯语言模型训练所使用的文本数据进行离线处理，第二小节将会介绍如何离线处理
Llava 训练数据。

.. warning::

   当切换了 tokenizer 或修改了数据处理中的超参数（如：单条数据的最大长度 ``max_length`` 等）时，需要重新离线处理数据，否则会导致训练报错。

语言模型训练数据离线处理
-------------------------

为便于介绍，本节以
`internlm2_7b_qlora_alpaca_e3.py <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/internlm/internlm2_7b/internlm2_7b_qlora_alpaca_e3.py>`__
配置文件为基础，介绍如何离线处理数据集，并使用离线处理的数据集进行训练。

步骤 1：导出目标 config 文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``internlm2_7b_qlora_alpaca_e3.py`` 是 XTuner 提供的使用 QLora 算法在
Alpaca 数据集上微调 Internlm2-7B 模型的配置文件。通过以下命令可以将该
config 拷贝至当前目录下：

.. code::

   xtuner copy-cfg internlm2_7b_qlora_alpaca_e3 .

.. tip::
   执行以上命令后，当前目录下会新增一个名为
   ``internlm2_7b_qlora_alpaca_e3_copy.py`` 的配置文件（与
   `internlm2_7b_qlora_alpaca_e3.py <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/internlm/internlm2_7b/internlm2_7b_qlora_alpaca_e3.py>`__
   完全一样）。

步骤 2：离线处理数据集
^^^^^^^^^^^^^^^^^^^^^^

使用以下命令可离线预处理原始数据：

.. code::

   python xtuner/tools/process_untokenized_datasets.py \
       internlm2_7b_qlora_alpaca_e3_copy.py  \
       --save-folder /folder/to/save/processed/dataset

.. note::
   这里的第一个参数为 Step 1 中修改过的 config
   文件，第二个参数为预处理过的数据集的保存路径。

.. note::

    上述命令会在 internlm2_7b_qlora_alpaca_e3_copy.py
    同级目录下新建一个 internlm2_7b_qlora_alpaca_e3_copy_modified.py
    文件，后续训练中需要使用该配置文件，而非
    ``internlm2_7b_qlora_alpaca_e3_copy.py`` 。

步骤 3：启动训练
^^^^^^^^^^^^^^^^

可以通过以下命令启动训练：

.. code:: console

   $ # On multiple GPUs(torchrun)
   $ NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_7b_qlora_alpaca_e3_copy_modified.py --deepspeed deepspeed_zero1
   $ # On multiple GPUs(slurm)
   $ srun ${SRUN_ARGS} xtuner train internlm2_7b_qlora_alpaca_e3_copy_modified.py --launcher slurm --deepspeed deepspeed_zero1


.. note::
   训练中需要使用步骤 2 新生成的
   internlm2_7b_qlora_alpaca_e3_copy_modified.py 文件，而非
   internlm2_7b_qlora_alpaca_e3_copy.py 文件。

Llava 训练数据离线处理
---------------------------

为便于介绍，本节以
`llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/internlm2_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py>`__
配置文件为基础，介绍如何离线处理数据集，并使用离线处理的数据集进行训练。


步骤 1：导出目标 config 文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py``
是 XTuner 提供的基于 internlm2-chat-7b 训练 Llava
模型配置文件。可以通过以下命令将该 config 拷贝至当前目录下：

.. code:: console

   $ xtuner copy-cfg llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain .

.. note::
   执行以上命令后，当前目录下会新增一个名为
   ``llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain_copy.py``
   的配置文件（与
   `llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/internlm2_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py>`__
   完全一样）。



步骤 2：离线处理数据集
^^^^^^^^^^^^^^^^^^^^^^

使用以下命令可离线预处理原始数据：

.. code:: console

   $ python xtuner/tools/process_untokenized_llava_data.py llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain_copy.py \
   $    --save-folder /folder/to/save/processed/llava/data

处理后可以读取离线处理后的数据集查看是否符合预期：

.. code:: python

   from datasets import load_from_disk
   ds = load_from_disk('/folder/to/save/processed/llava/data')
   print(ds)

步骤 3：修改 config 文件
^^^^^^^^^^^^^^^^^^^^^^^^

修改 config 文件以便程序运行时直接读取预处理的 Llava 数据：

.. code:: diff

   #######################################################################
   #                      PART 3  Dataset & Dataloader                   #
   #######################################################################
   llava_dataset = dict(
   -   data_path=data_path,
   -   tokenizer=tokenizer,
   +   offline_processed_text_folder=/folder/to/save/processed/llava/data
       ...)

.. note::
   其中，\ ``/folder/to/save/processed/llava/data`` 为步骤 2
   保存的离线处理数据路径。

步骤 4：开始训练
^^^^^^^^^^^^^^^^

使用步骤 3 修改得到的 config 训练即可：

.. code:: console

   $ # On a single GPU
   $ xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain_copy.py --deepspeed deepspeed_zero2

   $ # On multiple GPUs (torchrun)
   $ NPROC_PER_NODE=${GPU_NUM} xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain_copy.py --deepspeed deepspeed_zero2
   $ # On multiple GPUs (slurm)
   $ srun ${SRUN_ARGS} xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain_copy.py --launcher slurm --deepspeed deepspeed_zero2
