.. _case2:

Processed 数据集 + 其他模型
==========================================

.. warning::
   非 FTDP（一款闭源数据处理工具） 用户跳过此文档


使用尚未 token 化的 ftdp 数据训练其他模型（以 Mistral 为例），且需要用
Internlm2 对话模板覆盖原有对话模板以便让模型掌握 agent 、tool 能力。

步骤 1：离线处理数据集
----------------------

ftdp 把 sft
任务的数据处理划分为三个类型，原始数据（origin）、预处理数据（processed）和
token 过的数据（tokenized）。我们需要将预处理过的、具有统一格式的 ftdp
数据 token
化得到直接可以用于训练的格式。其中，预处理数据需要满足以下目录结构：

.. code::

   |-- processed-dir
       |-- data1
       |   |-- processed
       |       |-- sft_chat
       |           |-- data1.jsonl
       |-- data2
       |   |-- processed
       |       |-- sft_chat
       |           |-- data2.jsonl

使用以下命令可离线 token 化 ftdp 格式的预处理数据（processed）数据集：

.. code-block:: console

   $ python xtuner/tools/tokenize_ftdp_datasets.py \
   $    --processed-dir /path/to/preprocessed/data \
   $    --tokenized-dir /path/to/tokenized/data \
   $    --tokenizer-path pretrained_model_name_or_path

.. note::
   ``--processed-dir`` 需要指定预处理后的，具有 ftdp
   标准格式的数据路径

.. note::
   ``--tokenized-dir`` 需要指定为 token 化后的数据存储路径

.. note::
   ``--tokenizer-path pretrained_model_name_or_path`` 中的
   ``pretrained_model_name_or_path`` 同 ``from_pretrained`` 接口中的
   ``pretrained_model_name_or_path``\

.. note::
   上述命令执行成功后，会在 ``/path/to/tokenized/data/chatml_llamav13_32k``
   路径下保存两个子文件夹——``train`` 和 ``valid``\ 。

.. warning::
   由于除 Internlm2 外的其他模型（如 mistral 等）没有 internlm2-chat
   模型的智能体、工具调用等功能的对话模板，因此对于非 internlm2
   模型，需要将 internlm2-chat
   对话模板中的一些特殊字符（如：<\|im_start\|>、<\|plugin\|>等）加入到新模型的
   tokenizer 的 special tokens 中，需要通过
   ``--tokenizer-w-special-tokens-save-dir`` 指定新 tokenizer
   的存储路径。\ **同时，后续训练过程需要使用新保存的 tokenizer 而非原始
   tokenizer。**

步骤 2：导出模板 config 文件
----------------------------

XTuner 中目前提供了训练 Mistral 的模板 config，使用命令：

.. code-block:: console

   $ xtuner copy-cfg mistral_7b_w_tokenized_dataset .

.. note::
   当前目录下会有一个名为 ``mistral_7b_w_tokenized_dataset_copy.py`` 的新文件


步骤 3：修改模板 config 文件
----------------------------

.. note::
   修改模板 config 文件中的训练数据路径为真实数据路径，其中 `/path/to/tokenized/data` 需要基于 Step 1 中的 `/path/to/tokenized/data` 进一步指定 train folder，即 `/path/to/tokenized/data/chatml_llamav13_32k/train/` 。

.. note::
   需要修改 tokenizer 路径为步骤 1 保存的路径 `/path/to/save/new/tokenizer`。

.. warning::
   由于步骤 1 扩充了 tokenizer 的词表，因此需要将新 tokenizer 传入 `SupervisedFinetune` 中，以扩展语言模型的词表大小。

.. code:: diff

   ...

   #######################################################################
   #                          PART 1  Settings                           #
   #######################################################################
   # Model
   pretrained_model_name_or_path = 'mistralai/Mistral-7B-v0.1'
   # 已经使用 Internlm2 的对话模板覆盖了 Mistral 的原有模板，new tokenizer 中已经
   # 添加了 Internlm2 对话模板中的特殊字符。
   # 请参考 docs/zh_cn/user_guides/finetune_custom_dataset.md
   - tokenizer_path = '/new/tokenizer/path'
   + tokenizer_path = '/path/to/save/new/tokenizer'
   use_varlen_attn = True

   # Data
   - dataset_folder = '/path/to/sft/data/folder'
   + dataset_folder = '/path/to/tokenized/data/chatml_llamav13_32k/train'
   # 已经使用 Internlm2 的对话模板覆盖了 Mistral 的原有模板
   prompt_template = PROMPT_TEMPLATE.internlm2_chat
   max_length = 32768
   pack_to_max_length = True
   ...

   #######################################################################
   #                      PART 2  Model & Tokenizer                      #
   #######################################################################
   model = dict(
   +   tokenizer=tokenizer,
      ...)

.. tip::
   在使用 DeepSpeed 训练模型时，如需在保存 checkpoint
   时只保存模型权重，而不保存优化器状态，可参考以下步骤：

   1. 确保 mmengine 版本大于等于 0.10.3

   .. code-block:: console

      $ pip install 'mmengine>=0.10.3'

   2. 修改 Config 文件，CheckpointHook 增加 save_optimizer=False

   .. code:: diff

      default_hooks = dict(
         # record the time of every iteration.
         timer=dict(type=IterTimerHook),
         # print log every 100 iterations.
         logger=dict(type=LoggerHook, interval=1),
         # enable the parameter scheduler.
         param_scheduler=dict(type=ParamSchedulerHook),
         # save checkpoint per epoch.
         checkpoint=dict(
            type=CheckpointHook,
      +     save_optimizer=False,
            by_epoch=False,
            interval=save_steps,
            max_keep_ckpts=save_total_limit),
         # set sampler seed in distributed evrionment.
         sampler_seed=dict(type=DistSamplerSeedHook),
      )

.. warning::

    设置 ``save_optimizer=False`` 后，训练过程不可 resume 。


步骤 4：获取数据顺序 （可选）
-----------------------------

训练数据的提供顺序可能会对模型的最终训练成果产生影响。鉴于不同集群中通过
``os.walk``
方法所得到的结果可能存在差异，为了确保训练结果的稳定性和可控性，建议首先确立所有训练数据文件的相对次序，并在后续的训练阶段中，使用这一相对次序来替代
``os.walk`` 方法。

运行下面的代码可获取数据顺序，并存为 txt 文件：

.. code-block:: console

   $ python xtuner/tools/get_data_order.py \
   $    --data-folder /path/to/tokenized/data \
   $    --save-folder /folder/to/save/data/order \
   $    --file-type ${file_type}

.. tip::
   ``--file-type ${file_type}`` 表示需要统计所有以 ``${file_type}``
   为文件名后缀的文件的顺序。

   例如，需要获取 ``/path/to/tokenized/data`` 路径下所有以 ``.bin``
   结尾的文件的顺序，并保存在当前路径下，那么上述命令需要改为：

   .. code-block:: console

      $ python xtuner/tools/get_data_order.py \
      $    --data-folder /path/to/tokenized/data \
      $    --save-folder . \
      $    --file-type .bin

获得数据顺序文件后，还需要在 config 中设置数据顺序文件路径：

.. code:: diff

   ...
   #######################################################################
   #                      PART 3  Dataset & Dataloader                   #
   #######################################################################
   train_dataset = dict(
       type=build_packed_dataset,
       dataset_cfg=dict(
           type=load_intern_repo_tokenized_dataset,
   -       data_order_path=None,
   +       data_order_path='/folder/to/save/data/order/'+'data_order.txt',
           folder=dataset_folder,
           min_length=0,
           file_type='.bin'
       ),
       packed_length=max_length,
       seed=1024)


步骤 5：启动训练
----------------

注：训练前期（几十个 iters）loss 偏高是正常现象，因为模型需要时间学习
Internlm2 的对话模板。

在 slurm 集群调度系统中可以通过以下命令启动训练：

.. code-block:: console

   $ srun ${SRUN_ARGS} xtuner train mistral_7b_w_tokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero1

若出现 OOM 现象，可尝试使用 zero2 或 zero3。以下命令可以使用 zero 3
显存优化策略进行训练：

.. code-block:: console

   $ srun ${SRUN_ARGS} xtuner train internlm2_7b_w_tokenized_dataset_copy.py --launcher slurm --deepspeed deepspeed_zero3

在阿里云 DLC 中可通过以下命令启动训练：

.. code:: diff

   export NCCL_IB_TC=136
   export NCCL_IB_SL=5
   export NCCL_IB_GID_INDEX=3
   export NCCL_SOCKET_IFNAME=bond0
   export NCCL_DEBUG=INFO
   export NCCL_IB_HCA=mlx5
   export NCCL_IB_TIMEOUT=22
   export NCCL_IB_QPS_PER_CONNECTION=8
   export NCCL_NET_PLUGIN=none

   export NCCL_BUFFSIZE=2097152
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   - export EXP_NAME=debug
   + export EXP_NAME=your_exp_name
   export PYTHONPATH='.':$PYTHONPATH
   source ~/.bashrc
   + cd /path/to/xtuner
   + conda activate conda_env_name

   export NPROC_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
   export PORT=${MASTER_PORT}
   export NNODES=${WORLD_SIZE}
   export NODE_RANK=${RANK}
   export ADDR=${MASTER_ADDR}

   echo ${KUBERNETES_CONTAINER_RESOURCE_GPU}
   echo ${WORLD_SIZE}
   echo ${MASTER_PORT}
   echo ${MASTER_ADDR}
   echo ${RANK}
   xtuner train mistral_7b_w_tokenized_dataset_copy.py \
       --deepspeed deepspeed_zero1 \
       --work-dir work_dirs/${EXP_NAME}

Step 6, 转模型
--------------

deepspeed 转 hf：

.. code-block:: console

   $ python xtuner/tools/model_converters/pth_to_hf.py mistral_7b_w_tokenized_dataset_copy.py /src/model/path /hf/dst/model/path

hf 转 Turbomind：

.. code-block:: console

   $ lmdeploy convert internlm2-chat-7b /hf/dst/model/path --dst-path /turbomind/dst/model/path

Step 7，Turbomind 评测
----------------------

请参考 `OpenCompass LMDeploy
评测文档 <https://github.com/open-compass/opencompass/blob/e415ddf96ad5df4640310b12d71cf01e21f8fb32/docs/zh_cn/advanced_guides/evaluation_turbomind.md>`__\ 。
