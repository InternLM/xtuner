============
修改训练配置
============

XTuner 的训练由 MMEngine
的训练器提供支持，用户可以通过修改配置文件（config）中的特定参数，来修改对应的训练配置。以
`internlm2_chat_7b_qlora_oasst1_e3 <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/internlm/internlm2_chat_7b/internlm2_chat_7b_qlora_oasst1_e3.py>`__
为例，本节将首先速览配置文件中各个参数的含义，之后讲解常见配置的修改方式。

配置文件速览
============

XTuner 使用 MMEngine 的「纯 Python 风格的配置文件」，直接利用 ``import``
机制使用一些类或函数。

.. tip::

   如果您期望深入了解 MMEngine 「纯 Python
   风格的配置文件」的特性、优势，请参考
   `这里 <https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python-beta>`__\ 。

.. code:: python

   # Copyright (c) OpenMMLab. All rights reserved.
   import torch
   from datasets import load_dataset
   from mmengine.dataset import DefaultSampler
   from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                               LoggerHook, ParamSchedulerHook)
   from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
   from peft import LoraConfig
   from torch.optim import AdamW
   from transformers import (AutoModelForCausalLM, AutoTokenizer,
                             BitsAndBytesConfig)

   from xtuner.dataset import process_hf_dataset
   from xtuner.dataset.collate_fns import default_collate_fn
   from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
   from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                    VarlenAttnArgsToMessageHubHook)
   from xtuner.engine.runner import TrainLoop
   from xtuner.model import SupervisedFinetune
   from xtuner.utils import PROMPT_TEMPLATE

   #######################################################################
   #                          PART 1  Settings                           #
   #######################################################################
   # Model
   pretrained_model_name_or_path = 'internlm/internlm2-chat-7b'  # 设置 LLM 路径或 HuggingFace Hub ID
   use_varlen_attn = False  # 是否使用 varlen_attention

   # Data
   data_path = 'timdettmers/openassistant-guanaco'  # 设置 dataset 路径或 HuggingFace Hub ID，以用于 datasets.load_dataset
   prompt_template = PROMPT_TEMPLATE.internlm2_chat  # 设置对话模版
   max_length = 2048  # 设置训练数据截断长度
   pack_to_max_length = True  # 是否将多条样本打包为一条最长长度的样本

   # Scheduler & Optimizer
   batch_size = 1  # per_device  # 每个设备的样本个数
   accumulative_counts = 16  # 梯度累计数
   dataloader_num_workers = 0  # dataloader worker 数
   max_epochs = 3  # 训练迭代代数
   optim_type = AdamW  # 优化器
   lr = 2e-4  # 学习率
   betas = (0.9, 0.999)  # AdamW 优化器 betas
   weight_decay = 0  # AdamW 优化器权重衰减
   max_norm = 1  # grad clip  # 梯度裁剪
   warmup_ratio = 0.03  # warmup 比率

   # Save
   save_steps = 500  # checkpoint 保存间隔（iter 数）
   save_total_limit = 2  # 最大保存 checkpoint 个数，-1 表示无限制

   # Evaluate the generation performance during the training
   evaluation_freq = 500  # 验证对话效果的执行间隔（iter 数）
   SYSTEM = ''  # 验证对话效果的 system 字段
   evaluation_inputs = [  # 验证对话效果时的测试问题
       '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
   ]

   #######################################################################
   #                      PART 2  Model & Tokenizer                      #
   #######################################################################
   tokenizer = dict(  # 构建 tokenizer
       type=AutoTokenizer.from_pretrained,
       pretrained_model_name_or_path=pretrained_model_name_or_path,
       trust_remote_code=True,
       padding_side='right')

   model = dict(  # 构建 model
       type=SupervisedFinetune,
       use_varlen_attn=use_varlen_attn,
       llm=dict(  # 构建 LLM
           type=AutoModelForCausalLM.from_pretrained,
           pretrained_model_name_or_path=pretrained_model_name_or_path,
           trust_remote_code=True,
           torch_dtype=torch.float16,
           quantization_config=dict(  # 量化配置（保留则为 4 比特，删除则为正常浮点）
               type=BitsAndBytesConfig,
               load_in_4bit=True,
               load_in_8bit=False,
               llm_int8_threshold=6.0,
               llm_int8_has_fp16_weight=False,
               bnb_4bit_compute_dtype=torch.float16,
               bnb_4bit_use_double_quant=True,
               bnb_4bit_quant_type='nf4')),
       lora=dict(  # LoRA 配置（保留则使用 LoRA 微调，删除则使用全量微调）
           type=LoraConfig,
           r=64,
           lora_alpha=16,
           lora_dropout=0.1,
           bias='none',
           task_type='CAUSAL_LM'))

   #######################################################################
   #                      PART 3  Dataset & Dataloader                   #
   #######################################################################
   train_dataset = dict(  # 构建训练数据集
       type=process_hf_dataset,
       dataset=dict(type=load_dataset, path=data_path),  # 调用 datasets.load_dataset 接口
       tokenizer=tokenizer,
       max_length=max_length,
       dataset_map_fn=oasst1_map_fn,  # 选择匹配的数据集 map_fn
       template_map_fn=dict(
           type=template_map_fn_factory, template=prompt_template),
       remove_unused_columns=True,
       shuffle_before_pack=True,
       pack_to_max_length=pack_to_max_length,
       use_varlen_attn=use_varlen_attn)

   train_dataloader = dict(  # 构建训练数据集的 DataLoader
       batch_size=batch_size,
       num_workers=dataloader_num_workers,
       dataset=train_dataset,
       sampler=dict(type=DefaultSampler, shuffle=True),
       collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

   #######################################################################
   #                    PART 4  Scheduler & Optimizer                    #
   #######################################################################
   # optimizer
   optim_wrapper = dict(  # 构建优化器
       type=AmpOptimWrapper,
       optimizer=dict(
           type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
       clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
       accumulative_counts=accumulative_counts,
       loss_scale='dynamic',
       dtype='float16')

   # learning policy
   # More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
   param_scheduler = [  # 设置学习率 scheduler
       dict(
           type=LinearLR,  # warmup 阶段
           start_factor=1e-5,
           by_epoch=True,
           begin=0,
           end=warmup_ratio * max_epochs,
           convert_to_iter_based=True),
       dict(
           type=CosineAnnealingLR,  # Cosine 学习率衰减阶段
           eta_min=0.0,
           by_epoch=True,
           begin=warmup_ratio * max_epochs,
           end=max_epochs,
           convert_to_iter_based=True)
   ]

   # train, val, test setting
   train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)  # 设置训练迭代代数

   #######################################################################
   #                           PART 5  Runtime                           #
   #######################################################################
   # Log the dialogue periodically during the training process, optional
   custom_hooks = [  # 定义 Hooks
       dict(type=DatasetInfoHook, tokenizer=tokenizer),  # 在训练前打印可视化打印数据样本
       dict(
           type=EvaluateChatHook,  # 在训练时测试对话效果
           tokenizer=tokenizer,
           every_n_iters=evaluation_freq,
           evaluation_inputs=evaluation_inputs,
           system=SYSTEM,
           prompt_template=prompt_template)
   ]

   if use_varlen_attn:
       custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]  # vallen_attention 依赖的 Hook

   # 以下均为默认配置，如需调整请参考 MMEngine 文档及代码

   # configure default hooks
   default_hooks = dict(
       # record the time of every iteration.
       timer=dict(type=IterTimerHook),
       # print log every 10 iterations.
       logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
       # enable the parameter scheduler.
       param_scheduler=dict(type=ParamSchedulerHook),
       # save checkpoint per `save_steps`.
       checkpoint=dict(
           type=CheckpointHook,
           by_epoch=False,
           interval=save_steps,
           max_keep_ckpts=save_total_limit),
       # set sampler seed in distributed evrionment.
       sampler_seed=dict(type=DistSamplerSeedHook),
   )

   # configure environment
   env_cfg = dict(
       # whether to enable cudnn benchmark
       cudnn_benchmark=False,
       # set multi process parameters
       mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
       # set distributed parameters
       dist_cfg=dict(backend='nccl'),
   )

   # set visualizer
   visualizer = None

   # set log level
   log_level = 'INFO'

   # load from which checkpoint
   load_from = None

   # whether to resume training from the loaded checkpoint
   resume = False

   # Defaults to use random seed and disable `deterministic`
   randomness = dict(seed=None, deterministic=False)

   # set log processor
   log_processor = dict(by_epoch=False)

常见训练配置修改
=======================

模型
------------

使用其他 LLM 模型？
~~~~~~~~~~~~~~~~~~~~~~~~
1.  修改 ``pretrained_model_name_or_path``\ ，其将应用至 ``model.llm`` 和 ``tokenizer`` 的初始化中。
#.  修改 ``prompt_template`` 以适配所选择的 LLM。

使用 ModelScope 模型？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  参考 `文档 <../preparation/pretrained_model.md>`__ 将其下载至本地
2.  修改\ ``pretrained_model_name_or_path``\ 。

使用 openMind 模型？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
可在配置文件中新增 ``model_resource`` 参数， ``args`` 用作可变参数（如下载私有模型需传入token的情况）：

.. code:: python
   from openmind_hub import snapshot_download

   # Model
   pretrained_model_name_or_path = 'Tianjin_Ascend/Qwen1.5-4B'
   model_resource = {
      "fn": snapshot_download,
      "args":{
         # "token":"xxxxxxxxxx"
      }
   }

微调类型
-------------

.. tip::
   XTuner 内置的配置文件以 QLoRA 微调为主，但并不意味着 XTuner 仅支持 QLoRA
   微调。用户可以通过修改配置文件中的 ``model`` 来决定微调类型。


QLoRA 微调
~~~~~~~~~~~~~~~~~

.. code:: python

   model = dict(
         ......
         llm=dict(
            type=AutoModelForCausalLM.from_pretrained,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=dict(
               type=BitsAndBytesConfig,
               load_in_4bit=True,
               load_in_8bit=False,
               llm_int8_threshold=6.0,
               llm_int8_has_fp16_weight=False,
               bnb_4bit_compute_dtype=torch.float16,
               bnb_4bit_use_double_quant=True,
               bnb_4bit_quant_type='nf4')),
         lora=dict(
            type=LoraConfig,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'),
         ......)


LoRA 微调
~~~~~~~~~~~~~~~~

.. tip::

   在 QLoRA 设置的基础上，将 `quantization_config` 设置为 None，就切换成了 LoRA 微调

.. code:: python

   model = dict(
         ......
         llm=dict(
            type=AutoModelForCausalLM.from_pretrained,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=None),
         lora=dict(
            type=LoraConfig,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM'),
         ......)


全参数微调
~~~~~~~~~~~~~~~~~~
.. tip::

   将 `lora` 和 `quantization_config` 都设置为 None，就切换到了全参数训练模式

.. code:: python

   model = dict(
         ......
         llm=dict(
            type=AutoModelForCausalLM.from_pretrained,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=None),
         lora=None,
         ......)




数据集
--------------

请参考 `训练` 章节文档。

优化器
-----------

使用其他优化器？
~~~~~~~~~~~~~~~~~~~~

-  方法 1：修改 ``optim_type``\ （例如 ``optim_type=torch.optim.SGD``\ ），其将应用至 ``optim_wrapper.optimzer``\ 。
-  方法 2：忽略 ``optim_type``\ ，直接修改 ``optim_wrapper.optimzer``\ 。


修改优化器参数配置？
~~~~~~~~~~~~~~~~~~~~~~~~

-  方法 1：修改 ``lr``\ 、\ ``weight_decay`` 等参数，其将应用至 ``optim_wrapper.optimzer``\ 。
-  方法 2：直接修改 ``optim_wrapper.optimzer``\ 。

迭代次数
---------------

调整迭代次数？
~~~~~~~~~~~~~~~~~~~~~

-  修改 ``max_epochs`` 参数。

保存 Checkpoint 间隔
---------------------------

调整保存间隔？
~~~~~~~~~~~~~~~~~~~~~

-  修改 ``save_steps`` 参数。

调整最大保存 checkpoint 个数？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  修改 ``save_total_limit`` 参数。

训练间对话评测
----------------------

调整对话评测间隔？
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  修改 ``evaluation_freq`` 参数。

调整对话评测的 system 字段？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  修改 ``SYSTEM`` 参数。

调整对话评测的测试指令？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  修改 ``evaluation_inputs`` 参数。

GPU 数
--------------

XTuner
的多卡训练由启动命令决定，而非配置文件。用户可以参考下列命令启动多卡训练：

.. code:: bash

   # 单卡
   xtuner train ${CONFIG}
   # 多卡
   (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG}
   (SLURM) srun ${SRUN_ARGS} xtuner train ${CONFIG} --launcher slurm

DeepSpeed
------------------

XTuner 的 DeepSpeed
优化由启动命令决定，而非配置文件。用户可以参考下列命令启用 DeepSpeed
优化：

.. code:: bash

   xtuner train ${CONFIG} --deepspeed ${DS_CONFIG}

.. note::

   XTuner 内置了多个 DeepSpeed 配置文件（即命令中的
   ``${DS_CONFIG}``\ ），用户可以直接使用，具体文件见
   `这里 <https://github.com/InternLM/xtuner/tree/main/xtuner/configs/deepspeed>`__\ ：

   .. code:: bash

      xtuner train ${CONFIG} --deepspeed [deepspeed_zero1,deepspeed_zero2,deepspeed_zero2_offload,deepspeed_zero3,deepspeed_zero3_offload]

.. note::
   部分参数会在 DeepSpeed Config 和 XTuner Config 中重复定义（例如 batch
   size等）。此时相关配置会以 XTuner Config 为准：

   -  ``gradient_accumulation_steps`` 会被 XTuner Config 中的
      ``accumulative_counts`` 设置覆盖。

   -  ``train_micro_batch_size_per_gpu`` 会被 XTuner Config 中的
      ``train_dataloader.batch_size`` 设置覆盖。

   -  ``gradient_clipping`` 会被 XTuner Config 中的
      ``optim_wrapper.clip_grad.max_norm`` 设置覆盖。

   -  XTuner 会根据所使用的 GPU 架构自动选择 ``fp16`` 或 ``bf16`` 训练。

其他
----------

如有遗漏或特定需求，欢迎提出
`issue <https://github.com/InternLM/xtuner/issues>`__ 讨论。
