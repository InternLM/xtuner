===================================
自定义指令微调数据集（LLM）
===================================

XTuner 支持使用自定义数据集进行指令微调，为便于介绍，本节以
`internlm2_chat_7b_qlora_custom_sft_e1.py <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/custom_dataset/sft/internlm/internlm2_chat_7b_qlora_custom_sft_e1.py>`__
配置文件为基础进行介绍。

数据准备
=================

XTuner 采用 `OpenAI SFT
数据集格式 <https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset>`__
作为统一的自定义数据集格式，详细格式如下：

.. code:: json

   [{
       "messages": [
           { "role": "system", "content": "xxx."},
           { "role": "user", "content": "xxx." },
           { "role": "assistant", "content": "xxx."}
       ]
   },
   {
       "messages": [
           { "role": "system", "content": "xxx." },
           { "role": "user", "content": "xxx." },
           { "role": "assistant", "content": "xxx.", "loss": False},
           { "role": "user", "content": "xxx." },
           { "role": "assistant", "content": "xxx.", "loss": True}
       ]
   }]

.. note::
   每条数据除了 OpenAI 标准格式中的 ``role``
   字段和 ``content`` 字段外，XTuner 还额外扩充了一个 ``loss``
   字段，用于控制某轮 ``assistant`` 的输出不计算 loss。

.. note::
   - ``system`` 和 ``user`` 的 ``loss`` 默认为 False
   - ``assistant`` 的 ``loss`` 默认为 True

.. tip::

   若想令某轮对话 "assistant"
   部分的内容不参与 loss 计算，需要手动设置该数据 "loss" 字段的值为
   ``false``\ 。

训练
=============

步骤 1： 导出 config
--------------------------------

``xtuner/configs/custom_dataset/sft`` 目录下有所有 XTuner
支持的模型在自定义数据集下使用 QLora 算法训练的模板 config。可以通过
``xtuner list-cfg -p custom_sft`` 命令来查看候选 config。下面以
`internlm2_chat_7b_qlora_custom_sft_e1.py <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/custom_dataset/sft/internlm/internlm2_chat_7b_qlora_custom_sft_e1.py>`__
为例展开介绍。

可以通过以下命令将 ``internlm2_chat_7b_qlora_custom_sft_e1.py``
导出至当前目录下：

.. code:: console

   $ xtuner copy-cfg internlm2_chat_7b_qlora_custom_sft_e1 .

.. note::

   当前目录下会存在一个新 config
   ``internlm2_chat_7b_qlora_custom_sft_e1_copy.py`` 。

步骤 2：修改 config
----------------------------------

首先，需要修改数据集文件路径：

.. code:: diff

   - data_files = ['/path/to/json/file.json']
   + data_files = ['/path/to/custom_sft1.json', '/path/to/custom_sft2.json', ...]

若期望使用某个目录下所有的 json 文件作为训练数据集，可做如下修改：

.. code:: diff

   #######################################################################
   #                          PART 1  Settings                           #
   #######################################################################
   # Data
   - data_files = ['/path/to/json/file.json']
   + data_dir = '/dir/to/custom_sft'

   #######################################################################
   #                      PART 3  Dataset & Dataloader                   #
   #######################################################################
   train_dataset = dict(
   -   dataset=dict(type=load_dataset, path='json', data_files=data_files),
   +   dataset=dict(type=load_dataset, path='json', data_dir=data_dir),
       ...)

若期望使用 Lora 算法训练，可做如下修改：

.. code:: diff

   #######################################################################
   #                      PART 2  Model & Tokenizer                      #
   #######################################################################
   model = dict(
       type=SupervisedFinetune,
       use_varlen_attn=use_varlen_attn,
       llm=dict(
           type=AutoModelForCausalLM.from_pretrained,
           pretrained_model_name_or_path=pretrained_model_name_or_path,
           trust_remote_code=True,
           torch_dtype=torch.float16,
   -       quantization_config=dict(
   -           type=BitsAndBytesConfig,
   -           load_in_4bit=True,
   -           load_in_8bit=False,
   -           llm_int8_threshold=6.0,
   -           llm_int8_has_fp16_weight=False,
   -           bnb_4bit_compute_dtype=torch.float16,
   -           bnb_4bit_use_double_quant=True,
   -           bnb_4bit_quant_type='nf4')
       ),
       lora=dict(
           type=LoraConfig,
           r=64,
           lora_alpha=16,
           lora_dropout=0.1,
           bias='none',
           task_type='CAUSAL_LM'))

若期望进行全量参数训练，可做如下修改：

.. code:: diff

   #######################################################################
   #                      PART 2  Model & Tokenizer                      #
   #######################################################################
   model = dict(
       type=SupervisedFinetune,
       use_varlen_attn=use_varlen_attn,
       llm=dict(
           type=AutoModelForCausalLM.from_pretrained,
           pretrained_model_name_or_path=pretrained_model_name_or_path,
           trust_remote_code=True,
           torch_dtype=torch.float16,
   -       quantization_config=dict(
   -           type=BitsAndBytesConfig,
   -           load_in_4bit=True,
   -           load_in_8bit=False,
   -           llm_int8_threshold=6.0,
   -           llm_int8_has_fp16_weight=False,
   -           bnb_4bit_compute_dtype=torch.float16,
   -           bnb_4bit_use_double_quant=True,
   -           bnb_4bit_quant_type='nf4')
       ),
   -   lora=dict(
   -       type=LoraConfig,
   -       r=64,
   -       lora_alpha=16,
   -       lora_dropout=0.1,
   -       bias='none',
   -       task_type='CAUSAL_LM')
   )

步骤 3： 开始训练
-----------------------------

.. code:: console

   $ NPROC_PER_NODE=8 xtuner train internlm2_chat_7b_qlora_custom_sft_e1_copy.py --deepspeed deepspeed_zero1

.. tip::
   训练日志及 checkpoint 将默认保存在 ``./work_dirs/``\ ，可以通过命令
   ``xtuner train --work-dir ${SAVE_PATH}`` 指定保存路径。

步骤 4： 模型转换
------------------------------

模型训练后会自动保存成 PTH 模型（例如 ``iter_2000.pth``\ ，如果使用了
DeepSpeed，则将会是一个文件夹），我们需要利用
``xtuner convert pth_to_hf`` 将其转换为 HuggingFace
模型，以便于后续使用。具体命令为：

.. code:: bash

   xtuner convert pth_to_hf ${FINETUNE_CFG} ${PTH_PATH} ${SAVE_PATH}
   # 例如：xtuner convert pth_to_hf internlm2_chat_7b_qlora_custom_sft_e1_copy.py ./iter_2000.pth ./iter_2000_hf

对话
=================

用户可以利用 ``xtuner chat`` 实现与微调后的模型对话。如果使用的是 Lora
或 QLora 算法：

.. code:: console

   $ xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter {NAME_OR_PATH_TO_ADAPTER} --prompt-template ${PROMPT_TEMPLATE} [optional arguments]
   $ # 例如：xtuner chat internlm/internlm2-7b --adapter ./iter_2000_hf --prompt-template internlm2_chat


如果进行的是全量参数的微调：

.. code:: console

   $ xtuner chat ${PATH_TO_LLM} --prompt-template ${PROMPT_TEMPLATE} [optional arguments]
   $ # 例如：xtuner chat ./iter_2000_hf --prompt-template internlm2_chat

.. note::

   其中 ${PROMPT_TEMPLATE} 表示模型的对话模板，需要与训练用的 config 中的
   ``prompt_template`` 字段保持一致，例如
   ``internlm2_chat_7b_qlora_custom_sft_e1_copy.py`` 中的设置为：

   .. code:: python

      prompt_template = PROMPT_TEMPLATE.internlm2_chat

.. _模型合并可选）:

模型合并（可选）
======================

如果您使用了 LoRA / QLoRA 微调，则模型转换后将得到 adapter
参数，而并不包含原 LLM
参数。如果您期望获得合并后的模型权重（例如用于后续评测），那么可以利用
``xtuner convert merge`` ：

.. code:: console

   $ xtuner convert merge ${LLM} ${LLM_ADAPTER} ${SAVE_PATH}

.. tip::

   模型合并后，就得到了一个可以通过 ``AutoModelForCausalLM.from_pretrained`` 直接加载的模型，可以直接在各种下游工具中直接使用

评测
======================

推荐使用一站式平台
`OpenCompass <https://github.com/InternLM/opencompass>`__
来评测大语言模型，其目前已涵盖 50+ 数据集的约 30 万条题目。
