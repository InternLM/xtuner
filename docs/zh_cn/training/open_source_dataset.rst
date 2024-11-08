================================
开源指令微调数据集（LLM）
================================

HuggingFace Hub 中有众多优秀的开源数据，本节将以
`timdettmers/openassistant-guanaco <https://huggingface.co/datasets/timdettmers/openassistant-guanaco>`__
开源指令微调数据集为例，讲解如何开始训练。为便于介绍，本节以
`internlm2_chat_7b_qlora_oasst1_e3 <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/internlm/internlm2_chat_7b/internlm2_chat_7b_qlora_oasst1_e3.py>`__
配置文件为基础进行讲解。

适配开源数据集
=====================

不同的开源数据集有不同的数据「载入方式」和「字段格式」，因此我们需要针对所使用的开源数据集进行一些适配。

载入方式
-----------

XTuner 使用上游库 ``datasets`` 的统一载入接口 ``load_dataset``\ 。

.. code:: python

   data_path = 'timdettmers/openassistant-guanaco'
   train_dataset = dict(
       type=process_hf_dataset,
       dataset=dict(type=load_dataset, path=data_path),
       ...)

.. tip::
    一般来说，若想要使用不同的开源数据集，用户只需修改
    ``dataset=dict(type=load_dataset, path=data_path)`` 中的 ``path``
    参数即可。

    若想使用 openMind 数据集，可将 ``dataset=dict(type=load_dataset, path=data_path)`` 中的 ``type`` 替换为 ``openmind.OmDataset``。


字段格式
--------

为适配不同的开源数据集的字段格式，XTuner 开发并设计了一套 ``map_fn`` 机制，可以把不同的开源数据集转为统一的字段格式

.. code:: python

   from xtuner.dataset.map_fns import oasst1_map_fn
   train_dataset = dict(
       type=process_hf_dataset,
       ...
       dataset_map_fn=oasst1_map_fn,
       ...)

XTuner 内置了众多 map_fn
（\ `这里 <https://github.com/InternLM/xtuner/tree/main/xtuner/dataset/map_fns/dataset_map_fns>`__\ ），可以满足大多数开源数据集的需要。此处我们罗列一些常用
map_fn 及其对应的原始字段和参考数据集：

+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| map_fn                                                                                                                             | Columns                                           | Reference Datasets                                                                                                    |
+====================================================================================================================================+===================================================+=======================================================================================================================+
| `alpaca_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/alpaca_map_fn.py>`__           | ['instruction',  'input', 'output', ...]          | `tatsu-lab/alpaca <https://huggingface.co/datasets/tatsu-lab/alpaca>`__                                               |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `alpaca_zh_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/alpaca_zh_map_fn.py>`__     | ['instruction_zh',  'input_zh', 'output_zh', ...] | `silk-road/alpaca-data-gpt4-chinese <https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese>`__           |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `oasst1_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/oasst1_map_fn.py>`__           | ['text', ...]                                     | `timdettmers/openassistant-guanaco <https://huggingface.co/datasets/timdettmers/openassistant-guanaco>`__             |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `openai_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/openai_map_fn.py>`__           | ['messages',  ...]                                | `DavidLanz/fine_tuning_datraset_4_openai <https://huggingface.co/datasets/DavidLanz/fine_tuning_datraset_4_openai>`__ |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `code_alpaca_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/code_alpaca_map_fn.py>`__ | ['prompt',  'completion', ...]                    | `HuggingFaceH4/CodeAlpaca_20K <https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K>`__                       |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `medical_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/medical_map_fn.py>`__         | ['instruction',  'input', 'output', ...]          | `shibing624/medical <https://huggingface.co/datasets/shibing624/medical>`__                                           |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `tiny_codes_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/tiny_codes_map_fn.py>`__   | ['prompt',  'response', ...]                      | `nampdn-ai/tiny-codes <https://huggingface.co/datasets/nampdn-ai/tiny-codes>`__                                       |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| `default_map_fn <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/map_fns/dataset_map_fns/default_map_fn.py>`__         | ['input',  'output', ...]                         | /                                                                                                                     |
+------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+

例如，针对 ``timdettmers/openassistant-guanaco`` 数据集，XTuner 内置了
``oasst1_map_fn``\ ，以对其进行字段格式统一。具体实现如下：

.. code:: python

   def oasst1_map_fn(example):
       r"""Example before preprocessing:
           example['text'] = ('### Human: Can you explain xxx'
                              '### Assistant: Sure! xxx'
                              '### Human: I didn't understand how xxx'
                              '### Assistant: It has to do with a process xxx.')

       Example after preprocessing:
           example['conversation'] = [
               {
                   'input': 'Can you explain xxx',
                   'output': 'Sure! xxx'
               },
               {
                   'input': 'I didn't understand how xxx',
                   'output': 'It has to do with a process xxx.'
               }
           ]
       """
       data = []
       for sentence in example['text'].strip().split('###'):
           sentence = sentence.strip()
           if sentence[:6] == 'Human:':
               data.append(sentence[6:].strip())
           elif sentence[:10] == 'Assistant:':
               data.append(sentence[10:].strip())
       if len(data) % 2:
           # The last round of conversation solely consists of input
           # without any output.
           # Discard the input part of the last round, as this part is ignored in
           # the loss calculation.
           data.pop()
       conversation = []
       for i in range(0, len(data), 2):
           single_turn_conversation = {'input': data[i], 'output': data[i + 1]}
           conversation.append(single_turn_conversation)
       return {'conversation': conversation}

通过代码可以看到，\ ``oasst1_map_fn`` 对原数据中的 ``text``
字段进行处理，进而构造了一个 ``conversation``
字段，以此确保了后续数据处理流程的统一。

值得注意的是，如果部分开源数据集依赖特殊的
map_fn，则需要用户自行参照以提供的 map_fn
进行自定义开发，实现字段格式的对齐。

训练
=====

用户可以使用 ``xtuner train`` 启动训练。假设所使用的配置文件路径为
``./config.py``\ ，并使用 DeepSpeed ZeRO-2 优化。

单机单卡
--------

.. code:: console

    $ xtuner train ./config.py --deepspeed deepspeed_zero2

单机多卡
--------

.. code:: console

    $ NPROC_PER_NODE=${GPU_NUM} xtuner train ./config.py --deepspeed deepspeed_zero2

多机多卡（以 2 \* 8 GPUs 为例）
--------------------------------------

**方法 1：torchrun**

.. code:: console

    $ # excuete on node 0
    $ NPROC_PER_NODE=8 NNODES=2 PORT=$PORT ADDR=$NODE_0_ADDR NODE_RANK=0 xtuner train mixtral_8x7b_instruct_full_oasst1_e3 --deepspeed deepspeed_zero2

    $ # excuete on node 1
    $ NPROC_PER_NODE=8 NNODES=2 PORT=$PORT ADDR=$NODE_0_ADDR NODE_RANK=1 xtuner train mixtral_8x7b_instruct_full_oasst1_e3 --deepspeed deepspeed_zero2

.. note::

    \ ``$PORT`` 表示通信端口、\ ``$NODE_0_ADDR`` 表示 node 0 的 IP 地址。
    二者并不是系统自带的环境变量，需要根据实际情况，替换为实际使用的值

**方法 2：slurm**

.. code:: console

    $ srun -p $PARTITION --nodes=2 --gres=gpu:8 --ntasks-per-node=8 xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2

模型转换
=========

模型训练后会自动保存成 PTH 模型（例如 ``iter_500.pth``\ ），我们需要利用
``xtuner convert pth_to_hf`` 将其转换为 HuggingFace
模型，以便于后续使用。具体命令为：

.. code:: console

   $ xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
   $ # 例如：xtuner convert pth_to_hf ./config.py ./iter_500.pth ./iter_500_hf

.. _模型合并可选）:

模型合并（可选）
================

如果您使用了 LoRA / QLoRA 微调，则模型转换后将得到 adapter
参数，而并不包含原 LLM
参数。如果您期望获得合并后的模型权重，那么可以利用
``xtuner convert merge`` ：

.. code:: console

   $ xtuner convert merge ${LLM} ${ADAPTER_PATH} ${SAVE_PATH}
   $ # 例如：xtuner convert merge internlm/internlm2-chat-7b ./iter_500_hf ./iter_500_merged_llm

对话
=====

用户可以利用 ``xtuner chat`` 实现与微调后的模型对话：

.. code:: console

   $ xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter ${NAME_OR_PATH_TO_ADAPTER} --prompt-template ${PROMPT_TEMPLATE} [optional arguments]

.. tip::

   例如：

   .. code:: console

        $ xtuner chat internlm2/internlm2-chat-7b --adapter ./iter_500_hf --prompt-template internlm2_chat
        $ xtuner chat ./iter_500_merged_llm --prompt-template internlm2_chat
