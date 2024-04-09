快速上手
========

本节中，我们将演示如何使用 XTuner 微调模型，帮助您快速上手 XTuner。

在成功安装 XTuner
后，便可以开始进行模型的微调。在本节中，我们将演示如何使用 XTuner，应用
QLoRA 算法在 Colorist 数据集上微调 InternLM2-Chat-7B。

Colorist 数据集（\ `HuggingFace
链接 <https://huggingface.co/datasets/burkelibbey/colors>`__\ ；\ `ModelScope
链接 <https://www.modelscope.cn/datasets/fanqiNO1/colors/summary>`__\ ）是一个根据颜色描述提供颜色选择与建议的数据集，经过该数据集微调的模型可以做到根据用户对于颜色的描述，从而给出16进制下的颜色编码，如用户输入“宁静而又相当明亮的浅天蓝色，介于天蓝色和婴儿蓝之间，因其亮度而带有一丝轻微的荧光感。”，模型输出
|image1|\ ，该颜色很符合用户的描述。以下是该数据集的几条样例数据：

+-----------------------+-----------------------+-------------------+
| 英文描述              | 中文描述              | 颜色              |
+=======================+=======================+===================+
| Light Sky Blue: A     | 浅天蓝色              | #66ccff: |image8| |
| calming, fairly       | ：一种介于天蓝和婴儿  |                   |
| bright color that     | 蓝之间的平和、相当明  |                   |
| falls between sky     | 亮的颜色，由于明亮而  |                   |
| blue and baby blue,   | 带有一丝轻微的荧光。  |                   |
| with a hint of slight |                       |                   |
| fluorescence due to   |                       |                   |
| its brightness.       |                       |                   |
+-----------------------+-----------------------+-------------------+
| Bright red: This is a | 鲜红色：              | #ee0000: |image9| |
| very vibrant,         | 这是一种非常鲜        |                   |
| saturated and vivid   | 艳、饱和、生动的红色  |                   |
| shade of red,         | ，类似成熟苹果或新鲜  |                   |
| resembling the color  | 血液的颜色。它是标准  |                   |
| of ripe apples or     | RGB                   |                   |
| fresh blood. It is as | 调色板上的红色，不含  |                   |
| red as you can get on | 任何蓝色或绿色元素。  |                   |
| a standard RGB color  |                       |                   |
| palette, with no      |                       |                   |
| elements of either    |                       |                   |
| blue or green.        |                       |                   |
+-----------------------+-----------------------+-------------------+
| Bright Turquoise:     | 明亮的绿松石          | #00ffcc:          |
| This color mixes the  | 色：这种颜色融合了鲜  | |image10|         |
| freshness of bright   | 绿色的清新和淡蓝色的  |                   |
| green with the        | 宁静，呈现出一种充满  |                   |
| tranquility of light  | 活力的绿松石色调。它  |                   |
| blue, leading to a    | 让人联想到热带水域。  |                   |
| vibrant shade of      |                       |                   |
| turquoise. It is      |                       |                   |
| reminiscent of        |                       |                   |
| tropical waters.      |                       |                   |
+-----------------------+-----------------------+-------------------+

准备模型权重
------------

在微调模型前，首先要准备待微调模型的权重。

.. _从-huggingface-下载-1:

从 HuggingFace 下载
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install -U huggingface_hub

   # 拉取模型至 Shanghai_AI_Laboratory/internlm2-chat-7b
   huggingface-cli download internlm/internlm2-chat-7b \
                               --local-dir Shanghai_AI_Laboratory/internlm2-chat-7b \
                               --local-dir-use-symlinks False \
                               --resume-download

.. _从-modelscope-下载-1:

从 ModelScope 下载
~~~~~~~~~~~~~~~~~~

由于从 HuggingFace
拉取模型权重，可能存在下载过程不稳定、下载速度过慢等问题。因此在下载过程遇到网络问题时，我们则可以选择从
ModelScope 下载 InternLM2-Chat-7B 的权重。

.. code:: bash

   pip install -U modelscope

   # 拉取模型至当前目录
   python -c "from modelscope import snapshot_download; snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='.')"

在完成下载后，便可以开始准备微调数据集了。

此处附上 HuggingFace 链接与 ModelScope 链接：

-  HuggingFace
   链接位于：\ https://huggingface.co/internlm/internlm2-chat-7b

-  ModelScope
   链接位于：\ https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary

准备微调数据集
--------------

接下来，我们需要准备微调数据集。

.. _从-huggingface-下载-2:

从 HuggingFace 下载
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://huggingface.co/datasets/burkelibbey/colors

.. _从-modelscope-下载-2:

从 ModelScope 下载
~~~~~~~~~~~~~~~~~~

由于相同的问题，因此我们可以选择从 ModelScope 下载所需要的微调数据集。

.. code:: bash

   git clone https://www.modelscope.cn/datasets/fanqiNO1/colors.git

此处附上 HuggingFace 链接与 ModelScope 链接：

-  HuggingFace
   链接位于：\ https://huggingface.co/datasets/burkelibbey/colors

-  ModelScope 链接位于：\ https://modelscope.cn/datasets/fanqiNO1/colors

准备配置文件
------------

XTuner 提供了多个开箱即用的配置文件，可以通过 ``xtuner list-cfg``
查看。我们执行如下指令，以复制一个配置文件到当前目录。

.. code:: bash

   xtuner copy-cfg internlm2_7b_qlora_colorist_e5 .

配置文件名的解释：

======== ==============================
配置文件 internlm2_7b_qlora_colorist_e5
======== ==============================
模型名   internlm2_7b
使用算法 qlora
数据集   colorist
训练时长 5 epochs
======== ==============================

此时该目录文件结构应如下所示：

.. code:: bash

   .
   ├── colors
   │   ├── colors.json
   │   ├── dataset_infos.json
   │   ├── README.md
   │   └── train.jsonl
   ├── internlm2_7b_qlora_colorist_e5_copy.py
   └── Shanghai_AI_Laboratory
       └── internlm2-chat-7b
           ├── config.json
           ├── configuration_internlm2.py
           ├── configuration.json
           ├── generation_config.json
           ├── modeling_internlm2.py
           ├── pytorch_model-00001-of-00008.bin
           ├── pytorch_model-00002-of-00008.bin
           ├── pytorch_model-00003-of-00008.bin
           ├── pytorch_model-00004-of-00008.bin
           ├── pytorch_model-00005-of-00008.bin
           ├── pytorch_model-00006-of-00008.bin
           ├── pytorch_model-00007-of-00008.bin
           ├── pytorch_model-00008-of-00008.bin
           ├── pytorch_model.bin.index.json
           ├── README.md
           ├── special_tokens_map.json
           ├── tokenization_internlm2_fast.py
           ├── tokenization_internlm2.py
           ├── tokenizer_config.json
           └── tokenizer.model

修改配置文件
------------

| 在这一步中，我们需要修改待微调模型路径和数据路径为本地路径，并且修改数据集加载方式。
| 此外，由于复制得到的配置文件是基于基座（Base）模型的，所以还需要修改
  ``prompt_template`` 以适配对话（Chat）模型。

.. code:: diff

   #######################################################################
   #                          PART 1  Settings                           #
   #######################################################################
   # Model
   - pretrained_model_name_or_path = 'internlm/internlm2-7b'
   + pretrained_model_name_or_path = './Shanghai_AI_Laboratory/internlm2-chat-7b'

   # Data
   - data_path = 'burkelibbey/colors'
   + data_path = './colors/train.jsonl'
   - prompt_template = PROMPT_TEMPLATE.default
   + prompt_template = PROMPT_TEMPLATE.internlm2_chat

   ...
   #######################################################################
   #                      PART 3  Dataset & Dataloader                   #
   #######################################################################
   train_dataset = dict(
       type=process_hf_dataset,
   -   dataset=dict(type=load_dataset, path=data_path),
   +   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
       tokenizer=tokenizer,
       max_length=max_length,
       dataset_map_fn=colors_map_fn,
       template_map_fn=dict(
           type=template_map_fn_factory, template=prompt_template),
       remove_unused_columns=True,
       shuffle_before_pack=True,
       pack_to_max_length=pack_to_max_length)

因此在这一步中，修改了
``pretrained_model_name_or_path``\ 、\ ``data_path``\ 、\ ``prompt_template``
以及 ``train_dataset`` 中的 ``dataset`` 字段。

启动微调
--------

在完成上述操作后，便可以使用下面的指令启动微调任务了。

.. code:: bash

   # 单机单卡
   xtuner train ./internlm2_7b_qlora_colorist_e5_copy.py
   # 单机多卡
   NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm2_7b_qlora_colorist_e5_copy.py
   # slurm 情况
   srun ${SRUN_ARGS} xtuner train ./internlm2_7b_qlora_colorist_e5_copy.py --launcher slurm

正确输出的训练日志应类似如下所示：

.. code:: text

   01/29 21:35:34 - mmengine - INFO - Iter(train) [ 10/720]  lr: 9.0001e-05  eta: 0:31:46  time: 2.6851  data_time: 0.0077  memory: 12762  loss: 2.6900
   01/29 21:36:02 - mmengine - INFO - Iter(train) [ 20/720]  lr: 1.9000e-04  eta: 0:32:01  time: 2.8037  data_time: 0.0071  memory: 13969  loss: 2.6049  grad_norm: 0.9361
   01/29 21:36:29 - mmengine - INFO - Iter(train) [ 30/720]  lr: 1.9994e-04  eta: 0:31:24  time: 2.7031  data_time: 0.0070  memory: 13969  loss: 2.5795  grad_norm: 0.9361
   01/29 21:36:57 - mmengine - INFO - Iter(train) [ 40/720]  lr: 1.9969e-04  eta: 0:30:55  time: 2.7247  data_time: 0.0069  memory: 13969  loss: 2.3352  grad_norm: 0.8482
   01/29 21:37:24 - mmengine - INFO - Iter(train) [ 50/720]  lr: 1.9925e-04  eta: 0:30:28  time: 2.7286  data_time: 0.0068  memory: 13969  loss: 2.2816  grad_norm: 0.8184
   01/29 21:37:51 - mmengine - INFO - Iter(train) [ 60/720]  lr: 1.9863e-04  eta: 0:29:58  time: 2.7048  data_time: 0.0069  memory: 13969  loss: 2.2040  grad_norm: 0.8184
   01/29 21:38:18 - mmengine - INFO - Iter(train) [ 70/720]  lr: 1.9781e-04  eta: 0:29:31  time: 2.7302  data_time: 0.0068  memory: 13969  loss: 2.1912  grad_norm: 0.8460
   01/29 21:38:46 - mmengine - INFO - Iter(train) [ 80/720]  lr: 1.9681e-04  eta: 0:29:05  time: 2.7338  data_time: 0.0069  memory: 13969  loss: 2.1512  grad_norm: 0.8686
   01/29 21:39:13 - mmengine - INFO - Iter(train) [ 90/720]  lr: 1.9563e-04  eta: 0:28:36  time: 2.7047  data_time: 0.0068  memory: 13969  loss: 2.0653  grad_norm: 0.8686
   01/29 21:39:40 - mmengine - INFO - Iter(train) [100/720]  lr: 1.9426e-04  eta: 0:28:09  time: 2.7383  data_time: 0.0070  memory: 13969  loss: 1.9819  grad_norm: 0.9127

在训练开始前，可以看到模型的输出如下所示：

.. code:: text

   2024/01/29 21:34:58 - mmengine - INFO - before_train in EvaluateChatHook.
   2024/01/29 21:35:03 - mmengine - INFO - Sample output:
   <s><|im_start|>system
   You are a professional color designer. Please provide the corresponding colors based on the description of Human.
   <|im_end|>
   <|im_start|>user
   请给我一个像天空一样清澈透明的蓝色。<|im_end|>
   <|im_start|>assistant
   为了匹配您所描述的“像天空一样清澈透明的蓝色”，我建议您选择一种名为“天蓝”（Cerulean）的颜色。这种颜色通常被用来代表天空、海洋和清澈的水域，它具有明亮

   2024/01/29 21:35:07 - mmengine - INFO - Sample output:
   <s><|im_start|>system
   You are a professional color designer. Please provide the corresponding colors based on the description of Human.
   <|im_end|>
   <|im_start|>user
   Please give me a clear blue like the sky.<|im_end|>
   <|im_start|>assistant
   The color you're describing is a clear, bright blue that resembles the color of the sky on a sunny day. This color is often referred to as "sky blue" or "cerulean blue." Here are some specific color codes and names that

而在训练完全完成后，可以观察到模型的输出如下所示：

.. code:: text

   2024/01/29 22:08:07 - mmengine - INFO - Sample output:
   <s><|im_start|>system
   You are a professional color designer. Please provide the corresponding colors based on the description of Human.
   <|im_end|>
   <|im_start|>user
   请给我一个像天空一样清澈透明的蓝色。<|im_end|>
   <|im_start|>assistant
   #1099ee<|im_end|>

   2024/01/29 22:08:08 - mmengine - INFO - Sample output:
   <s><|im_start|>system
   You are a professional color designer. Please provide the corresponding colors based on the description of Human.
   <|im_end|>
   <|im_start|>user
   Please give me a clear blue like the sky.<|im_end|>
   <|im_start|>assistant
   #0066dd<|im_end|>

模型输出的颜色如下所示：

-  天空一样清澈透明的蓝色：\ |image11|

-  A clear blue like the sky: |image12|

不难发现，模型在经过训练后，其输出已经完全与数据集内容所对齐了。

.. _模型转换--lora-合并:

模型转换 + LoRA 合并
--------------------

在训练完成后，我们会得到几个 ``.pth`` 文件，这些文件存储了 QLoRA
算法训练过程所更新的参数，而\ **不是**\ 模型的全部参数。因此我们需要将这些
``.pth`` 文件转换为 HuggingFace 格式，并合并入原始的语言模型权重中。

模型转换
~~~~~~~~

XTuner 已经集成好了将模型转换为 HuggingFace 格式的工具，我们只需要执行

.. code:: bash

   # 创建存放 hf 格式参数的目录
   mkdir work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf

   # 转换格式
   xtuner convert pth_to_hf internlm2_7b_qlora_colorist_e5_copy.py \
                               work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720.pth \
                               work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf

该条转换命令将会根据配置文件 ``internlm2_7b_qlora_colorist_e5_copy.py``
的内容，将
``work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720.pth`` 转换为 hf
格式，并保存在
``work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf`` 位置。

LoRA 合并
~~~~~~~~~

XTuner 也已经集成好了合并 LoRA 权重的工具，我们只需执行如下指令：

.. code:: bash

   # 创建存放合并后的参数的目录
   mkdir work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged

   # 合并参数
   xtuner convert merge Shanghai_AI_Laboratory/internlm2-chat-7b \
                           work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf \
                           work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged \
                           --max-shard-size 2GB

与转换命令类似，该条合并参数命令会读取原始参数路径
``Shanghai_AI_Laboratory/internlm2-chat-7b`` 以及转换为 hf
格式的部分参数路径
``work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf``\ ，将两部分参数合并后保存于
``work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged``\ ，其中每个参数切片的最大文件大小为
2GB。

与模型对话
----------

在合并权重后，为了更好地体会到模型的能力，XTuner
也集成了与模型对话的工具。通过如下命令，便可以启动一个与模型对话的简易
Demo。

.. code:: bash

   xtuner chat work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged \
                   --prompt-template internlm2_chat \
                   --system-template colorist

当然，我们也可以选择不合并权重，而是直接与 LLM + LoRA Adapter
进行对话，我们只需要执行如下指令：

.. code:: bash

   xtuner chat Shanghai_AI_Laboratory/internlm2-chat-7b
                   --adapter work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf \
                   --prompt-template internlm2_chat \
                   --system-template colorist

其中 ``work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged``
是合并后的权重路径，\ ``--prompt-template internlm2_chat``
指定了对话模板为 InternLM2-Chat，\ ``--system-template colorist``
则是指定了与模型对话时的 System Prompt 为 Colorist 数据集所要求的模板。

以下是一个例子：

.. code:: text

   double enter to end input (EXIT: exit chat, RESET: reset history) >>> 宁静而又相当明亮的浅天蓝色，介于天蓝色和婴儿蓝之间，因其亮度而带有一丝轻微的荧光感。

   #66ccff<|im_end|>

其颜色如下所示：

宁静而又相当明亮的浅天蓝色，介于天蓝色和婴儿蓝之间，因其亮度而带有一丝轻微的荧光感。：\ |image13|

.. |image1| image:: https://img.shields.io/badge/%2366ccff-66CCFF
.. |image2| image:: https://img.shields.io/badge/%2366ccff-66CCFF
.. |image3| image:: https://img.shields.io/badge/%23ee0000-EE0000
.. |image4| image:: https://img.shields.io/badge/%2300ffcc-00FFCC
.. |image5| image:: https://img.shields.io/badge/%2366ccff-66CCFF
.. |image6| image:: https://img.shields.io/badge/%23ee0000-EE0000
.. |image7| image:: https://img.shields.io/badge/%2300ffcc-00FFCC
.. |image8| image:: https://img.shields.io/badge/%2366ccff-66CCFF
.. |image9| image:: https://img.shields.io/badge/%23ee0000-EE0000
.. |image10| image:: https://img.shields.io/badge/%2300ffcc-00FFCC
.. |image11| image:: https://img.shields.io/badge/天空一样清澈透明的蓝色-1099EE
.. |image12| image:: https://img.shields.io/badge/A_clear_blue_like_the_sky-0066DD
.. |image13| image:: https://img.shields.io/badge/宁静而又相当明亮的浅天蓝色，介于天蓝色和婴儿蓝之间，因其亮度而带有一丝轻微的荧光感。-66CCFF
