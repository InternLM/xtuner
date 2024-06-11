==================
预训练模型权重
==================

``HuggingFace`` 和 ``ModelScope``
提供了多种下载预训练模型权重的方法，本节将以下载 internlm2-chat-7b
为例，介绍如何快速下载预训练模型的权重。

.. note::

   若 HuggingFace 访问受限，请优先考虑使用 ModelScope 进行下载


[推荐] 方法 1：``snapshot_download``
========================================


HuggingFace
------------

``huggingface_hub.snapshot_download`` 支持下载特定的 HuggingFace Hub
模型权重，并且允许多线程。您可以利用下列代码并行下载模型权重：

.. code:: python

   from huggingface_hub import snapshot_download

   snapshot_download(repo_id='internlm/internlm2-chat-7b', local_dir='./internlm2-chat-7b', max_workers=20)

.. note::

   其中，\ ``repo_id`` 表示模型在 HuggingFace Hub 的名字、\ ``local_dir`` 表示期望存储到的本地路径、\ ``max_workers`` 表示下载的最大并行数。

.. tip::

   如果未指定 ``local_dir``\ ，则将下载至 HuggingFace 的默认 cache 路径中（\ ``~/.cache/huggingface/hub``\ ）。若要修改默认 cache 路径，需要修改相关环境变量：

   .. code:: console

      $ # 默认为 `~/.cache/huggingface/`
      $ export HF_HOME=XXXX

.. tip::
   如果觉得下载较慢（例如无法达到最大带宽等情况），可以尝试设置\ ``export HF_HUB_ENABLE_HF_TRANSFER=1`` 以获得更高的下载速度。

.. tip::
   关于环境变量的更多用法，可阅读\ `这里 <https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables>`__ 。


ModelScope
-----------

``modelscope.snapshot_download``
支持下载指定的模型权重，您可以利用下列命令下载模型：

.. code:: python

   from modelscope import snapshot_download

   snapshot_download(model_id='Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./internlm2-chat-7b')

.. note::
   其中，\ ``model_id`` 表示模型在 ModelScope 模型库的名字、\ ``cache_dir`` 表示期望存储到的本地路径。


.. note::
   ``modelscope.snapshot_download`` 不支持多线程并行下载。

.. tip::

   如果未指定 ``cache_dir``\ ，则将下载至 ModelScope 的默认 cache 路径中（\ ``~/.cache/huggingface/hub``\ ）。

   若要修改默认 cache 路径，需要修改相关环境变量：

   .. code:: console

      $ # 默认为 ~/.cache/modelscope/hub/
      $ export MODELSCOPE_CACHE=XXXX



方法 2： Git LFS
===================

HuggingFace 和 ModelScope 的远程模型仓库就是一个由 Git LFS 管理的 Git
仓库。因此，我们可以利用 ``git clone`` 完成权重的下载：

.. code:: console

   $ git lfs install
   $ # From HuggingFace
   $ git clone https://huggingface.co/internlm/internlm2-chat-7b
   $ # From ModelScope
   $ git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-7b.git


方法 3：``AutoModelForCausalLM``
=====================================================

``AutoModelForCausalLM.from_pretrained``
在初始化模型时，将尝试连接远程仓库并自动下载模型权重。因此，我们可以利用这一特性下载模型权重。

HuggingFace
------------

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)

.. tip::

   此时模型将会下载至 HuggingFace 的 cache 路径中（默认为\ ``~/.cache/huggingface/hub``\ ）。

   若要修改默认存储路径，需要修改相关环境变量：

   .. code:: console

      $ # 默认为 `~/.cache/huggingface/`
      $ export HF_HOME=XXXX

ModelScope
-----------

.. code:: python

   from modelscope import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained('Shanghai_AI_Laboratory/internlm2-chat-7b', trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained('Shanghai_AI_Laboratory/internlm2-chat-7b', trust_remote_code=True)

.. tip::

   此时模型将会下载至 ModelScope 的 cache 路径中（默认为\ ``~/.cache/modelscope/hub``\ ）。

   若要修改默认存储路径，需要修改相关环境变量：

   .. code:: console

      $ # 默认为 ~/.cache/modelscope/hub/
      $ export MODELSCOPE_CACHE=XXXX
