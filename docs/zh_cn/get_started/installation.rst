==================================
安装
==================================

本节中，我们将演示如何安装 XTuner。

最佳实践
========

我们推荐用户参照我们的最佳实践安装 XTuner。
推荐使用 Python-3.10 的 conda 虚拟环境安装 XTuner。

**步骤 0.** 使用 conda 先构建一个 Python-3.10 的虚拟环境

.. code-block:: console

    $ conda create --name xtuner-env python=3.10 -y
    $ conda activate xtuner-env

**步骤 1.** 安装 XTuner

方案a: 通过 pip 直接安装

.. code-block:: console

    $ pip install -U 'xtuner[deepspeed]'

方案b: 从源码安装

.. code-block:: console

   $ git clone https://github.com/InternLM/xtuner.git
   $ cd xtuner
   $ pip install -e '.[deepspeed]'

.. note::

   "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效

验证
========

为了验证 XTuner 是否安装正确，我们将使用命令打印配置文件。

**打印配置文件：** 在命令行中使用 ``xtuner list-cfg`` 验证是否能打印配置文件列表。

.. code-block:: console

   $ xtuner list-cfg
