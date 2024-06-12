.. _xtuner_rlhf_quick_start:

XTuner-RLHF 快速上手
===================================

RLHF 包括有监督指令微调（ SFT ）、训练奖励模型、近端策略优化（ PPO ），在完成前两步分别得到 Actor Model 和 Reward Model 后，
可通过XTuner 的 ``rlhf`` 命令进行第三步，即通过 PPO 强化学习算法训练 Actor Model 以对齐模型输出。

数据准备
--------

XTuner 采用如下的数据集格式进行 RLHF PPO 训练：

.. code:: json

   [{"role": "user", "content": "xxx"}]
   [{"role": "user", "content": "yyy"}]

训练
--------

Step 1, 获取配置文件
~~~~~~~~~~~~~~~~~~~

可以在 `配置文件目录 <https://github.com/InternLM/xtuner/tree/main/examples/rlhf>`__ 中获取相应的配置文件

Step 2, 修改配置
~~~~~~~~~~~~~~~~~~~

通过 ``dataset_config["ppo_datas"]`` 字段修改数据集文件路径，数据集文件后需加 ``::<prob>`` 后缀表明该数据集文件权重。例如：

.. code:: python

   dataset_config = {
        "num_samples_each_epoch": 64,
        "max_seq_len": 1024,
        "random_seed": 1024,
        "ppo_datas": [
            "Anthropic/hh-rlhf/helpful-base::1.0",
            "Anthropic/hh-rlhf/harmless-base::0.5"],
    }

表明所使用的数据有三分之二来自 ``Anthropic/hh-rlhf/helpful-base`` ，三分之一来自 ``Anthropic/hh-rlhf/harmless-base`` 。

通过 ``model_path`` 字段修改模型路径。例如：

.. code:: python

   model_configs=dict(
      actor = dict(
         model_path="internlm/internlm2-chat-1_8b-sft",
         ...
      ),
      ...
   )

表明 Actor Model 是 ``internlm/internlm2-chat-1_8b-sft``。

关于配置文件更详细的修改示例，见 :ref:`xtuner_rlhf_modify_settings`。

Step 3, 开始训练
~~~~~~~~~~~~~~~~

在单节点上：

.. code:: bash

   xtuner rlhf ${CONFIG_FILE}

在 Ray 集群：

.. code:: bash

   # on node 0
   ray start --head

   # on node 1
   ray start --address ${NODE_0_ADDR}:6379
   xtuner rlhf --address ${NODE_0_ADDR} ${CONFIG_FILE}

在Slurm集群：

.. code:: bash

   srun -p $PARTITION --job-name=rlhf --nodes=2 --gres=gpu:8 --ntasks-per-node=8 xtuner rlhf ${CONFIG_FILE}
