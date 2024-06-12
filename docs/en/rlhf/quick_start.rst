.. _xtuner_rlhf_quick_start:

XTuner-RLHF Quick Start
=======================

RLHF includes supervised instruction fine-tuning (SFT), training the reward model, and Proximal Policy Optimization (PPO). After completing the first two steps to obtain the Actor Model and Reward Model, we can take XTuner's ``rlhf`` command to train the Actor Model and align model outputs by PPO algorithm.

Data Preparation
----------------

XTuner uses the following dataset format for RLHF PPO training:

.. code:: json

   [{"role": "user", "content": "xxx"}]
   [{"role": "user", "content": "yyy"}]

Training
--------

Step 1: Obtain Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the corresponding configuration files from the `Configuration File Directory <https://github.com/InternLM/xtuner/tree/main/examples/rlhf>`__.

Step 2: Modify Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify the dataset file path through the ``dataset_config["ppo_datas"]`` field. The dataset file should be followed by the ``::<prob>`` suffix to indicate the weight of the dataset file. For example:

.. code:: python

   dataset_config = {
        "num_samples_each_epoch": 64,
        "max_seq_len": 1024,
        "random_seed": 1024,
        "ppo_datas": [
            "Anthropic/hh-rlhf/helpful-base::1.0",
            "Anthropic/hh-rlhf/harmless-base::0.5"],
    }

This indicates that two-thirds of the data come from ``Anthropic/hh-rlhf/helpful-base`` and one-third from ``Anthropic/hh-rlhf/harmless-base``.

Modify the model path through the ``model_path`` field. For example:

.. code:: python

   model_configs=dict(
      actor = dict(
         model_path="internlm/internlm2-chat-1_8b-sft",
         ...
      ),
      ...
   )

This indicates that the Actor Model is ``internlm/internlm2-chat-1_8b-sft``.

For more detailed examples of modifying configuration files, see :ref:`xtuner_rlhf_modify_settings`.

Step 3: Start Training
~~~~~~~~~~~~~~~~~~~~~~

On a single node:

.. code:: bash

   xtuner rlhf ${CONFIG_FILE}

On a Ray cluster:

.. code:: bash

   # on node 0
   ray start --head

   # on node 1
   ray start --address ${NODE_0_ADDR}:6379
   xtuner rlhf --address ${NODE_0_ADDR} ${CONFIG_FILE}

On a Slurm cluster:

.. code:: bash

   srun -p $PARTITION --job-name=rlhf --nodes=2 --gres=gpu:8 --ntasks-per-node=8 xtuner rlhf ${CONFIG_FILE}