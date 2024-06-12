.. _train_moe:

================
MoE 训练加速
================

MoE (Mixture of Experts) 模型可以在不提升理论计算量的前提下大幅度提升模型参数量，以实现大幅度的性能提升。

XTuner 针对 MoE 结构进行了优化，与原生训练方法相比得到了大幅度的速度提升。

使用方法
---------------------

下面以 DeepSeek V2 236B 模型为例，介绍如何使用 XTuner 优化方案加速训练。

安装必要依赖
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Git clone the latest xtuner
    git clone https://github.com/InternLM/xtuner.git

    # Install the latest xtuner
    cd xtuner
    pip install -e '.[all]'

    # Mixtral requires flash-attn
    pip install flash-attn

    # install the latest transformers
    pip install -U transformers


修改配置文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: diff

    + from xtuner.model.transformers_models.deepseek_v2 import DeepseekV2ForCausalLM

    #######################################################################
    #                      PART 2  Model & Tokenizer                      #
    #######################################################################

    model = dict(
        type=SupervisedFinetune,
        use_varlen_attn=use_varlen_attn,
        llm=dict(
            # Only full-finetune is supported in `DeepseekV2ForCausalLM``, XTuner.
            # Please use `AutoModelForCausalLM` for lora or qlora finetune.
            type=DeepseekV2ForCausalLM.from_pretrained,
    +       moe_implementation='shard',
    +       expert_in_one_shard=10,
            ...,
        )
    )
    ...


全量微调
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

全量微调 DeepSeek V2 236B 模型需要至少 64 A100-80GB。微调后的模型会通过 `HFCheckpointHook` 保存在 `${WORK_DIRS}/hf_model`。

slurm
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    srun -p $PARTITION --job-name=mixtral --nodes=8 --gres=gpu:8 --ntasks-per-node=8 xtuner train deepseek_v2_chat_full_alpaca_e3 --deepspeed deepspeed_zero3 --launcher slurm


torchrun
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # excuete on node 0
    NPROC_PER_NODE=8 NNODES=8 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=0 xtuner train deepseek_v2_chat_full_alpaca_e3 --deepspeed deepspeed_zero3 --launcher pytorch

    # excuete on node 1
    NPROC_PER_NODE=8 NNODES=8 PORT=29600 ADDR=$NODE_0_ADDR NODE_RANK=1 xtuner train deepseek_v2_chat_full_alpaca_e3 --deepspeed deepspeed_zero3 --launcher pytorch

    # excuete on node 2, 3, ..., 7


训练速度
---------------------

下表展示了使用 128 张 A100-80GB 全量微调 DeepSeek V2 236B 的训练速度对比：

.. list-table::
  :widths: 25 20 20 25 20
  :header-rows: 1

  * - Model
    - Seqlen
    - use_varlen_attn
    - sequence_parallel_size
    - Tokens per second
  * - deepseek v2 hf
    - 8k
    - False
    - 1
    - 60
  * - deepseek v2 XTuner
    - 8k
    - False
    - 1
    - 120 (2x)
  * - deepseek v2 hf
    - 8k
    - True
    - 1
    - 60
  * - deepseek v2 XTuner
    - 8k
    - True
    - 1
    - 130 (2.2x)
  * - deepseek v2 hf
    - 16k
    - False
    - 1
    - OOM
  * - deepseek v2 XTuner
    - 16k
    - False
    - 1
    - 148
  * - deepseek v2 hf
    - 16k
    - True
    - 1
    - 95
  * - deepseek v2 XTuner
    - 16k
    - True
    - 1
    - 180 (1.9x)
