.. _train_extreme_long_sequence:

序列并行：训练极长序列大模型的系统优化
===============================

.. raw:: html

    <p align="center">
        <img src="https://github.com/InternLM/xtuner/assets/41630003/e0460f39-7c06-4f46-b801-fdabb6c003c7" alt="XTuner"/>
    </p>

关键特性
----------

XTuner 的序列并行算法具有以下关键特性：

- 支持全量训练 **超过百万个 tokens** 的超长序列
- 支持 **百 B 级** 模型训练：XTuner 的序列并行不仅支持长序列训练，还可结合 ZeRO3 显存优化策略训练大尺寸模型
- 开箱即用：可直接训练 Transformers 算法库内和 HF Hub 上的模型
- 完全通用的序列并行 API 抽象

并行策略简介
---------------

随着生成性 AI 的不断发展，长序列训练正在变得非常重要。具有长上下文能力的大模型开始逐渐取代 RAG 成为信息检索的重要解决方案。代码库理解和例如 Sora 这种视频生成任务都需要在空间和时间层面对长上下文进行推理。

尽管序列长度的重要性不断增长，XTuner 现有的显存优化策略（如 ZeRO 系列），却不足以解决大模型、长序列训练问题。如表 1 所示，随着序列长度增大，训练过程中的显存开销主要来自激活值而非模型状态，因此使用 ZeRO-3 显存优化策略训练超长序列时，单纯增加 GPU 数量无法解决超长序列带来的 OOM 问题。

.. list-table:: **表 1 不同序列长度时，使用 ZeRO-3 训练 128k 上下文 yi-34B 模型的训练情况**
  :widths: 25 15 10 15 25
  :header-rows: 1

  * - Sequence Parallel Size
    - Model
    - ZeRO
    - GPU number
    - Tokens per second
  * - 1
    - yi-34B
    - ZeRO-3
    - 16
    - OOM
  * - 1
    - yi-34B
    - ZeRO-3
    - 32
    - OOM
  * - 1
    - yi-34B
    - ZeRO-3
    - 64
    - OOM
  * - 8
    - yi-34B
    - ZeRO-3
    - 16
    - OOM


为解决长序列训练过程中的显存问题，Megatron-LM 团队和 DeepSpeed 团队分别提出了两种序列并行算法，通过对长序列进行切分的方法来降低单 GPU 上计算的序列长度。XTuner 中的序列并行设计思路参考了 DeepSpeed 的工作 `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>`_，并加以优化， **以实现一键开启序列并行策略** 。三者的对比如下：

.. list-table:: **表 2 XTuner 与 Megatron-LM、DeepSpeed Ulysses 的序列并行实现对比**
  :widths: 25 50 50 25
  :header-rows: 1

  * -
    - Attention 通信量 （序列长度 N，并行度 P）
    - Yi-34B 32 * A100 80G 最大上下文长度
    - 代码侵入
  * - Megatron-LM
    - O(N)
    - 256K
    - 较高
  * - DeepSpeed Ulysses
    - O(N / P)
    - 1M
    - 较高
  * - XTuner
    - O(N / P)
    - 1M
    - 无

实现方案
---------------

XTuner 中的序列并行设计思路参考了 DeepSpeed 的工作 `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>`_，并加以优化，以达到直接基于 transformers 算法库或 Huggingface Hub 上的开源模型训练 1M 以上超长序列的目标。

.. raw:: html

    <p align="center">
        <img src="https://github.com/InternLM/xtuner/assets/41630003/3e0e1d49-e0fe-4966-93f4-32249d0cc398" alt="XTuner"/>
    </p>

.. raw:: html

    <p align="center">
        <b>图 1 序列并行实现方案</b>
    </p>

图 1 展示了序列并行策略的实现方案。由于 Transformer 结构较为规整，除 attention 计算外，其他计算过程中 token 之间不会互相影响（即每个 token 的计算是独立的），这一条件为序列并行提供了有利条件。上图展示了序列并行的核心设计。设由 P 个 GPUs 共同计算一个长度为 N 的长序列，在 Attention 计算的第一阶段，长度为 N / P 的子序列会通过线性层投影为 Query、Key、Value。接下来， QKV Tensor 会在参与序列并行计算的多个 GPUs 之间通过高度优化的 all-to-all 通信算子汇聚，得到序列长度为 N ，但更少注意力头的子序列。注意力计算后，通过另一个 all-to-all 通信算子将其转换为长度为 N / P 的子序列，进行后续计算。伪代码如下所示。

.. code-block:: python

    # Pseudo code for an Attention Layer
    # Input: hidden_states with shape (bs, seq_len, dim)
    # Output: attn_out with shape (bs, seq_len, dim)
    def attn_forward(hidden_states):
        q, k, v = qkv_proj(hidden_states)
        q, k, v = reshape(q, k, v)  # (bs, q_len, dim) -> (bs, q_len, nhead, hdim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        sp_size = get_sequence_parallel_world_size()
        # (bs, q_len, nhead, hdim) -> (bs, q_len * sp_size, nhead / sp_size, hdim)
        q, k, v = all_to_all(q, k, v, sp_size)
        attn_out = local_attn(q, k, v)
        # (bs, q_len * sp_size, nhead / sp_size, hdim) -> (bs, q_len, nhead, hdim)
        attn_out = all_to_all(attn_out)
        attn_out = reshape(attn_out)  # (bs, q_len, nhead, hdim) -> (bs, q_len, dim)
        attn_out = o_proj(attn_out)
        return attn_out

XTuner 序列并行支持情况
------------------------------

.. note::
    使用序列并行策略需要首先安装 `flash attn <https://github.com/Dao-AILab/flash-attention>`_ （参考 `flash attn 安装 <https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features>`_ ，安装过程需要 cuda）

    且要求 PyTorch 版本 >= 1.13.1 且 != 2.1 （PyTorch 2.1 loss 计算异常，如下图所示）

.. raw:: html

    <p align="center">
        <img src="https://github.com/InternLM/xtuner/assets/41630003/0a9fd60e-51dc-4650-ba60-8cf2ba8f773a" alt="XTuner" width="60%" />
    </p>

.. raw:: html

    <p align="center">
        <b>图 2 当序列并行度设为 2 时，使用不同 PyTorch 版本在 Alpaca 数据集上训练 32k 上下文 Llama2-7B 时的 loss 下降情况</b>
    </p>

.. list-table::
  :widths: 25 25
  :header-rows: 1

  * - 模型
    - 序列并行支持情况
  * - baichuan 1/2
    - ❌
  * - chatglm 2/3
    - ❌
  * - deepseek
    - ✅
  * - gemma
    - ❌
  * - internlm 2
    - ✅
  * - llama 2
    - ✅
  * - mistral
    - ❌
  * - qwen 1/1.5
    - ❌
  * - starcoder
    - ❌
  * - yi
    - ✅
  * - zephyr
    - ✅

其他模型的序列并行功能尚在开发中。

使用 XTuner 进行序列并行训练
------------------------------

Step 1 修改 config 文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以通过运行以下命令查看 XTuner 提供的训练不同模型的配置文件：

.. code-block:: bash

    xtuner list-cfg

针对任一 config 修改 `sequence_parallel_size` 即可使用序列并行策略：

.. code-block:: diff

    # parallel
    - sequence_parallel_size = 1
    + sequence_parallel_size = 4  # take `sequence_parallel_size = 4`` as an example

另外，若需要进一步拓展模型的长文本处理能力，需要进一步修改 config 中的 `max_position_embeddings` 字段。例如需要将模型的上下文长度拓展为 64K 时，可进行如下修改：

.. code-block:: diff

    + max_position_embeddings = 65536

    #######################################################################
    #                      PART 2  Model & Tokenizer                      #
    #######################################################################
    model = dict(
        type=SupervisedFinetune,
    +   max_position_embeddings = max_position_embeddings,
        ...)

Step 2 开始训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

需要使用 DeepSpeed 进行训练：

.. code-block:: bash

    (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_PATH} --deepspeed deepspeed_zero2
    (SLURM) srun ${SRUN_ARGS} xtuner train ${CONFIG_PATH} --launcher slurm --deepspeed deepspeed_zero2

- ${CONFIG_PATH} 为 Step 1 中修改得到的 config 文件路径
- 可根据实际情况选择使用不同的 zero 策略

序列并行 API 抽象
----------------------

为了方便在其他 repo 中使用序列并行策略，XTuner 中抽象出了序列并行所必须的五个 API 接口：

- 序列并行分布式环境初始化 (init_sequence_parallel)
- 适配序列并行的 Data Sampler (SequenceParallelSampler)
- 数据 Pad 与切分 (pad_for_sequence_parallel, split_for_sequence_parallel)
- 适配序列并行的 Attention (dispatch_modules)
- reduce loss 以正确打印训练损失 (reduce_sequence_parallel_loss)

序列并行分布式环境初始化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于序列并行算法会将长序列切分为 `sequence_parallel_world_size` 块，并将每个子序列分发给对应的 GPU 独立进行计算。因此需要在训练开始前初始化序列并行分布式环境，以指定哪几块 GPU 共同负责一个长序列输入的计算。

一个 `sequence_parallel_world_size = 4` 的示例如下：

.. code-block:: python

    # We have to initialize the distributed training environment first.
    # Here is an example when training on slurm scheduler
    # from xtuner.parallel.sequence import init_dist
    # init_dist('slurm', 'nccl', init_backend='deepspeed')
    from xtuner.parallel.sequence import init_sequence_parallel
    sequence_parallel_world_size = 4
    init_sequence_parallel(sequence_parallel_world_size)

上述过程在 `xtuner/engine/_strategy/deepspeed.py` 中实现。

Data Sampler 适配序列并行
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在使用序列并行后，Dataloader 的采样策略需要进一步调整。例如当 `sequence_parallel_world_size = 4` 时，4 块 GPU 从 Dataloader 拿到的数据需要是完全一样的。

在构建 Dataloader 时搭配 XTuner 中提供的 `SequenceParallelSampler` 使用即可：

.. code-block:: python

    from xtuner.parallel.sequence import SequenceParallelSampler
    dataloader = DataLoader(
        train_dataset, sampler=SequenceParallelSampler(train_dataset),
        **other_dataloader_params)

数据 Pad 与切分
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于每条训练数据的长度可能不尽相同，我们需要将数据进行 Pad 以使得序列长度可以被 `sequence_parallel_world_size` 整除，这样一条长数据才能被均等地分发给不同的 GPU 上。

训练过程中需要被 Pad 的 Tensor 往往有 input_ids, labels, position_ids, attention_mask 四个，pad 的过程可以通过以下方式实现：

.. code-block:: python

    from xtuner.parallel.sequence import pad_for_sequence_parallel
    input_ids, labels, position_ids, attention_mask = pad_for_sequence_parallel(
        input_ids, labels, position_ids, attention_mask)

如果训练过程用不到 attention_mask，那么可以：

.. code-block:: python

    input_ids, labels, position_ids, _ = pad_for_sequence_parallel(
        input_ids, labels, position_ids)

Pad 后，我们需要对长序列均等切分：

.. code-block:: python

    from xtuner.parallel.sequence import split_for_sequence_parallel
    # attention mask should not be split
    input_ids, labels, position_ids = split_for_sequence_parallel(
        input_ids, labels, position_ids)

以上两步在 `xtuner/dataset/collate_fns/defalut_collate_fn.py` 中实现。

Attention 适配序列并行
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 Attention 的计算过程中，序列中的不同 token 是不能独立运算的，但不同的 attention head 之间的计算却是独立的。因此，如第一节所述，需要在计算 Attention 前后（即 qkv_proj 后和 o_proj 前）分别插入一个 all-to-all 操作。

XTuner 提供了 dispatch_modules 接口以支持修改模型 Attention 的计算方式：

.. code-block:: python

    from xtuner.model.modules import dispatch_modules
    model: AutoModelForCausalLM
    dispatch_modules(model)

上述过程在 `xtuner/model/sft.py` 中实现。

Reduce Loss 以正确打印训练损失
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这个 API 对于保证训练的正确性不是必须的，但对于观测模型训练状态，打印训练 loss 是非常有用的。

.. code-block:: python

    from xtuner.parallel.sequence import reduce_sequence_parallel_loss
    outputs = llm(input_ids=input_ids, labels=labels, **kwargs)
    num_tokens_per_rank = (labels != -100).sum()
    # Suppose sequence parallel world size equals to 4,
    # losses on rank0, rank1, rank2, rank3 are different.
    loss = reduce_sequence_parallel_loss(outputs.loss, num_tokens_per_rank)
    # After loss reduction, losses on rank0, rank1, rank2, rank3 are the same.

上述过程在 `xtuner/model/sft.py` 中实现。
