.. _xtuner_rlhf_modify_settings:

修改 RLHF PPO 配置
============

本章节将从一个基础配置文件开始，给出一些常见训练场景下的配置文件修改示例。

配置文件速览
------------

以下是 XTuner-RLHF 通过 PPO 训练微调后的 InternLM2 1.8B 模型的配置。

.. code:: python

   import torch

   rollout_config=dict(
      actor_micro_bs=32,
      reward_micro_bs=32,
      clip_reward_min=-1.5,
      clip_reward_max=1.5,
      max_new_tokens=1024,
      generate_kwargs={
         "do_sample": True,
         "temperature": 1.0,
         "top_k": 0,
         "top_p": 0.9,
         "min_new_tokens": 1,
         "num_beams": 1,
         "early_stopping": True,
         "eos_token_id": 92542,
         "pad_token_id": 0,
      }
   )

   repeater_config=dict(
      actor_micro_bs=8,
      ref_micro_bs=8,
      critic_micro_bs=32,
      reward_scale=False,
      fine_grained_rm=False,
      value_ema=False,
      kl_coeff = 0.02,
      gamma = 1.0,
      gae_lambda = 0.95,
      answer_end_id = 92542,
      norm_adv = True,
   )

   train_config=dict(
      ppo_minibatch=64,
      value_minibatch=64,
      actor_micro_bs=2,
      critic_micro_bs=2,
      pretrain_step=0,
      save_interval=80,
   )

   model_configs=dict(
      actor = dict(
         model_path="internlm/internlm2-chat-1_8b-sft",
         model_type="actor",
         torch_dtype=torch.bfloat16,
         trainer_config=dict(
               trainer_type="huggingface",
               train_kwargs=dict(
                  micro_bsz=1,
                  lr=1e-6,
                  total_steps=1e9,
                  lr_decay_rate=1,
                  loss_type="per_seq",
               ),
               parallel=dict(
                  data=dict(size=1, mode="ddp"),
                  tensor=dict(size=1, mode="1d"),
                  pipeline=dict(size=1, interleaved_overlap=False),
                  sequence=False,
               ),
         ),
         generator_config=dict(
               shared_with_trainer=True,
         ),
      ),

      reference = dict(
         model_path="internlm/internlm2-chat-1_8b-sft",
         model_type="reference",
         torch_dtype=torch.bfloat16,
         trainer_config=dict(
               trainer_type="huggingface",
               parallel=dict(
                  data=dict(size=1, mode="ddp"),
                  tensor=dict(size=1, mode="1d"),
                  pipeline=dict(size=1, interleaved_overlap=False),
                  sequence=False,
               ),
         ),
      ),

      critic = dict(
         model_path="internlm/internlm2-chat-1_8b-reward",
         model_type="critic",
         torch_dtype=torch.bfloat16,
         trainer_config=dict(
               trainer_type="huggingface",
               train_kwargs=dict(
                  micro_bsz=1,
                  lr=1e-6,
                  total_steps=1e9,
                  lr_decay_rate=1,
                  loss_type="per_seq",
               ),
               parallel=dict(
                  data=dict(size=1, mode="ddp"),
                  tensor=dict(size=1, mode="1d"),
                  pipeline=dict(size=1, interleaved_overlap=False),
                  sequence=False,
               ),
         ),
      ),

      reward = dict(
         model_path="internlm/internlm2-chat-1_8b-reward",
         model_type="reward",
         torch_dtype=torch.bfloat16,
         trainer_config=dict(
               trainer_type="huggingface",
               parallel=dict(
                  data=dict(size=1, mode="ddp"),
                  tensor=dict(size=1, mode="1d"),
                  pipeline=dict(size=1, interleaved_overlap=False),
                  sequence=False,
               ),
         ),
      ),
   )

   dataset_config = {
         "num_samples_each_epoch": 64,
         "max_seq_len": 1024,
         "random_seed": 1024,
         "ppo_datas": [
               "Anthropic/hh-rlhf/helpful-base::1.0",
               "Anthropic/hh-rlhf/harmless-base::0.5",],
      }

场景一：从 InternLM2 1.8B 到 InternLM2 7B
----------------

- **修改模型路径**：actor/ref 的 model_path 从 ``internlm/internlm2-chat-1_8b-sft`` 改为 ``internlm/internlm2-chat-7b-sft``，critic/reward 的 model_path 从 ``internlm/internlm2-chat-1_8b-reward`` 改为 ``internlm/internlm2-chat-7b-reward``。

- **修改数据并行模式**：将 actor/critic 的 parallel 从 ``ddp`` 改为 ``deepspeed``，并相应配置 zero3 及其相关参数。

- **修改数据并行度**：根据全局的 batch size 和资源量，适当修改 ref/reward 模型的 data parallelism 程度，比如从 1 改为 2。

修改后的配置文件如下：

.. code:: python

   import torch

   ...

   model_configs=dict(
      actor = dict(
         model_path="internlm/internlm2-chat-7b-sft",
         ...
         trainer_config=dict(
               ...
               parallel=dict(
                data=dict(size=8, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
               ),
               deepspeed_config={
                  "bf16": {"enable": False},
                  "fp16": {"enable": False},
                  "zero_optimization": {
                     "stage": 3,
                     "stage3_gather_16bit_weights_on_model_save": True,
                  },
                  "gradient_accumulation_steps": 8,
                  "train_micro_batch_size_per_gpu": 2,
               },
         ),
         generator_config=dict(
               shared_with_trainer=True,
         ),
      ),

      # critic 同 actor 做类似修改
      critic = dict( ... )

      reward = dict(
         model_path="internlm/internlm2-chat-7b-reward",
         ...
         trainer_config=dict(
            torch_dtype="auto",
            trainer_type="huggingface",
            parallel=dict(
                data=dict(size=2, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
      ),

      # reference 同 reward 做类似修改
      reference = dict( ... )
   )

   ...

场景二：从 InternLM2 7B 到 LLaMA2 7B
----------------

- **修改模型路径**：修改 actor/ref 的 model_path 为 ``OpenLLMAI/Llama-2-7b-sft-model-ocra-500k``，修改 critic/reward 的 model_path 为 ``OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt``。

- **修改 Tokenizer 配置**：修改 tokenizer_config 以适配 LLaMA2 的模型。

.. code:: python

   tokenizer_config = dict(
      pad_token_id = 2,
      eos_token_id = 2,
      padding_side = 'left',
      chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{'Human:\n' + message['content'] + '\n'}}{% elif message['role'] == 'assistant' %}{{'Assistant:\n' + message['content'] + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:\n' }}{% endif %}",
   )

场景三：使用 vLLM 加速 LLaMA2 7B 的生成
----------------

从 DeepSpeed 生成 + DeepSpeed 训练，切换到 vLLM 生成 + DeepSpeed 训练，需增加 GPU 卡数以容纳 vLLM generator，并修改配置文件如下：

.. code:: python

   import torch

   ...

   model_configs=dict(
      actor = dict(
         ...
         generator_config=dict(
            shared_with_trainer=False,
            generator_type="vllm",
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=2, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
      ),
      ...
   )

   ...
