.. _xtuner_rlhf_modify_settings:

Modify RLHF PPO Configuration
=============================

This section starts with a basic configuration file and provides examples of modifications for common training scenarios.

Configuration File Overview
---------------------------

The following is the configuration of the InternLM2 1.8B model fine-tuned using PPO through XTuner-RLHF.

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

Scenario 1: From InternLM2 1.8B to InternLM2 7B
-----------------------------------------------

- **Modify model path**: Change the model_path of actor/ref from ``internlm/internlm2-chat-1_8b-sft`` to ``internlm/internlm2-chat-7b-sft`` and the model_path of critic/reward from ``internlm/internlm2-chat-1_8b-reward`` to ``internlm/internlm2-chat-7b-reward``.

- **Modify data parallel mode**: Change the parallel mode of actor/critic from ``ddp`` to ``deepspeed``, and configure zero3 and related parameters accordingly.

- **Modify data parallelism degree**: Adjust the data parallelism degree of the ref/reward model according to the global batch size and resource amount, for example, changing it from 1 to 2.

The modified configuration file is as follows:

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

      # critic same as actor modifications
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

      # reference same as reward modifications
      reference = dict( ... )
   )

   ...

Scenario 2: From InternLM2 7B to LLaMA2 7B
------------------------------------------

- **Modify model path**: Change the model_path of actor/ref to ``OpenLLMAI/Llama-2-7b-sft-model-ocra-500k`` and the model_path of critic/reward to ``OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt``.

- **Modify Tokenizer Configuration**: Update the tokenizer_config to adapt to the LLaMA2 model.

.. code:: python

   tokenizer_config = dict(
      pad_token_id = 2,
      eos_token_id = 2,
      padding_side = 'left',
      chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{'Human:\n' + message['content'] + '\n'}}{% elif message['role'] == 'assistant' %}{{'Assistant:\n' + message['content'] + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:\n' }}{% endif %}",
   )

Scenario 3: Using vLLM to Accelerate LLaMA2 7B Generation
--------------------------------------------------------

Switch from DeepSpeed generation + DeepSpeed training to vLLM generation + DeepSpeed training, and increase the number of GPUs to accommodate the vLLM generator, with the configuration modified as follows:

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