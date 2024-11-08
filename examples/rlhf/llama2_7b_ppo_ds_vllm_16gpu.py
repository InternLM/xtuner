#######################################################################
#                              Settings                               #
#######################################################################
RESUME_STEP = -1
MAX_PROMPT_LEN = 1024
MAX_ANSWER_LEN = 1024
MAX_PRETRAIN_LEN = 8192

PROMPT_BATCH_SIZE = 512
PRETRAIN_BATCH_SIZE = 0

GENERATE_MICRO_BATCH_SIZE = 16
INFER_MICRO_BATCH_SIZE = 16
TRAIN_MICRO_BATCH_SIZE = 4
REF_INFER_MICRO_BATCH_SIZE = 26

ZERO_STAGE = 3
POLICY_DP_SIZE = 8
CRITIC_DP_SIZE = 4
POLICY_GRADIENT_ACC_STEP = (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE
                            ) // POLICY_DP_SIZE // TRAIN_MICRO_BATCH_SIZE
CRITIC_GRADIENT_ACC_STEP = PROMPT_BATCH_SIZE // CRITIC_DP_SIZE // TRAIN_MICRO_BATCH_SIZE  # noqa: E501

# checkout generate config
assert PROMPT_BATCH_SIZE % GENERATE_MICRO_BATCH_SIZE == 0
assert PROMPT_BATCH_SIZE % POLICY_DP_SIZE == 0
# checkout infer config
assert PROMPT_BATCH_SIZE % (INFER_MICRO_BATCH_SIZE * POLICY_DP_SIZE) == 0
assert PROMPT_BATCH_SIZE % (INFER_MICRO_BATCH_SIZE * CRITIC_DP_SIZE) == 0
# checkout learn config
assert (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) % (TRAIN_MICRO_BATCH_SIZE *
                                                    POLICY_DP_SIZE) == 0
assert (PROMPT_BATCH_SIZE) % (TRAIN_MICRO_BATCH_SIZE * CRITIC_DP_SIZE) == 0

import torch  # noqa: E402

MODEL_DTYPE = torch.float16

POLICY_MODEL_PATH = 'meta-llama/Llama-2-7b-chat-hf'
REWARD_MODEL_PATH = 'meta-llama/Llama-2-7b-chat-hf'  # better using a well-trained reward model  # noqa: E501

tokenizer_config = dict(
    pad_token_id=2,
    eos_token_id=2,
    padding_side='left',
    chat_template=  # noqa: E251
    "{% for message in messages %}{% if message['role'] == 'user' %}{{'Human:\n' + message['content'] + '\n'}}{% elif message['role'] == 'assistant' %}{{'Assistant:\n' + message['content'] + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:\n' }}{% endif %}",  # noqa: E501
)

rollout_config = dict(
    policy_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    max_new_tokens=MAX_ANSWER_LEN,
    write_to_file=False,
    resume_step=RESUME_STEP,
    generate_kwargs={
        'do_sample': True,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 0.9,
        'min_new_tokens': 1,
        'num_beams': 1,
        'early_stopping': True,
        'eos_token_id': 92542,
        'pad_token_id': 0,
    },
)

repeater_config = dict(
    policy_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=REF_INFER_MICRO_BATCH_SIZE,
    kl_coeff=0.01,
    gamma=1.0,
    gae_lambda=0.99,
    clip_reward_min=-5,
    clip_reward_max=5,
    norm_rewards=True,
)

train_config = dict(
    policy_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    ppo_loss_weight=1.0,
    pretrain_loss_weight=0.5,
    critic_warmup_step=0,
    save_interval=40,
    max_train_step=400,
    resume_step=RESUME_STEP,
    async_learn=True,
)

model_configs = dict(
    policy=dict(
        model_path=POLICY_MODEL_PATH,
        model_type='policy',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            gradient_checkpointing=False,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
            ),
            parallel=dict(
                data=dict(size=POLICY_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True if MODEL_DTYPE == torch.bfloat16 else False
                },
                'fp16': {
                    'enabled': True if MODEL_DTYPE == torch.float16 else False
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': POLICY_GRADIENT_ACC_STEP,
                'train_batch_size': PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE,
            },
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type='vllm',
            parallel=dict(
                data=dict(size=1, mode='ddp'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
    critic=dict(
        model_path=REWARD_MODEL_PATH,
        model_type='critic',
        head_name='value_head',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            gradient_checkpointing=False,
            train_kwargs=dict(
                micro_bsz=1,
                lr=5e-6,
                total_steps=1e9,
                lr_decay_rate=1,
            ),
            parallel=dict(
                data=dict(size=CRITIC_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True if MODEL_DTYPE == torch.bfloat16 else False
                },
                'fp16': {
                    'enabled': True if MODEL_DTYPE == torch.float16 else False
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': CRITIC_GRADIENT_ACC_STEP,
                'train_batch_size': PROMPT_BATCH_SIZE,
            },
        ),
    ),
    reference=dict(
        model_path=POLICY_MODEL_PATH,
        model_type='reference',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=2, mode='ddp'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
    reward=dict(
        model_path=REWARD_MODEL_PATH,
        model_type='reward',
        head_name='value_head',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=1, mode='ddp'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
)

prompt_dataset_config = dict(
    samples_each_epoch=PROMPT_BATCH_SIZE,
    max_len=MAX_PROMPT_LEN,
    message_type='prompt',
    random_seed=1024,
    sample_strategy='in_batch',  # 'in_data'
    message_datasets=[
        '[HF]Anthropic/hh-rlhf/helpful-base::0.5[RM_PROMPT]:default',
    ])
