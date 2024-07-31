#######################################################################
#                              Settings                               #
#######################################################################
RESUME_STEP = -1
MAX_PROMPT_LEN = 1536
MAX_ANSWER_LEN = 512
MAX_PRETRAIN_LEN = 4096

PROMPT_BATCH_SIZE = 128
PRETRAIN_BATCH_SIZE = 128

PIPE_MICRO_BATCH_NUM = 4
assert PROMPT_BATCH_SIZE % PIPE_MICRO_BATCH_NUM == 0
PIPE_MICRO_BATCH_SIZE = PROMPT_BATCH_SIZE // PIPE_MICRO_BATCH_NUM   #32

GENERATE_MICRO_BATCH_SIZE = 8
INFER_MICRO_BATCH_SIZE = 2
TRAIN_MICRO_BATCH_SIZE = 1

ZERO_STAGE = 3
POLICY_DP_SIZE = 8
CRITIC_DP_SIZE = 8
REF_DP_SIZE = 4
REWARD_DP_SIZE = 4
VLLM_TP_SIZE=8
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

MODEL_DTYPE = 'auto'

tokenizer_config = dict(
    pad_token_id=0,
    eos_token_id=92542,
    padding_side='left',
)

rollout_config = dict(
    policy_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    max_new_tokens=MAX_ANSWER_LEN,
    write_to_file=True,
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
    async_reward=True,
)

repeater_config = dict(
    ref_micro_bs=INFER_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    kl_coeff=0.01,
    gamma=1.0,
    gae_lambda=0.99,
    clip_reward_min=-5,
    clip_reward_max=5,
    norm_rewards=True,
)

train_config = dict(
    pipe_micro_bs=PIPE_MICRO_BATCH_SIZE,
    policy_train_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_train_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    policy_infer_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_infer_micro_bs=INFER_MICRO_BATCH_SIZE,
    ppo_loss_weight=1.0,
    pretrain_loss_weight=0.5,
    # critic_warmup_step=40,
    critic_warmup_step=0, ## Debug-Only
    save_interval=200,
    max_train_step=800,
    resume_step=RESUME_STEP,
)

model_configs = dict(
    policy=dict(
        model_path=None,
        model_type='policy',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            gradient_checkpointing=True,
            train_kwargs=dict(
                micro_bsz=1,
                lr=5e-7,
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
                "zero_optimization": {
                    "stage": 3, 
                    "overlap_comm": True,
                    "stage3_gather_16bit_weights_on_model_save": True
                }, 
                'bf16': {
                    'enabled': True
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
                tensor=dict(size=VLLM_TP_SIZE, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    ),
    critic=dict(
        model_path=None,
        model_type="critic",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            gradient_checkpointing=True,
            train_kwargs=dict(
                micro_bsz=1,
                lr=9e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=CRITIC_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 3, 
                    "overlap_comm": True,
                    "stage3_gather_16bit_weights_on_model_save": True
                }, 
                'bf16': {
                    'enabled': True
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
        model_path=None,
        model_type="reference",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=REF_DP_SIZE, mode="deepspeed"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 3, 
                    "overlap_comm": True,
                    "stage3_gather_16bit_weights_on_model_save": True
                }, 
                "bf16": {
                    "enabled": True
                }, 
                "gradient_clipping": 1.0, 
                "prescale_gradients": False, 
                "wall_clock_breakdown": False, 
                "data_types": {
                    "grad_accum_dtype": "fp32"
                },
                "train_micro_batch_size_per_gpu": 2
            },
        ),
    ),
    reward=dict(
        model_path=None,
        model_type="reward",
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=REWARD_DP_SIZE, mode="deepspeed"),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
            deepspeed_config={
                "zero_optimization": {
                    "stage": 3, 
                    "overlap_comm": True,
                    "stage3_gather_16bit_weights_on_model_save": True
                }, 
                "bf16": {
                    "enabled": True
                }, 
                "gradient_clipping": 1.0, 
                "prescale_gradients": False, 
                "wall_clock_breakdown": False, 
                "data_types": {
                    "grad_accum_dtype": "fp32"
                },
                "train_micro_batch_size_per_gpu": 2
            },
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
        './examples/rlhf/demo_datas/prompt_data.json::0.01[SYS_PROMPT]:summarization',  # noqa: E501
        '[HF]Anthropic/hh-rlhf/helpful-base::0.5[RM_PROMPT]:default',
        '[HF]HuggingFaceH4/summarize_from_feedback::0.5',
    ])

pretrain_dataset_config = dict(
    samples_each_epoch=PRETRAIN_BATCH_SIZE,
    max_len=MAX_PRETRAIN_LEN,
    message_type='pretrain',
    random_seed=1024,
    sample_strategy='in_batch',  # 'in_data'
    message_datasets=[
        './examples/rlhf/demo_datas/pretrain_data.json::0.01',
        '[HF]Anthropic/hh-rlhf/helpful-base::0.5',
        '[HF]HuggingFaceH4/summarize_from_feedback::0.5',
    ],
)
