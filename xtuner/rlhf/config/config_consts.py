# keywords for config files

# model type (actor, critic, reward, reference, ...) for `model_type`
MODEL_TYPE_ACTOR = 'actor'
MODEL_TYPE_REFERENCE = 'reference'
MODEL_TYPE_REWARD = 'reward'
MODEL_TYPE_CRITIC = 'critic'

# training or generation engines for `trainer_type` and `generator_type`
ENGINE_HUGGINGFACE = 'huggingface'
ENGINE_INTERNEVO = 'internevo'
ENGINE_VLLM = 'vllm'
ENGINE_LMDEPLOY = 'lmdeploy'

# plugins for trainer engine (e.g., huggingface accelerate)
ENGINE_PLUGIN_DDP = 'ddp'
ENGINE_PLUGIN_FSDP = 'fsdp'
ENGINE_PLUGIN_DEEPSPEED = 'deepspeed'
