import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.utils.import_utils import (is_flash_attn_2_available,
                                             is_torch_sdpa_available)

from xtuner._lite.accelerate import LoadWoInit


def build_actor_model(model_path, dtype=torch.float32, trust_remote_code=True):

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if is_flash_attn_2_available():
        config.attn_implementation = 'flash_attention_2'
    elif is_torch_sdpa_available():
        config.attn_implementation = 'sdpa'

    with LoadWoInit():
        policy = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation='flash_attention_2',
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code)

    return policy


def build_reward_model(model_path, dtype=torch.float32, trust_remote_code=True):

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if is_flash_attn_2_available():
        config.attn_implementation = 'flash_attention_2'
    elif is_torch_sdpa_available():
        config.attn_implementation = 'sdpa'

    config.use_cache = False
    config.torch_dtype = dtype
    with LoadWoInit():
        reward = AutoModel.from_pretrained(
            model_path,
            attn_implementation='flash_attention_2',
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code)

    reward.model.use_cache = False
    
    return reward
