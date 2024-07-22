from .internlm2 import InternLM2Config, InternLM2ForCausalLM


def register_remote_code():
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register('internlm2', InternLM2Config, exist_ok=True)
    AutoModelForCausalLM.register(
        InternLM2Config, InternLM2ForCausalLM, exist_ok=True)
