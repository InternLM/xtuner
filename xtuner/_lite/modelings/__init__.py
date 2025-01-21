from .internlm2 import InternLM2Config, InternLM2ForCausalLM
from .internlm3 import InternLM3Config, InternLM3ForCausalLM, InternLM3Tokenizer
from .llava.modeling_llava import LlavaForConditionalGeneration
from .llava.configuration_llava import EnhancedLlavaConfig
from .llava.processing_llava import LlavaProcessor

def register_remote_code():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    AutoConfig.register('internlm2', InternLM2Config, exist_ok=True)
    AutoModelForCausalLM.register(
        InternLM2Config, InternLM2ForCausalLM, exist_ok=True)
    
    AutoConfig.register('internlm3', InternLM3Config, exist_ok=True)
    AutoModelForCausalLM.register(
        InternLM3Config, InternLM3ForCausalLM, exist_ok=True)
    AutoTokenizer.register(
        InternLM3Config, InternLM3Tokenizer, exist_ok=True)
