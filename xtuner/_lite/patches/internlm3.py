from xtuner._lite.chat import HybridChatTemplate
from xtuner._lite.modelings.internlm3.modeling_internlm3 import InternLM3ForCausalLM, InternLM3Attention
import types
from .llama import CUDAPatchedLlamaForCausalLM, cuda_patched_casual_forward, cuda_patched_llama_attn_training

class CUDAPatchedInternLM3ForCausalLM(CUDAPatchedLlamaForCausalLM):
    chat_template = HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>'])
    
    @staticmethod
    def dispatch_hf_code(model) -> InternLM3ForCausalLM:

        for name, module in model.named_modules():
            if isinstance(module, InternLM3Attention):
                module.forward = types.MethodType(cuda_patched_llama_attn_training, module)
            if isinstance(module, InternLM3ForCausalLM):
                module.forward = types.MethodType(cuda_patched_casual_forward, module)

        return model
   
class MLUPatchedInternLM3ForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    device_type = 'mlu'

class MuxiPatchedInternLM3ForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    device_type = 'muxi'