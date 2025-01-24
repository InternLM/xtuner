from xtuner._lite.chat import HybridChatTemplate
from xtuner._lite.modelings.internlm3.modeling_internlm3 import InternLM3ForCausalLM, InternLM3Attention

from .llama import CUDAPatchedLlamaForCausalLM

class CUDAPatchedInternLM3ForCausalLM(CUDAPatchedLlamaForCausalLM):
    
    attn_cls = InternLM3Attention
    causal_cls = InternLM3ForCausalLM

    chat_template = HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>'])
    
   
class MLUPatchedInternLM3ForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    device_type = 'mlu'

class MuxiPatchedInternLM3ForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    device_type = 'muxi'