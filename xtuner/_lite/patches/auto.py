from .base import PatchedCausalLM, FSDPConfig
from .llama import (CUDAPatchedLlamaForCausalLM, 
                    MLUPatchedLlamaForCausalLM,
                    MuxiPatchedLlamaForCausalLM)
from .internlm3 import (CUDAPatchedInternLM3ForCausalLM,
                        MLUPatchedInternLM3ForCausalLM,
                        MuxiPatchedInternLM3ForCausalLM)
from xtuner._lite.modelings.internlm3 import InternLM3ForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM

CUDA_PATCH_MAP = {
    LlamaForCausalLM: CUDAPatchedLlamaForCausalLM,
    InternLM3ForCausalLM: CUDAPatchedInternLM3ForCausalLM,
}

MLU_PATCH_MAP = {
    LlamaForCausalLM: MLUPatchedLlamaForCausalLM,
    InternLM3ForCausalLM: MLUPatchedInternLM3ForCausalLM,
}

MUXI_PATCH_MAP = {
    LlamaForCausalLM: MuxiPatchedLlamaForCausalLM,
    InternLM3ForCausalLM: MuxiPatchedInternLM3ForCausalLM,
}

class AutoPatch:

    @classmethod
    def from_causal_lm(cls, model, fsdp_config: FSDPConfig, device_type='cuda') -> PatchedCausalLM:

        if device_type == 'cuda':
            patch_cls = CUDA_PATCH_MAP[type(model)]
        elif device_type == 'mlu':
            patch_cls = MLU_PATCH_MAP[type(model)]
        elif device_type == 'muxi':
            patch_cls = MUXI_PATCH_MAP[type(model)]
        else:
            raise NotImplementedError
        
        patched_model = patch_cls(model, fsdp_config)

        return patched_model