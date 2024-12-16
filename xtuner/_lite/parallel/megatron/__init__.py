from .internlm2 import (megatron_internlm2, megatron_internlm2_casual,
                        megatron_internlm2_reward)
from .internlm3 import (megatron_internlm3, megatron_internlm3_casual,
                        megatron_internlm3_reward)

from .qwen2 import (megatron_qwen2_casual, megatron_qwen2, megatron_qwen2_reward)
from .internvl2 import megatron_internvl2_casual
from .minicpmv import megatron_minicpmv_casual
from .llama import megatron_llama, megatron_llama_casual
from .janus import megatron_janus_casual

MEGATRON_MAP = {
    'InternLM2ForCausalLM': megatron_internlm2_casual,
    'InternLM2ForRewardModel': megatron_internlm2_reward,
    'InternLM2Model': megatron_internlm2,
    'InternLM3ForCausalLM': megatron_internlm3_casual,
    'InternLM3ForRewardModel': megatron_internlm3_reward,
    'InternLM3Model': megatron_internlm3,
    'Qwen2ForCausalLM': megatron_qwen2_casual,
    'Qwen2Model': megatron_qwen2,
    'Qwen2ForRewardModel': megatron_qwen2_reward,
    'InternVLChatModel': megatron_internvl2_casual,
    'MiniCPMV': megatron_minicpmv_casual,
    'MultiModalityCausalLM': megatron_janus_casual,
    'LlamaModel': megatron_llama,
    'LlamaForCausalLM': megatron_llama_casual
}


def megatron_parallelize(model,
                         rank0_model,
                         dp_mesh,
                         tp_mesh=None,
                         pp_mesh=None,
                         mp_policy=None,
                         recompute_ratio=1.0,
                         reshard_after_forward=True,
                         **kwargs):

    cls_name = model.__class__.__name__
    if cls_name not in MEGATRON_MAP:
        raise NotImplementedError

    parallel_fn = MEGATRON_MAP[cls_name]

    model = parallel_fn(
        model,
        rank0_model,
        dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward,
        **kwargs)

    return model
