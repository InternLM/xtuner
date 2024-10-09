from .internlm2 import (megatron_internlm2, megatron_internlm2_casual,
                        megatron_internlm2_reward)

MEGATRON_MAP = {
    'InternLM2ForCausalLM': megatron_internlm2_casual,
    'InternLM2ForRewardModel': megatron_internlm2_reward,
    'InternLM2Model': megatron_internlm2,
}


def megatron_parallelize(model,
                         rank0_model,
                         dp_mesh,
                         tp_mesh=None,
                         pp_mesh=None,
                         mp_policy=None,
                         recompute_ratio=1.0,
                         reshard_after_forward=True):

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
        reshard_after_forward=reshard_after_forward)

    return model
