from functools import partial
from torch.distributed._composable.fsdp import fully_shard
from xtuner._lite import get_logger
from ..fsdp.lazy import lazy_init_megatron
from .utils import map_rank0_modules
from ..fsdp import checkpoint

logger = get_logger()


def megatron_minicpmv_casual(meta_model,
                             rank0_model,
                             dp_mesh,
                             tp_mesh=None,
                             pp_mesh=None,
                             mp_policy=None,
                             recompute_ratio=1.0,
                             reshard_after_forward=True):
    if tp_mesh.size() > 1:
        raise NotImplementedError

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(meta_model, rank0_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )

    # visual
    meta_model.vpm.apply(param_init_fn)
    fully_shard(
        meta_model.vpm,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    for i, layers in enumerate(meta_model.vpm.encoder.layers):
        checkpoint(layers)

    # resampler
    meta_model.resampler.apply(param_init_fn)
    fully_shard(
        meta_model.resampler,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )

    # llm
    num_layers = len(meta_model.llm.model.layers)
    num_recompute_layers = int(num_layers * recompute_ratio)
    for i, block in enumerate(meta_model.llm.model.layers):
        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    meta_model.llm.model.embed_tokens.apply(param_init_fn)
    meta_model.llm.model.norm.apply(param_init_fn)
    meta_model.llm.lm_head.apply(param_init_fn)

    model = fully_shard(
        meta_model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3
    return model
