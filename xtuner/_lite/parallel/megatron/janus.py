from functools import partial
from torch.distributed._composable.fsdp import fully_shard
from xtuner._lite import get_logger
from ..fsdp.lazy import lazy_init_megatron
from .utils import map_rank0_modules
from ..fsdp import checkpoint

logger = get_logger()


def megatron_janus_casual(meta_model,
                          rank0_model,
                          dp_mesh,
                          tp_mesh=None,
                          pp_mesh=None,
                          mp_policy=None,
                          recompute_ratio=1.0,
                          reshard_after_forward=True,
                          freeze_style='mode1'):
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

    num_layers = len(meta_model.language_model.model.layers)
    num_recompute_layers = int(num_layers * recompute_ratio)
    for i, block in enumerate(meta_model.language_model.model.layers):
        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    if freeze_style == 'mode1':
        meta_model.language_model.lm_head.apply(param_init_fn)
        fully_shard(
            meta_model.language_model.lm_head,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        meta_model.gen_head.apply(param_init_fn)
        fully_shard(
            meta_model.gen_head,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        meta_model.aligner.apply(param_init_fn)
        fully_shard(
            meta_model.aligner,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        meta_model.gen_aligner.apply(param_init_fn)
        fully_shard(
            meta_model.gen_aligner,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        meta_model.vision_model.apply(param_init_fn)
        meta_model.gen_vision_model.apply(param_init_fn)
        meta_model.gen_embed.apply(param_init_fn)
        meta_model.language_model.model.embed_tokens.apply(param_init_fn)
        meta_model.language_model.model.norm.apply(param_init_fn)

        model = fully_shard(
            meta_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3

        # TODO: Bug
        # model.set_reshard_after_backward(False)

    elif freeze_style == 'mode2':
        meta_model.gen_vision_model.apply(param_init_fn)
        fully_shard(
            meta_model.gen_vision_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)
        meta_model.gen_vision_model.set_reshard_after_backward(False)

        meta_model.vision_model.apply(param_init_fn)
        fully_shard(
            meta_model.vision_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)
        meta_model.vision_model.set_reshard_after_backward(False)

        meta_model.gen_head.apply(param_init_fn)
        meta_model.aligner.apply(param_init_fn)
        meta_model.gen_aligner.apply(param_init_fn)
        meta_model.gen_embed.apply(param_init_fn)
        meta_model.language_model.model.embed_tokens.apply(param_init_fn)
        meta_model.language_model.model.norm.apply(param_init_fn)
        meta_model.language_model.lm_head.apply(param_init_fn)
        model = fully_shard(
            meta_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)

    return model
