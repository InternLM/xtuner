from functools import partial
from torch.distributed._composable.fsdp import fully_shard
from xtuner._lite import get_logger
from ..fsdp.lazy import lazy_init_megatron
from .utils import map_rank0_modules
from ..fsdp import checkpoint

logger = get_logger()


def megatron_internvl2_casual(meta_model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True,
                              small_model=True):  # if 70b model, set to False
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

    has_forward_prefetch = hasattr(meta_model.language_model.model.layers[0], 'set_modules_to_forward_prefetch')
    if has_forward_prefetch:
        for layer_cur, layer_next in zip(meta_model.language_model.model.layers[:-1],
                                         meta_model.language_model.model.layers[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

    if small_model:
        meta_model.vision_model.apply(param_init_fn)
        fully_shard(
            meta_model.vision_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        for i, layers in enumerate(meta_model.vision_model.encoder.layers):
            checkpoint(layers)

        if has_forward_prefetch:
            meta_model.vision_model.set_modules_to_forward_prefetch([meta_model.language_model.model.layers[0]])
    else:
        # visual
        for i, block in enumerate(meta_model.vision_model.encoder.layers):
            block.apply(param_init_fn)

            fully_shard(
                block,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
            checkpoint(block)

        if has_forward_prefetch:
            for layer_cur, layer_next in zip(meta_model.vision_model.encoder.layers[:-1],
                                             meta_model.vision_model.encoder.layers[1:]):
                layer_cur.set_modules_to_forward_prefetch([layer_next])

            meta_model.vision_model.encoder.layers[-1].set_modules_to_forward_prefetch([meta_model.language_model.model.layers[0]])

        meta_model.vision_model.embeddings.apply(param_init_fn)
        fully_shard(
            meta_model.vision_model.embeddings,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        if has_forward_prefetch:
            meta_model.vision_model.embeddings.set_modules_to_forward_prefetch([meta_model.vision_model.encoder.layers[0]])

    meta_model.mlp1.apply(param_init_fn)
    try:
        meta_model.language_model.model.tok_embeddings.apply(param_init_fn)
    except AttributeError:
        meta_model.language_model.model.embed_tokens.apply(param_init_fn)

    meta_model.language_model.model.norm.apply(param_init_fn)
    try:
        meta_model.language_model.output.apply(param_init_fn)
    except AttributeError:
        meta_model.language_model.lm_head.apply(param_init_fn)

    model = fully_shard(
        meta_model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3

    if has_forward_prefetch and small_model:
        model.set_modules_to_forward_prefetch([model.vision_model])
    elif has_forward_prefetch:
        model.set_modules_to_forward_prefetch([model.vision_model.embeddings])
    return model
