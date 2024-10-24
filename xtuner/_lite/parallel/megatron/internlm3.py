from functools import partial

from torch import nn
from torch.distributed._tensor import Replicate, distribute_tensor
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               parallelize_module)

from xtuner._lite import get_logger
from ..fsdp.lazy import lazy_init_megatron
from .utils import map_rank0_modules

logger = get_logger()



def megatron_internlm3(model,
                       rank0_model,
                       dp_mesh,
                       tp_mesh=None,
                       pp_mesh=None,
                       mp_policy=None,
                       recompute_ratio=1.0,
                       reshard_after_forward=True):

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    if tp_mesh.size() > 1:
        raise NotImplementedError
        

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )

    from torch.distributed._composable import checkpoint
    from torch.distributed._composable.fsdp import fully_shard
    num_layers = len(model.cross_decoder.layers)
    num_recompute_layers = int(num_layers * recompute_ratio)

    for i, block in enumerate(model.self_decoder.layers):

        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    fully_shard(
        model.self_decoder,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)

    
    for i, block in enumerate(model.cross_decoder.layers):

        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    model.cross_decoder.wk.apply(param_init_fn)
    model.cross_decoder.wv.apply(param_init_fn)
    model.cross_decoder.norm.apply(param_init_fn)
    model.cross_decoder.rotary_emb.apply(param_init_fn)

    fully_shard(
        model.cross_decoder,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)

    model.tok_embeddings.apply(param_init_fn)
    model.norm.apply(param_init_fn)

    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)


def megatron_internlm3_casual(model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True):
    megatron_internlm3(
        model.model,
        rank0_model.model if dp_mesh.get_rank() == 0 else None,
        dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward)

    if tp_mesh.size() > 1:
        raise NotImplementedError
        # model = parallelize_module(
        #     module=model,
        #     device_mesh=tp_mesh,
        #     parallelize_plan={
        #         'output': ColwiseParallel(output_layouts=Replicate(), ),
        #     })

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )
    model.output.apply(param_init_fn)

    from torch.distributed._composable.fsdp import fully_shard
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)


def megatron_internlm3_reward(model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True):
    megatron_internlm3(
        model.model,
        rank0_model.model if dp_mesh.get_rank() == 0 else None,
        dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward)

    if tp_mesh.size() > 1:
        raise NotImplementedError
        # parallelize_module(
        #     module=model,
        #     device_mesh=tp_mesh,
        #     parallelize_plan={
        #         'v_head': ColwiseParallel(output_layouts=Replicate(), ),
        #     })

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )
    model.v_head.apply(param_init_fn)

    from torch.distributed._composable.fsdp import fully_shard
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)
