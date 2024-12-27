from functools import partial
from packaging import version

import torch
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


def _tp_qwen2(model, tp_mesh):

    layer_tp_plan = {
        
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        'self_attn.q_proj':
        ColwiseParallel(),
        'self_attn.k_proj':
        ColwiseParallel(),
        'self_attn.v_proj':
        ColwiseParallel(),
        'self_attn.o_proj':
        RowwiseParallel(),
        'input_layernorm':
        PrepareModuleInput(
            input_layouts=(Replicate(), ),
            desired_input_layouts=(Replicate(), ),
            use_local_output=True
        ),
        'mlp.up_proj':
        ColwiseParallel(),
        'mlp.down_proj':
        RowwiseParallel(),
        'mlp.gate_proj':
        ColwiseParallel(),
        'post_attention_layernorm':
        PrepareModuleInput(
            input_layouts=(Replicate(), ),
            desired_input_layouts=(Replicate(), ),
            use_local_output=True
        )
    }

    tp_size = tp_mesh.size()
    for layer in model.layers:
        attention = layer.self_attn
        num_key_value_heads = attention.num_key_value_heads
        num_heads = attention.num_heads
        hidden_size = attention.hidden_size

        attention.num_heads = num_heads // tp_size
        attention.num_key_value_heads = num_key_value_heads // tp_size
        attention.hidden_size = hidden_size // tp_size

        attn_norm = layer.input_layernorm
        attn_norm.register_parameter(
            'weight',
            nn.Parameter(
                distribute_tensor(attn_norm.weight, tp_mesh, [Replicate()])))

        ffn_norm = layer.post_attention_layernorm
        ffn_norm.register_parameter(
            'weight',
            nn.Parameter(
                distribute_tensor(ffn_norm.weight, tp_mesh, [Replicate()])))

        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    norm = model.norm
    dist_norm_w = nn.Parameter(
        distribute_tensor(norm.weight, tp_mesh, [Replicate()]))
    norm.register_parameter('weight', dist_norm_w)

    # emb = model.embed_tokens
    # dist_emb_w = nn.Parameter(
    #     distribute_tensor(emb.weight, tp_mesh, [Replicate()]))
    # emb.register_parameter('weight', dist_emb_w)

    # model.norm.apply(param_init_fn)
    # model.embed_tokens.apply(param_init_fn)

    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan={
            'model.embed_tokens':
            RowwiseParallel(input_layouts=Replicate(), ),
            'model.norm':PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
                use_local_output=True
            ),
        })


def megatron_qwen2(model,
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

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )

    
    if tp_mesh and tp_mesh.size() > 1:
        _tp_qwen2(model, tp_mesh)

    from torch.distributed._composable import checkpoint
    from torch.distributed._composable.fsdp import fully_shard
    num_layers = len(model.layers)
    num_recompute_layers = int(num_layers * recompute_ratio)

    for i, block in enumerate(model.layers):

        block.apply(param_init_fn)

        # # As an optimization, do not reshard after forward for the last
        # # transformer block since FSDP would prefetch it immediately
        # if i < num_layers - 1:
        #     _reshard = reshard_after_forward
        # else:
        #     _reshard = False

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    if version.parse(torch.__version__) >= version.parse("2.5.0"):
        for layer_cur, layer_next in zip(model.layers[:-1], model.layers[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

    model.embed_tokens.apply(param_init_fn)
    model.norm.apply(param_init_fn)    

    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)


def megatron_qwen2_casual(model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True):
    megatron_qwen2(
        model.model,
        rank0_model.model if dp_mesh.get_rank() == 0 else None,
        dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward)

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
    

    if tp_mesh and tp_mesh.size() > 1:
        model = parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan={
                'lm_head': ColwiseParallel(output_layouts=Replicate(), ),
            })

    model.lm_head.apply(param_init_fn)

    from torch.distributed._composable.fsdp import fully_shard
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)


def megatron_qwen2_reward(model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True):
    megatron_qwen2(
        model.model,
        rank0_model.model if dp_mesh.get_rank() == 0 else None,
        dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward)

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    if tp_mesh and tp_mesh.size() > 1:
        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan={
                'score.0': ColwiseParallel(),
                'score.2': RowwiseParallel(),
            })

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
