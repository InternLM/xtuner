# Copyright (c) OpenMMLab. All rights reserved.
import copy
import types
from functools import partial
from typing import Callable, List, Optional, Tuple, TypedDict, Union

import torch
from flash_attn import flash_attn_with_kvcache
from packaging import version
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed._tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.nn import functional as F
from tqdm import tqdm
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

from xtuner._lite.accelerate import liger_kernel_is_available
from xtuner._lite.chat import HybridChatTemplate
from xtuner._lite.parallel.sequence import split_for_sequence_parallel
from xtuner._lite.patches.base import (
    FSDPConfig,
    HFCheckpointLoader,
    ModelConfig,
    PatchedCausalLM,
    clip_grad_norm_,
    dense_model_init_weights,
    lazy_init_fn,
)
from xtuner._lite.patches.mixins import GenerateMixin
from xtuner._lite.patches.utils import pad_to_max_length, pad_to_multiple_of
from xtuner._lite.utils.misc import XTUNER_DETERMINISTIC

logger = logging.get_logger(__name__)


def all_to_all(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    mesh: DeviceMesh,
) -> torch.Tensor:
    world_size = mesh.size()
    split_size = input.size(scatter_dim) // world_size
    input_split_sizes = [split_size] * world_size
    output_split_sizes = input_split_sizes

    input = input.contiguous()
    input = input.movedim(scatter_dim, 0)

    output = all_to_all_single_autograd(
        input,
        group=mesh.get_group(),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )
    output = output.transpose(0, scatter_dim)

    output_list = [t for t in torch.tensor_split(output, world_size, scatter_dim)]
    output = torch.cat(output_list, dim=gather_dim).contiguous()
    return output


class FlashAttentionKwargs(TypedDict, total=False):
    """Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]
    block_table: Optional[torch.Tensor]
    prefilling: Optional[bool]


@torch.library.custom_op("xtuner::fill_paged_kv_cache", mutates_args=())
def fill_paged_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
    block_table: torch.Tensor,
) -> None:
    bs = block_table.size(0)
    from lmdeploy.pytorch.kernels import fill_kv_cache

    fill_kv_cache(
        key_states.transpose(1, 2)[:, : cu_seq_lens_k[bs]],
        value_states.transpose(1, 2)[:, : cu_seq_lens_k[bs]],
        key_cache,
        value_cache,
        cu_seq_lens_q[:bs],  # q_start_loc
        cu_seq_lens_q[1 : bs + 1] - cu_seq_lens_q[:bs],  # q_seq_length
        kv_seq_length=cu_seq_lens_k[1 : bs + 1] - cu_seq_lens_k[:bs],
        max_q_seq_length=max_length_q,
        block_offsets=block_table,
    )


@fill_paged_kv_cache.register_fake
def fill_paged_kv_cache_fake(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
    block_table: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op("xtuner::paged_attention_decoding", mutates_args=())
def paged_attention_decoding(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    bs = block_table.size(0)
    attn_outputs = flash_attn_with_kvcache(
        query_states.transpose(1, 2).transpose(0, 1)[:bs],
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    return attn_outputs


@paged_attention_decoding.register_fake
def paged_attention_decoding_fake(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
):
    bs = block_table.size(0)
    return torch.empty_like(query_states.transpose(1, 2).transpose(0, 1)[:bs])


class CUDAPatchedLlamaForCausalLM(PatchedCausalLM, GenerateMixin):
    device_type = "cuda"
    rotary_emb_cls = LlamaRotaryEmbedding
    attn_cls = LlamaAttention
    norm_cls = LlamaRMSNorm
    layer_cls = LlamaDecoderLayer
    causal_cls = LlamaForCausalLM

    layer_tp_plan = {
        "input_layernorm": SequenceParallel(),
        "self_attn": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "post_attention_layernorm": SequenceParallel(),
        "mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        "mlp.gate_proj": ColwiseParallel(),
    }

    casual_tp_plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "model.norm": SequenceParallel(),
        "lm_head": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Shard(1),),
        ),
    }

    chat_template = HybridChatTemplate(
        system=("<|start_header_id|>system<|end_header_id|>\n\n{system}" "<|eot_id|>"),
        user=(
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        assistant="{assistant}<|eot_id|>",
        sep="",
        stop_words=["<|eot_id|>"],
    )

    def __init__(
        self, model: LlamaForCausalLM, fsdp_config: Optional[FSDPConfig] = None
    ):
        super().__init__(model, fsdp_config)

        if dist.is_initialized() and dist.is_available():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            self._rank0_model = copy.deepcopy(model)
        else:
            self._rank0_model = None

        self._patched_model = self.dispatch_hf_code(model)

        self.init_model_config(fsdp_config)

        self._fsdp_config = fsdp_config
        if self._fsdp_config is not None:
            self.init_device_mesh(fsdp_config)

        if self._fsdp_config.enable_fp8:
            from xtuner._lite.accelerate.float8_gmm import Float8Handler

            self._float8_handler = Float8Handler(
                compile=fsdp_config.torch_compile,
                enable_fsdp_float8_all_gather=True,
                pad_inner_dim=False,
                scaling_granularity_gemm=fsdp_config.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=fsdp_config.scaling_granularity_grouped_gemm,
            )

    @property
    def patched_model(self) -> LlamaForCausalLM:
        return self._patched_model

    @property
    def rank0_model(self) -> LlamaForCausalLM:
        return self._rank0_model

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @property
    def fsdp_config(self) -> Optional[FSDPConfig]:
        return self._fsdp_config

    @property
    def data_parallel_mesh(self):
        return self.dp_mesh

    @property
    def data_mesh(self):
        return self._data_mesh

    @property
    def sequence_parallel_mesh(self):
        return self.sp_mesh

    def init_model_config(self, fsdp_config: FSDPConfig):
        assert self.patched_model.config.num_key_value_heads >= fsdp_config.tp_size
        assert self.patched_model.config.num_key_value_heads % fsdp_config.tp_size == 0

        self._model_config = ModelConfig(
            num_hidden_layers=self.patched_model.config.num_hidden_layers,
            num_attention_heads=self.patched_model.config.num_attention_heads,
            num_key_value_heads=self.patched_model.config.num_key_value_heads
            // fsdp_config.tp_size,
            hidden_size=self.patched_model.config.hidden_size,
            intermediate_size=self.patched_model.config.intermediate_size,
            vocab_size=self.patched_model.config.vocab_size,
            head_dim=self.patched_model.config.head_dim,
        )

    @classmethod
    def dispatch_hf_code(cls, model) -> LlamaForCausalLM:
        for name, module in model.named_modules():
            if isinstance(module, cls.attn_cls):
                module.forward = types.MethodType(cls.patched_attn_forward, module)
            if isinstance(module, cls.causal_cls):
                module.forward = types.MethodType(cls.patched_casual_forward, module)
            if isinstance(module, cls.layer_cls):
                module.forward = types.MethodType(cls.patched_layer_forward, module)

        return model

    def init_device_mesh(self, fsdp_config: FSDPConfig) -> None:
        if fsdp_config.ep_size > 1:
            raise NotImplementedError

        world_size = dist.get_world_size()
        sp_size = fsdp_config.sp_size
        tp_size = fsdp_config.tp_size

        if tp_size > sp_size:
            # add warning
            pass
        elif tp_size < sp_size:
            assert sp_size % tp_size == 0
            sp_size = sp_size // tp_size

        assert world_size % sp_size == 0
        assert world_size % tp_size == 0
        world_mesh_name = f"{fsdp_config.mesh_prefix}.world"
        fsdp_mesh_name = f"{fsdp_config.mesh_prefix}.fsdp"
        tp_mesh_name = f"{fsdp_config.mesh_prefix}.tp"
        dp_mesh_name = f"{fsdp_config.mesh_prefix}.dp"
        sp_mesh_name = f"{fsdp_config.mesh_prefix}.sp"
        data_mesh_name = f"{fsdp_config.mesh_prefix}.data"
        _tp_mesh_name = f"{fsdp_config.mesh_prefix}._tp"

        world_mesh = init_device_mesh(
            self.device_type,
            (world_size,),
            mesh_dim_names=[
                world_mesh_name,
            ],
        )
        self.world_mesh = world_mesh[world_mesh_name]

        model_mesh = init_device_mesh(
            self.device_type,
            (world_size // tp_size, tp_size),
            mesh_dim_names=[fsdp_mesh_name, tp_mesh_name],
        )

        fsdp_mesh = model_mesh[fsdp_mesh_name]
        tp_mesh = model_mesh[tp_mesh_name]

        self.tp_mesh = tp_mesh
        self.fsdp_mesh = fsdp_mesh

        data_mesh = init_device_mesh(
            self.device_type,
            (world_size // tp_size // sp_size, sp_size, tp_size),
            mesh_dim_names=[dp_mesh_name, sp_mesh_name, _tp_mesh_name],
        )
        self.dp_mesh = data_mesh[dp_mesh_name]
        self.sp_mesh = data_mesh[sp_mesh_name]

        _data_mesh = init_device_mesh(
            self.device_type,
            (world_size // tp_size // sp_size, sp_size * tp_size),
            mesh_dim_names=[dp_mesh_name, data_mesh_name],
        )
        self._data_mesh = _data_mesh[data_mesh_name]

    def fully_shard(self) -> None:
        if self._fsdp_config.enable_fp8:
            self._float8_handler.convert_to_float8_training(self.patched_model)

        if not getattr(self.patched_model.config, "skip_checkpoint", False):
            param_init_fn = partial(
                lazy_init_fn,
                module2name={
                    mod: name for name, mod in self.patched_model.named_modules()
                },
                checkpoint_loader=HFCheckpointLoader(
                    self.patched_model.config._name_or_path
                ),
                enable_fp8=self.fsdp_config.enable_fp8,
            )
        else:
            param_init_fn = partial(
                dense_model_init_weights, config=self.patched_model.config
            )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype,
            reduce_dtype=self.fsdp_config.reduce_dtype,
        )

        self.patched_model.model.rotary_emb = self.rotary_emb_cls(
            self.patched_model.config
        )

        num_recompute_layers = int(
            self.model_config.num_hidden_layers * self.fsdp_config.recompute_ratio
        )

        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(self.tp_mesh.get_group().group_name)

        if self.fsdp_config.torch_compile:
            compiled_layers = []

        for layer in tqdm(self.patched_model.model.layers):
            layer.apply(param_init_fn)

            attention = layer.self_attn

            if self.tp_mesh.size() > 1:
                parallelize_module(
                    module=layer,
                    device_mesh=self.tp_mesh,
                    parallelize_plan=self.layer_tp_plan,
                )

            if attention.layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(layer, preserve_rng_state=False)

            if self.fsdp_config.torch_compile:
                layer = torch.compile(layer, fullgraph=True)

            self.patched_model.model.layers.register_module(
                str(attention.layer_idx), layer
            )

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=CPUOffloadPolicy()
                if self.fsdp_config.cpu_offload
                else None,
            )

            if self.fsdp_config.torch_compile:
                compiled_layers.append(layer)

        if version.parse(torch.__version__) >= version.parse("2.5.0"):
            for layer_cur, layer_next in zip(
                self.patched_model.model.layers[:-1],
                self.patched_model.model.layers[1:],
            ):
                layer_cur.set_modules_to_forward_prefetch([layer_next])

        if self._patched_model.config.tie_word_embeddings:
            self.patched_model.model.embed_tokens.apply(param_init_fn)
            self.patched_model.lm_head.weight = (
                self.patched_model.model.embed_tokens.weight
            )
        else:
            self.patched_model.lm_head.apply(param_init_fn)
            self.patched_model.model.embed_tokens.apply(param_init_fn)
        self.patched_model.model.norm.apply(param_init_fn)

        if self.tp_mesh.size() > 1:
            _weight = self.patched_model.lm_head.weight
            _dtensor_weight = nn.Parameter(
                distribute_tensor(_weight, self.tp_mesh, [Replicate()])
            )
            self.patched_model.lm_head.register_parameter("weight", _dtensor_weight)

            parallelize_module(
                self.patched_model,
                self.tp_mesh,
                self.casual_tp_plan,
            )

        fully_shard(
            self.patched_model,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )

    @staticmethod
    def patched_attn_forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "block_table" in kwargs and kwargs["block_table"] is not None:
            # generating
            if "prefilling" in kwargs and kwargs["prefilling"]:
                return CUDAPatchedLlamaForCausalLM.patched_attn_prefilling(
                    self,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    sequence_parallel_mesh=sequence_parallel_mesh,
                    **kwargs,
                )
            else:
                return CUDAPatchedLlamaForCausalLM.patched_attn_decoding(
                    self,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    sequence_parallel_mesh=sequence_parallel_mesh,
                    **kwargs,
                )
        else:
            return CUDAPatchedLlamaForCausalLM.patched_attn_forward_training(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                position_ids=position_ids,
                cache_position=cache_position,
                output_attentions=output_attentions,
                sequence_parallel_mesh=sequence_parallel_mesh,
                **kwargs,
            )

    @staticmethod
    def patched_attn_forward_training(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention. This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            sp_size = sequence_parallel_mesh.size()
            num_kv_heads = key_states.size(1)
            if sp_size > num_kv_heads:
                assert sp_size % num_kv_heads == 0
                key_states = repeat_kv(key_states, sp_size // num_kv_heads)
                value_states = repeat_kv(value_states, sp_size // num_kv_heads)

            query_states = all_to_all(
                query_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh
            )
            key_states = all_to_all(
                key_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh
            )
            value_states = all_to_all(
                value_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh
            )

        if (
            XTUNER_DETERMINISTIC
            and "flash_attention" in self.config._attn_implementation
        ):
            kwargs["deterministic"] = True

        # (bs, n , qh // sp, d)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            position_ids=position_ids,
            **kwargs,
        )

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            attn_output = all_to_all(
                attn_output, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @staticmethod
    def patched_attn_prefilling(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        fill_paged_kv_cache(
            key_states,
            value_states,
            past_key_value[self.layer_idx][0],
            past_key_value[self.layer_idx][1],
            kwargs["cu_seq_lens_q"],
            kwargs["cu_seq_lens_k"],
            kwargs["max_length_q"],
            kwargs["max_length_k"],
            kwargs["block_table"],
        )

        assert self.config._attn_implementation == "flash_attention_2"
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @staticmethod
    def patched_attn_decoding(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        seq_lens_k = kwargs["cu_seq_lens_k"][1:] - kwargs["cu_seq_lens_k"][:-1]
        block_table = kwargs["block_table"]
        block_size = past_key_value[self.layer_idx][0].size(1)
        bs = block_table.size(0)
        assert kwargs["cu_seq_lens_k"].numel() - 1 == bs

        _key_states = key_states.transpose(1, 2).squeeze(0)
        _value_states = value_states.transpose(1, 2).squeeze(0)

        block_index = block_table[:, 0] + (seq_lens_k[:bs] - 1) // block_size
        past_key_value[self.layer_idx][0][
            block_index, (seq_lens_k[:bs] - 1) % block_size
        ] = _key_states
        past_key_value[self.layer_idx][1][
            block_index, (seq_lens_k[:bs] - 1) % block_size
        ] = _value_states

        assert self.config._attn_implementation == "flash_attention_2"

        attn_weights = None

        attn_output = paged_attention_decoding(
            query_states,
            past_key_value[self.layer_idx][0],
            past_key_value[self.layer_idx][1],
            kwargs["cu_seq_lens_k"][1:] - kwargs["cu_seq_lens_k"][:-1],
            block_table,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @staticmethod
    def patched_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "block_table" in kwargs and kwargs["block_table"] is not None:
            if "prefilling" in kwargs and kwargs["prefilling"]:
                return CUDAPatchedLlamaForCausalLM.patched_layer_forward_training(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            else:
                return CUDAPatchedLlamaForCausalLM.patched_layer_forward_decoding(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
        else:
            return CUDAPatchedLlamaForCausalLM.patched_layer_forward_training(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    @staticmethod
    # @torch.compile(fullgraph=True)
    def patched_layer_forward_training(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    @staticmethod
    @torch.compile(fullgraph=True)
    def patched_layer_forward_decoding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    @staticmethod
    def patched_casual_forward(
        self: LlamaForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        label_shifted=False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        if labels is None:
            loss = None

            logits = self.lm_head(hidden_states)
            if isinstance(logits, DTensor):
                logits = logits.to_local()
        else:
            if liger_kernel_is_available():
                # unable to return logits when using Liger Kernel
                logits = None

                if label_shifted:
                    shift_hidden_states = hidden_states
                    shift_labels = labels
                else:
                    shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                shift_hidden_states = shift_hidden_states.view(
                    -1, self.config.hidden_size
                )
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_hidden_states.device)

                from liger_kernel.transformers.fused_linear_cross_entropy import (
                    LigerFusedLinearCrossEntropyLoss,
                )

                loss_fct = LigerFusedLinearCrossEntropyLoss()

                # if self.config
                lm_head_weight = self.lm_head.weight
                lm_head_bias = self.lm_head.bias
                if isinstance(lm_head_weight, DTensor):
                    # NOTE: We ASSUME lm_head.weight has been fully sharded as model's
                    # outmost FSDP unit, so it will have been all gathered here as long
                    # as model's forward is called. The only device mesh that remains
                    # here is the TP mesh.
                    assert isinstance(shift_hidden_states, DTensor)
                    assert lm_head_bias is None or isinstance(lm_head_bias, DTensor)
                    assert (
                        lm_head_weight.device_mesh == shift_hidden_states.device_mesh
                    ), (
                        "Expected lm_head.weight to be on the same device mesh as shift_hidden_states, "
                        f"got {lm_head_weight.device_mesh} and {shift_hidden_states.device_mesh}"
                    )
                    tp_mesh = lm_head_weight.device_mesh
                    assert (
                        tp_mesh.ndim == 1 and "tp" in tp_mesh.mesh_dim_names[0]
                    ), f"Expected lm_head.weight placed on a 1d TP mesh, got {tp_mesh}"
                    shift_hidden_states = shift_hidden_states.to_local()
                    # Liger kernel interrupts the DTensor gradient placement that should be propagated
                    # to the last lm_head Linear module. Since the input is Shard(0) and the weight
                    # is Replicate(), the gradient should be Partial(). We manually set the gradient
                    # placement to make grad_norm calculation & optimizer.step() correct
                    lm_head_weight = lm_head_weight.to_local(
                        grad_placements=(Partial(),)
                    )
                    if lm_head_bias is not None:
                        lm_head_bias = lm_head_bias.to_local(
                            grad_placements=(Partial(),)
                        )

                loss = loss_fct(
                    lm_head_weight, shift_hidden_states, shift_labels, lm_head_bias
                )

            else:
                logits = self.lm_head(hidden_states)
                if isinstance(logits, DTensor):
                    logits = logits.to_local()

                if label_shifted:
                    shift_logits = logits
                    shift_labels = labels
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        label_shifted: bool = False,
        gather_logprobs: bool = False,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        block_table: Optional[torch.LongTensor] = None,
        prefilling: bool = False,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        # In order to unify the forward arguments of moe and dense model.
        # The moe models have an additional argument `aux_loss_global_average`.
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if gather_logprobs:
            assert labels is not None and label_shifted

        _input_ids = input_ids
        _labels = labels
        _position_ids = position_ids
        _cu_seq_lens_q = cu_seq_lens_q
        _cu_seq_lens_k = cu_seq_lens_k
        _max_length_q = max_length_q
        _max_length_k = max_length_k

        if self.fsdp_config.torch_compile or self.fsdp_config.enable_fp8:
            _input_ids = pad_to_max_length(
                _input_ids, 0, self.fsdp_config.max_length, 1
            )
            _position_ids = pad_to_max_length(
                _position_ids, 0, self.fsdp_config.max_length, 1
            )
            if labels is not None:
                _labels = pad_to_max_length(
                    _labels, -100, self.fsdp_config.max_length, 1
                )
        else:
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                multiple_of = sequence_parallel_mesh.size() * self.tp_mesh.size()
            else:
                multiple_of = self.tp_mesh.size()

            _input_ids = pad_to_multiple_of(_input_ids, 0, multiple_of, 1)
            _position_ids = pad_to_multiple_of(_position_ids, 0, multiple_of, 1)
            if labels is not None:
                _labels = pad_to_multiple_of(_labels, -100, multiple_of, 1)

        num_padded_tokens = _input_ids.numel() - input_ids.numel()

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _input_ids = split_for_sequence_parallel(
                _input_ids, dim=1, sp_mesh=sequence_parallel_mesh
            )
            _position_ids = split_for_sequence_parallel(
                _position_ids, dim=1, sp_mesh=sequence_parallel_mesh
            )

            if labels is not None:
                _labels = split_for_sequence_parallel(
                    _labels, dim=1, sp_mesh=sequence_parallel_mesh
                )

        if self.tp_mesh.size() > 1:
            if labels is not None:
                _labels = split_for_sequence_parallel(
                    _labels, dim=1, sp_mesh=self.tp_mesh
                )

        if self.training and num_padded_tokens > 0:
            assert torch.any(cu_seq_lens_k == cu_seq_lens_q)
            _cu_seq_lens_q = _cu_seq_lens_q.tolist()
            _cu_seq_lens_q.append(_cu_seq_lens_q[-1] + num_padded_tokens)

            _cu_seq_lens_q = torch.IntTensor(_cu_seq_lens_q).to(cu_seq_lens_q.device)
            _cu_seq_lens_k = _cu_seq_lens_q

            _max_length_q = max(_max_length_q, num_padded_tokens)
            _max_length_k = _max_length_q

        outputs = self.patched_model(
            _input_ids,
            attention_mask,
            _position_ids,
            past_key_values,
            inputs_embeds,
            _labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
            num_logits_to_keep,
            label_shifted=label_shifted,
            cu_seq_lens_q=_cu_seq_lens_q,
            cu_seq_lens_k=_cu_seq_lens_k,
            max_length_q=_max_length_q,
            max_length_k=_max_length_k,
            block_table=block_table,
            prefilling=prefilling,
            sequence_parallel_mesh=self.sequence_parallel_mesh,
        )

        if outputs.loss is not None:
            outputs.loss = outputs.loss * (_labels >= 0).sum()
            if self.tp_mesh.size() > 1:
                outputs.loss = DTensor.from_local(
                    outputs.loss, self.tp_mesh, placements=(Partial(),)
                ).full_tensor()
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                outputs.loss = dist.nn.all_reduce(
                    outputs.loss, group=sequence_parallel_mesh.get_group()
                )
            if (labels >= 0).sum() > 0:
                outputs.loss = outputs.loss / (labels >= 0).sum()

        return outputs

    @torch.no_grad()
    def sample(
        self,
        logits,
        cu_seq_lens,
        do_sample=True,
        top_k=0,
        top_p=0.9,
        temperature=1.0,
        vocab_size=None,
    ):
        last_token_inds = cu_seq_lens[1:] - 1
        rank_start = logits.size(0) * self.tp_mesh.get_local_rank()
        rank_end = logits.size(0) * (self.tp_mesh.get_local_rank() + 1)

        other_rank_mask = torch.logical_or(
            last_token_inds < rank_start, last_token_inds >= rank_end
        )
        last_token_inds -= rank_start
        last_token_inds = last_token_inds.clip(min=0, max=logits.size(0) - 1)

        logits = logits[last_token_inds]

        if vocab_size is not None:
            logits[:, vocab_size:] = -torch.inf

        if not do_sample:
            sampled = logits.argmax(-1)
            sampled[other_rank_mask] = 0
            if self.tp_mesh.size() > 1:
                dist.all_reduce(sampled, group=self.tp_mesh.get_group())
            return sampled

        # Apply temperature if necessary
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k if necessary
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            _, topk_indices = logits.topk(top_k, dim=-1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, False)
            logits.masked_fill_(mask, -torch.inf)

        # Apply top-p (nucleus sampling) if necessary
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            mask = cum_probs <= (1 - top_p)
            mask[:, -1] = False
            sorted_logits.masked_fill_(mask, -torch.inf)

            logits.scatter_(-1, sorted_indices, sorted_logits)

        probs = logits.softmax(-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        sampled[other_rank_mask] = 0
        if self.tp_mesh.size() > 1:
            dist.all_reduce(sampled, group=self.tp_mesh.get_group())

        return sampled

    def gather_logprobs(self, shifted_logits, shifted_labels, sequence_parallel_mesh):
        if self.fsdp_config.torch_compile:
            _labels = pad_to_max_length(
                shifted_labels, -100, self.fsdp_config.max_length, 1
            )
        else:
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                multiple_of = sequence_parallel_mesh.size() * self.tp_mesh.size()
            else:
                multiple_of = self.tp_mesh.size()

            _labels = pad_to_multiple_of(shifted_labels, -100, multiple_of, 1)

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _labels = split_for_sequence_parallel(
                _labels, dim=1, sp_mesh=sequence_parallel_mesh
            )

        if self.tp_mesh.size() > 1:
            _labels = split_for_sequence_parallel(_labels, dim=1, sp_mesh=self.tp_mesh)

        logprobs = F.log_softmax(shifted_logits, dim=-1)
        logprobs = logprobs.gather(
            dim=-1, index=_labels.clip(min=0).unsqueeze(-1)
        ).squeeze(-1)

        if self.tp_mesh.size() > 1:
            _logprobs = dist.nn.all_gather(logprobs, group=self.tp_mesh.get_group())
            logprobs = torch.cat(_logprobs, dim=1)

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _logprobs = dist.nn.all_gather(
                logprobs, group=sequence_parallel_mesh.get_group()
            )
            logprobs = torch.cat(_logprobs, dim=1)

        logprobs = logprobs[:, : shifted_labels.size(1)]

        return logprobs

    def trainable_parameters(self):
        _requried_grad_params = [
            param for param in self.patched_model.parameters() if param.requires_grad
        ]
        return _requried_grad_params

    def clip_grad_norm(self, max_norm):
        grad_norm = clip_grad_norm_(self.trainable_parameters(), max_norm)
        return grad_norm

    def precompute_float8_dynamic_scale_for_fsdp(self):
        if self._fsdp_config.enable_fp8:
            self._float8_handler.precompute_float8_dynamic_scale_for_fsdp(
                self._patched_model
            )


class MLUPatchedLlamaForCausalLM(CUDAPatchedLlamaForCausalLM):
    device_type = "mlu"


class MuxiPatchedLlamaForCausalLM(CUDAPatchedLlamaForCausalLM):
    device_type = "muxi"
