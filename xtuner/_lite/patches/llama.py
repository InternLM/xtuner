from xtuner._lite.patches.base import PatchedCausalLM, FSDPConfig, ModelConfig, HFCheckpointLoader, lazy_init_fn, clip_grad_norm_

from xtuner._lite.patches.utils import pad_to_multiple_of, pad_to_max_length
from typing import Callable, Optional, Tuple, TypedDict, Union, List
import types
import copy
from packaging import version
from tqdm import tqdm
from functools import partial
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaForCausalLM, apply_rotary_pos_emb, eager_attention_forward, repeat_kv
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable import checkpoint
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast
from xtuner._lite.accelerate import liger_kernel_is_available
from xtuner._lite.chat import HybridChatTemplate

from xtuner._lite.parallel.sequence import (
    pre_process_for_sequence_parallel_attn, post_process_for_sequence_parallel_attn,
    split_for_sequence_parallel)
from torch.distributed._tensor import Shard, Replicate, distribute_tensor, DTensor
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)
logger = logging.get_logger(__name__)


class CUDAPatchedLlamaForCausalLM(PatchedCausalLM):
    device_type = 'cuda'
    attn_cls = LlamaAttention
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
        "model.norm": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),
        "lm_head": PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ),
    }

    chat_template = HybridChatTemplate(
        system=('<|start_header_id|>system<|end_header_id|>\n\n{system}'
                '<|eot_id|>'),
        user=('<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>'
              '<|start_header_id|>assistant<|end_header_id|>\n\n'),
        assistant='{assistant}<|eot_id|>',
        sep='',
        stop_words=['<|eot_id|>']
    )

    def __init__(self, model: LlamaForCausalLM, fsdp_config: Optional[FSDPConfig]= None):
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
            self.fully_shard(fsdp_config)

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
            num_key_value_heads=self.patched_model.config.num_key_value_heads // fsdp_config.tp_size,
            hidden_size=self.patched_model.config.hidden_size,
            intermediate_size=self.patched_model.config.intermediate_size,
            vocab_size=self.patched_model.config.vocab_size,
            head_dim=self.patched_model.config.head_dim
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
    
    def fully_shard(self, fsdp_config: FSDPConfig) -> None:
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
        world_mesh_name = f'{fsdp_config.mesh_prefix}.world'
        fsdp_mesh_name = f'{fsdp_config.mesh_prefix}.fsdp'
        tp_mesh_name = f'{fsdp_config.mesh_prefix}.tp'
        dp_mesh_name = f'{fsdp_config.mesh_prefix}.dp'
        sp_mesh_name = f'{fsdp_config.mesh_prefix}.sp'
        data_mesh_name = f'{fsdp_config.mesh_prefix}.data'
        _tp_mesh_name = f'{fsdp_config.mesh_prefix}._tp'

        world_mesh = init_device_mesh(
            self.device_type, 
            (world_size ,),
            mesh_dim_names = [world_mesh_name, ]
        )
        self.world_mesh = world_mesh[world_mesh_name]

        model_mesh = init_device_mesh(
            self.device_type, 
            (world_size // tp_size, tp_size),
            mesh_dim_names = [fsdp_mesh_name, tp_mesh_name]
        )
        
        fsdp_mesh = model_mesh[fsdp_mesh_name]
        tp_mesh = model_mesh[tp_mesh_name]

        self.tp_mesh = tp_mesh
        self.fsdp_mesh = fsdp_mesh

        data_mesh = init_device_mesh(
            self.device_type, 
            (world_size // tp_size // sp_size, sp_size , tp_size),
            mesh_dim_names = [dp_mesh_name, sp_mesh_name, _tp_mesh_name]
        )
        self.dp_mesh = data_mesh[dp_mesh_name]
        self.sp_mesh = data_mesh[sp_mesh_name]

        _data_mesh = init_device_mesh(
            self.device_type, 
            (world_size // tp_size // sp_size, sp_size * tp_size),
            mesh_dim_names = [dp_mesh_name, data_mesh_name]
        )
        self._data_mesh = _data_mesh[data_mesh_name]
    

        param_init_fn = partial(
            lazy_init_fn,
            module2name = { mod: name for name, mod in self.patched_model.named_modules() },
            checkpoint_loader = HFCheckpointLoader(self.patched_model.config._name_or_path)
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype,
            reduce_dtype=fsdp_config.reduce_dtype)

        self.patched_model.model.rotary_emb = LlamaRotaryEmbedding(self.patched_model.config)

        num_recompute_layers = int(self.model_config.num_hidden_layers * fsdp_config.recompute_ratio)
        
        if fsdp_config.torch_compile:
            compiled_layers = []

            from torch.distributed._symmetric_memory import enable_symm_mem_for_group
            torch._inductor.config._micro_pipeline_tp = True
            enable_symm_mem_for_group(self.tp_mesh.get_group().group_name)

        for layer in tqdm(self.patched_model.model.layers):
            
            layer.apply(param_init_fn)
            attention = layer.self_attn
            
            if tp_mesh.size() > 1:
                
                parallelize_module(
                    module=layer,
                    device_mesh=tp_mesh,
                    parallelize_plan=self.layer_tp_plan
                )
                
            if attention.layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(layer, preserve_rng_state=False)
                # checkpoint(layer)
            
            if fsdp_config.torch_compile:
                layer = torch.compile(layer, fullgraph=True)
            
            self.patched_model.model.layers.register_module(str(attention.layer_idx), layer)
            
            fully_shard(
                layer,
                mesh=fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=fsdp_config.reshard_after_forward,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
            )

            if fsdp_config.torch_compile:
                compiled_layers.append(layer)

        if version.parse(torch.__version__) >= version.parse("2.5.0"):
            for layer_cur, layer_next in zip(self.patched_model.model.layers[:-1], self.patched_model.model.layers[1:]):
                layer_cur.set_modules_to_forward_prefetch([layer_next])

        self.patched_model.lm_head.apply(param_init_fn)
        self.patched_model.model.embed_tokens.apply(param_init_fn)
        self.patched_model.model.norm.apply(param_init_fn)

        if tp_mesh.size() > 1:
            _weight = self.patched_model.lm_head.weight
            _dtensor_weight = nn.Parameter(
                distribute_tensor(_weight, tp_mesh, [Replicate()]))
            self.patched_model.lm_head.register_parameter('weight', _dtensor_weight)
            
            _weight = self.patched_model.model.norm.weight
            _dtensor_weight = nn.Parameter(
                distribute_tensor(_weight, tp_mesh, [Replicate()]))
            self.patched_model.model.norm.register_parameter('weight', _dtensor_weight)
            
            parallelize_module(
                self.patched_model,
                tp_mesh,
                self.casual_tp_plan,
            )

        fully_shard(
            self.patched_model,
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

    @staticmethod
    def patched_attn_forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            
            
            # bs, qh, n_div_sp, d = query_states.shape
            # _, kvh, n_div_sp, d = key_states.shape
            
            # assert bs == 1
            # sp = sequence_parallel_mesh.size()
            # n = n_div_sp * sp
            # # (b, n // sp, qh, d) 
            # query_states = query_states.transpose(1,2)
            # key_states = key_states.transpose(1,2)
            # value_states = value_states.transpose(1,2)

            # # (qh, b * n // sp, d) 
            # query_states = query_states.flatten(0, 1).transpose(0,1).contiguous()
            # key_states = key_states.flatten(0, 1).transpose(0,1).contiguous()
            # value_states = value_states.flatten(0, 1).transpose(0,1).contiguous()

            # # (qh, b * n // sp, d) 
            # _query_states = query_states.new_empty(qh, bs * n // sp, d)
            # # (kvh, b * n // sp, d) 
            # _key_states = key_states.new_empty(kvh, bs * n // sp, d)
            # _value_states = value_states.new_empty(kvh, bs * n // sp, d)

            # # (qh, b * n // sp, d) 
            # _query_states = dist.nn.all_to_all_single(
            #     _query_states, query_states, group=sequence_parallel_mesh.get_group())
            # # (kvh, b * n // sp, d) 
            # _key_states = dist.nn.all_to_all_single(
            #     _key_states, key_states, group=sequence_parallel_mesh.get_group())
            # # (kvh, b * n // sp, d) 
            # _value_states = dist.nn.all_to_all_single(
            #     _value_states, value_states, group=sequence_parallel_mesh.get_group())
            
            # # (sp, qh // sp, b*n // sp, d)
            # _query_states = _query_states.view(sp, qh // sp, bs* n // sp, d)
            # # (sp, kvh // sp, b*n // sp, d)
            # _key_states = _key_states.view(sp, kvh // sp, bs * n // sp, d)
            # _value_states = _value_states.view(sp, kvh // sp, bs * n // sp, d)
            
            # query_states = _query_states.transpose(1,2).reshape(bs, n, qh // sp, d).transpose(1,2)
            # key_states = _key_states.transpose(1,2).reshape(bs, n, kvh // sp, d).transpose(1,2)
            # value_states = _value_states.transpose(1,2).reshape(bs, n, kvh // sp, d).transpose(1,2)
            
            # different from LlamaAttention.forward
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
            value_states = value_states.transpose(1,2)

            query_states, key_states, value_states = pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states, sequence_parallel_mesh
            )

            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
            value_states = value_states.transpose(1,2)



        # (bs, n , qh // sp, d)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            # # (bs * n , qh // sp, d)
            # attn_output = attn_output.flatten(0, 1).contiguous()
            # # (bs * n, qh // sp, d)
            # _attn_output = attn_output.new_empty(bs * n, qh // sp, d)

            # # (bs * n, qh // sp, d)
            # attn_output = dist.nn.all_to_all_single(
            #     _attn_output, attn_output, group=sequence_parallel_mesh.get_group())

            # # (sp, bs * n // sp, qh // sp, d)
            # attn_output = attn_output.view(sp, bs * n_div_sp, qh // sp, d)
            # # (bs * n // sp, sp, qh // sp, d)
            # attn_output = attn_output.transpose(0, 1)
            attn_output = post_process_for_sequence_parallel_attn(
                attn_output, sequence_parallel_mesh
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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
        label_shifted = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

                shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_hidden_states.device)

                from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
                loss_fct = LigerFusedLinearCrossEntropyLoss()
                
                lm_head_weight = self.lm_head.weight
                if isinstance(lm_head_weight, DTensor):
                    assert isinstance(shift_hidden_states, DTensor)
                    shift_hidden_states = shift_hidden_states.to_local()
                    lm_head_weight = self.lm_head.weight.to_local()
                
                loss = loss_fct(lm_head_weight, shift_hidden_states, shift_labels, self.lm_head.bias)

            else:
                logits = self.lm_head(hidden_states)

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
        
        if self.fsdp_config.torch_compile:
            _input_ids = pad_to_max_length(_input_ids, 0, self.fsdp_config.max_length, 1)
            _position_ids = pad_to_max_length(_position_ids, 0, self.fsdp_config.max_length, 1)
            if labels is not None:
                _labels = pad_to_max_length(_labels, -100, self.fsdp_config.max_length, 1)
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
                _input_ids, dim=1, sp_mesh=sequence_parallel_mesh)
            _position_ids = split_for_sequence_parallel(
                _position_ids, dim=1, sp_mesh=sequence_parallel_mesh)
            
            if labels is not None:
                _labels = split_for_sequence_parallel(
                    _labels, dim=1, sp_mesh=sequence_parallel_mesh)

        if self.tp_mesh.size() > 1:
            if labels is not None:
                _labels = split_for_sequence_parallel(
                        _labels, dim=1, sp_mesh=self.tp_mesh)
        
        if self.training and num_padded_tokens  > 0:
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

        if outputs.logits is not None:
           
            if self.tp_mesh.size() > 1:
                _logits = dist.nn.all_gather(outputs.logits, group=self.tp_mesh.get_group())
                outputs.logits = torch.cat(_logits, dim=1)
            # if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            #     _logits = dist.nn.all_gather(outputs.logits, group=sequence_parallel_mesh.get_group())
            #     outputs.logits = torch.cat(_logits, dim=1)
    
        if outputs.loss is not None:
            outputs.loss = outputs.loss * (_labels >= 0).sum()
            if self.tp_mesh.size() > 1:
                outputs.loss = dist.nn.all_reduce(outputs.loss, group=self.tp_mesh.get_group())
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                outputs.loss = dist.nn.all_reduce(outputs.loss, group=sequence_parallel_mesh.get_group())
            outputs.loss = outputs.loss / (labels >= 0).sum() 

        return outputs

    def gather_logprobs(self, shifted_logits, shifted_labels, sequence_parallel_mesh):
     
        logprobs = F.log_softmax(shifted_logits, dim=-1)
        logprobs = logprobs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

        # if self.tp_mesh.size() > 1:
        #     _logprobs = dist.nn.all_gather(logprobs, group=self.tp_mesh.get_group())
        #     logprobs = torch.cat(_logprobs, dim=1)
        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _logprobs = dist.nn.all_gather(logprobs, group=sequence_parallel_mesh.get_group())
            logprobs = torch.cat(_logprobs, dim=1)
    
        return logprobs
    
    def trainable_parameters(self):
        _requried_grad_params = [
            param for param in self.patched_model.parameters() if param.requires_grad
        ]
        return _requried_grad_params
    
    def clip_grad_norm(self, max_norm):

        if self.tp_mesh.size() > 1:
            dist.all_reduce(self.patched_model.lm_head.weight.grad.to_local(), group=self.tp_mesh.get_group())
            dist.all_reduce(self.patched_model.model.norm.weight.grad.to_local(), group=self.tp_mesh.get_group())
            self.patched_model.lm_head.weight.grad.div_(self.tp_mesh.size())
            self.patched_model.model.norm.weight.grad.div_(self.tp_mesh.size())
            for param in self.trainable_parameters():
                param.grad.div_(self.tp_mesh.size())

        grad_norm = clip_grad_norm_(self.trainable_parameters(), self.world_mesh, max_norm)
        return grad_norm

class MLUPatchedLlamaForCausalLM(CUDAPatchedLlamaForCausalLM):
    device_type = 'mlu'

class MuxiPatchedLlamaForCausalLM(CUDAPatchedLlamaForCausalLM):
    device_type = 'muxi'

class AscendPatchedLlamaForCausalLM(CUDAPatchedLlamaForCausalLM):
    device_type = 'npu'

    @staticmethod
    def patched_attn_forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            
            
            # bs, qh, n_div_sp, d = query_states.shape
            # _, kvh, n_div_sp, d = key_states.shape
            
            # assert bs == 1
            # sp = sequence_parallel_mesh.size()
            # n = n_div_sp * sp
            # # (b, n // sp, qh, d) 
            # query_states = query_states.transpose(1,2)
            # key_states = key_states.transpose(1,2)
            # value_states = value_states.transpose(1,2)

            # # (qh, b * n // sp, d) 
            # query_states = query_states.flatten(0, 1).transpose(0,1).contiguous()
            # key_states = key_states.flatten(0, 1).transpose(0,1).contiguous()
            # value_states = value_states.flatten(0, 1).transpose(0,1).contiguous()

            # # (qh, b * n // sp, d) 
            # _query_states = query_states.new_empty(qh, bs * n // sp, d)
            # # (kvh, b * n // sp, d) 
            # _key_states = key_states.new_empty(kvh, bs * n // sp, d)
            # _value_states = value_states.new_empty(kvh, bs * n // sp, d)

            # # (qh, b * n // sp, d) 
            # _query_states = dist.nn.all_to_all_single(
            #     _query_states, query_states, group=sequence_parallel_mesh.get_group())
            # # (kvh, b * n // sp, d) 
            # _key_states = dist.nn.all_to_all_single(
            #     _key_states, key_states, group=sequence_parallel_mesh.get_group())
            # # (kvh, b * n // sp, d) 
            # _value_states = dist.nn.all_to_all_single(
            #     _value_states, value_states, group=sequence_parallel_mesh.get_group())
            
            # # (sp, qh // sp, b*n // sp, d)
            # _query_states = _query_states.view(sp, qh // sp, bs* n // sp, d)
            # # (sp, kvh // sp, b*n // sp, d)
            # _key_states = _key_states.view(sp, kvh // sp, bs * n // sp, d)
            # _value_states = _value_states.view(sp, kvh // sp, bs * n // sp, d)
            
            # query_states = _query_states.transpose(1,2).reshape(bs, n, qh // sp, d).transpose(1,2)
            # key_states = _key_states.transpose(1,2).reshape(bs, n, kvh // sp, d).transpose(1,2)
            # value_states = _value_states.transpose(1,2).reshape(bs, n, kvh // sp, d).transpose(1,2)
            
            # different from LlamaAttention.forward
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
            value_states = value_states.transpose(1,2)

            query_states, key_states, value_states = pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states, sequence_parallel_mesh
            )

            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
            value_states = value_states.transpose(1,2)


        import torch_npu
        import numpy as np
        attention_mask = torch.triu(
            torch.ones(max_length_q, max_length_k), diagonal=1).bool().to(self.device_type)
        
        head_num = query_states.shape[1]
        attn_output = torch_npu.npu_fusion_attention(
            query_states,
            key_states,
            value_states,
            head_num,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=1.0 / query_states.shape[-1].sqrt().item(),
            keep_prob=1,
            input_layout='TND',
            actual_seq_qlen=tuple(cu_seq_lens_q[1:].tolist()),
            actual_seq_kvlen=tuple(cu_seq_lens_k[1:].tolist()),
            pre_tockens=2147483647,
            next_tockens=0,
            inner_precise=0)[0]

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            # # (bs * n , qh // sp, d)
            # attn_output = attn_output.flatten(0, 1).contiguous()
            # # (bs * n, qh // sp, d)
            # _attn_output = attn_output.new_empty(bs * n, qh // sp, d)

            # # (bs * n, qh // sp, d)
            # attn_output = dist.nn.all_to_all_single(
            #     _attn_output, attn_output, group=sequence_parallel_mesh.get_group())

            # # (sp, bs * n // sp, qh // sp, d)
            # attn_output = attn_output.view(sp, bs * n_div_sp, qh // sp, d)
            # # (bs * n // sp, sp, qh // sp, d)
            # attn_output = attn_output.transpose(0, 1)
            attn_output = post_process_for_sequence_parallel_attn(
                attn_output, sequence_parallel_mesh
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

if __name__ == '__main__':

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xtuner._lite.parallel import setup_parallel

    setup_parallel()
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_pretrained(
            '/fs-computility/llm/shared/llm_llama3_hf/Llama3.1-8B-Instruct/',
            attn_implementation='flash_attention_2'
        )
    tokenizer = AutoTokenizer.from_pretrained('/fs-computility/llm/shared/llm_llama3_hf/Llama3.1-8B-Instruct/')
    fsdp_config = FSDPConfig(reshard_after_forward=False, tp_size=8)
    patched_model = CUDAPatchedLlamaForCausalLM(model, fsdp_config)

    prompts = [
        '请给我介绍五个上海的景点'
    ] * 32 * 8

    input_ids = [tokenizer.encode(prompt, return_tensors="pt").cuda() for prompt in prompts]
    import time
    from xtuner._lite import get_logger
    logger = get_logger()
    start = time.time()
    patched_model.eval()
    import torch
    torch.manual_seed(42)
    response = patched_model.generate(
        input_ids,
        max_batch_size=len(prompts),
        max_new_tokens=1024,
        max_length=2048,
        cuda_graph=True
    )
    for res in response:
        logger.info(tokenizer.decode(res))

    logger.info(time.time() - start)