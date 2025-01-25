from xtuner._lite.patches.llama import CUDAPatchedLlamaForCausalLM
from xtuner._lite.patches.base import FSDPConfig, ModelConfig
from xtuner._lite.chat import HybridChatTemplate
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2ForCausalLM, apply_rotary_pos_emb, eager_attention_forward, repeat_kv


from typing import Callable, Optional, Tuple

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch


from torch.distributed.device_mesh import DeviceMesh
from transformers.utils import logging

from xtuner._lite.chat import HybridChatTemplate

from xtuner._lite.parallel.sequence import (
    pre_process_for_sequence_parallel_attn, post_process_for_sequence_parallel_attn)

logger = logging.get_logger(__name__)


class CUDAPatchedQwen2ForCausalLM(CUDAPatchedLlamaForCausalLM):

    attn_cls = Qwen2Attention
    layer_cls = Qwen2DecoderLayer
    causal_cls = Qwen2ForCausalLM

    chat_template = HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>', '<|endoftext|>']
    )

    def init_model_config(self, fsdp_config:FSDPConfig):
        
        assert self.patched_model.config.num_key_value_heads >= fsdp_config.tp_size
        assert self.patched_model.config.num_key_value_heads % fsdp_config.tp_size == 0
        assert self.patched_model.config.hidden_size % self.patched_model.config.num_attention_heads == 0

        self._model_config = ModelConfig(
            num_hidden_layers=self.patched_model.config.num_hidden_layers,
            num_attention_heads=self.patched_model.config.num_attention_heads,
            num_key_value_heads=self.patched_model.config.num_key_value_heads // fsdp_config.tp_size,
            hidden_size=self.patched_model.config.hidden_size,
            intermediate_size=self.patched_model.config.intermediate_size,
            vocab_size=self.patched_model.config.vocab_size,
            head_dim=self.patched_model.config.hidden_size // self.patched_model.config.num_attention_heads
        )

    @staticmethod
    def patched_attn_forward(
        self: Qwen2Attention,
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

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

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
            sliding_window=sliding_window,
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


