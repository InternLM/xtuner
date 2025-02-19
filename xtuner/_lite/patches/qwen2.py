from xtuner._lite.patches.llama import CUDAPatchedLlamaForCausalLM, all_to_all
from xtuner._lite.patches.base import FSDPConfig, ModelConfig
from xtuner._lite.chat import HybridChatTemplate
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2Attention,  Qwen2RMSNorm, Qwen2DecoderLayer, Qwen2ForCausalLM, apply_rotary_pos_emb, eager_attention_forward, repeat_kv


from typing import Callable, Optional, Tuple

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch


from torch.distributed.device_mesh import DeviceMesh
from transformers.utils import logging

from xtuner._lite.chat import HybridChatTemplate


logger = logging.get_logger(__name__)


class CUDAPatchedQwen2ForCausalLM(CUDAPatchedLlamaForCausalLM):

    rotary_emb_cls = Qwen2RotaryEmbedding
    attn_cls = Qwen2Attention
    layer_cls = Qwen2DecoderLayer
    causal_cls = Qwen2ForCausalLM
    norm_cls = Qwen2RMSNorm

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
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if 'block_table' in kwargs and kwargs['block_table'] is not None:
            if (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
            ):
                raise NotImplementedError

            # generating
            if 'prefilling' in kwargs and kwargs['prefilling']:
                return CUDAPatchedLlamaForCausalLM.patched_attn_prefilling(
                    self, 
                    hidden_states=hidden_states, position_embeddings=position_embeddings,
                    attention_mask=attention_mask, past_key_value=past_key_value,
                    position_ids=position_ids, 
                    cache_position=cache_position, output_attentions=output_attentions,
                    sequence_parallel_mesh=sequence_parallel_mesh, 
                    **kwargs
                )
            else:
                return CUDAPatchedLlamaForCausalLM.patched_attn_decoding(
                    self, 
                    hidden_states=hidden_states, position_embeddings=position_embeddings,
                    attention_mask=attention_mask, past_key_value=past_key_value, 
                    position_ids=position_ids,
                    cache_position=cache_position, output_attentions=output_attentions,
                    sequence_parallel_mesh=sequence_parallel_mesh, 
                    **kwargs
                )
        else:
            return CUDAPatchedQwen2ForCausalLM.patched_attn_forward_training(
                    self, 
                    hidden_states=hidden_states, position_embeddings=position_embeddings,
                    attention_mask=attention_mask, past_key_value=past_key_value, 
                    position_ids=position_ids,
                    cache_position=cache_position, output_attentions=output_attentions,
                    sequence_parallel_mesh=sequence_parallel_mesh, 
                    **kwargs
                )

    @staticmethod
    def patched_attn_forward_training(
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
            sp_size = sequence_parallel_mesh.size()
            num_kv_heads = key_states.size(1)
            if sp_size > num_kv_heads:
                assert sp_size % num_kv_heads == 0
                key_states = repeat_kv(key_states, sp_size // num_kv_heads)
                value_states = repeat_kv(value_states, sp_size // num_kv_heads)

            query_states = all_to_all(query_states, scatter_dim=1, gather_dim=2,  mesh=sequence_parallel_mesh)
            key_states = all_to_all(key_states, scatter_dim=1, gather_dim=2,  mesh=sequence_parallel_mesh)
            value_states = all_to_all(value_states, scatter_dim=1, gather_dim=2,  mesh=sequence_parallel_mesh)

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
            attn_output = all_to_all(attn_output, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh)
           

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



if __name__ == '__main__':

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xtuner._lite.parallel import setup_parallel
    from xtuner._lite.patches import FSDPConfig, AutoPatch
    
    torch._dynamo.config.cache_size_limit = 16384 
    setup_parallel()
    
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-7B-Instruct',
            attn_implementation='flash_attention_2',
        )

        # model.lm_head = torch.nn.Linear(model.config.hidden_size, 1)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
    fsdp_config = FSDPConfig(reshard_after_forward=False, torch_compile=False, tp_size=4)
    patched_model = AutoPatch.from_causal_lm(model, fsdp_config)

    prompts = [
        '<|im_start|>user\n陈金山同志在上海市委有什么突出贡献？<|im_end|>\n<|im_start|>assistant\n',
        '<|im_start|>user\n我的前任男友伤害了我，我想报复他，但我不知道应该采取什么措施来打击他。<|im_end|>\n<|im_start|>assistant\n'
    ] * 32

    # prompts = ['Hello'] * 256

    input_ids = [tokenizer.encode(prompt, return_tensors="pt").cuda() for prompt in prompts]
    import time
    from xtuner._lite import get_logger
    logger = get_logger()
    start = time.time()
    patched_model.eval()
    import torch
    torch.manual_seed(42)

    from torch.profiler import ProfilerActivity, profile, record_function
    
    response = patched_model.generate(
        input_ids,
        stop_token_ids=[151645, 151643],
        max_batch_size=len(prompts),
        max_prefill_batch=len(prompts),
        max_new_tokens=1024,
        max_length=1024 + 256,
        cuda_graph=True,
        do_sample=True
    )
    for res in response:
        logger.info(tokenizer.decode(res))

    logger.info(time.time() - start)

    start = time.time()
    response = patched_model.generate(
        input_ids,
        stop_token_ids=[], #[151645, 151643],
        max_batch_size=len(prompts),
        max_prefill_batch=len(prompts),
        max_new_tokens=512,
        max_length=1024 + 256,
        cuda_graph=True,
        do_sample=True
    )
    logger.info(time.time() - start)


    # start = time.time()
    # patched_model.eval()
    
    from torch.profiler import ProfilerActivity, profile, record_function
    with profile(
            record_shapes=True,
            profile_memory=True,
            activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA
            ]) as prof:
        response = patched_model.generate(
            input_ids,
            stop_token_ids=[], #[151645, 151643],
            max_batch_size=len(prompts),
            max_prefill_batch=len(prompts),
            max_new_tokens=3,
            max_length=1024 + 256,
            cuda_graph=True,
            do_sample=False
        )


    prof.export_chrome_trace(f'xpuyu_tp4_{torch.distributed.get_rank()}.json')
    # for res in response:
    #     logger.info(tokenizer.decode(res))

    # logger.info(time.time() - start)

