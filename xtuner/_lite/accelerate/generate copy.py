from xtuner._lite.chat import ChatMessages
import copy
import torch
from mmengine import MessageHub
from xtuner._lite import get_logger
import math
import sys
from .packed import packed_cumulative_length, pack_sequence
logger = get_logger()



def sample(logits, top_k=40, top_p=0, temperature=1.0):
    return logits.argmax(-1)
    # Apply temperature if necessary
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k if necessary
    if top_k > 0:
        top_k = min(top_k, logits.size(-1)) 
        _, indices = logits.topk(top_k)
        mask = torch.ones_like(logits, dtype=torch.bool) 
        mask.scatter_(-1, indices, False)
        logits.masked_fill_(mask, -torch.inf)

    # # Apply top-p (nucleus sampling) if necessary
    # if top_p > 0.0:
    #     cum_probs = logits.softmax(dim=-1).cumsum(dim=-1)
    #     mask = cum_probs < top_p
    #     mask = torch.cat([torch.ones_like(mask[..., :1]), mask[..., :-1]], dim=-1)
    #     logits.masked_fill_(mask, -torch.inf)
    
    probs = logits.softmax(-1)
    # breakpoint()
    return torch.multinomial(probs, 1).squeeze(-1)

@torch.no_grad()
def _zero_padding_generate(model, 
                          input_ids,
                          past_key_values,
                          block_table,
                          stop_token_ids = [],
                          max_new_tokens = 128,
                          max_length=2048,
                          device='cuda'):

    
    
    _packed_ids, _num_tokens = pack_sequence(input_ids)
    _position_ids = [torch.arange(seq.numel()) for seq in input_ids]
    _packed_pos_ids = torch.cat(_position_ids, dim=0).unsqueeze(0)
    _cumulative_length = packed_cumulative_length(_num_tokens)

    next_input_ids = _packed_ids.to(device)
    next_position_ids = _packed_pos_ids.to(device)
    next_start_pos = _cumulative_length[:-1].to(device)
    next_end_pos = (_cumulative_length[1:] - 1).to(device)
    next_query_length = _num_tokens.to(device)
    next_cache_length = _num_tokens.to(device)
    next_block_table = block_table.to(device).to(torch.int32)
    next_cumulative_length = _cumulative_length.to(device)
    next_is_prefilling = True

    num_sessions = len(input_ids)
    computing = [i for i in range(num_sessions)]
    
    responses = [[] for _ in range(num_sessions)]

    while len(computing):

        attn_ctx = MessageHub.get_instance('paged_attention')
        attn_ctx.update_info('block_offsets', next_block_table)
        attn_ctx.update_info('kv_seq_length', next_cache_length)
        attn_ctx.update_info('q_seq_length', next_query_length)
        attn_ctx.update_info('max_kv_seq_length', next_cache_length.max())
        attn_ctx.update_info('max_q_seq_length', next_query_length.max())
        attn_ctx.update_info('q_start_loc', next_start_pos)
        attn_ctx.update_info('cumulative_length', next_cumulative_length)
        attn_ctx.update_info('is_prefilling', next_is_prefilling)
        
        logger.info('Begin Prefilling')

        outputs = model(
            input_ids=next_input_ids,
            position_ids=next_position_ids,
            past_key_values=past_key_values,
            cache_position=next_position_ids,
        )
    
        for key in list(attn_ctx.runtime_info.keys()):
            attn_ctx.pop_info(key)
        
        # TODO (pppppM) support sampling
        sampled = sample(outputs.logits[0, next_end_pos])
        
        _next_input_ids = []
        _next_position_ids = []
        _next_computing = []
        _next_query_length = []
        _next_cache_length = []
        _next_block_table = []

        for i, sess_id in enumerate(computing):
            token_id = sampled[i]
            responses[sess_id].append(token_id.item())

            _sess_new_tokens = len(responses[sess_id])
            _sess_len = _sess_new_tokens + input_ids[sess_id].numel()

            stop = (_sess_new_tokens >= max_new_tokens or 
                    _sess_len >= max_length or
                    token_id in stop_token_ids)
            
            if not stop:
                # next step is decoding
                _next_computing.append(sess_id)
                _next_block_table.append(next_block_table[i])
                _next_input_ids.append(token_id.reshape(1,-1))
                _next_position_ids.append(torch.arange(_sess_len - 1, _sess_len))
                _next_query_length.append(1)
                _next_cache_length.append(_sess_len)

        computing = _next_computing
        if len(computing) == 0:
            # All sessions have ended.
            break

        _packed_ids, _num_tokens = pack_sequence(_next_input_ids)
        _cumulative_length = packed_cumulative_length(_num_tokens)

        next_input_ids = _packed_ids.to(device)
        next_position_ids = torch.cat(_next_position_ids, dim=0).unsqueeze(0)
        next_position_ids = next_position_ids.to(device)
        next_start_pos = _cumulative_length[:-1].to(device)
        next_end_pos = (_cumulative_length[1:] - 1).to(device)
        next_query_length = torch.IntTensor(_next_query_length).to(device)
        next_cache_length = torch.IntTensor(_next_cache_length).to(device)
        next_block_table = torch.stack(_next_block_table).to(device)

        next_cumulative_length = _cumulative_length.to(device)
        next_is_prefilling = False

    return [torch.IntTensor(res).to(device) for res in responses]

@torch.no_grad()
def contiguous_batching_generate(model, 
                                input_ids,
                                stop_token_ids = [],
                                max_batch_size = 64,
                                max_new_tokens = 128,
                                max_length=2048,
                                tp_size=1,
                                device='cuda'): 
    
    model.config.use_cache = True

    from lmdeploy.pytorch.engine.cache_engine import CacheEngine
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    
    block_size = 256
    max_batch_size = min(max_batch_size, len(input_ids))
    num_blocks = max_length // block_size * max_batch_size
    cache_config = CacheConfig(max_batch_size, block_size, num_blocks, num_blocks)
    model_config = ModelConfig.from_hf_config(model.config)
    cache_engine = CacheEngine(cache_config, model_config, world_size=tp_size)
    block_table = torch.arange(num_blocks).reshape(max_batch_size, -1)

    num_sessions = len(input_ids)
    responses = []
    for i in range(0, num_sessions, max_batch_size):
        batch_input_ids = input_ids[i: i+max_batch_size]
        _responses = _zero_padding_generate(
            model, batch_input_ids, cache_engine.gpu_cache,
            block_table, stop_token_ids, max_new_tokens, 
            max_length, device
        )
        responses.extend(_responses)
    
    for i in range(len(cache_engine.gpu_cache)):
        cache_engine.gpu_cache.pop()

    del cache_engine
    torch.cuda.empty_cache()
    
    model.config.use_cache = False

    return responses