# Copyright (c) OpenMMLab. All rights reserved.
import torch

from xtuner._lite import get_logger
from xtuner._lite.patches.utils import pack_sequence, packed_cumulative_length

logger = get_logger()


class GenerateMixin:
    @torch.no_grad()
    def build_kv_cache(
        self,
        max_batch_size,
        max_length,
        block_size=256,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        num_blocks = max(
            max_batch_size, (max_length + block_size - 1) // block_size * max_batch_size
        )
        head_dim = self.model_config.head_dim
        num_heads = self.model_config.num_key_value_heads
        past_key_values = []
        for _ in range(self.model_config.num_hidden_layers):
            cache_k = torch.zeros(
                num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device
            )
            cache_v = torch.zeros(
                num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device
            )

            past_key_values.append((cache_k, cache_v))

        block_table = torch.arange(num_blocks).reshape(max_batch_size, -1)
        return past_key_values, block_table

    @torch.no_grad()
    def prefilling(
        self,
        input_ids,
        position_ids,
        past_key_values,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
    ):
        outputs = self(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        return outputs.logits

    @torch.no_grad()
    def init_cuda_graph(
        self,
        input_ids,
        position_ids,
        past_key_values,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        block_table,
    ):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        self.graph_block_table = block_table
        self.graph_cu_seq_lens_q = cu_seq_lens_q
        self.graph_cu_seq_lens_k = cu_seq_lens_k
        self.graph_max_length_q = max_length_q
        self.graph_max_length_k = max_length_k
        self.graph_input_ids = input_ids
        self.graph_position_ids = position_ids
        self.graph_cache_position = position_ids.clone()

        # 在新 stream 中预热
        with torch.cuda.stream(s):
            with torch.no_grad():
                self.graph_logits = self(
                    input_ids=self.graph_input_ids,
                    position_ids=self.graph_position_ids,
                    past_key_values=past_key_values,
                    cache_position=self.graph_cache_position,
                    cu_seq_lens_q=self.graph_cu_seq_lens_q,
                    cu_seq_lens_k=self.graph_cu_seq_lens_k,
                    max_length_q=self.graph_max_length_q,
                    max_length_k=self.graph_max_length_k,
                    block_table=self.graph_block_table,
                    prefilling=False,
                ).logits

        # 等待预热完成
        torch.cuda.current_stream().wait_stream(s)
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            with torch.no_grad():
                self.graph_logits = self(
                    input_ids=self.graph_input_ids,
                    position_ids=self.graph_position_ids,
                    past_key_values=past_key_values,
                    cache_position=self.graph_cache_position,
                    cu_seq_lens_q=self.graph_cu_seq_lens_q,
                    cu_seq_lens_k=self.graph_cu_seq_lens_k,
                    max_length_q=self.graph_max_length_q,
                    max_length_k=self.graph_max_length_k,
                    block_table=self.graph_block_table,
                    prefilling=False,
                ).logits

    @torch.no_grad()
    def cuda_graph_decoding(
        self,
        input_ids,
        position_ids,
        past_key_values,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        block_table,
    ):
        if self.cuda_graph is None:
            self.init_cuda_graph(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                block_table=block_table,
            )

        self.graph_block_table.copy_(block_table)
        self.graph_cu_seq_lens_q.copy_(cu_seq_lens_q)
        self.graph_cu_seq_lens_k.copy_(cu_seq_lens_k)
        self.graph_max_length_q.copy_(max_length_q)
        self.graph_max_length_k.copy_(max_length_k)
        self.graph_input_ids.copy_(input_ids)
        self.graph_position_ids.copy_(position_ids)
        self.graph_cache_position.copy_(position_ids)
        self.cuda_graph.replay()

        return self.graph_logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        stop_token_ids=[],
        max_batch_size=64,
        max_new_tokens=128,
        max_prefill_batch=16,
        max_length=2048,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        cuda_graph=False,
        vocab_size=None,
    ):
        assert max_batch_size % max_prefill_batch == 0
        self.patched_model.config.use_cache = True

        past_key_values, block_table = self.build_kv_cache(
            max_batch_size, max_length, block_size=256, device=self.device_type
        )

        next_input_ids = []
        next_position_ids = []
        next_cu_seq_lens_q = []
        next_cu_seq_lens_k = []
        next_max_length_q = []
        next_max_length_k = []
        next_block_table = []

        for start in range(0, max_batch_size, max_prefill_batch):
            _packed_ids, _num_tokens = pack_sequence(
                input_ids[start : start + max_prefill_batch]
            )
            _position_ids = [
                torch.arange(seq.numel())
                for seq in input_ids[start : start + max_prefill_batch]
            ]
            _packed_pos_ids = torch.cat(_position_ids, dim=0).unsqueeze(0)
            _cumulative_length = packed_cumulative_length(_num_tokens)

            next_input_ids.append(_packed_ids.to(self.device_type))
            next_position_ids.append(_packed_pos_ids.to(self.device_type))
            next_cu_seq_lens_q.append(_cumulative_length.to(self.device_type))
            next_cu_seq_lens_k.append(_cumulative_length.to(self.device_type))

            next_max_length_q.append(_num_tokens.max().item())
            next_max_length_k.append(_num_tokens.max().item())

            next_block_table.append(
                block_table[start : start + max_prefill_batch]
                .to(self.device_type)
                .to(torch.int32)
            )

        next_is_prefilling = True

        num_sessions = len(input_ids)
        stopped = []
        responses = [[] for _ in range(num_sessions)]

        self.cuda_graph = None
        self.compiled_model = None
        while True:
            all_rank_stopped = torch.IntTensor([len(stopped) >= num_sessions]).to(
                self.device_type
            )
            torch.distributed.all_reduce(
                all_rank_stopped, torch.distributed.ReduceOp.MIN
            )
            if all_rank_stopped:
                break

            if next_is_prefilling:
                if isinstance(next_input_ids, list):
                    sampled = []
                    seq_lens_q = []
                    seq_lens_k = []
                    for (
                        chunk_input_ids,
                        chunk_pos_ids,
                        chunk_cu_seq_lens_q,
                        chunk_max_length_q,
                        chunk_cu_seq_lens_k,
                        chunk_max_length_k,
                        chunk_block_table,
                    ) in zip(
                        next_input_ids,
                        next_position_ids,
                        next_cu_seq_lens_q,
                        next_max_length_q,
                        next_cu_seq_lens_k,
                        next_max_length_k,
                        next_block_table,
                        strict=True,
                    ):
                        chunk_outputs = self(
                            input_ids=chunk_input_ids,
                            position_ids=chunk_pos_ids,
                            past_key_values=past_key_values,
                            cache_position=chunk_pos_ids,
                            cu_seq_lens_q=chunk_cu_seq_lens_q,
                            cu_seq_lens_k=chunk_cu_seq_lens_k,
                            max_length_q=chunk_max_length_q,
                            max_length_k=chunk_max_length_k,
                            block_table=chunk_block_table,
                            prefilling=next_is_prefilling,
                        )
                        chunk_logits = chunk_outputs.logits

                        chunk_sampled = self.sample(
                            chunk_logits[0],
                            cu_seq_lens=chunk_cu_seq_lens_q,
                            do_sample=do_sample,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                        )

                        chunk_seq_lens_q = (
                            chunk_cu_seq_lens_q[1:] - chunk_cu_seq_lens_q[:-1]
                        )
                        chunk_seq_lens_k = (
                            chunk_cu_seq_lens_k[1:] - chunk_cu_seq_lens_k[:-1]
                        )

                        sampled.append(chunk_sampled)
                        seq_lens_q.append(chunk_seq_lens_q)
                        seq_lens_k.append(chunk_seq_lens_k)

                    sampled = torch.cat(sampled)
                    next_input_ids = torch.cat(next_input_ids, dim=1)
                    next_position_ids = torch.cat(next_position_ids, dim=1)
                    next_block_table = torch.cat(next_block_table, dim=0)

                    seq_lens_q = torch.cat(seq_lens_q)
                    seq_lens_k = torch.cat(seq_lens_k)

                    next_cu_seq_lens_q = packed_cumulative_length(seq_lens_q)
                    next_cu_seq_lens_k = packed_cumulative_length(seq_lens_k)
                    next_max_length_q = seq_lens_q.max()
                    next_max_length_k = seq_lens_k.max()

            else:
                if cuda_graph:
                    logits = self.cuda_graph_decoding(
                        input_ids=next_input_ids,
                        position_ids=next_position_ids,
                        past_key_values=past_key_values,
                        cu_seq_lens_q=next_cu_seq_lens_q,
                        cu_seq_lens_k=next_cu_seq_lens_k,
                        max_length_q=next_max_length_q,
                        max_length_k=next_max_length_k,
                        block_table=next_block_table,
                    )
                else:
                    outputs = self(
                        input_ids=next_input_ids,
                        position_ids=next_position_ids,
                        past_key_values=past_key_values,
                        cache_position=next_position_ids,
                        cu_seq_lens_q=next_cu_seq_lens_q,
                        cu_seq_lens_k=next_cu_seq_lens_k,
                        max_length_q=next_max_length_q,
                        max_length_k=next_max_length_k,
                        block_table=next_block_table,
                        prefilling=next_is_prefilling,
                    )
                    logits = outputs.logits

                sampled = self.sample(
                    logits[0],
                    cu_seq_lens=next_cu_seq_lens_q,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )

            _next_input_ids = []
            _next_position_ids = []
            _next_seq_lens_q = []
            _next_seq_lens_k = []
            _next_block_table = []

            for sess_id in range(num_sessions):
                if sess_id not in stopped:
                    token_id = sampled[sess_id]
                    responses[sess_id].append(token_id)
                else:
                    token_id = responses[sess_id][-1]

                _sess_new_tokens = len(responses[sess_id])
                _sess_len = _sess_new_tokens + input_ids[sess_id].numel()

                stop = (
                    _sess_new_tokens >= max_new_tokens
                    or _sess_len >= max_length
                    or token_id in stop_token_ids
                )

                if stop and sess_id not in stopped:
                    stopped.append(sess_id)

                _next_block_table.append(next_block_table[sess_id])
                _next_input_ids.append(token_id.reshape(1, -1))
                _next_position_ids.append(torch.arange(_sess_len - 1, _sess_len))
                _next_seq_lens_q.append(1)
                _next_seq_lens_k.append(_sess_len)

            _packed_ids, _num_tokens = pack_sequence(_next_input_ids)
            _cumulative_length = packed_cumulative_length(_num_tokens)

            next_input_ids = _packed_ids.to(self.device_type)
            next_position_ids = torch.cat(_next_position_ids, dim=0).unsqueeze(0)
            next_position_ids = next_position_ids.to(self.device_type)

            _next_seq_lens_q = torch.IntTensor([0] + _next_seq_lens_q).to(
                self.device_type
            )
            _next_seq_lens_k = torch.IntTensor([0] + _next_seq_lens_k).to(
                self.device_type
            )

            next_max_length_q = _next_seq_lens_q.max()
            next_max_length_k = _next_seq_lens_k.max()

            next_cu_seq_lens_q = torch.cumsum(_next_seq_lens_q, dim=0).int()
            next_cu_seq_lens_k = torch.cumsum(_next_seq_lens_k, dim=0).int()

            next_block_table = torch.stack(_next_block_table).to(self.device_type)

            next_is_prefilling = False

        self.patched_model.config.use_cache = False

        del past_key_values
        self.cuda_graph = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return [torch.stack(res).tolist() for res in responses]
