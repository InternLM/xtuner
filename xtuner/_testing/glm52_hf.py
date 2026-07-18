from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def apply_glm52_hf_numeric_oracle_patch() -> None:
    """Backport the HF GLM-MoE-DSA oracle path used by tests.

    The test environment may run an older transformers release. This patch keeps
    the monkeypatch local to tests while matching HF PR #46842 for the GLM DSA
    indexer: interleaved RoPE, ReLU-scored top-k, and shared top-k layers.
    """

    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf_glm

    if getattr(hf_glm.GlmMoeDsaAttention, "_xtuner_glm52_oracle_patched", False):
        return

    def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
        sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)

        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_embed, k_embed

    def is_shared_layer(config, layer_idx: int) -> bool:
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None and layer_idx < len(indexer_types):
            return indexer_types[layer_idx] == "shared"

        freq = getattr(config, "index_topk_freq", 1) or 1
        offset = getattr(config, "index_skip_topk_offset", 0) or 0
        return freq > 1 and (max(layer_idx + 1 - offset, 0) % freq) != 0

    def source_layer(config, layer_idx: int) -> int:
        for idx in range(layer_idx, -1, -1):
            if not is_shared_layer(config, idx):
                return idx
        raise ValueError(f"GLM-5.2 shared DSA layer {layer_idx} has no preceding full indexer layer.")

    def patched_indexer_forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

        if past_key_values is not None and hasattr(past_key_values, "update_indexer"):
            k = past_key_values.update_indexer(k, self.layer_idx)
        elif use_cache:
            if seq_len > 1:
                self._cached_keys = None
            k = torch.cat([self._cached_keys, k], dim=1) if getattr(self, "_cached_keys", None) is not None else k
            self._cached_keys = k

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
            causal = key_positions[None, None, :] > position_ids[:, :, None]
            index_scores = index_scores.masked_fill(causal, float("-inf"))

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)

    original_attention_init = hf_glm.GlmMoeDsaAttention.__init__

    def patched_attention_init(self, config, layer_idx: int):
        original_attention_init(self, config, layer_idx)
        self.skip_topk = is_shared_layer(config, layer_idx)
        if self.skip_topk and hasattr(self, "indexer"):
            del self.indexer
        if not hasattr(self, "indexer"):
            self.indexer = None

    def patched_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        kv_expanded = self.kv_b_proj(k_pass)
        kv_expanded = kv_expanded.view(
            batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_pass, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pass = k_pass.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)

        query_states = torch.cat([q_pass, q_rot], dim=-1)
        key_states = torch.cat([k_pass, k_rot], dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        holder = getattr(self.config, "_xtuner_glm52_topk_holder", None)
        if holder is None:
            holder = {}
            self.config._xtuner_glm52_topk_holder = holder

        if self.indexer is None:
            src = source_layer(self.config, self.layer_idx)
            if src not in holder:
                raise ValueError(f"Shared GLM-5.2 DSA layer {self.layer_idx} requires layer {src} top-k indices.")
            topk_indices = holder[src]
        else:
            indexer_mask = (
                attention_mask[:, 0, :, :] if attention_mask is not None and attention_mask.dim() == 4 else None
            )
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            holder[self.layer_idx] = topk_indices

        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            torch.finfo(query_states.dtype).min,
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices.long(), 0.0)
        index_mask = index_mask.unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            combined_mask = index_mask + attention_mask[..., :total_len]
        else:
            combined_mask = index_mask

        attention_interface = hf_glm.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf_glm.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=None,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    hf_glm.GlmMoeDsaIndexer.forward = patched_indexer_forward
    hf_glm.GlmMoeDsaAttention.__init__ = patched_attention_init
    hf_glm.GlmMoeDsaAttention.forward = patched_attention_forward
    hf_glm.GlmMoeDsaAttention._xtuner_glm52_oracle_patched = True


def load_glm52_hf_oracle_model(
    model_path: str | Path,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load GLM-5.2 HF model with test-only PR #46842 oracle semantics."""

    from transformers.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    apply_glm52_hf_numeric_oracle_patch()

    config = GlmMoeDsaConfig.from_pretrained(model_path)
    config.head_dim = config.qk_rope_head_dim
    config._attn_implementation = "eager"

    model = GlmMoeDsaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype)
    model.to(device)
    model.eval()

    def clear_topk_holder(_module, _inputs):
        config._xtuner_glm52_topk_holder = {}

    model.register_forward_pre_hook(clear_topk_holder)
    return model
