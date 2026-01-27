import re
from pathlib import Path
from typing import Literal

import torch
from pydantic import computed_field
from typing_extensions import Self

from transformers import PretrainedConfig
from transformers.models.gpt_oss import GptOssConfig as HFGptOssConfig
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig

from .moe import MoE


class GptOss(MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)
        if "gate" in key:
            key = key.replace("gate", "router")

        if "fused_w1w3.weight" in key:
            key = key.replace("fused_w1w3.weight", "gate_up_proj")
        elif "fused_w2.weight" in key:
            key = key.replace("fused_w2.weight", "down_proj")
        if "fused_w1w3.bias" in key:
            key = key.replace("fused_w1w3.bias", "gate_up_proj_bias")
        elif "fused_w2.bias" in key:
            key = key.replace("fused_w2.bias", "down_proj_bias")

        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        else:
            return [key]

    def safetensors_to_params(
        self,
        safetensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        param_name: str,
        start: int | None,
        end: int | None,
        dim: int | None,
    ):
        if len(safetensors) > 1:
            assert dim is not None, "Internal Error dim must not be None when len(safetensors) > 1"
            loaded_tensor = torch.cat(safetensors, dim=dim)
        else:
            loaded_tensor = safetensors[0]

        if "fused_w1w3.weight" in param_name:
            # hf: num_experts, hidden_size, expert_dim * 2
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            num_experts, hidden_size = loaded_tensor.shape[:2]
            loaded_tensor = loaded_tensor.transpose(1, 2)  # num_experts, expert_dim * 2, hidden_size
            loaded_tensor = loaded_tensor.reshape(num_experts, -1, 2, hidden_size)
            # # num_experts *2 * expert_dim, hidden_size
            loaded_tensor = loaded_tensor.transpose(1, 2).reshape(-1, hidden_size)

        elif "fused_w2.weight" in param_name:
            # hf: num_experts, expert_dim, hidden_size
            # xtuner: num_experts * hidden_size, expert_dim
            loaded_tensor = loaded_tensor.transpose(1, 2).flatten(0, 1)

        if "fused_w1w3.bias" in param_name:
            # hf: num_experts, expert_dim * 2
            # xtuner: num_experts, 2 * expert_dim
            num_experts = loaded_tensor.size(0)
            loaded_tensor = loaded_tensor.reshape(num_experts, -1, 2)
            loaded_tensor = loaded_tensor.transpose(1, 2).reshape(num_experts, -1)

        if start is not None and end is not None:
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            loaded_tensor_slice = loaded_tensor.index_select(
                dim=self.FSDP_SHARD_DIM, index=torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            )
            non_pad_len = end - start
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)

            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        else:
            local_tensor.copy_(loaded_tensor)

    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        assert isinstance(hf_param_name, str)
        if "gate_up_proj" in hf_param_name and "gate_up_proj_bias" not in hf_param_name:
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            # hf: num_experts, hidden_size, expert_dim * 2
            num_experts = self.config.n_routed_experts
            hidden_size = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, 2, -1, hidden_size)  # num_experts, 2, expert_dim, hidden_size
            safetensor = safetensor.permute(0, 3, 2, 1).reshape(num_experts, hidden_size, -1)
        elif "down_proj" in hf_param_name and "down_proj_bias" not in hf_param_name:
            # xtuner: num_experts * hidden_size, expert_dim
            # hf: num_experts, expert_dim, hidden_size
            num_experts = self.config.n_routed_experts
            expert_dim = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, -1, expert_dim).transpose(1, 2).contiguous()
        elif "gate_up_proj_bias" in hf_param_name:
            # xtuner: num_experts, 2 * expert_dim
            # hf: num_experts, expert_dim * 2
            num_experts = self.config.n_routed_experts
            safetensor = safetensor.reshape(num_experts, 2, -1).transpose(1, 2).reshape(num_experts, -1)
        return safetensor


class GptOssConfig(MoEConfig):
    gate_bias: bool = True
    moe_bias: bool = True
    tie_word_embeddings: bool = False
    n_shared_experts: int = 0
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(act_type="clipped_swiglu", clip_alpha=1.702, clip_limit=7)

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention"]]:
        return ["sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)]

    def build(self) -> GptOss:
        return GptOss(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path | None = None, hf_config: PretrainedConfig | None = None) -> Self:
        if hf_path is not None:
            cfg = HFGptOssConfig.from_pretrained(hf_path)
            assert isinstance(cfg, HFGptOssConfig)
        else:
            cfg = hf_config
        assert hf_config is not None and isinstance(cfg, PretrainedConfig)

        config = cls(
            hf_config=cfg,
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=cfg.pad_token_id,
            eos_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            moe_intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            attention=MHAConfig(
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                rms_norm_eps=cfg.rms_norm_eps,
                sliding_window=cfg.sliding_window,
                with_sink=True,
                qkv_bias=True,
                o_bias=True,
            ),
            n_routed_experts=cfg.num_local_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            tie_word_embeddings=cfg.tie_word_embeddings,
            router=GreedyRouterConfig(
                scoring_func="softmax",
                norm_topk_prob=True,
                router_scaling_factor=1.0,
            ),
        )

        return config

    @property
    def hf_config(self) -> HFGptOssConfig:
        assert isinstance(self.router, GreedyRouterConfig), "Only support saving GreedyRouter to HF GptOss format."

        return HFGptOssConfig(
            architectures=["GptOssForCausalLM"],
            layer_types=self.layers_type,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_key_value_heads,
            head_dim=self.attention.head_dim,
            sliding_window=self.attention.sliding_window,
            tie_word_embeddings=self.tie_word_embeddings,
            num_local_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.router.norm_topk_prob,
            qkv_bias=True,
            o_bias=True,
            dtype=torch.bfloat16,
            swiglu_limit=self.moe_act_fn_cfg.clip_limit,
        )


class GptOss21BA3P6Config(GptOssConfig):
    vocab_size: int = 201088
    max_position_embeddings: int = 131072
    pad_token_id: int = 199999
    eos_token_id: int = 200002
    num_hidden_layers: int = 24
    hidden_size: int = 2880
    intermediate_size: int = 2880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        with_sink=True,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        sliding_window=128,
        qkv_bias=True,
        o_bias=True,
        rms_norm_eps=1e-5,
    )
    n_routed_experts: int = 32
    num_experts_per_tok: int = 4
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 2880
    router: GreedyRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()


class GptOss117BA5P8Config(GptOssConfig):
    vocab_size: int = 201088
    max_position_embeddings: int = 131072
    pad_token_id: int = 199999
    eos_token_id: int = 200002
    num_hidden_layers: int = 36
    hidden_size: int = 2880
    intermediate_size: int = 2880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        with_sink=True,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        sliding_window=128,
        qkv_bias=True,
        o_bias=True,
        rms_norm_eps=1e-5,
    )
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 2880
    router: GreedyRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
