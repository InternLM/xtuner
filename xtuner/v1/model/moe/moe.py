# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, cast, overload

import torch
from mmengine import is_installed
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial
from torch.nn import functional as F

from transformers.activations import ACT2FN
from xtuner.v1.config.base_model import MoEConfig, MoEModelOutputs
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import RMSNorm, RotaryEmbedding, RouterResults, build_attnention, build_router
from xtuner.v1.module.dispatcher import DecodingDispatchResult, PrefillingDispatchResult, get_dispatcher
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.utils import ForwardState, HFCheckpointLoader, get_logger


logger = get_logger()


# TODO: (yehaochen) Maybe could be optimized
class _Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer."""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        return F.linear(input, w, b)


class MoEMLP(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.gate_proj = _Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = _Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = _Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEBlock(nn.Module):
    def __init__(self, config: MoEConfig, ep_mesh: DeviceMesh | None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_routed_experts = config.n_routed_experts

        self.ep_mesh = ep_mesh
        # self.fused_w1 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        # self.fused_w3 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)

        self.fused_w1w3 = GroupedLinear(self.hidden_size, 2 * self.intermediate_size, self.num_routed_experts, ep_mesh)
        self.fused_w2 = GroupedLinear(self.intermediate_size, self.hidden_size, self.num_routed_experts, self.ep_mesh)

    def forward(self, x, tokens_per_expert, decoding):
        gate_up_out = self.fused_w1w3(x, tokens_per_expert, decoding)
        gate_out, up_out = gate_up_out.chunk(2, dim=-1)
        # up_out = self.fused_w1(x, tokens_per_expert, decoding)
        # gate_out = self.fused_w3(x, tokens_per_expert, decoding)
        gate_out = F.silu(gate_out)
        out = gate_out * up_out

        res = self.fused_w2(out, tokens_per_expert, decoding)
        return res


class MoEGate(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        self.router = build_router(config)

    # @torch.compile(fullgraph=True)
    def forward(self, hidden_states) -> RouterResults:
        _, _, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local(grad_placements=(Partial("avg"),))
        else:
            weight = self.weight
        logits = F.linear(hidden_states.float(), weight.float(), None)

        return self.router(logits)


class MoELayer(nn.Module):
    """InternLM3MoELayer is a mixed expert layer with shared experts."""

    def __init__(self, config: MoEConfig, ep_mesh: DeviceMesh | None):
        super().__init__()
        self.ep_mesh = ep_mesh
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.experts = MoEBlock(config, ep_mesh)
        self.dispatcher = get_dispatcher(config.dispatcher)(
            config=config,
            process_group=ep_mesh.get_group() if ep_mesh is not None else None,
        )
        self.gate = MoEGate(config)

        if config.n_shared_experts > 0:
            self.shared_experts = MoEMLP(config)

    def forward(self, hidden_states: torch.Tensor):
        raise NotImplementedError("MoE layer will not be directly called since `async` and `sync` will be more  ")


class MoEDecoderLayer(nn.Module):
    """MoE decoder layer."""

    def __init__(self, config: MoEConfig, layer_idx: int, ep_mesh: DeviceMesh | None):
        super().__init__()
        self.ep_mesh = ep_mesh
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = build_attnention(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.shared_experts = MoEMLP(config) if config.n_shared_experts > 0 else None

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gate = MoEGate(config)
        self.experts = MoEBlock(config, ep_mesh)
        # TODO: (yehaochen) Maybe should be replaced by build_dispatcher
        process_group = ep_mesh.get_group() if ep_mesh is not None else None
        self.dispatcher = get_dispatcher(config.dispatcher)(config=config, process_group=process_group)

    # TODO: decouple the training and decoding interface. since arg like past_key_value will never be used in training
    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[list[torch.Tensor]] | None = None,
        state: ForwardState = ForwardState.TRAINING,
    ) -> Tuple[torch.Tensor, RouterResults]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            past_key_values=past_key_values,
            state=ForwardState.TRAINING,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_results: RouterResults = self.gate(hidden_states)

        origin_shape = hidden_states.shape

        # reshape hidden_states to (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=router_results["topk_ids"],
            topk_weights=router_results["topk_weights"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            decoding=state == ForwardState.DECODING,
        )  # type: ignore[call-overload]
        experts_out: torch.Tensor = self.experts(
            dispatched["hidden_states"],
            dispatched["tokens_per_experts"],
            decoding=False,
        )

        if state == ForwardState.DECODING:
            dispatched = cast(DecodingDispatchResult, dispatched)
            combined = self.dispatcher.combine(
                hidden_states=experts_out,
                pre_dispatched=pre_dispatched,
                dispatch_result=dispatched,
                decoding=True,
            )
        else:
            dispatched = cast(PrefillingDispatchResult, dispatched)
            combined = self.dispatcher.combine(
                hidden_states=experts_out,
                pre_dispatched=pre_dispatched,
                dispatch_result=dispatched,
                decoding=False,
            )
        hidden_states = self.dispatcher.combine_post_process(
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            combine_result=combined,
        )
        hidden_states = hidden_states.view(*origin_shape)

        if self.config.n_shared_experts > 0:
            assert self.shared_experts is not None, "Shared experts should be initialized when n_shared_experts > 0"
            shared_experts_out = self.shared_experts(hidden_states)
            hidden_states += shared_experts_out

        hidden_states = residual + hidden_states * self.config.hidden_factor
        return hidden_states, router_results


class DenseMLP(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = _Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = _Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DenseDecoderLayer(nn.Module):
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = build_attnention(config, layer_idx)
        self.mlp = DenseMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # TODO: decouple the training and decoding interface. since arg like past_key_value will never be used in training
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: Optional[List[List[torch.Tensor]]] = None,
        state: ForwardState = ForwardState.TRAINING,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            past_key_values=past_key_values,
            state=ForwardState.TRAINING,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoE(nn.Module):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`InternLM3DecoderLayer`]

    Args:
        config: MoEModelConfig
    """

    def __init__(self, config: MoEConfig, ep_mesh: DeviceMesh | None = None):
        super().__init__()
        self.ep_mesh = ep_mesh
        self.config = config
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = _Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)

        self.fp32_layers = [self.rotary_emb]

        self.chunked_loss = config.chunked_loss
        if self.chunked_loss:
            assert is_installed("liger_kernel"), "Liger kernel is required for chunked loss."

    def forward(
        self,
        seq_ctx: SequenceContext,
        labels: torch.LongTensor,
        return_router_results: bool = False,
        return_hidden_states: bool = False,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        hidden_states = self.embed_tokens(input_ids)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output = {}  # type: ignore
        if return_hidden_states:
            output["hidden_states"] = []

        if return_router_results:
            output["router_logits"] = {}

        for idx, decoder_layer in self.layers.items():
            if isinstance(decoder_layer, MoEDecoderLayer):
                hidden_states, router_results = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                )
                if return_router_results:
                    output["router_logits"][f"layer{idx}"] = router_results
            elif isinstance(decoder_layer, DenseDecoderLayer):
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                )
            else:
                raise ValueError(f"Unsupported decoder layer type: {type(decoder_layer)}")

            if return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        logits: torch.Tensor | None = None
        loss: torch.Tensor

        if self.chunked_loss:
            shift_hidden_states = hidden_states.view(-1, self.config.hidden_size)
            shift_labels = labels.view(-1)

            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            if isinstance(self.lm_head.weight, DTensor):
                _weight = self.lm_head.weight.to_local(grad_placements=(Partial(),))
            else:
                _weight = self.lm_head.weight

            if isinstance(self.lm_head.bias, DTensor):
                _bias = self.lm_head.bias.to_local(grad_placements=(Partial(),))
            else:
                _bias = self.lm_head.bias

            loss_fct = LigerFusedLinearCrossEntropyLoss()
            loss = loss_fct(_weight, shift_hidden_states, shift_labels, _bias)
        else:
            logits = cast(torch.Tensor, self.lm_head(hidden_states))

            shift_logits = logits.view(-1, self.config.vocab_size)
            shift_labels = labels.view(-1)

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        output["loss"] = loss

        return MoEModelOutputs(**output)  # type: ignore[typeddict-item]

    def get_hf_key(self, key: str) -> str:
        raise NotImplementedError

    def trainable_parameters(self):
        params = [param for param in self.parameters() if param.requires_grad]
        return params

    def build_embeddings(self, config: MoEConfig):
        return nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)

    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        # 让 layers 是一个 nn.ModuleDict 方便做 pipeline parallel 的参数切分，
        # 这样可以保证部分 layer 被切掉后，idx 保持不变
        layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = DenseDecoderLayer(config, layer_idx)
            else:
                layers[str(layer_idx)] = MoEDecoderLayer(config, layer_idx, self.ep_mesh)
        return layers

    def build_rotary_embedding(self, config: MoEConfig) -> RotaryEmbedding:
        return RotaryEmbedding(config=config)

    # NOTE: Add this overload for inferring the return type for easier type checking and using
    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: SequenceContext,
        labels: torch.LongTensor,
        return_router_results: bool = False,
        return_hidden_states: bool = False,
    ) -> MoEModelOutputs: ...

    __call__ = nn.Module.__call__

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn)
        self.rotary_emb.to(torch.float32)
        return self

    def to_hf_key_list(self, key: str) -> str | List[str]:
        raise NotImplementedError()

    def from_hf(self, hf_path: str, device: torch.device | None = None, strict=True):
        hf_loader = HFCheckpointLoader(hf_path)

        if device is None:
            # TODO: NPU support (need `get_available_device`)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ep_rank = self.ep_mesh.get_local_rank() if self.ep_mesh is not None else 0
        ep_size = self.ep_mesh.size() if self.ep_mesh is not None else 1

        cur_device = next(iter(self.parameters())).device
        if cur_device == torch.device("meta"):
            self.to_empty(device=device)
            self.rotary_emb = self.build_rotary_embedding(self.config).to(device)

        not_matched = []
        not_loaded = []
        loaded = []

        with torch.no_grad():
            for name, value in self.state_dict().items():
                if isinstance(value, DTensor):
                    value = value.to_local()
                hf_keys = self.to_hf_key_list(name)
                if isinstance(hf_keys, list):
                    n_experts_per_rank = len(hf_keys) // ep_size

                    hf_values = []
                    start_idx = ep_rank * n_experts_per_rank
                    end_idx = start_idx + n_experts_per_rank
                    for idx in range(start_idx, end_idx):
                        hf_key = hf_keys[idx]
                        _value = hf_loader.load(hf_key).to(device)
                        if _value is None:
                            not_loaded.append(f"{name}")
                            logger.warning(f"Parameter {f'{name}'} -> {hf_key} not found in HF checkpoint.")
                            break
                        hf_values.append(_value)
                    hf_value = torch.cat(hf_values, dim=0)

                    if hf_value.shape != value.shape:
                        not_matched.append(f"{f'{name}'} {hf_value.shape} != {value.shape}")
                        logger.warning(
                            f"Parameter {f'{name}'} shape mismatch: expected {value.shape}, got {hf_value.shape}."
                        )
                        continue
                    value.copy_(hf_value)
                    loaded.extend(hf_keys)
                else:
                    hf_value = hf_loader.load(hf_keys)
                    if hf_value is None:
                        not_loaded.append(f"{name}")
                        logger.warning(f"Parameter {f'{name}'} -> {hf_keys} not found in HF checkpoint.")
                        continue

                    if hf_value.shape != value.shape:
                        not_matched.append(
                            f"Parameter {f'{name}'} -> {hf_keys}: {f'{name}'} {hf_value.shape} != {value.shape}"
                        )
                        logger.warning(
                            f"Parameter {f'{name}'} shape mismatch: expected {value.shape}, got {hf_value.shape}."
                        )
                    value.copy_(hf_value)
                    loaded.append(hf_keys)

        missing = set(hf_loader.weight_map) - set(loaded)

        if strict:
            if not_matched:
                raise RuntimeError(f"Some parameters from {hf_path} do not match the model: {not_matched}. ")
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {missing}. ")
