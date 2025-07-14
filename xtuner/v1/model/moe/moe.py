# Copyright (c) OpenMMLab. All rights reserved.
from typing import cast, overload

import torch
from mmengine import is_installed
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from xtuner.v1.config.base_model import MoEConfig, MoEModelOutputs
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import RMSNorm, RotaryEmbedding
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer
from xtuner.v1.module.linear.linear import _Linear
from xtuner.v1.utils import HFCheckpointLoader, get_logger


logger = get_logger()


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
            if int(idx) < self.config.first_k_dense_replace:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                )
            else:
                hidden_states, router_results = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                )
                if return_router_results:
                    output["router_logits"][f"layer{idx}"] = router_results

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
                _weight = self.lm_head.weight.to_local()
            else:
                _weight = self.lm_head.weight

            if isinstance(self.lm_head.bias, DTensor):
                _bias = self.lm_head.bias.to_local()
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

    def to_hf_key_list(self, key: str) -> str | list[str]:
        raise NotImplementedError()

    def from_hf(self, hf_path: str, prefix: str = "", device: torch.device | None = None, strict=True):
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
                        hf_key = prefix + hf_keys[idx]
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
                    loaded.extend([prefix + hf_key for hf_key in hf_keys])
                else:
                    hf_keys = prefix + hf_keys
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
