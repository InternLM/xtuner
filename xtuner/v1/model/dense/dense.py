# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from typing_extensions import overload, override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.base import (
    DEFAULT_FLOAT8_CFG,
    BaseModel,
    ModelOutputs,
    TorchCompileOption,
    TransformerConfig,
)
from xtuner.v1.model.utils import checkpoint_wrapper
from xtuner.v1.module import LMHead, RMSNorm, RotaryEmbeddingProtocol, get_rope_embedding
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.utils import (
    get_device,
    get_logger,
)


DEVICE = get_device()
logger = get_logger()


DENSE_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


class Dense(BaseModel):
    config: TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)

        # Make sure it works properly when not using fsdp
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # TODO(@yehaochen): 把这两行移除 _maybe_compile_layers 要把 compile 相关的 setting 放到 fsdp_config 之外
        # _init_load_spec 放到 post init 里
        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)

    def forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: CELossContext,
    ) -> ModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds

        # create position embeddings to be shared across the decoder layers
        assert position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        for idx, decoder_layer in self.layers.items():
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings,
                seq_ctx,
            )

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx)
        output["loss"] = loss
        output["logits"] = logits
        output["extra_info"] = extra_info
        return ModelOutputs(**output)  # type: ignore[typeddict-item]

    def build_embeddings(self, config: TransformerConfig):
        return nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def build_layers(self, config: TransformerConfig) -> nn.ModuleDict:
        # 让 layers 是一个 nn.ModuleDict 方便做 pipeline parallel 的参数切分，
        # 这样可以保证部分 layer 被切掉后，idx 保持不变
        layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            layers[str(layer_idx)] = DenseDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mlp_bias=config.mlp_bias,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                attention_config=config.attention,
                generate_config=config.generate_config,
                rope_scaling_cfg=config.rope_scaling_cfg,
                float8_cfg=config.float8_cfg,
                layer_type=config.layers_type[layer_idx] if config.layers_type is not None else None,
                layer_idx=layer_idx,
            )
        return layers

    def build_rotary_embedding(self, config: TransformerConfig) -> RotaryEmbeddingProtocol:
        with torch.device(DEVICE):
            return get_rope_embedding(config=config)

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return DENSE_COMPILE_CFG

    # NOTE: Add this overload for inferring the return type for easier type checking and using
    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: SequenceContext,
        loss_ctx: CELossContext,
    ) -> ModelOutputs: ...

    __call__ = nn.Module.__call__

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn)
        self.rotary_emb.to(torch.float32)  # type: ignore
        return self

    @override
    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict)
        # If model is built on meta device, we need to rebuild rotary embedding since from_hf will not
        # load the `inv_freq` of RotaryEmbedding which is a inpersisitent buffer.
        # xTODO: remove this line below when with torch.device(DEVICE) in __init__()
        # self.rotary_emb = self.build_rotary_embedding(self.config)
        return loaded_keys, unloaded_keys, missing_keys

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ) -> "Dense":
        self.fsdp_config = fsdp_config
        self._init_device_mesh(fsdp_config)

        if float8_handler is not None:
            # As we modify the shape of the model's parameters,
            # we need to reinitialize the load spec mapping.
            float8_handler.pad_for_fsdp(
                self, cast(DeviceMesh, self.fsdp_mesh), callback_after_pad=self._init_load_spec
            )

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        if not checkpoint_preserve_rng_state and self.config.attention.dropout > 0.0:
            checkpoint_preserve_rng_state = True
            logger.warning("When using dropout, checkpoint_preserve_rng_state is set to True to avoid issues.")

        # Just for narrowing the type of self.fsdp_mesh and self.ep_mesh
        assert self.fsdp_mesh is not None
        assert self.fsdp_config is not None

        if self.fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        # xTODO: remove this line below when with torch.device(DEVICE) in __init__()
        # self.rotary_emb = self.build_rotary_embedding(self.config)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        if self.fsdp_config.lm_head_fp32:
            lm_head_mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        else:
            lm_head_mp_policy = mp_policy
        num_recompute_layers = int(self.config.num_hidden_layers * self.fsdp_config.recompute_ratio)

        generator = torch.Generator()
        generator.manual_seed(dist.get_rank())
        shuffled_layers_idxs = torch.randperm(len(self.layers), generator=generator)

        for layer_idx in tqdm(shuffled_layers_idxs, desc="[FSDP Sharding]"):
            layer = self.layers[str(int(layer_idx))]
            layer_idx = int(layer_idx)
            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(
                    layer, preserve_rng_state=checkpoint_preserve_rng_state, checkpoint_impl=CheckpointImpl.REENTRANT
                )
                # __class__ without self attribute

                if self.compile_cfg:
                    layer.forward = torch.compile(layer.forward, fullgraph=True)

            self.layers[str(layer_idx)] = layer
            fully_shard(
                layer,
                mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
            )

        for layer_cur, layer_next in zip(
            list(self.layers.values())[:-1],
            list(self.layers.values())[1:],
        ):
            layer_cur.set_modules_to_forward_prefetch([layer_next])  # type: ignore

        fully_shard(
            self.embed_tokens,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )

        fully_shard(
            self.norm,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )

        fully_shard(
            self.lm_head,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=lm_head_mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )

        fully_shard(
            self,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )
        self.set_modules_to_forward_prefetch([self.embed_tokens, self.layers["0"]])  # type: ignore

        self._to_empty_meta()

        # Make sure it works properly when using fsdp
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        return self

    # TODO: 支持 tp
    def _init_device_mesh(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config

        device = DEVICE
        world_size = dist.get_world_size()

        if self.fsdp_config.hsdp_sharding_size is None:
            model_mesh = init_device_mesh(
                device,
                (world_size, 1),
                mesh_dim_names=(f"{self.config.mesh_prefix}.fsdp", f"{self.config.mesh_prefix}.others"),
            )
            self.fsdp_mesh = model_mesh[f"{self.config.mesh_prefix}.fsdp"]
        else:
            self.hsdp_mesh = init_device_mesh(
                device,
                (
                    world_size // self.fsdp_config.hsdp_sharding_size,
                    self.fsdp_config.hsdp_sharding_size,
                ),
                mesh_dim_names=(
                    f"{self.config.mesh_prefix}.hsdp_replicate",
                    f"{self.config.mesh_prefix}.hsdp_shard",
                ),
            )
            self.fsdp_mesh = self.hsdp_mesh[f"{self.config.mesh_prefix}.hsdp_shard"]

    # TODO: Remove patch before opensource
    @staticmethod
    def patched_rms_norm_forward(self, input):
        if hasattr(self, "weight"):
            if isinstance(self.weight, DTensor):
                w = self.weight.to_local()
            else:
                w = self.weight
        else:
            if isinstance(self.norm.weight, DTensor):
                w = self.norm.weight.to_local()
            else:
                w = self.norm.weight
        return F.rms_norm(input, w.shape, w, self.variance_epsilon)

    @staticmethod
    def patched_emb_forward(self, input):
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
        else:
            w = self.weight
        return F.embedding(
            input,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
