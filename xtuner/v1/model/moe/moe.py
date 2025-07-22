# Copyright (c) OpenMMLab. All rights reserved.
import types
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import is_installed
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from tqdm import tqdm
from typing_extensions import overload, override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.config.base_model import MoEConfig, MoEModelOutputs
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.model import BaseModel
from xtuner.v1.module import RMSNorm, RotaryEmbedding
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEBlock, MoEDecoderLayer
from xtuner.v1.module.linear.linear import _Linear
from xtuner.v1.utils import (
    get_device,
    get_logger,
    get_torch_device_module,
)
from xtuner.v1.utils.compile import maybe_compile


DEVICE_MODULE = get_torch_device_module()
DEVICE = get_device()
logger = get_logger()


class MoE(BaseModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`InternLM3DecoderLayer`]

    Args:
        config: MoEModelConfig
    """

    config: MoEConfig
    ep_mesh: DeviceMesh | None = None

    def __init__(self, config: MoEConfig):
        super().__init__()
        if config.ep_size is not None and config.ep_size > 1:
            world_size = dist.get_world_size()
            self.ep_mesh = init_device_mesh(
                DEVICE,
                (world_size // config.ep_size, config.ep_size),
                mesh_dim_names=("dp", "ep"),
            )["ep"]
        else:
            self.ep_mesh = None
        self.config = config

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = _Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)

        self.fp32_layers = [self.rotary_emb]

        self.chunked_loss = config.chunked_loss
        if self.chunked_loss:
            assert is_installed("liger_kernel"), "Liger kernel is required for chunked loss."
        self.load_spec_mapping = self._init_load_spec()
        self._maybe_compile_layers()

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

    def build_embeddings(self, config: MoEConfig):
        return nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)

    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        # 让 layers 是一个 nn.ModuleDict 方便做 pipeline parallel 的参数切分，
        # 这样可以保证部分 layer 被切掉后，idx 保持不变
        layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = DenseDecoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_config=config.attention,
                    generate_config=config.generate_config,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                )
            else:
                layers[str(layer_idx)] = MoEDecoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    moe_intermediate_size=config.moe_intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    num_experts_per_tok=config.num_experts_per_tok,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    hidden_factor=config.hidden_factor,
                    attention_config=config.attention,
                    generate_config=config.generate_config,
                    router_config=config.router,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    dispatcher=config.dispatcher,
                    ep_mesh=self.ep_mesh,
                )
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

    @override
    def from_hf(self, hf_path: str | Path, prefix="", strict: bool = True):
        super().from_hf(hf_path, prefix, strict)
        self.rotary_emb = self.build_rotary_embedding(self.config)

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        assert self.fsdp_config.ep_size == self.config.ep_size
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        device = "cpu" if self.fsdp_config.cpu_offload else str(DEVICE)
        self._init_device_mesh(fsdp_config)

        with torch.device("meta"):
            self.layers = self.build_layers(self.config)

        if float8_handler is not None:
            float8_handler.pad_for_fsdp(self, cast(DeviceMesh, self.fsdp_mesh))

        # Since ep size could be changed by `fsdp_config`, we need to re-initialize the load spec
        self.load_spec_mapping = self._init_load_spec()

        # Just for narrowing the type of self.fsdp_mesh and self.ep_mesh
        assert self.fsdp_mesh is not None
        assert self.ep_mesh is not None
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

        if self.ep_mesh.size() > 1:
            self._replicate_other_params(self)

        self.rotary_emb = self.build_rotary_embedding(self.config)

        self._maybe_compile_layers()
        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        num_recompute_layers = int(self.config.num_hidden_layers * self.fsdp_config.recompute_ratio)

        for layer_idx, layer in tqdm(self.layers.items(), desc="[FSDP Sharding]"):
            layer_idx = int(layer_idx)
            layer.to_empty(device=DEVICE_MODULE.current_device())
            if layer_idx < num_recompute_layers - 1:
                layer = ptd_checkpoint_wrapper(
                    layer,
                    preserve_rng_state=False,
                    checkpoint_impl=CheckpointImpl.REENTRANT,
                )

            self.layers[str(layer_idx)] = layer
            if layer_idx >= len(self.layers) - 1:
                reshard_after_forward = False
            else:
                reshard_after_forward = self.fsdp_config.reshard_after_forward
            fully_shard(
                layer,
                mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
            )
            layer.to_empty(device=device)

        for layer_cur, layer_next in zip(
            list(self.layers.values())[:-1],
            list(self.layers.values())[1:],
        ):
            layer_cur.set_modules_to_forward_prefetch([layer_next])  # type: ignore

        self.embed_tokens.to_empty(device=DEVICE_MODULE.current_device())
        fully_shard(
            self.embed_tokens,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )
        self.embed_tokens.to_empty(device=device)

        self.norm.to_empty(device=DEVICE_MODULE.current_device())
        fully_shard(
            self.norm,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )
        self.norm.to_empty(device=device)

        self.lm_head.to_empty(device=DEVICE_MODULE.current_device())
        fully_shard(
            self.lm_head,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )
        self.lm_head.to_empty(device=device)

        fully_shard(
            self,
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )
        self.set_modules_to_forward_prefetch([self.embed_tokens, self.layers["0"]])  # type: ignore

        for _, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                module.forward = types.MethodType(self.patched_emb_forward, module)  # type: ignore
            elif isinstance(module, RMSNorm):
                module.forward = types.MethodType(self.patched_rms_norm_forward, module)  # type: ignore
        return self

    @torch.no_grad  # type: ignore
    def scale_and_reduce_grad(self):
        for name, param in self.trainable_parameters():
            if param.grad is None:
                continue

            ep_enabled = self.ep_mesh is not None and self.ep_mesh.size() > 1
            # Scale moe parameters
            if ep_enabled and ".experts" in name:
                param.grad.div_(self.ep_mesh.size())  # type: ignore
                continue

            # Reduce gradients for other parameters
            if ep_enabled:
                grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                dist.all_reduce(
                    grad.div_(self.ep_mesh.size()),  # type: ignore
                    ReduceOp.SUM,
                    group=self.ep_mesh.get_group(mesh_dim=0),  # type: ignore
                )

    def _init_device_mesh(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config

        device = DEVICE if not fsdp_config.cpu_offload else "cpu"
        world_size = dist.get_world_size()
        experts_fsdp_size = world_size // self.fsdp_config.ep_size

        if self.fsdp_config.hsdp_sharding_size is None:
            model_mesh = init_device_mesh(
                device,
                (experts_fsdp_size, self.fsdp_config.ep_size),
                mesh_dim_names=(f"{self.fsdp_config.mesh_prefix}.fsdp", f"{self.fsdp_config.mesh_prefix}.ep"),
            )
            if self.ep_mesh is not None:
                assert torch.equal(self.ep_mesh.mesh, model_mesh[f"{self.fsdp_config.mesh_prefix}.ep"].mesh), (
                    "FSDP enabled, it requires the `ep_size` of model config equals to the `ep_size` of FSDPConfig."
                )
            self.ep_mesh = model_mesh[f"{self.fsdp_config.mesh_prefix}.ep"]
            self.fsdp_mesh = model_mesh[f"{self.fsdp_config.mesh_prefix}.fsdp"]
        else:
            assert self.fsdp_config.ep_size == 1, "Currently, HSDP requires expert parallel size to be 1"
            # We can not init ep_mesh and fsdp_mesh like this.
            # This will lead to "RuntimeError: Cannot create a submesh from a submesh."
            # in FSDPParam.shard_mesh, as fsdp_mesh is not the root mesh. The root mesh is model_mesh.
            # So we have to init the ep_mesh and fsdp_mesh separately.
            # model_mesh = init_device_mesh(
            #     device,
            #     (
            #         experts_fsdp_size // self.fsdp_config.hsdp_sharding_size,
            #         self.fsdp_config.hsdp_sharding_size,
            #         self.fsdp_config.ep_size,
            #     ),
            #     mesh_dim_names=(
            #         f"{self.fsdp_config.mesh_prefix}.hsdp_replicate",
            #         f"{self.fsdp_config.mesh_prefix}.hsdp_shard",
            #         f"{self.fsdp_config.mesh_prefix}.ep",
            #     ),
            # )
            # self.ep_mesh = model_mesh[f"{self.fsdp_config.mesh_prefix}.ep"]
            # self.fsdp_mesh = model_mesh[
            #     (f"{self.fsdp_config.mesh_prefix}.hsdp_replicate", f"{self.fsdp_config.mesh_prefix}.hsdp_shard")
            # ]
            ep_mesh = init_device_mesh(
                device, (world_size, 1), mesh_dim_names=("_", f"{self.fsdp_config.mesh_prefix}.ep")
            )[f"{self.fsdp_config.mesh_prefix}.ep"]
            if self.ep_mesh is not None:
                assert self.ep_mesh == ep_mesh, "ep_mesh should be the same as the previous one"
            self.ep_mesh = ep_mesh
            self.hsdp_mesh = init_device_mesh(
                device,
                (
                    experts_fsdp_size // self.fsdp_config.hsdp_sharding_size,
                    self.fsdp_config.hsdp_sharding_size,
                ),
                mesh_dim_names=(
                    f"{self.fsdp_config.mesh_prefix}.hsdp_replicate",
                    f"{self.fsdp_config.mesh_prefix}.hsdp_shard",
                ),
            )
            self.fsdp_mesh = self.hsdp_mesh[f"{self.fsdp_config.mesh_prefix}.hsdp_shard"]

    def _replicate_other_params(self, model: nn.Module):
        def traverse(module):
            if isinstance(module, MoEBlock):
                return
            for name, param in module.named_parameters(recurse=False):
                dist_param = nn.Parameter(distribute_tensor(param, self.ep_mesh, [Replicate()]))
                module.register_parameter(name, dist_param)
            for child in module.children():
                traverse(child)

        traverse(model)

    def _maybe_compile_layers(self):
        if self.fsdp_config is not None:
            if self.fsdp_config.torch_compile:
                if self.fsdp_config.compile_targets is None:
                    if self.ep_mesh.size() > 1:
                        # all_to_all_single_autograd in TorchAll2AllDispatcher.dispatch can not be compiled even if the fullgraph=False
                        # ref: https://github.com/pytorch/pytorch/issues/155205
                        # todo: decorate MoEDecoderLayer.forward with @torch.compile(fullgraph=False) when the bug is fixed
                        # so that we do not need to remove the compile target
                        maybe_compile.remove_compile_target(
                            "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward"
                        )
                else:
                    maybe_compile.clear_compile_targets()
                    for target in self.fsdp_config.compile_targets:
                        maybe_compile.set_compile_target(target)
            else:
                maybe_compile.clear_compile_targets()
        else:
            if self.ep_mesh is not None and self.ep_mesh.size() > 1:
                # all_to_all_single_autograd in TorchAll2AllDispatcher.dispatch can not be compiled even if the fullgraph=False
                # ref: https://github.com/pytorch/pytorch/issues/155205
                # todo: decorate MoEDecoderLayer.forward with @torch.compile(fullgraph=False) when the bug is fixed
                # so that we do not need to remove the compile target
                maybe_compile.remove_compile_target(
                    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward"
                )

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
