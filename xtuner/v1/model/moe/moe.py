# Copyright (c) OpenMMLab. All rights reserved.
import types
from itertools import accumulate
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
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
from xtuner.v1.loss import BalancingLoss, CELossContext, ZLoss
from xtuner.v1.model.base import BaseModel
from xtuner.v1.model.utils import checkpoint_wrapper
from xtuner.v1.module import LMHead, RMSNorm, RotaryEmbedding
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEBlock, MoEDecoderLayer
from xtuner.v1.module.router import NoAuxRouter, NoAuxRouterConfig
from xtuner.v1.utils import (
    get_device,
    get_logger,
)
from xtuner.v1.utils.compile import maybe_compile


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
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)

        self.fp32_layers = [self.rotary_emb]

        # TODO(@yehaochen): 把这两行移除 _maybe_compile_layers 要把 compile 相关的 setting 放到 fsdp_config 之外
        # _init_load_spec 放到 post init 里
        self._init_load_spec()
        self._maybe_compile_layers()

        self.balancing_loss: BalancingLoss | None
        self.z_loss: ZLoss | None
        if self.config.balancing_loss_cfg is not None:
            self.balancing_loss = self.config.balancing_loss_cfg.build(self.config.router.scoring_func)
        else:
            self.balancing_loss = None
        if self.config.z_loss_cfg is not None:
            self.z_loss = self.config.z_loss_cfg.build()
        else:
            self.z_loss = None

    def _select_non_pad_router_logits(
        self,
        router_logits_list: list[list[torch.Tensor]] | list[torch.Tensor],
        attn_mask_list: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        assert len(router_logits_list) > 0, "router_logits_list should not be empty"
        if isinstance(router_logits_list[0], torch.Tensor):
            router_logits_list = [cast(list[torch.Tensor], router_logits_list)]  # intra_layer_micro_batch is 1
            attn_mask_list = [cast(torch.Tensor, attn_mask_list)]
        # router_logits_list [intra_layer_micro_batch, num_layers][seq, num_experts]
        # attn_mask_list [intra_layer_micro_batch, ][1, seq]
        intra_layer_micro_batch = len(router_logits_list)
        num_layers = len(router_logits_list[0])

        router_logits_list_new = []  # [num_layers, intra_layer_micro_batch] -> [num_layers * intra_layer_micro_batch]
        for layer_idx in range(num_layers):
            for micro_batch_idx in range(intra_layer_micro_batch):
                router_logits_list_new.append(router_logits_list[micro_batch_idx][layer_idx])

        router_logits = torch.stack(
            router_logits_list_new, dim=0
        )  # [num_layers * intra_layer_micro_batch, seq, num_experts]
        router_logits = router_logits.view(
            num_layers, -1, router_logits.shape[-1]
        )  # [num_layers, intra_layer_micro_batch * seq, num_experts]
        attn_mask = torch.stack(attn_mask_list, dim=0)  # type: ignore  # [intra_layer_micro_batch, 1, seq]
        attn_mask = attn_mask.flatten()
        router_logits = router_logits[:, attn_mask].contiguous().float()  # [num_layers, non_pad_seq, num_experts]
        return router_logits

    @torch.no_grad()
    def _cal_tokens_per_expert(self, router_logits: torch.Tensor):
        scoring_func = self.config.router.scoring_func
        n_routed_experts = self.config.n_routed_experts
        num_experts_per_tok = self.config.num_experts_per_tok
        num_layers = router_logits.shape[0]
        router_logits = router_logits.float()  # (nlayers, seq, ne)
        if scoring_func == "softmax":
            routing_weights = F.softmax(router_logits, dim=-1)
        elif scoring_func == "sigmoid":
            routing_weights = router_logits / torch.sum(router_logits, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown scoring function: {scoring_func}")
        _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
        selected_experts_flat = selected_experts.view(num_layers, -1)
        offset = torch.arange(num_layers, device=router_logits.device).unsqueeze(1) * n_routed_experts
        selected_experts_offset = selected_experts_flat + offset
        tokens_per_expert_flat = torch.histc(
            selected_experts_offset.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)  # (nlayers, ne)
        if dist.is_initialized():
            tokens_per_expert_global_for_bias = all_reduce(tokens_per_expert, "sum", dist.group.WORLD)  # type: ignore
        else:
            tokens_per_expert_global_for_bias = tokens_per_expert
        return tokens_per_expert_global_for_bias

    @torch.no_grad()
    def update_bias(self, total_expert_counts_pre_iter, expected_loads):
        """Implementation for the following paper:
        Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
        https://arxiv.org/abs/2408.15664

        TODO: refactor it later.
        """
        first_k_dense_replace = self.config.first_k_dense_replace
        bias_update_speed = cast(NoAuxRouterConfig, self.config.router).router_bias_update_speed
        n_layer, _ = total_expert_counts_pre_iter.size()

        for i_layer in range(n_layer):
            # 前 l 层是 mlp 层，跳过
            gate = cast(MoEDecoderLayer, self.layers[first_k_dense_replace + i_layer]).gate
            e_score_correction_bias = cast(NoAuxRouter, gate).e_score_correction_bias
            expected_load = expected_loads[i_layer]
            current_loads = total_expert_counts_pre_iter[i_layer]

            load_diff = current_loads - expected_load
            update_mask = load_diff != 0  # 只更新需要调整的专家
            updates = torch.where(load_diff > 0, -bias_update_speed, bias_update_speed) * update_mask.float()

            e_score_correction_bias.add_(updates)

    def forward(
        self,
        seq_ctx: list[SequenceContext] | SequenceContext,
        loss_ctx: list[CELossContext] | CELossContext | None,
    ):
        # TODO: caoweihan: Recover this assertion after the refactor of LossContext
        if isinstance(seq_ctx, SequenceContext):
            # assert isinstance(loss_ctx, (CELossContext, LossContext)) or loss_ctx is None, (
            #     f"If seq_ctx_list is a single SequenceContext, loss_ctx_list must be a single CELossContext or None, but got {type(loss_ctx)}"
            # )
            # NOTE: @caoweihan, this type ignore should be remove after the refactor of LossContext
            return self._forward(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,  # type: ignore
            )
        else:
            assert isinstance(loss_ctx, list) and len(loss_ctx) == len(seq_ctx), (
                "seq_ctx_list and loss_ctx_list must be lists of the same length"
            )
            if loss_ctx is None:
                raise NotImplementedError("loss_ctx must be provided for intra-layer bsz > 1")
            return self._micro_batch_forward(
                seq_ctx_list=seq_ctx,
                loss_ctx_list=loss_ctx,
            )

    def _micro_batch_forward(
        self,
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[CELossContext],
    ) -> MoEModelOutputs:
        """Micro-batch forward pass for MoE model.

        This method processes multiple micro-batches in parallel, similar to how MoEDecoderLayer handles micro-batching
        at the layer level.
        """
        if self.config.return_hidden_states:
            raise NotImplementedError

        assert len(seq_ctx_list) == len(loss_ctx_list), "seq_ctx and loss_ctx must have same length"

        # Prepare input embeddings for all micro-batches
        hidden_states_list: list[torch.Tensor] = []
        position_embeddings_list = []

        for ctx in seq_ctx_list:
            input_ids = ctx.input_ids
            position_ids = ctx.position_ids

            if input_ids is not None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = ctx.inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            hidden_states_list.append(hidden_states)
            position_embeddings_list.append(position_embeddings)

        # Initialize output containers
        output: dict = {}

        router_logits_list: list[dict[str, torch.Tensor]] = [{} for _ in range(len(seq_ctx_list))]

        # Process through layers
        cat_seq_ctx: SequenceContext | None = None
        cat_position_embeddings: torch.Tensor | None = None
        cat_hidden_states: torch.Tensor | None = None

        for idx, decoder_layer in self.layers.items():
            layer_idx = int(idx)

            if layer_idx < self.config.first_k_dense_replace:
                if cat_seq_ctx is None:
                    cat_seq_ctx = self._cat_seq_ctx(seq_ctx_list)
                    cat_position_embeddings = torch.cat(position_embeddings_list, dim=1)
                    cat_hidden_states = torch.cat(hidden_states_list, dim=1)
                # Dense decoder layer - process concated hidden states
                cat_hidden_states = decoder_layer(
                    cat_hidden_states,
                    position_embeddings=cat_position_embeddings,
                    seq_ctx=cat_seq_ctx,
                )
            else:
                if cat_hidden_states is not None:
                    hidden_states_list = cat_hidden_states.split(len(seq_ctx_list), dim=1)

                layer_results = decoder_layer(
                    *hidden_states_list,
                    position_embeddings=position_embeddings_list,
                    seq_ctx=seq_ctx_list,
                )
                hidden_states = layer_results[: len(hidden_states_list)]
                router_logits = layer_results[len(hidden_states_list) :]

                # Update hidden states and collect router results
                for i, hidden_states in enumerate(hidden_states):
                    hidden_states_list[i] = hidden_states
                    router_logits_list[i][f"layer{idx}"] = router_logits[i]

        # Apply final norm to all micro-batches
        for i, hidden_states in enumerate(hidden_states_list):
            hidden_states_list[i] = self.norm(hidden_states)

        # Process final outputs for each micro-batch
        loss_list: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []

        for hidden_states, loss_ctx_single in zip(hidden_states_list, loss_ctx_list):
            loss, logits = self.lm_head(hidden_states, loss_ctx_single)  # type: ignore
            loss_list.append(loss)
            if logits is not None:
                logits_list.append(logits)

        # Aggregate losses (mean across micro-batches)
        output["loss"] = torch.stack(loss_list).sum() if loss_list else None

        # Handle router results for all micro-batches
        all_router_logits = []

        for micro_batch_idx, micro_batch_router_logits in enumerate(router_logits_list):
            if micro_batch_router_logits:
                _router_logits_list = list(micro_batch_router_logits.values())
                attn_mask = seq_ctx_list[micro_batch_idx].mask
                router_logits = self._select_non_pad_router_logits(_router_logits_list, attn_mask)
                all_router_logits.append(router_logits)

        if all_router_logits:
            # Concatenate router logits from all micro-batches
            combined_router_logits = torch.cat(all_router_logits, dim=1)  # [num_layers, total_seq, num_experts]

            # Calculate balancing loss across all micro-batches
            if self.balancing_loss:
                balancing_loss = self.balancing_loss(
                    router_logits=combined_router_logits,
                    n_routed_experts=self.config.n_routed_experts,
                    num_experts_per_tok=self.config.num_experts_per_tok,
                )
                output["balancing_loss"] = balancing_loss

            # Calculate z-loss across all micro-batches
            if self.z_loss:
                z_loss = self.z_loss(router_logits=combined_router_logits)
                output["z_loss"] = z_loss

            # Calculate tokens per expert for bias update (if applicable)
            if isinstance(self.config.router, NoAuxRouterConfig) and self.config.router.router_bias_update_speed > 0:
                tokens_per_expert_global = self._cal_tokens_per_expert(combined_router_logits)
                output["tokens_per_expert_global"] = tokens_per_expert_global

            del combined_router_logits

        # Return logits for all micro-batches
        if all(logits is not None for logits in logits_list):
            final_logits = torch.cat(logits_list, dim=0) if logits_list else None
        else:
            final_logits = None

        if self.config.return_router_results:
            raise NotImplementedError

            # TODO: Return router logits is costy

            # router_logits_dict: dict[str, torch.Tensor] = {}
            # layer_names = list(router_logits_list[0].keys())
            #
            # for layer_name in layer_names:
            #     layer_router_logits_list: list[torch.Tensor] = []
            #     for micro_batch_idx in range(len(seq_ctx_list)):
            #         layer_router_logits_list.append(router_logits_list[micro_batch_idx][layer_name].clone().detach())
            #     router_logits = torch.stack(layer_router_logits_list, dim=0).unsqueeze(0)
            #     router_logits_dict["router_logits"] = router_logits
            #
            # output["router_logits"] = router_logits_dict

        return MoEModelOutputs(**output, logits=final_logits)  # type: ignore[typeddict-item]

    def _forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: CELossContext | None,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}  # type: ignore
        if self.config.return_hidden_states:
            output["hidden_states"] = []

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
                output["router_logits"][f"layer{idx}"] = router_results

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        loss, logits = self.lm_head(hidden_states, loss_ctx)  # type: ignore
        output["loss"] = loss
        output["logits"] = logits

        router_logits_list = list(output["router_logits"].values())  # type: ignore
        router_logits = self._select_non_pad_router_logits(router_logits_list, seq_ctx.mask)

        if self.balancing_loss:
            balancing_loss = self.balancing_loss(
                router_logits=router_logits,
                n_routed_experts=self.config.n_routed_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
            )
            output["balancing_loss"] = balancing_loss

        if self.z_loss:
            z_loss = self.z_loss(router_logits=router_logits)
            output["z_loss"] = z_loss

        if isinstance(self.config.router, NoAuxRouterConfig) and self.config.router.router_bias_update_speed > 0:
            tokens_per_expert_global = self._cal_tokens_per_expert(router_logits)
            output["tokens_per_expert_global"] = tokens_per_expert_global

        del router_logits

        if self.config.return_router_results:
            raise NotImplementedError
            # TODO: Move router logits to CPU is cost
            # for layer_name, router_logits in output["router_logits"].items():
            #     output["router_logits"][layer_name] = router_logits.detach().cpu().unsqueeze(0)
        else:
            output["router_logits"] = None

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

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn)
        self.rotary_emb.to(torch.float32)
        return self

    @override
    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict)
        self.rotary_emb = self.build_rotary_embedding(self.config)
        return loaded_keys, unloaded_keys, missing_keys

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ) -> "MoE":
        self.fsdp_config = fsdp_config
        assert self.fsdp_config.ep_size == self.config.ep_size
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        self._init_device_mesh(fsdp_config)
        self._maybe_compile_layers()

        # TODO: 一定不能少，因为在模型 init 时候会构建一套 ep_mesh，如果不重新构建，fsdp_mesh 和 ep_mesh 会没有任何联系
        # fully_shard 时候会出现： AssertionError: FSDP requires the DP and TP mesh to have the same parent mesh
        with torch.device("meta"):
            self.layers = self.build_layers(self.config)

        if float8_handler is not None:
            # As we modify the shape of the model's parameters,
            # we need to reinitialize the load spec mapping.
            float8_handler.pad_for_fsdp(
                self, cast(DeviceMesh, self.fsdp_mesh), callback_after_pad=self._init_load_spec
            )

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

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        num_recompute_layers = int(self.config.num_hidden_layers * self.fsdp_config.recompute_ratio)

        for layer_idx, layer in tqdm(self.layers.items(), desc="[FSDP Sharding]"):
            layer_idx = int(layer_idx)
            if layer_idx < num_recompute_layers - 1:
                layer = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.REENTRANT)

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
            mp_policy=mp_policy,
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

        device = DEVICE
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
                torch._dynamo.config.cache_size_limit = 128
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

    def _cat_seq_ctx(self, seq_ctx_list: list[SequenceContext]) -> SequenceContext:
        """Concatenate multiple SequenceContext objects into one."""
        if len(seq_ctx_list) == 1:
            return seq_ctx_list[0]

        input_ids = torch.cat([ctx.input_ids for ctx in seq_ctx_list], dim=0)
        position_ids = torch.cat([cast(torch.Tensor, ctx.position_ids) for ctx in seq_ctx_list], dim=0)

        cu_seq_len_offset = [0] + [seq_ctx.cu_seq_lens_q[-1] for seq_ctx in seq_ctx_list[:-1]]
        cu_offsets = torch.tensor(list(accumulate(cu_seq_len_offset)))
        shifted_offsets = [seq_ctx.cu_seq_lens_q + offset for offset, seq_ctx in zip(cu_offsets, seq_ctx_list)]
        cat_seq_lens_q = torch.cat(
            [ctx.cu_seq_lens_q + offset for ctx, offset in zip(seq_ctx_list, shifted_offsets)], dim=0
        )

        return SequenceContext(
            input_ids=input_ids,  # type: ignore[arg-type]
            position_ids=position_ids,  # type: ignore[arg-type]
            cu_seq_lens_q=cat_seq_lens_q,  # type: ignore[arg-type]
            cu_seq_lens_k=cat_seq_lens_q,  # type: ignore[arg-type]
            max_length_q=(cat_seq_lens_q[1:] - cat_seq_lens_q[:-1]).max().item(),  # type: ignore[arg-type]
            max_length_k=(cat_seq_lens_q[1:] - cat_seq_lens_q[:-1]).max().item(),  # type: ignore[arg-type]
        )

    # NOTE: Add this overload for inferring the return type for easier type checking and using
    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: SequenceContext,
        loss_ctx: CELossContext | None,
    ) -> MoEModelOutputs: ...

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: list[SequenceContext],
        loss_ctx: list[CELossContext],
    ) -> MoEModelOutputs: ...

    __call__ = nn.Module.__call__
