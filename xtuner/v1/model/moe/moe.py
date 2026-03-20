# Copyright (c) OpenMMLab. All rights reserved.
import os
import types
from pathlib import Path
from typing import Annotated, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
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
from typing_extensions import NotRequired, overload, override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import BalancingLoss, CELossContext, ZLoss
from xtuner.v1.model.base import (
    DEFAULT_FLOAT8_CFG,
    BaseModel,
    ModelOutputs,
    TorchCompileOption,
    TransformerConfig,
)
from xtuner.v1.model.utils import ModelForwardExtraLogInfo, checkpoint_wrapper, module_dict_repr
from xtuner.v1.module import (
    GreedyRouterConfig,
    LMHead,
    NoAuxRouter,
    NoAuxRouterConfig,
    RMSNorm,
    RotaryEmbeddingProtocol,
    get_rope_embedding,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock, MoEDecoderLayer
from xtuner.v1.utils import (
    get_device,
    get_logger,
)
from xtuner.v1.utils.activation_offload import async_save_on_cpu


DEVICE = get_device()
logger = get_logger()


MOE_NON_EP_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._pre_moe_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._shared_experts_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._post_moe_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}

MOE_EP_COMPILE_CFG = MOE_NON_EP_COMPILE_CFG.copy()
MOE_EP_COMPILE_CFG.pop("xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward")


class MoEModelOutputs(ModelOutputs):
    router_logits: NotRequired[dict[str, torch.Tensor]]
    balancing_loss: NotRequired[torch.Tensor]
    z_loss: NotRequired[torch.Tensor]
    tokens_per_expert_global: NotRequired[torch.Tensor]


class BalancingLossConfig(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")
    balancing_loss_alpha: float = 0.001
    balancing_loss_global_average: bool = True

    def build(self, router_scoring_func) -> BalancingLoss:
        return BalancingLoss(
            self.balancing_loss_alpha,
            self.balancing_loss_global_average,
            router_scoring_func=router_scoring_func,
        )


class ZLossConfig(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")
    z_loss_alpha: float = 0.001
    z_loss_global_average: bool = True

    def build(self) -> "ZLoss":
        from xtuner.v1.loss import ZLoss

        return ZLoss(
            self.z_loss_alpha,
            self.z_loss_global_average,
        )


class MoEConfig(TransformerConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    n_routed_experts: Annotated[int, Parameter(group="moe")]
    n_shared_experts: Annotated[int, Parameter(group="moe")]
    num_experts_per_tok: Annotated[int, Parameter(group="moe")]
    first_k_dense_replace: Annotated[int, Parameter(group="moe")] = 0
    hidden_factor: Annotated[float, Parameter(group="moe")] = 1.0
    moe_intermediate_size: Annotated[int, Parameter(group="moe")]
    ep_size: Annotated[int, Parameter(group="moe")] = 1
    dispatcher: Annotated[Literal["deepep", "all2all", "agrs"] | None, Parameter(group="moe")] = None
    router: GreedyRouterConfig | NoAuxRouterConfig
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None
    return_router_results: bool = False
    gate_bias: bool = False
    moe_bias: bool = False
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig()
    freeze_routers: bool = False

    def build(self) -> "MoE":
        from xtuner.v1.model.moe.moe import MoE

        if self.dispatcher == "agrs":
            assert self.router.use_grouped_router, "AGRS dispatcher requires grouped router"
            assert self.ep_size == self.router.router_n_groups == 8, (
                "Currently, AGRS dispatcher requires ep_size and router_n_groups to be 8"
            )

        return MoE(self)


class MoE(BaseModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`InternLM3DecoderLayer`]

    Args:
        config: MoEModelConfig
    """

    config: MoEConfig
    ep_mesh: DeviceMesh | None = None

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        if config.ep_size is not None and config.ep_size > 1:
            world_size = dist.get_world_size()
            self.ep_mesh = init_device_mesh(
                DEVICE,
                (world_size // config.ep_size, config.ep_size),
                mesh_dim_names=(f"{self.config.mesh_prefix}.dp", f"{self.config.mesh_prefix}.ep"),
            )[f"{self.config.mesh_prefix}.ep"]
        else:
            self.ep_mesh = None

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)

        self.fp32_layers = [self.rotary_emb]

        # TODO(@yehaochen): 把这两行移除 _maybe_compile_layers 要把 compile 相关的 setting 放到 fsdp_config 之外
        # _init_load_spec 放到 post init 里
        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)

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

        self.offload_stream = torch.cuda.Stream()

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

        # router_logits = router_logits[:, attn_mask].contiguous().float()
        indices = torch.nonzero(attn_mask, as_tuple=True)[0]
        router_logits = (
            torch.index_select(router_logits, 1, indices).contiguous().float()
        )  # [num_layers, non_pad_seq, num_experts]

        return router_logits

    @torch.no_grad()
    def _cal_tokens_per_expert(self, router_weights: torch.Tensor):
        n_routed_experts = self.config.n_routed_experts
        num_experts_per_tok = self.config.num_experts_per_tok
        num_layers = router_weights.shape[0]
        router_weights = router_weights.float()  # (nlayers, seq, ne)
        _, selected_experts = torch.topk(router_weights, num_experts_per_tok, dim=-1)
        selected_experts_flat = selected_experts.view(num_layers, -1)
        offset = torch.arange(num_layers, device=router_weights.device).unsqueeze(1) * n_routed_experts
        selected_experts_offset = selected_experts_flat + offset
        tokens_per_expert_flat = torch.histc(
            selected_experts_offset.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)  # (nlayers, ne)
        tokens_per_expert_global = tokens_per_expert.to(torch.long)  # (nlayers, ne)
        if dist.is_initialized():
            tokens_per_expert_global = all_reduce(tokens_per_expert_global, "sum", dist.group.WORLD)  # type: ignore
        return tokens_per_expert_global

    @torch.no_grad()
    def update_bias(self, total_expert_counts_pre_iter, expected_loads):
        """Implementation for the following paper:
        Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
        https://arxiv.org/abs/2408.15664

        TODO: refactor it later.
        """
        if self.config.freeze_routers:
            return

        first_k_dense_replace = self.config.first_k_dense_replace
        bias_update_speed = cast(NoAuxRouterConfig, self.config.router).router_bias_update_speed
        n_layer, _ = total_expert_counts_pre_iter.size()

        for i_layer in range(n_layer):
            # 前 l 层是 mlp 层，跳过
            gate = cast(MoEDecoderLayer, self.layers[str(first_k_dense_replace + i_layer)]).gate
            e_score_correction_bias = cast(NoAuxRouter, gate.router).e_score_correction_bias
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
        return_router_logits: bool = False,
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
                return_router_logits=return_router_logits,
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
                return_router_logits=return_router_logits,
            )

    def _micro_batch_forward(
        self,
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[CELossContext],
        return_router_logits: bool = False,
    ) -> MoEModelOutputs:
        """Micro-batch forward pass for MoE model.

        This method processes multiple micro-batches in parallel, similar to how MoEDecoderLayer handles micro-batching
        at the layer level.
        """
        if self.config.return_hidden_states:
            raise NotImplementedError

        assert len(seq_ctx_list) == len(loss_ctx_list), "seq_ctx and loss_ctx must have same length"

        # Prepare input embeddings for all micro-batches
        if seq_ctx_list[0].input_ids is None:
            cat_hidden_states = torch.cat([ctx.inputs_embeds for ctx in seq_ctx_list], dim=1)  # type: ignore
        else:
            cat_input_ids = torch.cat([ctx.input_ids for ctx in seq_ctx_list], dim=1)  # type: ignore
            cat_hidden_states = self.embed_tokens(cat_input_ids)
        cat_position_ids = torch.cat([ctx.position_ids for ctx in seq_ctx_list], dim=1)  # type: ignore
        cat_position_embeddings = self.rotary_emb(cat_hidden_states, cat_position_ids)  # type: ignore
        position_embeddings_list = list(
            zip(
                cat_position_embeddings[0].chunk(len(seq_ctx_list), dim=1),
                cat_position_embeddings[1].chunk(len(seq_ctx_list), dim=1),
            )
        )

        # Initialize output containers
        output: dict = {}

        router_logits_list: list[dict[str, torch.Tensor]] = [{} for _ in range(len(seq_ctx_list))]
        router_weights_list: list[dict[str, torch.Tensor]] = [{} for _ in range(len(seq_ctx_list))]

        # Process through layers
        cat_seq_ctx: SequenceContext | None = None

        moe_forward = False
        for idx, decoder_layer in self.layers.items():
            layer_idx = int(idx)

            if layer_idx < self.config.first_k_dense_replace:
                if cat_seq_ctx is None:
                    cat_seq_ctx = SequenceContext.cat(seq_ctx_list)
                # Dense decoder layer - process concated hidden states
                cat_hidden_states = decoder_layer(
                    cat_hidden_states,
                    position_embeddings=cat_position_embeddings,
                    seq_ctx=cat_seq_ctx,
                )
            else:
                if not moe_forward:
                    # TODO: `i.clone()` here is weird. However, the current Implementation of
                    # `async_save_on_cpu` is not friendly with `chunk` op (maybe caused by shared storage? not sure),
                    # resulting in nan grad norm. So we have to clone the chunked tensors here to make sure each
                    # hidden state has its own storage. This workaround may introduce extra memory and time cost, and
                    # should be optimized in the future.
                    hidden_states_list = [i.clone() for i in cat_hidden_states.chunk(len(seq_ctx_list), dim=1)]
                    moe_forward = True

                if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
                    with async_save_on_cpu(
                        h2d_stream=self.offload_stream,
                        d2h_stream=self.offload_stream,
                        block_idx=layer_idx - self.config.first_k_dense_replace,
                        depth=len(self.layers) - self.config.first_k_dense_replace,
                        custom_check_fn=lambda x: x.data_ptr()
                        in [hidden_states.data_ptr() for hidden_states in hidden_states_list],
                        prefetch=True,
                    ):
                        layer_results = decoder_layer(
                            *hidden_states_list,
                            position_embeddings=position_embeddings_list,
                            seq_ctx=seq_ctx_list,
                        )
                else:
                    layer_results = decoder_layer(
                        *hidden_states_list,
                        position_embeddings=position_embeddings_list,
                        seq_ctx=seq_ctx_list,
                    )
                hidden_states = layer_results[: len(hidden_states_list)]
                router_logits = layer_results[len(hidden_states_list) : len(hidden_states_list) * 2]
                router_weights = layer_results[len(hidden_states_list) * 2 :]

                # Update hidden states and collect router results
                for i, hidden_states in enumerate(hidden_states):
                    hidden_states_list[i] = hidden_states
                    router_logits_list[i][f"layer{idx}"] = router_logits[i]
                    router_weights_list[i][f"layer{idx}"] = router_weights[i]

        # Apply final norm to all micro-batches
        cat_hidden_states = torch.cat(hidden_states_list, dim=1)
        cat_hidden_states = self.norm(cat_hidden_states)

        # Process final outputs for each micro-batch
        cat_loss_ctx = CELossContext.cat(loss_ctx_list)
        loss, (logits, extra_info) = self.lm_head(cat_hidden_states, cat_loss_ctx)

        # Aggregate losses (mean across micro-batches)
        output["loss"] = loss.sum()
        moe_extra_info = ModelForwardExtraLogInfo()
        if extra_info:
            moe_extra_info.append(extra_info)
        output["extra_info"] = moe_extra_info

        # Handle router results for all micro-batches
        all_router_logits = []
        all_router_weights = []

        for micro_batch_idx, (micro_batch_router_logits, micro_batch_router_weights) in enumerate(
            zip(router_logits_list, router_weights_list)
        ):
            if micro_batch_router_logits:
                _router_logits_list = list(micro_batch_router_logits.values())
                _router_weights_list = list(micro_batch_router_weights.values())

                attn_mask = seq_ctx_list[micro_batch_idx].mask
                router_logits = self._select_non_pad_router_logits(_router_logits_list, attn_mask)
                router_weights = self._select_non_pad_router_logits(_router_weights_list, attn_mask)
                all_router_logits.append(router_logits)
                all_router_weights.append(router_weights)

        if all_router_logits:
            # Concatenate router logits from all micro-batches
            combined_router_logits = torch.cat(all_router_logits, dim=1)  # [num_layers, total_seq, num_experts]
            combined_router_weights = torch.cat(all_router_weights, dim=1)

            # Calculate balancing loss across all micro-batches
            if self.balancing_loss:
                balancing_loss = self.balancing_loss(
                    router_weights=combined_router_weights,
                    n_routed_experts=self.config.n_routed_experts,
                    num_experts_per_tok=self.config.num_experts_per_tok,
                )
                output["balancing_loss"] = balancing_loss

            # Calculate z-loss across all micro-batches
            if self.z_loss:
                z_loss = self.z_loss(router_logits=combined_router_logits)
                output["z_loss"] = z_loss

            # Calculate tokens per expert for bias update (if applicable)
            tokens_per_expert_global = self._cal_tokens_per_expert(combined_router_logits)
            output["tokens_per_expert_global"] = tokens_per_expert_global

            del combined_router_logits

        if self.config.return_router_results or return_router_logits:
            # raise NotImplementedError

            # TODO: Return router logits is costy

            router_logits_dict: dict[str, torch.Tensor] = {}
            layer_names = list(router_logits_list[0].keys())

            for layer_name in layer_names:
                layer_router_logits_list: list[torch.Tensor] = []
                for micro_batch_idx in range(len(seq_ctx_list)):
                    layer_router_logits_list.append(router_logits_list[micro_batch_idx][layer_name].detach())
                router_logits = torch.stack(layer_router_logits_list, dim=0).unsqueeze(0)
                router_logits_dict[layer_name] = router_logits

            output["router_logits"] = router_logits_dict

        return MoEModelOutputs(**output, logits=logits)  # type: ignore[typeddict-item]

    def _forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: CELossContext | None,
        return_router_logits: bool = False,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds

        # create position embeddings to be shared across the decoder layers
        assert position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}  # type: ignore
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        output["router_logits"] = {}
        output["router_weights"] = {}

        for idx, decoder_layer in self.layers.items():
            if int(idx) < self.config.first_k_dense_replace:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                )
            else:
                if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
                    with async_save_on_cpu(
                        h2d_stream=self.offload_stream,
                        d2h_stream=self.offload_stream,
                        block_idx=int(idx),
                        depth=len(self.layers),
                        custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
                    ):
                        layer_results = decoder_layer(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            seq_ctx=seq_ctx,
                        )

                else:
                    layer_results = decoder_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        seq_ctx=seq_ctx,
                    )
                hidden_states, router_results, router_weights = layer_results
                output["router_logits"][f"layer{idx}"] = router_results
                output["router_weights"][f"layer{idx}"] = router_weights

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx)  # type: ignore
        output["loss"] = loss
        output["logits"] = logits
        output["extra_info"] = extra_info

        router_logits_list = list(output["router_logits"].values())  # type: ignore
        router_weights_list = list(output["router_weights"].values())  # type: ignore
        router_logits = self._select_non_pad_router_logits(router_logits_list, seq_ctx.mask)
        router_weights = self._select_non_pad_router_logits(router_weights_list, seq_ctx.mask)

        if self.balancing_loss:
            balancing_loss = self.balancing_loss(
                router_weights=router_weights,
                n_routed_experts=self.config.n_routed_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
            )
            output["balancing_loss"] = balancing_loss

        if self.z_loss:
            z_loss = self.z_loss(router_logits=router_logits)
            output["z_loss"] = z_loss

        tokens_per_expert_global = self._cal_tokens_per_expert(router_logits)
        output["tokens_per_expert_global"] = tokens_per_expert_global

        del router_logits

        if self.config.return_router_results or return_router_logits:
            # raise NotImplementedError
            # TODO: Move router logits to CPU is cost
            for layer_name, router_logits in output["router_logits"].items():
                output["router_logits"][layer_name] = router_logits.detach().unsqueeze(0)
        else:
            output["router_logits"] = None

        return MoEModelOutputs(**output)  # type: ignore[typeddict-item]

    def build_embeddings(self, config: MoEConfig):
        return nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

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
                    layer_type=config.layers_type[layer_idx],
                    rope_scaling_cfg=config.rope_scaling_cfg,
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
                    gate_bias=config.gate_bias,
                    moe_bias=config.moe_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    num_experts_per_tok=config.num_experts_per_tok,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    hidden_factor=config.hidden_factor,
                    layer_type=config.layers_type[layer_idx],
                    attention_config=config.attention,
                    rope_scaling_cfg=config.rope_scaling_cfg,
                    generate_config=config.generate_config,
                    router_config=config.router,
                    moe_act_fn_cfg=config.moe_act_fn_cfg,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    dispatcher=config.dispatcher,
                    ep_mesh=self.ep_mesh,
                )
                if self.config.freeze_routers:
                    layers[str(layer_idx)].gate.requires_grad_(False)
                    layers[str(layer_idx)].gate.eval()
                    logger.info(f"Freeze MoE Router in layer {layer_idx}")

        layers.__class__.__repr__ = module_dict_repr  # type: ignore[method-assign]
        return layers

    def build_rotary_embedding(self, config: MoEConfig) -> RotaryEmbeddingProtocol:
        with torch.device(DEVICE):
            return get_rope_embedding(config=config)

    @override
    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        # If model is built on meta device, we need to rebuild rotary embedding since from_hf will not
        # load the `inv_freq` of RotaryEmbedding which is a inpersisitent buffer.
        # This is used for training without FSDP.
        # remove this because init with actual device already
        # For FoPE, the rotary_emb parameters (such as sin_coef) were already sharded by full_shard previously.
        # However, rebuilding them here would revert sin_coef back to its pre-sharded state, causing issues like dimension mismatches during loading.
        # self.rotary_emb = self.build_rotary_embedding(self.config).to(self.device)

        # logger.debug(f"before load hf: self.rotary_emb.sin_coef = {self.rotary_emb.sin_coef}, self.rotary_emb.cos_coef = {self.rotary_emb.cos_coef}")
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict)
        # logger.debug(f"after load hf: self.rotary_emb.sin_coef = {self.rotary_emb.sin_coef}, self.rotary_emb.cos_coef = {self.rotary_emb.cos_coef}")

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

        # Although rotary_emb was already constructed in __init__, it was built on the meta device.
        # Here we need to rebuild it on the actual device to calculate coefficients like inv_freq.
        # xTODO: remove this because init with actual device already, Check it
        # self.rotary_emb = self.build_rotary_embedding(self.config).to(self.device)

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

        self._to_empty_meta()
        return self

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        if self.config.ep_size > 1:
            return MOE_EP_COMPILE_CFG
        else:
            return MOE_NON_EP_COMPILE_CFG

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
                mesh_dim_names=(f"{self.config.mesh_prefix}.fsdp", f"{self.config.mesh_prefix}.ep"),
            )
            if self.ep_mesh is not None:
                # WARN: This assertion is **VERY** important.
                # FSDP requires that `device_mesh` shares the same root mesh across all mesh dimensions.
                # If not, it will raise an AssertionError:
                # "FSDP requires the DP and TP mesh to have the same parent mesh but got:
                #  DP's global mesh: {dp_global_mesh}\nTP's global mesh: {tp_global_mesh}"
                # ...
                # For MoE models that can perform inference independently without FSDP,
                # they build their own `ep_mesh`, which may not initially share the same root mesh
                # as `fsdp_mesh`. However, PyTorch's mesh management uses global logic: when a
                # submesh with an existing name is accessed (e.g., `model_mesh[f"{self.config.mesh_prefix}.ep"]`),
                # it creating a new submesh with the same **hash** as the existing `ep_mesh`
                # ...
                # FSDP's mesh manage the parent-child mapping by _mesh_resources, of which the key is the child mesh
                # and the value is the parent mesh, then, something interesting happened:
                # >>> print(id(old_ep_mesh), hash(old_ep_mesh))
                # 9753864, 6644214454873602895
                # >>> print(id(new_ep_mesh), hash(new_ep_mesh))
                # 9753878, 6644214454873602895
                # >>> _mesh_resources.get_root_mesh(old_ep_mesh) == _mesh_resources.get_root_mesh(new_ep_mesh)
                # True
                # Aha, although `old_ep_mesh` and `new_ep_mesh` are two different mesh, but `_mesh_resources` think
                # they share the same root mesh, which follows FSDP's assumption.
                # ...
                # Although I think it is an unexpected behavior of PyTorch's mesh management, but we can take
                # advantage of it to satisfy FSDP's requirement without changing the original `ep_mesh`.
                _new_created_ep_mesh = model_mesh[f"{self.config.mesh_prefix}.ep"]
                assert _new_created_ep_mesh.mesh_dim_names == self.ep_mesh.mesh_dim_names, (
                    f"FSDP enabled, it requires the name of new created `ep_mesh`: {_new_created_ep_mesh.mesh_dim_names}"  # noqa: E501
                    f"equals to the origin one: {self.ep_mesh.mesh_dim_names}"
                )
                assert torch.equal(self.ep_mesh.mesh, model_mesh[f"{self.config.mesh_prefix}.ep"].mesh), (
                    "FSDP enabled, it requires the `ep_size` of model config equals to the `ep_size` of FSDPConfig."
                )
            else:
                self.ep_mesh = model_mesh[f"{self.config.mesh_prefix}.ep"]

            self.fsdp_mesh = model_mesh[f"{self.config.mesh_prefix}.fsdp"]
        else:
            assert self.fsdp_config.ep_size == 1, "Currently, HSDP requires expert parallel size to be 1"
            ep_mesh = init_device_mesh(device, (world_size, 1), mesh_dim_names=("_", f"{self.config.mesh_prefix}.ep"))[
                f"{self.config.mesh_prefix}.ep"
            ]
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
                    f"{self.config.mesh_prefix}.hsdp_replicate",
                    f"{self.config.mesh_prefix}.hsdp_shard",
                ),
            )
            self.fsdp_mesh = self.hsdp_mesh[f"{self.config.mesh_prefix}.hsdp_shard"]

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
