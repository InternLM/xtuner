# Copyright (c) OpenMMLab. All rights reserved.
import os
import types
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self, Sequence, TypedDict, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import ConfigDict, model_validator
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from tqdm import tqdm
from typing_extensions import overload, override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import (
    AuxLossConfig,
    AuxLossContext,
    BalancingLossConfig,
    BaseLossContext,
    LMHeadLossContext,
    MTPLossContext,
    ZLossConfig,
)
from xtuner.v1.loss.mtp_loss import MTPLossConfig
from xtuner.v1.model.base import (
    DEFAULT_FLOAT8_CFG,
    BaseModel,
    BatchForwardInfo,
    ModelOutputs,
    TorchCompileOption,
    TransformerConfig,
)
from xtuner.v1.model.utils import ModelForwardExtraLogInfo, checkpoint_wrapper, module_dict_repr
from xtuner.v1.module import (
    GatedDeltaNetConfig,
    GreedyRouterConfig,
    LMHead,
    MHAConfig,
    MLAConfig,
    NoAuxRouter,
    NoAuxRouterConfig,
    RMSNorm,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock, MoEDecoderLayer
from xtuner.v1.module.mtp import MTPBlock, MTPConfig, MTPLayer
from xtuner.v1.utils import (
    get_device,
    get_logger,
    log_rank0,
)
from xtuner.v1.utils.activation_offload import async_save_on_cpu
from xtuner.v1.utils.router_offload import async_offload_to_cpu


if TYPE_CHECKING:
    from xtuner.v1.datasets.collator import ColateItem


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


def _total_backward_loss(
    llm_loss: torch.Tensor,
    balancing_loss: torch.Tensor | None,
    mtp_loss: torch.Tensor | None,
) -> torch.Tensor:
    """Sum the graph-carrying loss terms into the single scalar the engine
    backprops.

    Z-loss is not added here: it is injected into the main graph via ``AuxLossScaler`` at
    accumulate() time, so its gradient already flows through ``llm_loss``. Only the terms whose
    backward graph is NOT already on the main path (balancing, MTP) are summed in.

    Args:
        llm_loss (torch.Tensor): The LM (or RL policy) loss.
        balancing_loss (torch.Tensor | None): Balancing loss carrying its graph, or ``None``.
        mtp_loss (torch.Tensor | None): MTP loss carrying its graph, or ``None``.

    Returns:
        torch.Tensor: The total backward loss.
    """
    total = llm_loss
    if balancing_loss is not None:
        total = total + balancing_loss
    if mtp_loss is not None:
        total = total + mtp_loss
    return total


def _build_loss_log(
    lm_display: torch.Tensor,
    aux_calibrated: dict[str, torch.Tensor],
    mtp_display: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Assemble the per-loss display dict from the LM, aux, and MTP terms.

    Stored on ``output.loss_log`` and aggregated over micro-batches by the display pipeline. The aux
    terms (``balancing_loss`` / ``balancing_loss_global`` / ``z_loss``) are produced by
    :meth:`AuxLossContext.calibrate`; this only prepends ``llm_loss`` and appends ``mtp_loss``.

    Args:
        lm_display (torch.Tensor): This rank's LM loss display value.
        aux_calibrated (dict[str, torch.Tensor]): Aux display values from ``AuxLossContext.calibrate``.
        mtp_display (torch.Tensor | None): MTP loss display value, or ``None``.

    Returns:
        dict[str, torch.Tensor]: Display values keyed by loss name.
    """
    loss_log: dict[str, torch.Tensor] = {"llm_loss": lm_display.detach()}
    loss_log |= aux_calibrated
    if mtp_display is not None:
        loss_log["mtp_loss"] = mtp_display.detach()
    return loss_log


class MoEModelOutputs(ModelOutputs):
    router_logits: dict[str, torch.Tensor] | None = None
    router_weights: dict[str, torch.Tensor] | None = None
    # Aux losses (balancing / z / MTP) are folded into `loss` for backward and into `loss_log` for
    # display; only the routing statistics needed by metrics / bias update are surfaced separately.
    tokens_per_expert_global: torch.Tensor

    def free_nongrad_feature(self):
        """Release large intermediate tensors not needed for backward or
        logging.

        This method is called immediately after forward() in the micro-batch loop.
        It releases large tensors (logits, hidden_states) while keeping:
        - loss: needed for backward pass
        - extra_info: lightweight logging info needed by post_micro_batch_forward()
        """
        super().free_nongrad_feature()
        self.router_logits = None
        self.router_weights = None


class MoEBatchForwardInfo(BatchForwardInfo):
    step_balancing_loss: float
    z_loss: float
    tokens_per_expert_global: int
    maxvio: float


class MoELossContextDict(TypedDict):
    lm: BaseLossContext
    aux: AuxLossContext | None
    mtp: list[BaseLossContext] | None


class MoEConfig(TransformerConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    n_routed_experts: Annotated[int, Parameter(group="moe")]
    n_shared_experts: Annotated[int, Parameter(group="moe")]
    with_shared_expert_gate: bool = False  # enable when n_shared_experts > 0
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
    router_compute_dtype: Literal["float32", "native"] = "float32"
    moe_bias: bool = False
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig()
    mtp_config: MTPConfig | None = None
    freeze_routers: bool = False
    router_async_offload: bool = False
    aux_loss_cfg: AuxLossConfig = AuxLossConfig()
    # TODO: `FSDPConfig` should be model-specific; temporarily keep
    # `embed_reshard_after_forward` here until per-submodule FSDP config is supported.
    # Compose models call `self.embed_tokens` multiple times per step, so default to
    # keeping it unsharded after forward to avoid repeated all-gathers.
    embed_reshard_after_forward: bool = True

    @model_validator(mode="after")
    def _populate_aux_loss_cfg(self) -> "MoEConfig":
        # aux_loss_cfg is the internal aggregate AuxLossConfig.build(data, sp_mesh) reads: mirror the
        # routing dims and the user-facing balancing / z-loss switches into it so build() is
        # self-contained. Users configure aux losses via balancing_loss_cfg / z_loss_cfg on the model
        # config; this keeps aux_loss_cfg the single, in-sync source build() consumes.
        self.aux_loss_cfg = self.aux_loss_cfg.model_copy(
            update={
                "n_routed_experts": self.n_routed_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "balancing_loss_cfg": self.balancing_loss_cfg,
                "z_loss_cfg": self.z_loss_cfg,
            }
        )
        return self

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

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type=config.rms_norm_type)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)
        self.mtp_block = self.build_mtp_block(config) if config.mtp_config is not None else None

        self.fp32_layers = [self.rotary_emb]

        # TODO(@yehaochen): 把这两行移除 _maybe_compile_layers 要把 compile 相关的 setting 放到 fsdp_config 之外
        # _init_load_spec 放到 post init 里
        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)

        self.offload_stream = torch.cuda.Stream()

    def _maybe_offload_router(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.router_async_offload:
            return async_offload_to_cpu(tensor, self.offload_stream)
        return tensor

    def _resolve_aux_ctx(
        self,
        loss_ctx: list[MoELossContextDict | None] | MoELossContextDict | None,
        seq_ctx: list[SequenceContext] | SequenceContext,
    ) -> AuxLossContext:
        # The whole-batch aux hub for this forward. Aux stats are whole-batch quantities: under
        # intra_layer_micro_batch the per-micro-batch hubs built at build_loss_ctx_batch time are
        # merged via cat(), which concatenates their nonpad indices into nonpad_list.
        loss_ctx_list = loss_ctx if isinstance(loss_ctx, list) else [loss_ctx]
        aux_chunks = [lc.get("aux") if lc is not None else None for lc in loss_ctx_list]
        if aux_chunks and all(aux is not None for aux in aux_chunks):
            return AuxLossContext.cat([cast(AuxLossContext, aux) for aux in aux_chunks])
        # Fallback: the loss context carries no aux hub (forward_only / reference model, which pass a
        # bare {"lm": ...} or None). Build a tokens-only hub -- balancing / z sub-configs stripped, so
        # no aux backward and no aux collective -- so tokens_per_expert is still produced for maxvio /
        # bias update. The forward seq_ctx mask is already SP-local, so build runs with no sp_split.
        seq_ctx_list = seq_ctx if isinstance(seq_ctx, list) else [seq_ctx]
        tokens_only_cfg = self.config.aux_loss_cfg.model_copy(update={"balancing_loss_cfg": None, "z_loss_cfg": None})
        chunks = [tokens_only_cfg.build({"seq_ctx": sc}) for sc in seq_ctx_list]
        return chunks[0] if len(chunks) == 1 else AuxLossContext.cat(chunks)

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

    def build_loss_ctx_batch(  # type: ignore[override]
        self,
        data_batch: list["ColateItem"],
        sp_mesh: DeviceMesh | None = None,
    ) -> list[MoELossContextDict]:  # type: ignore[override]
        """Build and calibrate loss contexts for MoE model.

        Args:
            data_batch (list[dict]): All microbatch data
            sp_mesh (DeviceMesh | None): Sequence parallel mesh
            cu_seq_lens_list (list[torch.IntTensor] | None): For calibration

        Returns:
            list[dict]: Loss context dict for each microbatch.
                Each dict contains:
                - "lm": LM loss context
                - "aux": Per-micro-batch auxiliary-loss hub (routing stats + optional balancing / z)
                - "mtp": MTP loss contexts (if configured)
        """
        # Build LM loss context
        _data_batch: list[dict] = data_batch  # type: ignore[assignment]
        res: list[dict] = super().build_loss_ctx_batch(_data_batch, sp_mesh)
        cu_seq_lens_list = [data["seq_ctx"].cu_seq_lens_k for data in data_batch]

        # Build the per-micro-batch aux hub (routing stats + optional balancing / z sub-contexts),
        # data-bound like the lm / mtp contexts: one AuxLossConfig.build(data, sp_mesh) per micro-batch.
        for loss_ctx_dict, data in zip(res, _data_batch):
            loss_ctx_dict["aux"] = self.config.aux_loss_cfg.build(data, sp_mesh)

        # Add MTP loss contexts if MTP is enabled
        if self.config.mtp_config is not None:
            for mtp_idx in range(self.config.mtp_config.num_layers):
                mtp_loss_cfg = MTPLossConfig(
                    **self.config.lm_loss_cfg.model_dump(),
                    mtp_depth=mtp_idx + 1,
                    detach_mtp_lm_head_weight=self.config.mtp_config.detach_mtp_lm_head_weight,
                )
                mtp_loss_ctx_list = self._build_loss_ctx(mtp_loss_cfg, _data_batch, sp_mesh)
                if mtp_loss_ctx_list is not None:
                    mtp_loss_ctx_list = MTPLossContext.build_batches(  # type: ignore[assignment]
                        cast(list[MTPLossContext], mtp_loss_ctx_list),  # type: ignore[arg-type]
                        cu_seq_lens_list=cu_seq_lens_list,
                        sp_mesh=sp_mesh,
                    )
                    for i, mtp_loss_ctx in enumerate(mtp_loss_ctx_list):
                        if "mtp" not in res[i]:
                            res[i]["mtp"] = []
                        res[i]["mtp"].append(mtp_loss_ctx)  # type: ignore[union-attr]

            # Ensure all microbatches have mtp key
            for loss_ctx_dict in res:
                if "mtp" not in loss_ctx_dict:
                    loss_ctx_dict["mtp"] = None
        else:
            for loss_ctx_dict in res:
                loss_ctx_dict["mtp"] = None

        return res  # type: ignore[return-value]

    def forward(
        self,
        seq_ctx: list[SequenceContext] | SequenceContext,
        loss_ctx: list[MoELossContextDict] | MoELossContextDict | None,
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

    def post_micro_batch_forward(self, batch_outputs: Sequence[MoEModelOutputs]) -> MoEBatchForwardInfo:
        base_info = super().post_micro_batch_forward(batch_outputs)
        logs_info = base_info["logs_info"]

        first_tokens_per_expert = batch_outputs[0]["tokens_per_expert_global"]
        tokens_per_expert_global = torch.zeros_like(first_tokens_per_expert)
        for output in batch_outputs:
            tokens_per_expert_global += output["tokens_per_expert_global"]

        avg_count_load = tokens_per_expert_global.float().mean(1)
        max_load_i, _ = torch.max(tokens_per_expert_global, dim=1)
        maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
        maxvio = maxvio_all_layers.mean()
        logs_info["maxvio"] = maxvio.item()

        if self.need_update_bias:
            self.update_bias(tokens_per_expert_global, avg_count_load)  # type: ignore

        moe_info = cast(MoEBatchForwardInfo, base_info)
        return moe_info

    def _micro_batch_forward(
        self,
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[MoELossContextDict],
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
        # M-RoPE position_ids are 3D [axes, batch, seq] for VL while text-only ones are 2D
        # [batch, seq]; -1 selects the seq dim in both cases. Hard-coded dim=1 was a text-only
        # assumption and produced a wrong-length cos/sin for VL under intra_layer_micro_batch.
        cat_position_ids = torch.cat([ctx.position_ids for ctx in seq_ctx_list], dim=-1)  # type: ignore
        cat_position_embeddings = self.rotary_emb(cat_hidden_states, cat_position_ids)  # type: ignore
        position_embeddings_list = list(
            zip(
                cat_position_embeddings[0].chunk(len(seq_ctx_list), dim=1),
                cat_position_embeddings[1].chunk(len(seq_ctx_list), dim=1),
            )
        )
        # Initialize output containers
        output: dict = {}

        # Only the logits side is ever exposed to callers in the micro-batch path; the
        # weights side is not part of the returned schema, so we never accumulate it.
        keep_router = self.config.return_router_results or return_router_logits
        router_logits_list: list[dict[str, torch.Tensor]] = (
            [{} for _ in range(len(seq_ctx_list))] if keep_router else []
        )
        # Whole-batch aux hub: padding removal / stat accumulation / z-loss injection all happen
        # inside it; the model feeds one layer's raw router outputs for every micro-batch as lists and
        # the hub zips them against its own per-micro-batch nonpad_list (built at build_loss_ctx_batch).
        aux_ctx = self._resolve_aux_ctx(loss_ctx_list, seq_ctx_list)  # type: ignore[arg-type]

        # Process through layers
        cat_seq_ctx: SequenceContext | None = None

        hidden_states_list: list[torch.Tensor] = []
        moe_forward = False
        # Dense 0-based ordinal of the MoE layers actually processed (immune to PP layer subsets and
        # to the absolute dict index); it keys the aux hub's per-layer slots and continues into MTP.
        moe_layer_ordinal = 0

        for seq_ctx in seq_ctx_list:
            self._mark_dynamic(seq_ctx)

        for idx, decoder_layer in self.layers.items():
            layer_idx = int(idx)

            if layer_idx < self.config.first_k_dense_replace:
                if cat_seq_ctx is None:
                    cat_seq_ctx = SequenceContext.cat(seq_ctx_list)
                    self._mark_dynamic(cat_seq_ctx)
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
                assert hidden_states_list, "XTuner Internal Error, found empty hidden states for domino EP"

                if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
                    with async_save_on_cpu(
                        h2d_stream=self.offload_stream,
                        d2h_stream=self.offload_stream,
                        block_idx=layer_idx - self.config.first_k_dense_replace,
                        group="text",
                        custom_check_fn=lambda x: x.data_ptr()
                        in [hidden_states.data_ptr() for hidden_states in hidden_states_list],
                        prefetch=True,
                        reserve_pin_memory=True,
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
                layer_hidden = layer_results[: len(hidden_states_list)]
                router_logits = layer_results[len(hidden_states_list) : len(hidden_states_list) * 2]
                router_weights = layer_results[len(hidden_states_list) * 2 :]

                # Per micro-batch: stash router logits if requested, then hand this layer's raw router
                # outputs to the aux hub. The hub strips padding with the micro-batch's own nonpad
                # indices and adds the stats into this layer's slot -- summing over micro-batches is
                # equal to concatenating them and accumulating once. The z-loss rides on each
                # micro-batch's own carrier; all carriers converge into the same total_loss backward,
                # so each aux node fires exactly once.
                for i in range(len(layer_hidden)):
                    hidden_states_list[i] = layer_hidden[i]
                    if keep_router:
                        router_logits_list[i][f"layer{idx}"] = self._maybe_offload_router(router_logits[i])
                hidden_states_list = aux_ctx.accumulate(
                    layer_idx=moe_layer_ordinal,
                    router_weights_list=list(router_weights),
                    router_logits_list=list(router_logits),
                    hidden_states_list=hidden_states_list,
                )
                moe_layer_ordinal += 1

        assert hidden_states_list, "XTuner Internal Error, found empty hidden states for domino EP"

        mtp_loss: torch.Tensor | None = None
        mtp_calibrated: torch.Tensor | None = None
        if self.mtp_block is not None:
            assert self.config.mtp_config is not None

            # Build a per-microbatch SequenceContext clone for MTP. We always run the MTP
            # block on every micro-batch so domino EP can overlap dispatch/combine across
            # micro-batches at each MTP depth; per-microbatch loss aggregation below skips
            # the ones whose loss context is absent.
            mtp_seq_ctx_list: list[SequenceContext] = []
            for seq_ctx in seq_ctx_list:
                assert seq_ctx.position_ids is not None
                mtp_seq_ctx_list.append(
                    seq_ctx.copy(
                        input_ids=seq_ctx.input_ids.clone() if seq_ctx.input_ids is not None else None,
                        position_ids=seq_ctx.position_ids.clone(),
                        inputs_embeds=seq_ctx.inputs_embeds.clone() if seq_ctx.inputs_embeds is not None else None,
                    )
                )

            mtp_outputs_per_mb = self.mtp_block(
                *hidden_states_list,
                embed_tokens_fn=self.embed_tokens,
                position_embeddings=position_embeddings_list,
                seq_ctx=mtp_seq_ctx_list,
            )

            mtp_losses = torch.tensor(0.0, device=DEVICE)
            # Per-rank display value (from each depth's calibrate()), mirroring `mtp_losses`.
            mtp_display = torch.tensor(0.0, device=DEVICE)
            has_mtp_loss = False
            for micro_batch_idx, (loss_ctx_dict, mtp_outputs) in enumerate(zip(loss_ctx_list, mtp_outputs_per_mb)):
                mtp_loss_ctx_list = loss_ctx_dict.get("mtp")
                if mtp_loss_ctx_list is None:
                    continue

                micro_batch_mtp_losses = torch.tensor(0.0, device=DEVICE)
                micro_batch_mtp_display = torch.tensor(0.0, device=DEVICE)
                for mtp_idx, (mtp_hidden, mtp_ctx) in enumerate(zip(mtp_outputs, mtp_loss_ctx_list)):
                    mtp_hidden_states, mtp_router_results, _ = mtp_hidden
                    mtp_loss, _ = self.lm_head(mtp_hidden_states, cast(MTPLossContext, mtp_ctx))
                    micro_batch_mtp_losses += mtp_loss
                    micro_batch_mtp_display += cast(MTPLossContext, mtp_ctx).calibrate()

                    if keep_router:
                        router_logits_list[micro_batch_idx][f"mtp_layer{mtp_idx}"] = mtp_router_results

                mtp_losses += micro_batch_mtp_losses / len(mtp_loss_ctx_list)
                mtp_display += micro_batch_mtp_display / len(mtp_loss_ctx_list)
                has_mtp_loss = True

            if has_mtp_loss:
                # MTP routed experts feed the same balancing / z aux loss as the main MoE layers
                # (mirrors the single-microbatch path in `_forward`); without this they are silently
                # excluded from the aux loss and from `tokens_per_expert` in this path. Each MTP depth
                # is one aux slot, accumulated per micro-batch into that slot (summing over micro-
                # batches equals concatenating them and accumulating once). MTP shares the main micro-
                # batch masks (mtp_seq_ctx_list is copied from seq_ctx_list), so the same per-mb nonpad
                # indices apply. The depth slots continue the dense MoE-layer ordinal. The z-loss rides
                # on each micro-batch's own carrier -- the same main-loss path the per-layer aux loss
                # already rides on -- so backward traverses each MTP aux node exactly once.
                n_moe_layers = moe_layer_ordinal
                for mtp_idx in range(self.config.mtp_config.num_layers):
                    hidden_states_list = aux_ctx.accumulate(
                        layer_idx=n_moe_layers + mtp_idx,
                        router_weights_list=[mb_outputs[mtp_idx][2] for mb_outputs in mtp_outputs_per_mb],
                        router_logits_list=[mb_outputs[mtp_idx][1] for mb_outputs in mtp_outputs_per_mb],
                        hidden_states_list=hidden_states_list,
                    )

                mtp_loss = mtp_losses * self.config.mtp_config.loss_scaling_factor
                mtp_calibrated = mtp_display * self.config.mtp_config.loss_scaling_factor

        # Apply final norm to all micro-batches
        cat_hidden_states = torch.cat(hidden_states_list, dim=1)
        cat_hidden_states = self.norm(cat_hidden_states)

        # Process final outputs for each micro-batch
        # Extract LM loss context from dict
        lm_loss_ctx_list = [loss_ctx_dict["lm"] for loss_ctx_dict in loss_ctx_list]
        cat_loss_ctx = type(lm_loss_ctx_list[0]).cat(lm_loss_ctx_list)
        # All micro-batch lm contexts of a step share the same display coefficient (build_batches
        # computes one per step); carry it onto the concatenated context so calibrate() works.
        cat_loss_ctx._display_coeff = lm_loss_ctx_list[0]._display_coeff
        loss, (logits, extra_info) = self.lm_head(cat_hidden_states, cast(LMHeadLossContext, cat_loss_ctx))

        moe_extra_info = ModelForwardExtraLogInfo()
        if extra_info:
            moe_extra_info.append(extra_info)
        output["extra_info"] = moe_extra_info

        aux_out = aux_ctx.finalize()
        # output.loss is the single total the engine backprops; per-term values are display-only.
        output["loss"] = _total_backward_loss(loss.sum(), aux_out["balancing_loss"], mtp_loss)
        output["tokens_per_expert_global"] = aux_out["tokens_per_expert_global"]
        output["loss_log"] = _build_loss_log(
            cast(LMHeadLossContext, cat_loss_ctx).calibrate(),
            aux_ctx.calibrate(),
            mtp_calibrated,
        )

        if keep_router:
            # TODO: Returning router logits is costly.
            router_logits_dict: dict[str, torch.Tensor] = {}
            layer_names = list(router_logits_list[0].keys())

            for layer_name in layer_names:
                layer_router_logits_list: list[torch.Tensor] = []
                for micro_batch_idx in range(len(seq_ctx_list)):
                    layer_router_logits_list.append(router_logits_list[micro_batch_idx][layer_name].detach())
                router_logits = torch.stack(layer_router_logits_list, dim=0).unsqueeze(0)
                router_logits_dict[layer_name] = router_logits

            output["router_logits"] = router_logits_dict

        return MoEModelOutputs(**output, logits=logits)

    def _forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: MoELossContextDict | None,
        return_router_logits: bool = False,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            assert seq_ctx.inputs_embeds is not None, "inputs_embeds should not be None when input_ids is None"
            # The clone here is mainly for ActivationOffload. The current offload implementation modifies
            # the input tensor in-place, causing subsequent accesses to input_embeds to get a tensor with
            # empty storage and trigger errors. So we clone here to ensure later accesses to input_embeds
            # won't fail. However, there are two remaining caveats:
            # 1. The extra clone may introduce a slight performance overhead.
            # 2. hidden_states itself still cannot be reused, as offload will leave it with empty storage.
            hidden_states = seq_ctx.inputs_embeds.clone()

        # create position embeddings to be shared across the decoder layers
        assert position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}  # type: ignore
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        # Router logits / weights are only retained when a downstream consumer
        # (config flag or per-call kwarg) asked for them; otherwise we skip the
        # per-layer dict population and the optional D2H offload entirely.
        keep_router = self.config.return_router_results or return_router_logits
        if keep_router:
            output["router_logits"] = {}
            output["router_weights"] = {}
        else:
            output["router_logits"] = None
            output["router_weights"] = None
        self._mark_dynamic(seq_ctx)
        # Whole-batch aux hub (single micro-batch here): padding removal / stat accumulation / z-loss
        # injection all happen inside it; the model feeds each layer's raw router outputs as
        # single-element lists and the hub zips them against its own nonpad_list.
        aux_ctx = self._resolve_aux_ctx(loss_ctx, seq_ctx)
        # Dense 0-based ordinal of the MoE layers actually processed; keys the aux hub's per-layer
        # slots and continues into MTP. Immune to PP layer subsets / the absolute dict index.
        moe_layer_ordinal = 0

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
                        group="text",
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
                if keep_router:
                    output["router_logits"][f"layer{idx}"] = self._maybe_offload_router(router_results)
                    output["router_weights"][f"layer{idx}"] = self._maybe_offload_router(router_weights)
                hidden_states = aux_ctx.accumulate(
                    layer_idx=moe_layer_ordinal,
                    router_weights_list=[router_weights],
                    router_logits_list=[router_results],
                    hidden_states_list=[hidden_states],
                )[0]
                moe_layer_ordinal += 1

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        layer_hidden_states = hidden_states
        hidden_states = self.norm(hidden_states)

        # Get LM loss context from dict
        lm_loss_ctx = loss_ctx["lm"] if loss_ctx is not None else None
        loss, (logits, extra_info) = self.lm_head(hidden_states, lm_loss_ctx)  # type: ignore
        output["logits"] = logits
        output["extra_info"] = extra_info

        # MTP forward pass and loss computation
        mtp_loss_total: torch.Tensor | None = None
        mtp_calibrated: torch.Tensor | None = None
        if (
            self.mtp_block is not None
            and loss_ctx is not None
            and (mtp_loss_ctx_list := loss_ctx.get("mtp")) is not None
        ):
            mtp_seq_ctx = seq_ctx.copy(
                input_ids=input_ids.clone() if input_ids is not None else None,
                position_ids=position_ids.clone(),
                inputs_embeds=seq_ctx.inputs_embeds.clone() if seq_ctx.inputs_embeds is not None else None,
            )
            # MTP shares the main padding mask (mtp_seq_ctx is a copy that does not override the
            # cu_seq_lens / num_padding the mask derives from), so the main non-pad indices and the
            # token counts already set on the aux contexts apply directly.

            # Forward through MTP block
            mtp_outputs = self.mtp_block(
                layer_hidden_states,
                embed_tokens_fn=self.embed_tokens,
                position_embeddings=position_embeddings,
                seq_ctx=mtp_seq_ctx,
            )

            # Compute MTP losses for each depth
            mtp_losses = torch.tensor(0.0, device=DEVICE)
            # Per-rank display value (from each depth's calibrate()), mirroring `mtp_losses`.
            mtp_display = torch.tensor(0.0, device=DEVICE)
            # MTP depth slots continue the dense MoE-layer ordinal; MTP shares the main mask, so the
            # main nonpad indices apply directly.
            n_moe_layers = moe_layer_ordinal
            for depth, (mtp_hidden, mtp_ctx) in enumerate(zip(mtp_outputs, mtp_loss_ctx_list)):
                mtp_hidden_states, mtp_router_results, mtp_router_weights = mtp_hidden

                if keep_router:
                    output["router_logits"][f"mtp_layer{depth}"] = mtp_router_results
                    output["router_weights"][f"mtp_layer{depth}"] = mtp_router_weights
                # Inject this MTP layer's z-loss before lm_head so backward through mtp_loss
                # traverses the AuxLossScaler node and releases this layer's logsumexp activations.
                mtp_hidden_states = aux_ctx.accumulate(
                    layer_idx=n_moe_layers + depth,
                    router_weights_list=[mtp_router_weights],
                    router_logits_list=[mtp_router_results],
                    hidden_states_list=[mtp_hidden_states],
                )[0]
                mtp_loss, _ = self.lm_head(mtp_hidden_states, cast(MTPLossContext, mtp_ctx))
                mtp_losses += mtp_loss
                mtp_display += cast(MTPLossContext, mtp_ctx).calibrate()

            # Average MTP losses across depths and scale
            mtp_losses = mtp_losses / len(mtp_loss_ctx_list)
            scaled_mtp_loss = mtp_losses * self.config.mtp_config.loss_scaling_factor  # type: ignore

            mtp_loss_total = scaled_mtp_loss
            mtp_calibrated = mtp_display / len(mtp_loss_ctx_list) * self.config.mtp_config.loss_scaling_factor  # type: ignore

        aux_out = aux_ctx.finalize()
        output["tokens_per_expert_global"] = aux_out["tokens_per_expert_global"]
        if lm_loss_ctx is not None:
            assert loss is not None
            # output.loss is the single total the engine backprops; per-term values are display-only.
            output["loss"] = _total_backward_loss(loss, aux_out["balancing_loss"], mtp_loss_total)
            output["loss_log"] = _build_loss_log(
                cast(LMHeadLossContext, lm_loss_ctx).calibrate(),
                aux_ctx.calibrate(),
                mtp_calibrated,
            )

        if keep_router:
            # TODO: Moving router logits to CPU is costly.
            for layer_name, router_logits in output["router_logits"].items():
                output["router_logits"][layer_name] = router_logits.detach().unsqueeze(0)

        return MoEModelOutputs(**output)

    def build_embeddings(self, config: MoEConfig):
        return nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        # 让 layers 是一个 nn.ModuleDict 方便做 pipeline parallel 的参数切分，
        # 这样可以保证部分 layer 被切掉后，idx 保持不变
        layers = nn.ModuleDict()
        attention_config: GatedDeltaNetConfig | MLAConfig | MHAConfig | None = None
        for layer_idx in range(config.num_hidden_layers):
            if config.layers_type[layer_idx] in ["full_attention", "sliding_attention"]:
                attention_config = config.attention
            elif config.layers_type[layer_idx] == "linear_attention":
                attention_config = config.linear_attention
                assert attention_config is not None, (
                    "linear_attention config must be provided for linear_attention layer"
                )
            else:
                raise ValueError(
                    f"Unsupported layer type {config.layers_type[layer_idx]} at layer {layer_idx}. Only 'full_attention', 'sliding_attention' and 'linear_attention' are supported."
                )

            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = DenseDecoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    attention_config=attention_config,
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
                    rms_norm_type=config.rms_norm_type,
                    num_experts_per_tok=config.num_experts_per_tok,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    with_shared_expert_gate=config.with_shared_expert_gate,
                    hidden_factor=config.hidden_factor,
                    layer_type=config.layers_type[layer_idx],
                    attention_config=attention_config,
                    rope_scaling_cfg=config.rope_scaling_cfg,
                    generate_config=config.generate_config,
                    router_config=config.router,
                    router_compute_dtype=config.router_compute_dtype,
                    moe_act_fn_cfg=config.moe_act_fn_cfg,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    dispatcher=config.dispatcher,
                    ep_mesh=self.ep_mesh,
                )
                if self.config.freeze_routers:
                    layers[str(layer_idx)].gate.requires_grad_(False)
                    layers[str(layer_idx)].gate.eval()
                    log_rank0.info(f"Freeze MoE Router in layer {layer_idx}")

        layers.__class__.__repr__ = module_dict_repr  # type: ignore[method-assign]
        return layers

    def build_mtp_block(self, config: MoEConfig) -> MTPBlock:
        """Build MTP block with MoE decoder layers.

        Args:
            config (MoEConfig): Model configuration.

        Returns:
            MTPBlock: Constructed MTP block.
        """
        mtp_config = config.mtp_config
        assert mtp_config is not None, "mtp_config must be provided"

        mtp_layers = []
        # Get attention config for MTP layers (use last layer's config)
        last_layer_idx = config.num_hidden_layers - 1
        layers_type_list = config.layers_type
        attention_config: MLAConfig | MHAConfig | GatedDeltaNetConfig
        if layers_type_list[last_layer_idx] in ["full_attention", "sliding_attention"]:
            attention_config = config.attention
        elif layers_type_list[last_layer_idx] == "linear_attention":
            assert config.linear_attention is not None, (
                "linear_attention config must be provided for linear_attention layer"
            )
            attention_config = config.linear_attention
        else:
            raise ValueError(f"Unsupported layer type {layers_type_list[last_layer_idx]}")

        num_physical_layer = 1 if mtp_config.share_weights else mtp_config.num_layers
        for i in range(num_physical_layer):
            # Build MoE decoder layer for MTP
            decoder_layer = MoEDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                moe_intermediate_size=config.moe_intermediate_size,
                mlp_bias=config.mlp_bias,
                gate_bias=config.gate_bias,
                moe_bias=config.moe_bias,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                rms_norm_type=config.rms_norm_type,
                num_experts_per_tok=config.num_experts_per_tok,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                with_shared_expert_gate=config.with_shared_expert_gate,
                hidden_factor=config.hidden_factor,
                layer_type=layers_type_list[last_layer_idx],
                attention_config=attention_config,
                rope_scaling_cfg=config.rope_scaling_cfg,
                generate_config=config.generate_config,
                router_config=config.router,
                router_compute_dtype=config.router_compute_dtype,
                moe_act_fn_cfg=config.moe_act_fn_cfg,
                float8_cfg=config.float8_cfg,
                layer_idx=config.num_hidden_layers + i,
                dispatcher=config.dispatcher,
                ep_mesh=self.ep_mesh,
            )

            # Wrap decoder layer in MTPLayer
            mtp_layer = MTPLayer(
                hidden_size=config.hidden_size,
                rms_norm_eps=config.rms_norm_eps,
                rms_norm_type=config.rms_norm_type,
                decoder_layer=decoder_layer,
                float8_cfg=config.float8_cfg,
            )
            mtp_layers.append(mtp_layer)

        return MTPBlock(mtp_config=mtp_config, mtp_layers=mtp_layers)

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
    ) -> Self:
        self.fsdp_config = fsdp_config
        assert self.fsdp_config.ep_size == self.config.ep_size
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        if self.fsdp_config.fp32_lm_head:
            lm_head_mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        else:
            lm_head_mp_policy = self.mp_policy
        self._init_device_mesh(fsdp_config)

        if self.config.float8_cfg is not None:
            # As we modify the shape of the model's parameters,
            # we need to reinitialize the load spec mapping.
            Float8Handler.pad_for_fsdp(self, cast(DeviceMesh, self.fsdp_mesh), callback_after_pad=self._init_load_spec)

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

        for layer_idx, layer in tqdm(self.layers.items(), desc="[FSDP Sharding]"):
            layer_idx = int(layer_idx)
            if self._should_recompute(
                layer_idx=layer_idx,
                mtp_idx=None,
            ):
                layer = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.REENTRANT)

            self.layers[str(layer_idx)] = layer
            if layer_idx >= len(self.layers) - 1 and self.mtp_block is None:
                reshard_after_forward = False
            else:
                reshard_after_forward = self.fsdp_config.reshard_after_forward

            self._fully_shard(
                mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
                module=layer,
            )

        for layer_cur, layer_next in zip(
            list(self.layers.values())[:-1],
            list(self.layers.values())[1:],
        ):
            layer_cur.set_modules_to_forward_prefetch([layer_next])  # type: ignore

        self._fully_shard(
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.config.embed_reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
            module=self.embed_tokens,
        )

        self._fully_shard(
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
            module=self.norm,
        )

        self._fully_shard(
            mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
            mp_policy=lm_head_mp_policy,
            reshard_after_forward=False,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
            module=self.lm_head,
        )

        # Shard MTP block if it exists
        if self.mtp_block is not None:
            for mtp_idx, mtp_layer in enumerate(self.mtp_block.layers):
                if self._should_recompute(None, mtp_idx=mtp_idx) or (
                    self.config.mtp_config is not None and self.config.mtp_config.share_weights
                ):  # share mtp head must recompute
                    mtp_layer = checkpoint_wrapper(mtp_layer, checkpoint_impl=CheckpointImpl.REENTRANT)
                self.mtp_block.layers[mtp_idx] = mtp_layer

                reshard_after_forward = mtp_idx != len(self.mtp_block.layers) - 1
                self._fully_shard(
                    mesh=self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
                    module=mtp_layer,
                )
                if mtp_idx == 0:
                    layer_next.set_modules_to_forward_prefetch([mtp_layer])  # type: ignore

            if self.config.mtp_config is not None and self.config.mtp_config.num_layers > 0:
                for prev_mtp_layer, next_mtp_layer in zip(
                    list(self.mtp_block.layers)[:-1],
                    list(self.mtp_block.layers)[1:],
                ):
                    prev_mtp_layer.set_modules_to_forward_prefetch([next_mtp_layer])  # type: ignore

        self._fully_shard(
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
        # Reduce-scatter gradients with pure SUM (no divide) for every sharded submodule; the
        # expert / replicated grads not covered by reduce-scatter are handled without division in
        # scale_and_reduce_grad. See BaseModel.set_gradient_reduce_sum.
        self.set_gradient_reduce_sum()
        return self

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        if self.config.ep_size > 1:
            return MOE_EP_COMPILE_CFG
        else:
            return MOE_NON_EP_COMPILE_CFG

    @property
    def need_update_bias(self) -> bool:
        router_config = self.config.router
        return isinstance(router_config, NoAuxRouterConfig) and router_config.router_bias_update_speed > 0

    @torch.no_grad  # type: ignore
    def scale_and_reduce_grad(self):
        ep_enabled = self.ep_mesh is not None and self.ep_mesh.size() > 1

        # Bucket gradients that need a cross-rank reduction by their target process
        # group. Each bucket is reduced with a single coalesced NCCL all_reduce
        # instead of one launch per parameter, which used to dominate latency for
        # models with many small replicated tensors.
        grads_by_group: dict[dist.ProcessGroup, list[torch.Tensor]] = {}

        for name, param in self.trainable_parameters():
            if param.grad is None:
                continue

            # Expert parameters live on a unique EP rank; their FSDP sharding is only over the
            # experts_fsdp sub-dim, already SUM-reduced by reduce-scatter. No cross-rank reduction
            # and no rescaling: under reduce-sum the local-component gradient is what we keep.
            if ep_enabled and ".experts" in name:
                continue

            if not isinstance(param, DTensor):
                continue

            replicate_dim_names = tuple(
                param.device_mesh.mesh_dim_names[i] for i, p in enumerate(param.placements) if isinstance(p, Replicate)
            )
            if not replicate_dim_names:
                continue

            # `DeviceMesh.get_group()` only supports a single mesh dimension,
            # so calling it directly on a multi-dim sub-mesh raises RuntimeError.
            # `_flatten()` collapses all Replicate dims into a 1D mesh whose
            # process group covers every rank across those dimensions, allowing
            # a single all_reduce regardless of how many Replicate dims exist.
            if len(replicate_dim_names) > 1:
                flat_mesh = param.device_mesh[replicate_dim_names]._flatten()
            else:
                # In the case that only one replicate dim, in pt2.8 _flatten is worked due to a bug.
                # in pt2.9.1 this bug is fixed and _flatten will raise error when the mesh is already 1D,
                # which means replicate_dim_names represents an existing single mesh dimension
                # so we directly get the submesh without flatten in this case.
                flat_mesh = param.device_mesh[replicate_dim_names[0]]

            grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            # Replicated params get no reduce-scatter; SUM their per-rank local-component grads
            # across the replicate group with NO pre-divide, matching the reduce-sum invariant.
            grads_by_group.setdefault(flat_mesh.get_group(), []).append(grad)  # type: ignore

        # One coalesced all_reduce per process group covers all replicated grads.
        for group, grads in grads_by_group.items():
            with dist._coalescing_manager(group=group):
                for grad in grads:
                    dist.all_reduce(grad, ReduceOp.SUM, group=group)

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
            self._world_mesh = model_mesh
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
                dist_param = nn.Parameter(
                    distribute_tensor(param, self.ep_mesh, [Replicate()]), requires_grad=param.requires_grad
                )
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

    def _should_recompute(
        self,
        layer_idx: int | None,
        mtp_idx: int | None,
    ) -> bool:
        """Determine if a layer should use gradient checkpointing
        (recomputation).

        The recomputation strategy treats decoder layers and MTP layers as a single
        sequence. The recompute_ratio is applied to the total layer count. The last
        layer in the entire model is never recomputed to avoid unnecessary overhead.

        Args:
            layer_idx (int | None): Index of the decoder layer (0-based). None if this
                is an MTP layer.
            mtp_idx (int | None): Index of the MTP layer (0-based). None if this is a
                decoder layer.

        Returns:
            bool: True if the layer should use gradient checkpointing, False otherwise.

        Example:
            Configuration: 7 decoder layers, 3 MTP layers, recompute_ratio=0.8
            - Total layers: 10
            - Recompute layers: int(10 * 0.8) = 8
            - Layer mapping:
                * Decoder 0-6 → global index 0-6 (7 layers)
                * MTP 0-2 → global index 7-9 (3 layers)
            - Recomputation decision:
                * Global 0-7 (decoder 0-6, MTP 0): recompute ✓
                * Global 8 (MTP 1): no recompute
                * Global 9 (MTP 2, last layer): no recompute (forced)
        """
        num_layers = self.config.num_hidden_layers
        if self.config.mtp_config is not None:
            mtp_layers = 1 if self.config.mtp_config.share_weights else self.config.mtp_config.num_layers
        else:
            mtp_layers = 0
        recompute_ratio = self.fsdp_config.recompute_ratio if self.fsdp_config is not None else 0.0

        total_layers = num_layers + mtp_layers
        num_recompute_layers = int(total_layers * recompute_ratio)

        # Determine the global layer index (0-based)
        if layer_idx is not None:
            # This is a decoder layer
            global_idx = layer_idx
        else:
            # This is an MTP layer (comes after all decoder layers)
            assert mtp_idx is not None, "Either layer_idx or mtp_idx must be provided"
            global_idx = num_layers + mtp_idx

        # Last layer is never recomputed
        if global_idx == total_layers - 1:
            return False

        # Recompute if within the recompute range
        return global_idx < num_recompute_layers

    # NOTE: Add this overload for inferring the return type for easier type checking and using
    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: SequenceContext,
        loss_ctx: MoELossContextDict | None,
    ) -> MoEModelOutputs: ...

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: list[SequenceContext],
        loss_ctx: list[MoELossContextDict],
    ) -> MoEModelOutputs: ...

    __call__ = nn.Module.__call__
