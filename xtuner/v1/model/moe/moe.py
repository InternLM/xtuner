# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os
import types
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self, Sequence, TypedDict, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import ConfigDict
from torch import nn
from torch.distributed._functional_collectives import all_reduce
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
    AuxLossInputs,
    BalancingLossConfig,
    BalancingLossContext,
    BaseLossContext,
    LMHeadLossContext,
    MTPLossContext,
    ZLossConfig,
    ZLossContext,
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


class MoEModelOutputs(ModelOutputs):
    router_logits: dict[str, torch.Tensor] | None = None
    router_weights: dict[str, torch.Tensor] | None = None
    balancing_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    # Optional so models with no MoE-routed layer this step (e.g. V4 smoke
    # configs where `num_hash_layers >= num_hidden_layers` makes every layer
    # hash-routed) can emit `None` instead of fabricating a tensor.
    # `internal_metrics.py` already short-circuits on `is None`.
    tokens_per_expert_global: torch.Tensor | None = None
    # Per-row routed-layer name aligned with ``tokens_per_expert_global`` rows (``"layer{idx}"``
    # or ``"mtp_layer{idx}"``), so ``update_bias`` resolves each row back to its router by name
    # instead of re-deriving the score-routed layer order positionally.
    aux_loss_layer_names: list[str] | None = None
    mtp_loss: torch.Tensor | None = None

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
    balancing: BalancingLossContext | None
    z_loss: ZLossContext | None
    mtp: list[BaseLossContext] | None


class LayerInput(TypedDict):
    """Everything a decoder layer needs for the single-sequence forward path.

    Produced by :meth:`MoE._prepare_hidden_states`. Subclasses that thread extra per-layer inputs
    extend this with their own fields (see DeepSeek-V4's ``V4LayerInput``); the base loop only reads
    the fields it declares, so subclass extras ride along untouched.
    """

    hidden_states: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]


class LayerInputMB(TypedDict):
    """Micro-batch counterpart of :class:`LayerInput`; every field is a per-
    microbatch list."""

    hidden_states_list: list[torch.Tensor]
    position_embeddings: list[tuple[torch.Tensor, torch.Tensor]]


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
        self.aux_loss: AuxLossContext = self.config.aux_loss_cfg.build(
            n_routed_experts=self.config.n_routed_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
        )

    def _maybe_offload_router(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.router_async_offload:
            return async_offload_to_cpu(tensor, self.offload_stream)
        return tensor

    def _z_loss_dist_token_count(
        self,
        z_ctx: list[ZLossContext] | ZLossContext | None,
        num_tokens_local: int,
        device: torch.device | str | int,
    ) -> tuple[torch.Tensor | None, int]:
        """Compute the cross-rank non-padding token count needed by the z-loss
        inline path.

        Returns ``(num_tokens_global, world_size)``. ``num_tokens_global`` is ``None`` (i.e. skip
        global averaging) when there is no z-loss context, when the configured z-loss is not
        global-average, or when no process group is initialized.
        """
        if z_ctx is None:
            return None, 1
        first = z_ctx[0] if isinstance(z_ctx, list) else z_ctx
        if not first.loss_cfg.z_loss_global_average or not dist.is_initialized():
            return None, 1
        n = torch.tensor(num_tokens_local, device=device, dtype=torch.int64)
        group = dist.group.WORLD
        assert group is not None
        n_global = all_reduce(n, "sum", group)
        return n_global, dist.get_world_size()

    def _extract_aux_loss_ctx(
        self,
        loss_ctx: list[MoELossContextDict] | MoELossContextDict | None,
    ) -> tuple[
        list[BalancingLossContext] | BalancingLossContext | None,
        list[ZLossContext] | ZLossContext | None,
    ]:
        if loss_ctx is None:
            return None, None

        if isinstance(loss_ctx, list):
            balancing_ctx: list[BalancingLossContext] = []
            z_ctx: list[ZLossContext] = []
            for ctx in loss_ctx:
                ctx_bal = ctx.get("balancing")
                if ctx_bal is not None:
                    balancing_ctx.append(ctx_bal)
                ctx_z = ctx.get("z_loss")
                if ctx_z is not None:
                    z_ctx.append(ctx_z)
            # Collapse empty fan-out lists to None so downstream guards
            # (`if ctx is None`, `_z_loss_dist_token_count`, AuxLoss.accumulate fan-out)
            # can treat "no context across any micro-batch" as the no-op case.
            return (balancing_ctx or None), (z_ctx or None)

        return loss_ctx.get("balancing"), loss_ctx.get("z_loss")

    def _build_aux_loss_inputs(
        self,
        loss_ctx: list[MoELossContextDict] | MoELossContextDict | None,
        mask: torch.Tensor,
    ) -> AuxLossInputs:
        """Build the per-forward aux-loss inputs from a routing mask.

        Both forward paths share this: split the aux-loss sub-contexts, locate non-pad tokens, and
        bundle the non-pad selection with the balancing / z-loss contexts and token bookkeeping that
        stay invariant across the decoder layers of one forward.

        Args:
            loss_ctx (list[MoELossContextDict] | MoELossContextDict | None): Per-microbatch loss
                context(s); ``None`` when no aux loss is configured.
            mask (torch.Tensor): Boolean token mask for this forward — a single sequence or the
                micro-batches concatenated — used to select non-pad tokens.

        Returns:
            AuxLossInputs: the per-forward aux-loss inputs (non-pad indices, sub-contexts, and token
            bookkeeping) reused at every ``accumulate`` and at ``finalize``.
        """
        balancing_ctx, z_ctx = self._extract_aux_loss_ctx(loss_ctx)
        nonpad_indices = torch.nonzero(mask, as_tuple=True)[1]
        non_pad_token = nonpad_indices.numel()
        num_tokens_global, z_world_size = self._z_loss_dist_token_count(z_ctx, non_pad_token, mask.device)
        return AuxLossInputs(
            nonpad_indices=nonpad_indices,
            balancing_ctx=balancing_ctx,
            z_ctx=z_ctx,
            num_tokens_local=non_pad_token,
            num_tokens_global=num_tokens_global,
            world_size=z_world_size,
        )

    def _should_compute_aux_loss(self, layer_idx: int) -> bool:
        # Extension hook for routers whose `router_results` is not shape-compatible with
        # `aux_loss.accumulate` (e.g. HashRouter emits a `[1]` dummy logits tensor since
        # it never scores). DeepSeekV4 overrides this for layers wired to HashRouter.
        # Default keeps the existing score-routed behaviour unchanged.
        return True

    def get_layer_by_name(self, name: str) -> MoEDecoderLayer:
        """Resolve a routed-layer name to the decoder layer that owns its
        router.

        Names follow the router-stats convention shared with ``MoEModelOutputs.router_logits``:
        ``"layer{idx}"`` for a main decoder layer, ``"mtp_layer{idx}"`` for an MTP depth (whose
        routed decoder layer is nested under ``MTPLayer.decoder_layer``). Centralizing the mapping
        keeps the naming a single contract shared by aux-loss accumulation and bias update.

        Args:
            name (str): Routed-layer name, ``"layer{idx}"`` or ``"mtp_layer{idx}"``.

        Returns:
            MoEDecoderLayer: The decoder layer whose ``gate.router`` produced that name's count rows.
        """
        mtp_prefix = "mtp_layer"
        if name.startswith(mtp_prefix):
            assert self.mtp_block is not None, "Got an MTP routed-layer name but this model has no MTP block."
            return cast(MoEDecoderLayer, self.mtp_block.layers[int(name[len(mtp_prefix) :])].decoder_layer)
        layer_prefix = "layer"
        assert name.startswith(layer_prefix), f"Unrecognized routed-layer name: {name!r}"
        return cast(MoEDecoderLayer, self.layers[name[len(layer_prefix) :]])

    @torch.no_grad()
    def update_bias(
        self,
        total_expert_counts_pre_iter: torch.Tensor,
        expected_loads: torch.Tensor,
        layer_names: list[str],
    ) -> None:
        """Implementation for the following paper:
        Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
        https://arxiv.org/abs/2408.15664

        TODO: refactor it later.
        """
        if self.config.freeze_routers:
            return

        bias_update_speed = cast(NoAuxRouterConfig, self.config.router).router_bias_update_speed
        # Each count row carries the routed-layer name it was accumulated for (``layer_names``,
        # aligned with the rows of ``total_expert_counts_pre_iter`` / ``expected_loads``). Resolving
        # the name back to its layer keeps the row-to-router mapping explicit instead of re-deriving
        # the score-routed layer order positionally.
        for layer_name, expected_load, current_loads in zip(layer_names, expected_loads, total_expert_counts_pre_iter):
            router = self.get_layer_by_name(layer_name).gate.router
            # Only NoAuxRouter owns a learnable bias; other routers (e.g. HashRouter) never
            # accumulate a count row, but guard anyway in case a future hybrid router does.
            if not isinstance(router, NoAuxRouter):
                continue

            load_diff = current_loads - expected_load
            update_mask = load_diff != 0  # 只更新需要调整的专家
            updates = torch.where(load_diff > 0, -bias_update_speed, bias_update_speed) * update_mask.float()

            router.e_score_correction_bias.add_(updates)

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
                - "balancing": Balancing loss context (if configured)
                - "z_loss": Z-loss context (if configured)
                - "mtp": MTP loss contexts (if configured)

        Note:
            Auxiliary loss contexts are built without parameters.
            All data is passed to forward() at runtime:
            - balancing_ctx(router_weights, n_routed_experts, num_experts_per_tok)
            - z_loss_ctx(router_logits)
        """
        # Build LM loss context
        _data_batch: list[dict] = data_batch  # type: ignore[assignment]
        res: list[dict] = super().build_loss_ctx_batch(_data_batch, sp_mesh)
        cu_seq_lens_list = [data["seq_ctx"].cu_seq_lens_k for data in data_batch]

        # Add auxiliary losses
        self._add_auxiliary_loss("balancing", self.config.balancing_loss_cfg, _data_batch, res)
        self._add_auxiliary_loss("z_loss", self.config.z_loss_cfg, _data_batch, res)

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

        # No-MoE-layer step (e.g. V4 smoke where every layer is hash-routed):
        # `tokens_per_expert_global` is None across micro-batches. Nothing to aggregate —
        # logs simply omit `maxvio` for the step and bias updates have nothing to apply.
        first_tokens_per_expert = batch_outputs[0]["tokens_per_expert_global"]
        if first_tokens_per_expert is None:
            return cast(MoEBatchForwardInfo, base_info)

        tokens_per_expert_global = torch.zeros_like(first_tokens_per_expert)
        for output in batch_outputs:
            tokens_per_expert_global += output["tokens_per_expert_global"]

        avg_count_load = tokens_per_expert_global.float().mean(1)
        max_load_i, _ = torch.max(tokens_per_expert_global, dim=1)
        maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
        maxvio = maxvio_all_layers.mean()
        logs_info["maxvio"] = maxvio.item()

        if self.need_update_bias:
            # Row order is identical across micro-batches (same layers accumulate every MB),
            # so the layer-name mapping from any output describes the summed count rows.
            aux_loss_layer_names = batch_outputs[0]["aux_loss_layer_names"]
            self.update_bias(tokens_per_expert_global, avg_count_load, aux_loss_layer_names)  # type: ignore

        moe_info = cast(MoEBatchForwardInfo, base_info)
        return moe_info

    def _prepare_hidden_states_mb(self, seq_ctx_list: list[SequenceContext]) -> LayerInputMB:
        """Embed every micro-batch into a per-MB hidden-state list (mb
        counterpart of :meth:`_prepare_hidden_states`).

        Embedding and rope run once on the concatenated batch (one kernel for all micro-batches)
        and are then split per micro-batch, so the decoder stack works in per-MB-list form
        throughout. Subclasses with a transformed residual stream or extra per-layer inputs
        override this and return the matching :class:`LayerInputMB` extended with their fields.

        Args:
            seq_ctx_list (list[SequenceContext]): One packed sequence per micro-batch.

        Returns:
            LayerInputMB: the per-microbatch ``hidden_states_list`` and per-MB rope list, threaded
            to :meth:`_call_decoder_layer_mb` for every layer.
        """
        n_mb = len(seq_ctx_list)
        if seq_ctx_list[0].input_ids is None:
            cat_hidden_states = torch.cat([ctx.inputs_embeds for ctx in seq_ctx_list], dim=1)  # type: ignore
        else:
            cat_input_ids = torch.cat([ctx.input_ids for ctx in seq_ctx_list], dim=1)  # type: ignore
            cat_hidden_states = self.embed_tokens(cat_input_ids)
        # M-RoPE position_ids are 3D [axes, batch, seq] for VL while text-only ones are 2D
        # [batch, seq]; -1 selects the seq dim in both cases.
        cat_position_ids = torch.cat([ctx.position_ids for ctx in seq_ctx_list], dim=-1)  # type: ignore
        cos, sin = self.rotary_emb(cat_hidden_states, cat_position_ids)  # type: ignore
        position_embeddings_list = list(zip(cos.chunk(n_mb, dim=1), sin.chunk(n_mb, dim=1)))
        # Clone the chunks so each micro-batch has its own storage: chunk() returns views into
        # one tensor, and async_save_on_cpu's in-place offload aliases that shared storage
        # (it produced nan grad norm). Dense-prefix models re-clone after the dense phase.
        hidden_states_list = [c.clone() for c in cat_hidden_states.chunk(n_mb, dim=1)]
        for seq_ctx in seq_ctx_list:
            self._mark_dynamic(seq_ctx)
        return {"hidden_states_list": hidden_states_list, "position_embeddings": position_embeddings_list}

    def _call_decoder_layer_mb(
        self,
        decoder_layer,
        idx: str,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        layer_input: LayerInputMB,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Invoke one MoE layer across all micro-batches (mb counterpart of
        :meth:`_call_decoder_layer`).

        The layer is called once carrying every micro-batch (it micro-batches internally for
        Domino EP) and returns a flat ``3 * n_mb`` tuple. Subclasses that thread extra per-layer
        inputs (from ``layer_input``) override this.

        Returns:
            tuple[list, list, list]: Per-micro-batch ``(hidden_states, router_logits, router_weights)``.
        """
        n_mb = len(hidden_states_list)
        layer_results = decoder_layer(
            *hidden_states_list,
            position_embeddings=layer_input["position_embeddings"],
            seq_ctx=seq_ctx_list,
        )
        return (
            list(layer_results[:n_mb]),
            list(layer_results[n_mb : 2 * n_mb]),
            list(layer_results[2 * n_mb :]),
        )

    def _micro_batch_forward(
        self,
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[MoELossContextDict],
        return_router_logits: bool = False,
    ) -> MoEModelOutputs:
        """Micro-batch forward pass for MoE model.

        Mirrors :meth:`_forward` but over a list of micro-batches. Dense-prefix layers (if any)
        run once on the concatenated batch; MoE layers run per micro-batch so Domino EP can
        overlap dispatch/combine across micro-batches.
        """
        if self.config.return_hidden_states:
            raise NotImplementedError

        assert len(seq_ctx_list) == len(loss_ctx_list), "seq_ctx and loss_ctx must have same length"
        n_mb = len(seq_ctx_list)

        layer_input = self._prepare_hidden_states_mb(seq_ctx_list)
        hidden_states_list = layer_input["hidden_states_list"]
        # Non-pad lookup + aux-loss inputs are invariant across layers; concatenate the per-MB masks
        # so accumulate sees the same global token set the single-sequence path does.
        cat_mask = torch.cat([ctx.mask for ctx in seq_ctx_list], dim=1)
        aux_inputs = self._build_aux_loss_inputs(loss_ctx_list, cat_mask)

        # Only the logits side is exposed per-MB (keep_router); the weights side is consumed by
        # accumulate only, so it is never stashed per-MB.
        keep_router = self.config.return_router_results or return_router_logits
        router_logits_list: list[dict[str, torch.Tensor]] = [{} for _ in range(n_mb)] if keep_router else []

        # Dense prefix runs on the concatenated batch; no-op when first_k_dense_replace == 0.
        hidden_states_list = self._run_dense_layers_mb(hidden_states_list, layer_input, seq_ctx_list)

        for idx, decoder_layer in self.layers.items():
            layer_idx = int(idx)
            if layer_idx < self.config.first_k_dense_replace:
                continue
            # One call per layer carrying all MBs. Offload stages the per-MB inputs on CPU; the
            # block ring is indexed within the MoE sub-stack (dense layers never offload).
            with self._activation_offload_ctx(
                layer_idx - self.config.first_k_dense_replace, hidden_states_list, reserve_pin_memory=True
            ):
                hidden_states_list, router_logits, router_weights = self._call_decoder_layer_mb(
                    decoder_layer, idx, hidden_states_list, seq_ctx_list, layer_input
                )

            if keep_router:
                for mb_idx in range(n_mb):
                    router_logits_list[mb_idx][f"layer{idx}"] = self._maybe_offload_router(router_logits[mb_idx])

            self._accumulate_layer_router_stats_mb(
                layer_idx, router_logits, router_weights, hidden_states_list, aux_inputs
            )

        # Assemble the result from each stage's slice. MTP runs on the per-MB streams before they
        # are concatenated for the final head.
        output: dict = {}
        output |= self._mtp_outputs_mb(
            hidden_states_list, seq_ctx_list, loss_ctx_list, layer_input, router_logits_list
        )

        # Cat once across MBs so the final norm + lm_head run as single kernels. ``_finalize_hidden_states``
        # is identity for the base; subclasses on an expanded residual stream collapse it here.
        cat_hidden_states = self.norm(self._finalize_hidden_states(torch.cat(hidden_states_list, dim=1)))
        lm_loss_ctx_list = [loss_ctx_dict["lm"] for loss_ctx_dict in loss_ctx_list]
        cat_loss_ctx = type(lm_loss_ctx_list[0]).cat(lm_loss_ctx_list)
        loss, (logits, extra_info) = self.lm_head(cat_hidden_states, cast(LMHeadLossContext, cat_loss_ctx))
        output["loss"] = loss.sum()
        moe_extra_info = ModelForwardExtraLogInfo()
        if extra_info:
            moe_extra_info.append(extra_info)
        output["extra_info"] = moe_extra_info

        output |= self._finalize_aux_loss_outputs(aux_inputs)
        if keep_router:
            output["router_logits"] = self._stack_router_logits_mb(router_logits_list, n_mb)

        return MoEModelOutputs(**output, logits=logits)

    def _accumulate_layer_router_stats_mb(
        self,
        layer_idx: int,
        router_logits: list[torch.Tensor],
        router_weights: list[torch.Tensor],
        hidden_states_list: list[torch.Tensor],
        aux_inputs: AuxLossInputs,
    ) -> None:
        """Concatenate a layer's per-MB router stats into the global token set
        and accumulate aux loss.

        No-op for layers that don't contribute (e.g. hash-routed). The z-loss carrier is pinned to MB0's hidden_states:
        all MBs converge into the same total_loss backward, so MB0's path traverses every aux-loss node exactly once.
        """
        if not self._should_compute_aux_loss(layer_idx):
            return
        hidden_states_list[0] = self.aux_loss.accumulate(
            router_weights=torch.cat(router_weights, dim=0),
            router_logits=torch.cat(router_logits, dim=0),
            hidden_states=hidden_states_list[0],
            layer_name=f"layer{layer_idx}",
            inputs=aux_inputs,
        )

    def _stack_router_logits_mb(
        self, router_logits_list: list[dict[str, torch.Tensor]], n_mb: int
    ) -> dict[str, torch.Tensor]:
        """Stack the retained per-MB router logits into ``[1, n_mb, ...]``
        tensors keyed by layer name."""
        # TODO: Returning router logits is costly.
        return {
            layer_name: torch.stack(
                [router_logits_list[mb_idx][layer_name].detach() for mb_idx in range(n_mb)], dim=0
            ).unsqueeze(0)
            for layer_name in router_logits_list[0]
        }

    def _run_dense_layers_mb(
        self, hidden_states_list: list[torch.Tensor], layer_input: LayerInputMB, seq_ctx_list: list[SequenceContext]
    ) -> list[torch.Tensor]:
        """Run the dense-prefix layers on the concatenated batch, returning a
        fresh per-MB list.

        Dense layers carry no routing and don't need per-MB separation, so they run once on the
        concatenated sequence (one call rather than ``n_mb``). No-op when there is no dense prefix,
        so models like DeepSeek-V4 (``first_k_dense_replace == 0``) never pay the cat/split round
        trip.
        """
        dense_count = self.config.first_k_dense_replace
        if dense_count <= 0:
            return hidden_states_list
        n_mb = len(seq_ctx_list)
        cat_hidden_states = torch.cat(hidden_states_list, dim=1)
        pe_list = layer_input["position_embeddings"]
        cat_position_embeddings = (
            torch.cat([pe[0] for pe in pe_list], dim=1),
            torch.cat([pe[1] for pe in pe_list], dim=1),
        )
        cat_seq_ctx = SequenceContext.cat(seq_ctx_list)
        self._mark_dynamic(cat_seq_ctx)
        for idx, decoder_layer in self.layers.items():
            if int(idx) >= dense_count:
                break
            cat_hidden_states = decoder_layer(
                cat_hidden_states,
                position_embeddings=cat_position_embeddings,
                seq_ctx=cat_seq_ctx,
            )
        # Clone the chunks: async_save_on_cpu is not friendly with chunk's shared storage (it
        # produced nan grad norm), so each micro-batch needs its own storage downstream.
        return [c.clone() for c in cat_hidden_states.chunk(n_mb, dim=1)]

    def _mtp_outputs_mb(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[MoELossContextDict],
        layer_input: LayerInputMB,
        router_logits_list: list[dict[str, torch.Tensor]],
    ) -> dict:
        """Run the MTP block over every micro-batch and return ``{"mtp_loss":
        ...}`` (empty when off).

        Per-depth router logits are appended into ``router_logits_list`` (per-MB) when it is retained
        (non-empty). The block runs on all micro-batches so Domino EP can overlap dispatch/combine
        across them at each depth; per-MB aggregation skips micro-batches whose loss context is absent.
        """
        if self.mtp_block is None:
            return {}
        assert self.config.mtp_config is not None

        # Build a per-microbatch SequenceContext clone for MTP.
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
            position_embeddings=layer_input["position_embeddings"],
            seq_ctx=mtp_seq_ctx_list,
        )

        mtp_losses = torch.tensor(0.0, device=DEVICE)
        has_mtp_loss = False
        for micro_batch_idx, (loss_ctx_dict, mtp_outputs) in enumerate(zip(loss_ctx_list, mtp_outputs_per_mb)):
            mtp_loss_ctx_list = loss_ctx_dict.get("mtp")
            if mtp_loss_ctx_list is None:
                continue

            micro_batch_mtp_losses = torch.tensor(0.0, device=DEVICE)
            for mtp_idx, (mtp_hidden, mtp_ctx) in enumerate(zip(mtp_outputs, mtp_loss_ctx_list)):
                mtp_hidden_states, mtp_router_results, _ = mtp_hidden
                mtp_loss, _ = self.lm_head(mtp_hidden_states, cast(MTPLossContext, mtp_ctx))
                micro_batch_mtp_losses += mtp_loss

                if router_logits_list:
                    router_logits_list[micro_batch_idx][f"mtp_layer{mtp_idx}"] = mtp_router_results

            mtp_losses += micro_batch_mtp_losses / len(mtp_loss_ctx_list)
            has_mtp_loss = True

        if not has_mtp_loss:
            return {}
        return {"mtp_loss": mtp_losses * self.config.mtp_config.loss_scaling_factor}

    def _prepare_hidden_states(self, seq_ctx: SequenceContext) -> LayerInput:
        """Embed inputs, build shared position embeddings, and mark dynamic
        shapes.

        Template-method seam for the single-sequence :meth:`_forward` path. Subclasses
        whose decoder layers consume a transformed hidden state (e.g. an extra
        residual-stream axis) or need extra per-layer inputs (e.g. a second rope)
        override this and return the matching :class:`LayerInput` extended with their fields.

        Args:
            seq_ctx (SequenceContext): The packed sequence to embed.

        Returns:
            LayerInput: the initial ``hidden_states`` and the shared position embeddings, threaded
            to :meth:`_call_decoder_layer` for every layer.
        """
        input_ids = seq_ctx.input_ids
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
        assert seq_ctx.position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, seq_ctx.position_ids)
        self._mark_dynamic(seq_ctx)
        return {"hidden_states": hidden_states, "position_embeddings": position_embeddings}

    def _activation_offload_ctx(
        self, block_idx: int, tensors: list[torch.Tensor], *, reserve_pin_memory: bool = False
    ) -> "contextlib.AbstractContextManager":
        """Activation-offload context that stages ``tensors`` (a layer's
        inputs) on CPU.

        Returns a null context when ``XTUNER_ACTIVATION_OFFLOAD`` is off, so callers can ``with``
        it unconditionally instead of branching on the env flag. Both the single-sequence and
        micro-batch paths build their offload window through here.

        Args:
            block_idx (int): Position in the offload buffer ring (the MoE sub-stack index).
            tensors (list[torch.Tensor]): The layer-input tensors to stage; the offload check
                matches any of them by storage pointer.
            reserve_pin_memory (bool): Reuse a pinned CPU buffer per (block, tensor) across steps.

        Returns:
            contextlib.AbstractContextManager: The offload window, or a null context when off.
        """
        if (
            int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) != 1
            or block_idx + self.config.first_k_dense_replace == len(self.layers) - 1
        ):
            return contextlib.nullcontext()
        return async_save_on_cpu(
            h2d_stream=self.offload_stream,
            d2h_stream=self.offload_stream,
            block_idx=block_idx,
            group="text",
            custom_check_fn=lambda x, _ts=tensors: x.data_ptr() in [t.data_ptr() for t in _ts],
            prefetch=True,
            reserve_pin_memory=reserve_pin_memory,
        )

    def _decoder_layer_offload_ctx(self, decoder_layer, idx: str, layer_input: torch.Tensor):
        """Activation-offload context wrapping one single-sequence decoder-
        layer call.

        Dense layers never offload; otherwise delegate to :meth:`_activation_offload_ctx`.
        Subclasses that use a different offload stream or bucketing override this.
        """
        if int(idx) < self.config.first_k_dense_replace:
            return contextlib.nullcontext()
        return self._activation_offload_ctx(int(idx), [layer_input])

    def _call_decoder_layer(
        self,
        decoder_layer,
        idx: str,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        layer_input: LayerInput,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Invoke one decoder layer and normalise the return to a triple.

        Default: dense layers (``idx < first_k_dense_replace``) return only the hidden
        state; MoE layers return ``(hidden, router_logits, router_weights)``. Subclasses
        that thread extra per-layer inputs (from ``layer_input``) or whose layers always
        return router stats override this. ``hidden_states`` is the live residual stream (the
        ``layer_input["hidden_states"]`` field only carries the pre-loop seed).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            ``(hidden_states, router_logits, router_weights)`` with the router entries
            ``None`` for layers that do not route.
        """
        position_embeddings = layer_input["position_embeddings"]
        if int(idx) < self.config.first_k_dense_replace:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )
            return hidden_states, None, None
        hidden_states, router_results, router_weights = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
        return hidden_states, router_results, router_weights

    def _post_layer(self, hidden_states: torch.Tensor, idx: str, seq_ctx: SequenceContext) -> torch.Tensor:
        """Per-layer post-processing hook (default: identity).

        Runs after each decoder layer and its aux-loss accumulation. Subclasses that inject per-layer side inputs (e.g.
        multi-scale visual embeds) override this.
        """
        return hidden_states

    def _finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform the stacked hidden state before the final norm (default:
        identity).

        Subclasses whose decoder stack runs on an expanded residual stream collapse it
        back to ``[B, S, D]`` here.
        """
        return hidden_states

    def _should_finalize_aux_loss(self) -> bool:
        """Whether to run :meth:`AuxLossContext.finalize` after the stack
        (default: True).

        Subclasses where no layer accumulates routing stats (e.g. an all-hash-routed sub-stack) override this to skip
        the finalize, which would otherwise raise.
        """
        return True

    def _forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: MoELossContextDict | None,
        return_router_logits: bool = False,
    ) -> MoEModelOutputs:
        layer_input = self._prepare_hidden_states(seq_ctx)
        hidden_states = layer_input["hidden_states"]
        # Non-pad lookup + aux-loss inputs are invariant across layers, so hoist them out of the
        # per-layer accumulate path.
        aux_inputs = self._build_aux_loss_inputs(loss_ctx, seq_ctx.mask)

        # Router stats are retained only when a downstream consumer asked (config flag or kwarg);
        # otherwise the per-layer collection and its optional D2H offload are skipped entirely.
        keep_router = self.config.return_router_results or return_router_logits
        router_logits: dict[str, torch.Tensor] | None = {} if keep_router else None
        router_weights: dict[str, torch.Tensor] | None = {} if keep_router else None
        collected_hidden: list[torch.Tensor] | None = [] if self.config.return_hidden_states else None

        for idx, decoder_layer in self.layers.items():
            with self._decoder_layer_offload_ctx(decoder_layer, idx, hidden_states):
                hidden_states, router_logits_l, router_weights_l = self._call_decoder_layer(
                    decoder_layer, idx, hidden_states, seq_ctx, layer_input
                )

            # Dense layers report ``None`` router stats — skip the routing bookkeeping.
            if router_logits_l is not None:
                # Routed layers return both stats together (dense layers return neither).
                assert router_weights_l is not None
                if router_logits is not None:
                    router_logits[f"layer{idx}"] = self._maybe_offload_router(router_logits_l)
                    router_weights[f"layer{idx}"] = self._maybe_offload_router(router_weights_l)  # type: ignore[index]

                if self._should_compute_aux_loss(int(idx)):
                    hidden_states = self.aux_loss.accumulate(
                        router_weights=router_weights_l,
                        router_logits=router_logits_l,
                        hidden_states=hidden_states,
                        layer_name=f"layer{idx}",
                        inputs=aux_inputs,
                    )

            hidden_states = self._post_layer(hidden_states, idx, seq_ctx)
            if collected_hidden is not None:
                collected_hidden.append(hidden_states)

        layer_hidden_states = hidden_states
        hidden_states = self.norm(self._finalize_hidden_states(hidden_states))

        # Assemble the result from each stage's returned slice. Call order matters for the aux-loss
        # side effects only: MTP accumulates its depths (inside ``_mtp_outputs``) before finalize.
        output: dict = {}
        output |= self._lm_head_outputs(hidden_states, loss_ctx)
        output |= self._mtp_outputs(
            layer_hidden_states,
            seq_ctx,
            loss_ctx,
            layer_input["position_embeddings"],
            aux_inputs,
            router_logits,
            router_weights,
        )
        output |= self._finalize_aux_loss_outputs(aux_inputs)

        # Detach + add a leading dim only when router logits were retained; MTP depths were appended
        # into the same dict above, so this covers layer and mtp entries alike.
        if router_logits is not None:
            router_logits = {name: rl.detach().unsqueeze(0) for name, rl in router_logits.items()}
        output["router_logits"] = router_logits
        output["router_weights"] = router_weights
        if collected_hidden is not None:
            output["hidden_states"] = collected_hidden

        return MoEModelOutputs(**output)

    def _lm_head_outputs(self, hidden_states: torch.Tensor, loss_ctx: "MoELossContextDict | None") -> dict:
        """Run the LM head and return its output slice: ``loss``, ``logits``, ``extra_info``."""
        lm_loss_ctx = loss_ctx["lm"] if loss_ctx is not None else None
        loss, (logits, extra_info) = self.lm_head(hidden_states, lm_loss_ctx)  # type: ignore
        return {"loss": loss, "logits": logits, "extra_info": extra_info}

    def _finalize_aux_loss_outputs(self, aux_inputs: AuxLossInputs) -> dict:
        """Finalize the split aux losses and return their output slice.

        Skips the finalize when no layer accumulated routing stats (e.g. an all-hash-routed step),
        still reporting ``tokens_per_expert_global`` / ``aux_loss_layer_names`` as ``None`` so
        downstream logging short-circuits. ``balancing_loss`` / ``z_loss`` are present only when
        configured.
        """
        if not self._should_finalize_aux_loss():
            return {"tokens_per_expert_global": None, "aux_loss_layer_names": None}
        balancing_loss, z_loss, tokens_per_expert_global, aux_loss_layer_names = self.aux_loss.finalize(
            inputs=aux_inputs,
        )
        outputs: dict = {
            "tokens_per_expert_global": tokens_per_expert_global,
            "aux_loss_layer_names": aux_loss_layer_names,
        }
        if balancing_loss is not None:
            outputs["balancing_loss"] = balancing_loss
        if z_loss is not None:
            outputs["z_loss"] = z_loss
        return outputs

    def _mtp_outputs(
        self,
        layer_hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        loss_ctx: "MoELossContextDict | None",
        position_embeddings,
        aux_inputs: AuxLossInputs,
        router_logits: dict[str, torch.Tensor] | None,
        router_weights: dict[str, torch.Tensor] | None,
    ) -> dict:
        """Run the MTP block and return ``{"mtp_loss": ...}`` (empty dict when
        MTP is off).

        Per-depth router stats are appended into the caller's retained ``router_logits`` /
        ``router_weights`` collections when those are not ``None``. MTP uses its own mask, so it
        recomputes non-pad token counts; each depth's z-loss is injected before its lm_head so
        backward through ``mtp_loss`` traverses the AuxLossScaler node and releases that depth's
        logsumexp activations.
        """
        if self.mtp_block is None or loss_ctx is None:
            return {}
        mtp_loss_ctx_list = loss_ctx.get("mtp")
        if mtp_loss_ctx_list is None:
            return {}

        input_ids = seq_ctx.input_ids
        assert seq_ctx.position_ids is not None
        mtp_seq_ctx = seq_ctx.copy(
            input_ids=input_ids.clone() if input_ids is not None else None,
            position_ids=seq_ctx.position_ids.clone(),
            inputs_embeds=seq_ctx.inputs_embeds.clone() if seq_ctx.inputs_embeds is not None else None,
        )
        mtp_nonpad_indices = torch.nonzero(mtp_seq_ctx.mask, as_tuple=True)[1]
        mtp_non_pad_token = mtp_nonpad_indices.numel()
        mtp_num_tokens_global, mtp_z_world_size = self._z_loss_dist_token_count(
            aux_inputs.z_ctx, mtp_non_pad_token, mtp_seq_ctx.mask.device
        )
        # Same balancing / z-loss contexts as the main path, but MTP re-masks its inputs, so it
        # carries its own non-pad token counts.
        mtp_inputs = replace(
            aux_inputs,
            nonpad_indices=mtp_nonpad_indices,
            num_tokens_local=mtp_non_pad_token,
            num_tokens_global=mtp_num_tokens_global,
            world_size=mtp_z_world_size,
        )

        mtp_outputs = self.mtp_block(
            layer_hidden_states,
            embed_tokens_fn=self.embed_tokens,
            position_embeddings=position_embeddings,
            seq_ctx=mtp_seq_ctx,
        )

        mtp_losses = torch.tensor(0.0, device=DEVICE)
        for idx, (mtp_hidden, mtp_ctx) in enumerate(zip(mtp_outputs, mtp_loss_ctx_list)):
            mtp_hidden_states, mtp_router_results, mtp_router_weights = mtp_hidden

            if router_logits is not None:
                router_logits[f"mtp_layer{idx}"] = mtp_router_results
                router_weights[f"mtp_layer{idx}"] = mtp_router_weights  # type: ignore[index]
            mtp_hidden_states = self.aux_loss.accumulate(
                router_weights=mtp_router_weights,
                router_logits=mtp_router_results,
                hidden_states=mtp_hidden_states,
                layer_name=f"mtp_layer{idx}",
                inputs=mtp_inputs,
            )
            mtp_loss, _ = self.lm_head(mtp_hidden_states, cast(MTPLossContext, mtp_ctx))
            mtp_losses += mtp_loss

        mtp_losses = mtp_losses / len(mtp_loss_ctx_list)
        return {"mtp_loss": mtp_losses * self.config.mtp_config.loss_scaling_factor}  # type: ignore

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

        # Patch nn.Embedding and nn.Linear forwards on the non-MoE backbone so that
        # weights pre-wrapped as Replicate-on-ep DTensor by `_replicate_other_params`
        # get .to_local()'d before F.embedding / F.linear. Without this, callers that
        # pass plain-Tensor activations (V4 HC helpers, attn_block / ffn_block hops
        # that bypass MoEDecoderLayer.forward's torch.compile coercion) crash with
        # "got mixed torch.Tensor and DTensor". The walk skips MoEBlock to leave the
        # expert / gate / shared-expert weights untouched — those are ep-sharded on
        # purpose by the MoE dispatch path.
        def _patch_non_moe_block_linears_and_embeds(module: nn.Module) -> None:
            from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEBlock

            if isinstance(module, MoEBlock):
                return
            # Patch only when the module's forward is the stock implementation —
            # subclasses with a custom forward (e.g. xtuner.v1.module.lm_head.LMHead
            # inherits nn.Linear with a 2-arg forward(hidden_states, loss_ctx))
            # must not be replaced. `isinstance` is needed (instead of `type is`)
            # because FSDP-managed modules can be reachable through synthetic
            # subclasses without ever overriding `forward`.
            if isinstance(module, nn.Embedding) and module.__class__.forward is nn.Embedding.forward:
                module.forward = types.MethodType(self.patched_emb_forward, module)  # type: ignore
            elif isinstance(module, nn.Linear) and module.__class__.forward is nn.Linear.forward:
                module.forward = types.MethodType(self.patched_linear_forward, module)  # type: ignore
            for child in module.children():
                _patch_non_moe_block_linears_and_embeds(child)

        _patch_non_moe_block_linears_and_embeds(self)

        self._to_empty_meta()
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

            # Expert parameters live on a unique EP rank, so no cross-rank reduction
            # is needed — just rescale by `ep_size` to keep the effective average.
            if ep_enabled and ".experts" in name:
                param.grad.div_(self.ep_mesh.size())  # type: ignore
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
            # Pre-scale locally so the SUM all_reduce below yields the mean across replicas.
            grad.div_(flat_mesh.size())  # type: ignore
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

    @staticmethod
    def patched_linear_forward(self, input):
        # Same shape as `patched_emb_forward`. After FSDP unshards on the fsdp_mesh
        # dim, an ep-replicated parameter is still a DTensor with Replicate placement
        # on ep_mesh. F.linear's DTensor dispatch only auto-promotes scalar plain
        # tensors, so callers with plain-Tensor activations crash. .to_local() is a
        # no-op for plain Tensors; for Replicate DTensors it costs nothing (every
        # rank already holds the full copy locally).
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
        else:
            w = self.weight
        if isinstance(self.bias, DTensor):
            b = self.bias.to_local()
        else:
            b = self.bias
        return F.linear(input, w, b)

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
