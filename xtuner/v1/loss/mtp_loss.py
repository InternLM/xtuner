# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce
from torch.utils.checkpoint import checkpoint

from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class MTPLossKwargs(CELossKwargs):
    """Keyword arguments for MTP loss computation.

    Inherits all fields from CELossKwargs. The ``shifted_labels`` field is
    expected to be pre-rolled by ``MTPLossConfig.build()`` before this object
    is constructed, so no additional fields are required.

    Args:
        shifted_labels (torch.Tensor): The shifted and rolled labels for MTP
            loss computation.
        loss_weight (torch.Tensor | None): Per-token loss weight.
        logprobs (torch.Tensor | None): Log probabilities
            for KL loss computation in RL training. When present, MTPLossContext
            computes KL loss instead of CE loss.
    """

    logprobs: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> "MTPLossKwargs":
        super().sp_split(sp_mesh)
        if self.logprobs is not None:
            self.logprobs = sp_split(self.logprobs, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        return self

    def to(self, device: torch.device | str) -> "MTPLossKwargs":
        super().to(device)
        if self.logprobs is not None:
            self.logprobs = self.logprobs.to(device)
        return self


class MTPLossConfig(CELossConfig):
    """Loss configuration for Multi-Token Prediction (MTP).

    Extends ``CELossConfig`` with a ``mtp_depth`` field that controls how many
    additional positions the labels are rolled during ``build()``. This class
    is intended for internal use by the model and is not exposed to users.

    Args:
        mtp_depth (int): 1-indexed MTP layer depth. The first MTP layer uses
            ``mtp_depth=1`` (shift=-1 on top of the existing label shift).
        detach_mtp_lm_head_weight (bool): Whether to detach the LM head weight.
            This is used in RL training. Default is False.
    """

    mtp_depth: int
    detach_mtp_lm_head_weight: bool = False

    @property
    def loss_ctx_cls(self) -> type["MTPLossContext"]:
        return MTPLossContext

    @property
    def _loss_kwargs_cls(self) -> type["MTPLossKwargs"]:
        return MTPLossKwargs

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "MTPLossContext | None":
        """Build MTPLossContext from data dict.

        Rolls ``shifted_labels`` (and optionally ``logprobs``) by
        ``-mtp_depth`` positions (per-sequence, respecting packed-sequence
        boundaries) before constructing the loss context. The roll is performed
        on the full sequence prior to any sequence-parallel split so that
        boundary positions and ``cu_seq_lens`` are always consistent.

        Args:
            data (dict): Data dict containing loss-related fields.
                Required keys: ``shifted_labels``, ``seq_ctx``.
                Optional keys: ``logprobs``.
            sp_mesh (DeviceMesh | None): Sequence parallel mesh.

        Returns:
            MTPLossContext | None: Built loss context, or ``None`` if
                ``shifted_labels`` is not present in ``data``.
        """
        # TODO: Should move the common utils function to public package to avoid from circular import.
        from xtuner.v1.module.mtp.utils import roll_packed_tensor

        if "shifted_labels" not in data:
            return None

        shifted_labels = data["shifted_labels"]
        cu_seq_lens = data["seq_ctx"].cu_seq_lens_k

        # cu_seq_lens[-1] may be larger than shifted_labels.shape[-1] when seq_ctx
        # was split for sequence parallelism (padding is added to make the sequence
        # length a multiple of sp_size). Pad with -100 so roll_packed_tensor does
        # not go out of bounds.
        padded_len = int(cu_seq_lens[-1].item())
        seq_len = shifted_labels.shape[-1]
        if padded_len > seq_len:
            pad = torch.full(
                (*shifted_labels.shape[:-1], padded_len - seq_len),
                fill_value=-100,
                dtype=shifted_labels.dtype,
                device=shifted_labels.device,
            )
            shifted_labels = torch.cat([shifted_labels, pad], dim=-1)

        rolled = roll_packed_tensor(shifted_labels, cu_seq_lens, shifts=-self.mtp_depth, dim=-1, fill_value=-100)

        # Roll logprobs by the same amount as shifted_labels
        logprobs = data.get("logprobs", None)
        rolled_logprobs = None
        if logprobs is not None:
            rp_seq_len = logprobs.shape[-1]
            if padded_len > rp_seq_len:
                rp_pad = torch.zeros(
                    (*logprobs.shape[:-1], padded_len - rp_seq_len),
                    dtype=logprobs.dtype,
                    device=logprobs.device,
                )
                logprobs = torch.cat([logprobs, rp_pad], dim=-1)
            rolled_logprobs = roll_packed_tensor(logprobs, cu_seq_lens, shifts=-self.mtp_depth, dim=-1, fill_value=0)

        loss_kwargs = MTPLossKwargs(
            shifted_labels=rolled,
            logprobs=rolled_logprobs,
        ).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)

        return MTPLossContext(self, loss_kwargs)


class MTPLossContext(LMHeadLossContext):
    """Loss context for Multi-Token Prediction (MTP).

    Supports two modes:
    - **CE mode** (default): Standard cross-entropy loss on rolled labels.
      Used during SFT/pretraining.
    - **KL mode**: When ``logprobs`` is available (RL training),
      computes KL divergence between MTP's log-probabilities and the
      rolled rollout log-probabilities.

    Both modes support chunk mode for memory-efficient computation via the
    base class's ``forward() → eager_mode()/chunk_mode() → loss_fn()`` dispatch.

    Args:
        loss_cfg (MTPLossConfig): The MTP loss configuration.
        loss_kwargs (MTPLossKwargs): Pre-rolled keyword arguments for loss
            computation.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        if self.loss_cfg.detach_mtp_lm_head_weight:
            head_weight = head_weight.detach()
            head_bias = head_bias.detach() if head_bias is not None else None
        # Dispatch to eager_mode/chunk_mode via base class, which calls loss_fn per chunk
        return super().forward(hidden_states, head_weight, head_bias)

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: MTPLossKwargs,  # type: ignore[override]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        if loss_kwargs.logprobs is not None:
            return self._kl_loss_fn(hidden_states, head_weight, head_bias, loss_kwargs)
        return super().loss_fn(hidden_states, head_weight, head_bias, loss_kwargs)

    def _kl_loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: MTPLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Compute KL loss between MTP logprobs and rolled rollout logprobs.

        Called per-chunk in chunk mode, so tensors here may be a slice of the full sequence.
        """
        from xtuner.v1.rl.loss import kl_penalty
        from xtuner.v1.rl.utils import gather_logprobs

        logits = F.linear(hidden_states, head_weight, head_bias).float()

        shifted_labels = loss_kwargs.shifted_labels
        loss_weight = loss_kwargs.loss_weight
        rollout_logprobs = loss_kwargs.logprobs

        assert rollout_logprobs is not None
        assert loss_weight is not None, "loss_weight can not be None"

        mtp_logprobs = gather_logprobs(logits, shifted_labels)
        loss_weight = loss_weight.flatten()

        kl_loss = kl_penalty(
            mtp_logprobs.flatten(),
            rollout_logprobs.flatten(),
            loss_weight,
            "low_var_kl",
        )

        return kl_loss, (None, {})


class MTPE2ETVLossKwargs(CELossKwargs):
    """Inputs used to mask and globally normalize the joint MTP TV loss."""

    sp_mesh: DeviceMesh | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> "MTPE2ETVLossKwargs":
        super().sp_split(sp_mesh)
        self.sp_mesh = sp_mesh
        return self


class MTPE2ETVLossConfig(CELossConfig):
    """End-to-end TV objective for multi-step rejection-sampling acceptance.

    The objective follows Eq. 13 of https://arxiv.org/abs/2606.12370. Unlike
    :class:`MTPLossConfig`, one context covers every logical MTP depth because
    the prefix-acceptance products couple the per-depth distribution overlaps.
    """

    num_steps: int
    detach_mtp_lm_head_weight: bool = False

    @property
    def loss_ctx_cls(self) -> type["MTPE2ETVLossContext"]:
        return MTPE2ETVLossContext

    @property
    def _loss_kwargs_cls(self) -> type["MTPE2ETVLossKwargs"]:
        return MTPE2ETVLossKwargs

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "MTPE2ETVLossContext | None":
        from xtuner.v1.module.mtp.utils import roll_packed_tensor

        if "shifted_labels" not in data:
            return None

        shifted_labels = data["shifted_labels"]
        cu_seq_lens = data["seq_ctx"].cu_seq_lens_k
        padded_len = int(cu_seq_lens[-1].item())
        seq_len = shifted_labels.shape[-1]
        if padded_len > seq_len:
            pad = torch.full(
                (*shifted_labels.shape[:-1], padded_len - seq_len),
                fill_value=self.ignore_idx,
                dtype=shifted_labels.dtype,
                device=shifted_labels.device,
            )
            shifted_labels = torch.cat([shifted_labels, pad], dim=-1)

        # Eq. 13 assumes a fixed gamma. Restrict training to starting positions
        # for which every one of the configured MTP steps has a supervised target.
        valid_mask = torch.ones_like(shifted_labels, dtype=torch.bool)
        for depth in range(1, self.num_steps + 1):
            rolled_labels = roll_packed_tensor(
                shifted_labels,
                cu_seq_lens,
                shifts=-depth,
                dim=-1,
                fill_value=self.ignore_idx,
            )
            valid_mask.logical_and_(rolled_labels != self.ignore_idx)

        # Reuse LMHeadLossContext.build_batches() for token/sample weighting and
        # DP/SP/global-batch calibration. Label values are immaterial to TV loss;
        # only ignore_idx positions are consumed by the weighting code.
        mask_labels = torch.where(
            valid_mask,
            torch.zeros_like(shifted_labels),
            torch.full_like(shifted_labels, self.ignore_idx),
        )
        loss_kwargs = MTPE2ETVLossKwargs(shifted_labels=mask_labels, sp_mesh=sp_mesh).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        return MTPE2ETVLossContext(self, loss_kwargs)


def _shift_target_hidden_states(
    target_hidden_states: torch.Tensor,
    num_steps: int,
    sp_mesh: DeviceMesh | None,
) -> list[torch.Tensor]:
    """Left-shift target states with a small halo across sequence-parallel ranks."""

    target_hidden_states = target_hidden_states.detach()
    batch_size, local_length, hidden_size = target_hidden_states.shape
    if local_length == 0:
        raise ValueError("MTP e2e TV loss requires a non-empty local sequence shard.")

    sp_size = 1 if sp_mesh is None else sp_mesh.size()
    if sp_size == 1:
        halo = target_hidden_states.new_zeros((batch_size, num_steps, hidden_size))
        extended = torch.cat((target_hidden_states, halo), dim=1)
    elif local_length >= num_steps:
        assert dist.is_initialized(), "Sequence parallelism requires torch.distributed to be initialized."
        assert sp_mesh is not None
        group = sp_mesh.get_group()
        sp_rank = dist.get_rank(group)
        prefix = target_hidden_states[:, :num_steps].contiguous()
        prefixes = [torch.empty_like(prefix) for _ in range(sp_size)]
        dist.all_gather(prefixes, prefix, group=group)
        halo = prefixes[sp_rank + 1] if sp_rank + 1 < sp_size else torch.zeros_like(prefix)
        extended = torch.cat((target_hidden_states, halo), dim=1)
    else:
        # This path is only relevant to tiny tests or extremely short shards.
        # Gather complete shards because one horizon can cross several ranks.
        assert dist.is_initialized(), "Sequence parallelism requires torch.distributed to be initialized."
        assert sp_mesh is not None
        group = sp_mesh.get_group()
        sp_rank = dist.get_rank(group)
        shards = [torch.empty_like(target_hidden_states) for _ in range(sp_size)]
        dist.all_gather(shards, target_hidden_states.contiguous(), group=group)
        global_hidden_states = torch.cat(shards, dim=1)
        start = sp_rank * local_length
        tail = global_hidden_states[:, start : start + local_length + num_steps]
        if tail.shape[1] < local_length + num_steps:
            tail = torch.cat(
                (
                    tail,
                    target_hidden_states.new_zeros(
                        (batch_size, local_length + num_steps - tail.shape[1], hidden_size)
                    ),
                ),
                dim=1,
            )
        extended = tail

    return [extended[:, depth : depth + local_length] for depth in range(1, num_steps + 1)]


class MTPE2ETVLossContext(LMHeadLossContext):
    """Compute exact full-vocabulary e2e TV loss over all MTP depths."""

    loss_cfg: MTPE2ETVLossConfig
    loss_kwargs: MTPE2ETVLossKwargs

    @staticmethod
    def _tv_overlap(
        draft_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        # The target distribution is a teacher for this objective. Gradients are
        # defined only with respect to the draft distribution (Bebop Appendix F).
        with torch.no_grad():
            target_logits = F.linear(target_hidden_states, head_weight.detach()).float()
            target_probs = F.softmax(target_logits, dim=-1)

        draft_logits = F.linear(draft_hidden_states, head_weight).float()
        draft_probs = F.softmax(draft_logits, dim=-1)
        return torch.minimum(target_probs, draft_probs).sum(dim=-1).clamp_(0.0, 1.0)

    def forward(  # type: ignore[override]
        self,
        hidden_states: tuple[torch.Tensor, Sequence[torch.Tensor]],
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[None, dict[str, Any]]]:
        if head_bias is not None:
            raise NotImplementedError("MTP e2e TV loss does not support head_bias.")

        target_hidden_states, draft_hidden_states = hidden_states
        if len(draft_hidden_states) != self.loss_cfg.num_steps:
            raise ValueError(
                f"Expected {self.loss_cfg.num_steps} MTP outputs for e2e TV loss, got {len(draft_hidden_states)}."
            )
        if self.loss_cfg.detach_mtp_lm_head_weight:
            head_weight = head_weight.detach()

        shifted_targets = _shift_target_hidden_states(
            target_hidden_states,
            self.loss_cfg.num_steps,
            self.loss_kwargs.sp_mesh,
        )
        loss_weight = self.loss_kwargs.loss_weight
        assert loss_weight is not None, "loss_weight can not be None"
        valid_mask = loss_weight != 0

        if not valid_mask.any():
            loss = head_weight.sum() * 0.0 + sum(draft.sum() * 0.0 for draft in draft_hidden_states)
        else:
            selected_weight = loss_weight[valid_mask]
            alpha_per_step: list[torch.Tensor] = []
            chunk_size = self.loss_cfg.chunk_size
            assert chunk_size is not None and chunk_size > 0, "A positive chunk_size is required for e2e TV loss."

            for draft_hidden, target_hidden in zip(draft_hidden_states, shifted_targets):
                selected_draft = draft_hidden[valid_mask]
                selected_target = target_hidden[valid_mask]
                alpha_chunks: list[torch.Tensor] = []
                for draft_chunk, target_chunk in zip(
                    torch.split(selected_draft, chunk_size, dim=0),
                    torch.split(selected_target, chunk_size, dim=0),
                ):
                    if torch.is_grad_enabled() and (draft_chunk.requires_grad or head_weight.requires_grad):
                        alpha = checkpoint(
                            self._tv_overlap,
                            draft_chunk,
                            target_chunk,
                            head_weight,
                            use_reentrant=False,
                        )
                    else:
                        alpha = self._tv_overlap(draft_chunk, target_chunk, head_weight)
                    alpha_chunks.append(alpha)
                alpha_per_step.append(torch.cat(alpha_chunks, dim=0))

            alphas = torch.stack(alpha_per_step, dim=-1)
            normalized_expected_acceptance = torch.cumprod(alphas, dim=-1).mean(dim=-1)
            loss = ((1.0 - normalized_expected_acceptance) * selected_weight).sum()

        if dist.is_initialized():
            loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        return loss, (None, {})
