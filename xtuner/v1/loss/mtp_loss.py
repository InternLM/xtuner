# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

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
