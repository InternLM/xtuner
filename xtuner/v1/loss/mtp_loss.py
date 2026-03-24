# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext
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
    """


class MTPLossConfig(CELossConfig):
    """Loss configuration for Multi-Token Prediction (MTP).

    Extends ``CELossConfig`` with a ``mtp_depth`` field that controls how many
    additional positions the labels are rolled during ``build()``. This class
    is intended for internal use by the model and is not exposed to users.

    Args:
        mtp_depth (int): 1-indexed MTP layer depth. The first MTP layer uses
            ``mtp_depth=1`` (shift=-1 on top of the existing label shift).
    """

    mtp_depth: int

    @property
    def loss_ctx_cls(self) -> type["MTPLossContext"]:
        return MTPLossContext

    @property
    def _loss_kwargs_cls(self) -> type["MTPLossKwargs"]:
        return MTPLossKwargs

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "MTPLossContext | None":
        """Build MTPLossContext from data dict.

        Rolls ``shifted_labels`` by ``-mtp_depth`` positions (per-sequence,
        respecting packed-sequence boundaries) before constructing the loss
        context. The roll is performed on the full sequence prior to any
        sequence-parallel split so that boundary positions and ``cu_seq_lens``
        are always consistent.

        Args:
            data (dict): Data dict containing loss-related fields.
                Required keys: ``shifted_labels``, ``seq_ctx``.
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

        loss_kwargs = MTPLossKwargs(shifted_labels=rolled).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)

        return MTPLossContext(self, loss_kwargs)


class MTPLossContext(LMHeadLossContext):
    """Loss context for Multi-Token Prediction (MTP).

    Inherits all computation logic from ``LMHeadLossContext``. The label
    rolling is handled upstream in ``MTPLossConfig.build()``, so no override
    is needed here.

    Args:
        loss_cfg (MTPLossConfig): The MTP loss configuration.
        loss_kwargs (MTPLossKwargs): Pre-rolled keyword arguments for loss
            computation.
    """
