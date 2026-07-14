from typing import Annotated, Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss.base_loss_ctx import BaseLossConfig, BaseLossKwargs
from xtuner.v1.loss.ce_loss import LMHeadLossContext
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    loss_weight: torch.Tensor,
    *,
    loss_type: Literal["mse", "clipped"],
    old_values: torch.Tensor | None = None,
    value_clip: float = 0.2,
) -> torch.Tensor:
    """Compute globally calibrated MSE or clipped PPO value loss.

    Args:
        values (torch.Tensor): Current scalar value predictions.
        returns (torch.Tensor): Frozen critic regression targets.
        loss_weight (torch.Tensor): Per-token weights, normally mask divided by global valid count.
        loss_type (Literal["mse", "clipped"]): Value loss variant.
        old_values (torch.Tensor | None): Frozen pre-update values required by clipped loss.
        value_clip (float): Symmetric value clipping range.

    Returns:
        torch.Tensor: Scalar weighted value loss.
    """
    if values.shape != returns.shape or values.shape != loss_weight.shape:
        raise ValueError(
            f"values, returns, and loss_weight must have the same shape, got {values.shape}, {returns.shape}, "
            f"and {loss_weight.shape}"
        )
    if value_clip < 0:
        raise ValueError(f"value_clip must be non-negative, got {value_clip}")

    predictions = values.float()
    targets = returns.detach().to(device=predictions.device, dtype=torch.float32)
    weights = loss_weight.to(device=predictions.device, dtype=torch.float32)
    squared_error = (predictions - targets).square()
    if loss_type == "mse":
        per_token_loss = 0.5 * squared_error
    elif loss_type == "clipped":
        if old_values is None:
            raise ValueError("old_values is required for clipped value loss.")
        if old_values.shape != values.shape:
            raise ValueError(f"old_values must match values, got {old_values.shape} and {values.shape}")
        frozen_old_values = old_values.detach().to(device=predictions.device, dtype=torch.float32)
        clipped_values = frozen_old_values + torch.clamp(
            predictions - frozen_old_values,
            min=-value_clip,
            max=value_clip,
        )
        clipped_squared_error = (clipped_values - targets).square()
        per_token_loss = 0.5 * torch.maximum(squared_error, clipped_squared_error)
    else:
        raise ValueError(f"Unsupported value loss type: {loss_type}")
    return (per_token_loss * weights).sum()


class ValueLossConfig(BaseLossConfig):
    """Configuration for scalar critic value loss.

    Args:
        loss_type (Literal["mse", "clipped"]): Value loss variant. Defaults to ``"clipped"``.
        value_clip (float): Symmetric PPO value clipping range. Defaults to 0.2.
        ignore_idx (int): Inherited and unused; ``value_mask`` controls valid positions.
        mode (Literal["eager", "chunk"]): Loss computation mode.
        chunk_size (int | None): Chunk size when mode is ``"chunk"``.
    """

    loss_type: Annotated[
        Literal["mse", "clipped"],
        Parameter(help="critic value loss type"),
    ] = "clipped"
    value_clip: Annotated[float, Parameter(help="symmetric PPO value clipping range")] = 0.2

    @property
    def loss_ctx_cls(self) -> type["ValueLossContext"]:
        return ValueLossContext

    @property
    def _loss_kwargs_cls(self) -> type["ValueLossKwargs"]:
        return ValueLossKwargs

    def model_post_init(self, _context: Any) -> None:
        if self.value_clip < 0:
            raise ValueError(f"value_clip must be non-negative, got {self.value_clip}")

    def build(
        self,
        data: dict[str, Any],
        sp_mesh: DeviceMesh | None = None,
    ) -> "ValueLossContext | None":
        """Build a scalar value loss context from training data.

        Args:
            data (dict[str, Any]): Requires ``returns`` and ``value_mask``. Clipped loss also requires
                ``old_values``.
            sp_mesh (DeviceMesh | None): Optional sequence-parallel device mesh.

        Returns:
            ValueLossContext | None: Built context, or ``None`` when common required fields are missing.
        """
        if "returns" not in data or "value_mask" not in data:
            return None
        old_values = data.get("old_values")
        if self.loss_type == "clipped" and old_values is None:
            raise ValueError("old_values is required when loss_type='clipped'.")

        returns = cast(torch.Tensor, data["returns"])
        value_mask = cast(torch.Tensor, data["value_mask"])
        if returns.ndim != 2 or value_mask.ndim != 2:
            raise ValueError(
                f"returns and value_mask must be two-dimensional, got {returns.shape} and {value_mask.shape}"
            )
        if returns.shape != value_mask.shape:
            raise ValueError(
                f"returns and value_mask must have the same shape, got {returns.shape} and {value_mask.shape}"
            )
        if old_values is not None and old_values.shape != returns.shape:
            raise ValueError(
                f"old_values and returns must have the same shape, got {old_values.shape} and {returns.shape}"
            )

        loss_kwargs = ValueLossKwargs(
            returns=returns.detach(),
            value_mask=value_mask.bool(),
            old_values=old_values.detach() if old_values is not None else None,
        ).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        return ValueLossContext(self, loss_kwargs)


class ValueLossKwargs(BaseLossKwargs):
    """Keyword arguments for scalar critic value loss.

    Args:
        returns (torch.Tensor): Frozen critic regression targets.
        value_mask (torch.Tensor): Valid critic action-token mask.
        old_values (torch.Tensor | None): Frozen pre-update values for PPO clipping.
        loss_weight (torch.Tensor | None): Globally calibrated per-token weights.
        global_valid_count (torch.Tensor | None): Global valid-token count for the optimizer update.
    """

    returns: torch.Tensor
    value_mask: torch.Tensor
    old_values: torch.Tensor | None = None
    loss_weight: torch.Tensor | None = None
    global_valid_count: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        self.returns = sp_split(self.returns, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        self.value_mask = sp_split(self.value_mask, sp_mesh=sp_mesh, split_dim=1, padding_value=False)
        if self.old_values is not None:
            self.old_values = sp_split(self.old_values, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.loss_weight is not None:
            self.loss_weight = sp_split(self.loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        return self

    def to(self, device: torch.device | str) -> Self:
        self.returns = self.returns.to(device)
        self.value_mask = self.value_mask.to(device)
        if self.old_values is not None:
            self.old_values = self.old_values.to(device)
        if self.loss_weight is not None:
            self.loss_weight = self.loss_weight.to(device)
        if self.global_valid_count is not None:
            self.global_valid_count = self.global_valid_count.to(device)
        return self


class ValueLossContext(LMHeadLossContext):
    """Loss context for an independent scalar critic head.

    Args:
        loss_cfg (ValueLossConfig): Scalar value loss configuration.
        loss_kwargs (ValueLossKwargs): Per-trajectory value targets and masks.
    """

    loss_cfg: ValueLossConfig  # type: ignore[assignment]
    loss_kwargs: ValueLossKwargs  # type: ignore[assignment]

    def __init__(self, loss_cfg: ValueLossConfig, loss_kwargs: ValueLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def build_batches(
        loss_ctx_list: list["ValueLossContext"],
        *args: Any,
        **kwargs: Any,
    ) -> list["ValueLossContext"]:
        """Calibrate value losses by the global valid action-token count.

        Args:
            loss_ctx_list (list[ValueLossContext]): Contexts in one optimizer update.
            *args (Any): Unused compatibility arguments.
            **kwargs (Any): Unused compatibility arguments.

        Returns:
            list[ValueLossContext]: The same contexts with calibrated loss weights.
        """
        del args, kwargs
        if not loss_ctx_list:
            raise ValueError("loss_ctx_list must not be empty.")

        rank_valid_count = sum(context.loss_kwargs.value_mask.sum() for context in loss_ctx_list)
        global_valid_count = cast(torch.Tensor, rank_valid_count).to(dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
        denominator = global_valid_count.clamp_min(1.0)

        for context in loss_ctx_list:
            context._batch_size = len(loss_ctx_list)
            context.loss_kwargs.loss_weight = context.loss_kwargs.value_mask.float() / denominator
            context.loss_kwargs.global_valid_count = global_valid_count
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: ValueLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        """Apply the scalar head and compute value loss.

        Args:
            hidden_states (torch.Tensor): Critic backbone hidden states.
            head_weight (torch.Tensor): Scalar value head weight.
            head_bias (torch.Tensor | None): Optional scalar value head bias.
            loss_kwargs (ValueLossKwargs): Frozen value targets and calibrated weights.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]: Scalar loss, values, and extra metrics.
        """
        values = F.linear(hidden_states, head_weight, head_bias).float()
        if values.size(-1) != 1:
            raise ValueError(f"Value head must produce one scalar, got output shape {values.shape}")
        values = values.squeeze(-1)
        if values.shape != loss_kwargs.returns.shape:
            raise ValueError(
                f"Value predictions and returns must have the same shape, got {values.shape} and "
                f"{loss_kwargs.returns.shape}"
            )
        if loss_kwargs.loss_weight is None:
            raise RuntimeError("ValueLossContext.build_batches must be called before forward.")

        loss = value_loss(
            values,
            loss_kwargs.returns,
            loss_kwargs.loss_weight,
            loss_type=self.loss_cfg.loss_type,
            old_values=loss_kwargs.old_values,
            value_clip=self.loss_cfg.value_clip,
        )
        return loss, (values, {})
