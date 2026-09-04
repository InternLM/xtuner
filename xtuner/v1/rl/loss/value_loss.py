"""Value-function loss for a PPO critic with a scalar head."""

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
    loss_type: Literal["mse", "clipped"] = "clipped",
    old_values: torch.Tensor | None = None,
    value_clip: float = 0.2,
) -> torch.Tensor:
    """Compute the globally calibrated value regression loss.

    ``clipped`` is the PPO value objective: the prediction is confined to a
    trust region around the pre-update value, and the larger of the clipped and
    unclipped errors is taken, so a critic cannot move too far on any single
    batch of reused rollouts.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Frozen regression targets, usually GAE returns.
        loss_weight (torch.Tensor): Per-token weights, normally the value mask
            divided by the global valid-token count.
        loss_type (Literal["mse", "clipped"]): Objective variant.
        old_values (torch.Tensor | None): Pre-update predictions, required by
            ``clipped``.
        value_clip (float): Half-width of the symmetric trust region.

    Returns:
        torch.Tensor: Scalar weighted loss.
    """
    if values.shape != returns.shape or values.shape != loss_weight.shape:
        raise ValueError(
            "values, returns and loss_weight must have the same shape, got "
            f"{values.shape}, {returns.shape} and {loss_weight.shape}"
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
            raise ValueError("old_values is required when loss_type='clipped'.")
        if old_values.shape != values.shape:
            raise ValueError(f"old_values must match values, got {old_values.shape} and {values.shape}")
        frozen = old_values.detach().to(device=predictions.device, dtype=torch.float32)
        clipped = frozen + torch.clamp(predictions - frozen, min=-value_clip, max=value_clip)
        per_token_loss = 0.5 * torch.maximum(squared_error, (clipped - targets).square())
    else:
        raise ValueError(f"Unsupported value loss type: {loss_type}")

    return (per_token_loss * weights).sum()


class ValueLossConfig(BaseLossConfig):
    """Configuration for the scalar critic value loss.

    Args:
        loss_type (Literal["mse", "clipped"]): Objective variant. Defaults to
            ``"clipped"``, the PPO value objective.
        value_clip (float): Half-width of the symmetric value trust region.
            Defaults to 0.2.
        ignore_idx (int): Inherited and unused; ``value_mask`` selects the valid
            positions instead.
        mode (Literal["eager"]): Only ``"eager"`` is supported. Chunking exists
            to avoid materializing ``[tokens, vocab_size]`` logits, which does
            not apply to a ``[tokens, 1]`` value head.
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
        if self.mode != "eager":
            raise ValueError(
                f"ValueLossConfig only supports mode='eager', got {self.mode!r}. A scalar value head "
                "produces one output per token, so chunking saves no memory."
            )

    def build(
        self,
        data: dict[str, Any],
        sp_mesh: DeviceMesh | None = None,
    ) -> "ValueLossContext | None":
        """Build a value loss context from training data.

        Args:
            data (dict[str, Any]): Requires ``returns`` and ``value_mask``.
                ``clipped`` additionally requires ``old_values``.
            sp_mesh (DeviceMesh | None): Sequence parallel mesh.

        Returns:
            ValueLossContext | None: Built context, or ``None`` when the
                required fields are absent.
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
    """Keyword arguments for the scalar critic value loss.

    Args:
        returns (torch.Tensor): Frozen regression targets.
        value_mask (torch.Tensor): Valid action-token mask.
        old_values (torch.Tensor | None): Pre-update predictions for clipping.
        loss_weight (torch.Tensor | None): Globally calibrated per-token weights,
            populated by :meth:`ValueLossContext.build_batches`.
        global_valid_count (torch.Tensor | None): Global valid-token count for
            this optimizer step, retained for logging.
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
        # `loss_weight` is derived from the already-split `value_mask` in
        # `build_batches`, so it is never split here.
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
    """Loss context for a critic's scalar value head.

    Args:
        loss_cfg (ValueLossConfig): Value loss configuration.
        loss_kwargs (ValueLossKwargs): Targets, masks and calibrated weights.
    """

    loss_cfg: ValueLossConfig  # type: ignore[assignment]
    loss_kwargs: ValueLossKwargs  # type: ignore[assignment]

    def __init__(self, loss_cfg: ValueLossConfig, loss_kwargs: ValueLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["ValueLossContext"],
        *args: Any,
        **kwargs: Any,
    ) -> list["ValueLossContext"]:
        """Calibrate the loss by the global valid action-token count.

        Implements step 1 of the calibration protocol in
        :mod:`xtuner.v1.loss.base_loss_ctx`: one denominator shared by every
        micro-batch, data-parallel rank and gradient-accumulation step, so the
        result matches single-rank training without accumulation.

        Args:
            loss_ctx_list (list[ValueLossContext]): Contexts in one optimizer step.
            *args (Any): Unused, for signature compatibility.
            **kwargs (Any): Unused, for signature compatibility.

        Returns:
            list[ValueLossContext]: The same contexts, with weights populated.
        """
        del args, kwargs
        if not loss_ctx_list:
            raise ValueError("loss_ctx_list must not be empty.")

        rank_valid_count = sum(ctx.loss_kwargs.value_mask.sum() for ctx in loss_ctx_list)
        global_valid_count = cast(torch.Tensor, rank_valid_count).to(dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
        denominator = global_valid_count.clamp_min(1.0)

        for ctx in loss_ctx_list:
            ctx._batch_size = len(loss_ctx_list)
            ctx.loss_kwargs.loss_weight = ctx.loss_kwargs.value_mask.float() / denominator
            ctx.loss_kwargs.global_valid_count = global_valid_count
        return loss_ctx_list

    def loss_fn(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: ValueLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        """Apply the scalar head and compute the value loss.

        Args:
            hidden_states (torch.Tensor): Critic backbone hidden states.
            head_weight (torch.Tensor): Scalar head weight.
            head_bias (torch.Tensor | None): Optional scalar head bias.
            loss_kwargs (ValueLossKwargs): Targets and calibrated weights.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]: Scalar
                loss, the predicted values, and diagnostic metrics.
        """
        values = F.linear(hidden_states, head_weight, head_bias).float()
        if values.size(-1) != 1:
            raise ValueError(f"A value head must emit one scalar per token, got output shape {tuple(values.shape)}")
        values = values.squeeze(-1)
        if values.shape != loss_kwargs.returns.shape:
            raise ValueError(
                f"value predictions and returns must have the same shape, got {values.shape} "
                f"and {loss_kwargs.returns.shape}"
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
        return loss, (values, self._metrics(values, loss_kwargs))

    def _metrics(self, values: torch.Tensor, loss_kwargs: ValueLossKwargs) -> dict[str, Any]:
        """Collect per-rank sums for critic health metrics.

        Sums rather than means are emitted so the caller can reduce them across
        ranks and divide by the global count exactly once.
        """
        with torch.no_grad():
            mask = loss_kwargs.value_mask
            if not bool(mask.any()):
                return {}
            selected_values = values.detach().masked_select(mask)
            selected_returns = loss_kwargs.returns.masked_select(mask)
            metrics: dict[str, Any] = {
                "reduced_critic_valid_count": mask.sum(),
                "reduced_critic_value_sum": selected_values.sum(),
                "reduced_critic_value_square_sum": selected_values.square().sum(),
                "reduced_critic_return_sum": selected_returns.sum(),
                "reduced_critic_return_square_sum": selected_returns.square().sum(),
                "reduced_critic_error_square_sum": (selected_values - selected_returns).square().sum(),
            }
            if loss_kwargs.old_values is not None and self.loss_cfg.loss_type == "clipped":
                deviation = (selected_values - loss_kwargs.old_values.masked_select(mask)).abs()
                metrics["reduced_critic_clip_count"] = (deviation > self.loss_cfg.value_clip).sum()
            return metrics


def explained_variance(
    return_square_sum: float,
    return_sum: float,
    error_square_sum: float,
    count: float,
) -> float | None:
    """Compute ``1 - Var(returns - values) / Var(returns)`` from reduced sums.

    The primary measure of whether a critic is learning anything: 0 means it
    predicts no better than the mean return, 1 means perfect prediction, and
    negative values mean it is worse than the mean.

    Args:
        return_square_sum (float): Sum of squared returns.
        return_sum (float): Sum of returns.
        error_square_sum (float): Sum of squared prediction errors.
        count (float): Number of valid tokens.

    Returns:
        float | None: The explained variance, or ``None`` when the return
            variance is too small for the ratio to be meaningful.
    """
    if count < 2:
        return None
    return_variance = max(return_square_sum / count - (return_sum / count) ** 2, 0.0)
    if return_variance <= 1e-12:
        return None
    # E[(returns - values)^2] upper-bounds the residual variance and equals it
    # when the critic is unbiased, which is the intended operating point.
    residual_variance = error_square_sum / count
    return 1.0 - residual_variance / return_variance
