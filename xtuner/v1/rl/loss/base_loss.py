from typing import Any, Literal, Protocol, Sequence, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss.ce_loss import CELossConfig, CELossContext, CELossKwargs
from xtuner.v1.loss.utils import sp_gather, sp_split

# from ..utils import sp_split
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
logger = get_logger()


def compute_kl_loss_weight(
    shifted_labels: torch.Tensor, global_grad_tokens: torch.Tensor, kl_loss_coef: float, ignore_idx: int = -100
) -> torch.Tensor:
    kl_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens * kl_loss_coef
    kl_loss_weight[shifted_labels == ignore_idx] = 0.0
    return kl_loss_weight


class BaseRLLossConfig(CELossConfig):
    """Base configuration for reinforcement learning loss functions in XTuner
    RL.

    Configuration base class for RL loss computations, providing a framework for
    policy optimization objectives with optional KL divergence regularization.
    Serves as the foundation for various RL algorithms including PPO, GRPO, and custom implementations.

    Args:
        policy_loss_cfg (dict[str, Any]): Configuration parameters for the main policy loss.
            Contains algorithm-specific parameters for policy optimization.
        use_kl_loss (bool): Whether to include KL divergence penalty in the loss.
            When True, requires a reference model for KL computation. Defaults to False.
        kl_loss_coef (float): Coefficient for weighting the KL divergence penalty.
            Controls the strength of regularization against the reference policy. Defaults to 0.001.
        kl_loss_type (Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None):
            Type of KL penalty computation method. Different types provide various
            regularization behaviors and numerical stability properties. Defaults to None.
        rollout_is (RolloutImportanceSampling): Configuration parameters for the rollout importance sampling.
            Contains algorithm-specific parameters for rollout importance sampling.
            Defaults to RolloutImportanceSampling().
    **Abstract Method:**
        loss_ctx_cls: Must be implemented by subclasses to return the appropriate
        loss context class for the specific RL algorithm.

    **Examples:**

    Example configuration for basic RL loss ::

        config = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.2,
                cliprange_low=0.2,
                loss_type='vanilla',
            ),
            use_kl_loss=False
        )

    Example configuration RL loss with KL regularization::

        config = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.2,
                cliprange_low=0.2,
                loss_type='vanilla',
            ),
            use_kl_loss=True,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl"
        )

    .. note::
       When ``use_kl_loss=True``, ensure that the training worker is configured
       with a reference model for KL divergence computation.
    """

    policy_loss_cfg: dict[str, Any]
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    rollout_is: RolloutImportanceSampling = RolloutImportanceSampling()

    @property
    def loss_ctx_cls(self) -> type["BaseRLLossContext"]:
        raise NotImplementedError

    @property
    def required_loss_keys(self) -> list[str]:
        """Optional pack fields this loss requires on every training pack.

        ``TrainBatch`` loads exactly these optional fields (in addition to the batch-level ones) and fails when a
        sample lacks one of them.

        Returns:
            list[str]: Field names such as ``rollout_logprobs`` or ``sampler_topk_ids``.
        """
        return []

    @property
    def _loss_kwargs_cls(self) -> type["BaseRLLossKwargs"]:
        raise NotImplementedError

    def finalize_metrics(self, extra_info_dict: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
        """Finalize metrics emitted by the policy loss."""
        return finalize_train_policy_metrics(extra_info_dict, device)

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
        device: str | torch.device | None = None,
    ) -> "BaseRLLossContext | None":
        """Build RL loss context from data dict.

        Args:
            data (dict): Data dictionary containing RL-specific fields:
                - shifted_labels (torch.Tensor): The shifted labels
                - advantages (torch.Tensor): Advantage estimates
                - rollout_logprobs (torch.Tensor | None): Rollout log probabilities
                - old_logprobs (torch.Tensor | None): Old policy log probabilities (optional, can be set later)
                - rollout_is_weights (torch.Tensor | None): Importance sampling weights
                - ref_logprobs (torch.Tensor | None): Reference model log probabilities
                - sampler_topk_ids (torch.Tensor | None): Sampler top-k token ids, ``[1, T, k]``
                - sampler_topk_logprobs (torch.Tensor | None): Sampler top-k log probabilities, ``[1, T, k]``
            sp_mesh (DeviceMesh | None): Sequence parallel device mesh
            device (str | torch.device): Target device of the built loss kwargs. Defaults to the
                accelerator device; pass ``"cpu"`` to build the context before any device transfer.

        Returns:
            BaseRLLossContext | None: The built loss context, or None if required fields are missing
        """
        # Check for required fields
        if "shifted_labels" not in data or "advantages" not in data:
            return None
        if device is None:
            device = DEVICE

        # Extract RL-specific fields from data
        shifted_labels = data["shifted_labels"]
        advantages = data["advantages"]
        rollout_logprobs = data.get("rollout_logprobs", None)
        old_logprobs = data.get("old_logprobs", None)
        rollout_is_weights = data.get("rollout_is_weights", None)
        ref_logprobs = data.get("ref_logprobs", None)

        LossKwargs = self._loss_kwargs_cls
        loss_kwargs = LossKwargs(
            shifted_labels=shifted_labels,
            old_logprobs=old_logprobs,
            advantages=advantages,
            rollout_logprobs=rollout_logprobs,
            is_weights=rollout_is_weights,
            ref_logprobs=ref_logprobs,
            sampler_topk_ids=data.get("sampler_topk_ids"),
            sampler_topk_logprobs=data.get("sampler_topk_logprobs"),
        ).to(device)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)

        LossContext = self.loss_ctx_cls
        return LossContext(self, loss_kwargs)


class BaseRLLossKwargs(CELossKwargs):
    """Keyword arguments for reinforcement learning loss computation.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        old_logprobs (torch.Tensor): Log probabilities from the old policy.
        advantages (torch.Tensor): Advantage estimates for the actions taken.
        policy_loss_weight (torch.Tensor): Weights for each token in the policy loss computation.
        ref_logprobs (torch.Tensor | None): Reference log probabilities for KL penalty, if used.
        kl_loss_weight (torch.Tensor | None): Weights for each token in the KL loss computation, if used.
        rollout_logprobs (torch.Tensor | None): Rollout log probabilities from inference engine, used for importance sampling.
        is_weights (torch.Tensor | None): Importance sampling weights. If None, importance sampling is not used.
        sampler_topk_ids (torch.Tensor | None): Sampler top-k token ids ``[1, T, k]``, ``-1`` on unused slots.
        sampler_topk_logprobs (torch.Tensor | None): Full-vocabulary sampler log probabilities of
            ``sampler_topk_ids``, ``[1, T, k]``, ``-inf`` on unused slots.
    """

    rollout_logprobs: torch.Tensor | None = None
    advantages: torch.Tensor
    old_logprobs: torch.Tensor | None = None
    policy_loss_weight: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    kl_loss_weight: torch.Tensor | None = None
    global_grad_tokens: torch.Tensor | None = None
    is_weights: torch.Tensor | None = None
    sampler_topk_ids: torch.Tensor | None = None
    sampler_topk_logprobs: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        # Call parent class to handle shifted_labels
        super().sp_split(sp_mesh)
        # Handle RL-specific fields
        self.advantages = sp_split(self.advantages, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = sp_split(self.rollout_logprobs, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.is_weights is not None:
            self.is_weights = sp_split(self.is_weights, sp_mesh=sp_mesh, split_dim=1, padding_value=1.0)
        if self.sampler_topk_ids is not None:
            self.sampler_topk_ids = sp_split(self.sampler_topk_ids, sp_mesh=sp_mesh, split_dim=1, padding_value=-1)
        if self.sampler_topk_logprobs is not None:
            self.sampler_topk_logprobs = sp_split(
                self.sampler_topk_logprobs, sp_mesh=sp_mesh, split_dim=1, padding_value=float("-inf")
            )
        # 1. 这里不用对old_logprobs和ref_logprobs进行sp_split，因为他是模型 fwd 生成的，
        # 因为模型 fwd 前一定会对 seq_ctx 进行 sp_split。
        # 2. global_grad_tokens 是scalar Tensor, 不用进行sp_split。
        # 3. 这里也不用对各种weight(policy_loss_weight, kl_loss_weight, is_weights)进行sp_split，
        # 因为他们在LossContext.build_batches()中生成时也保证是sp_split的。
        return self

    def to(self, device: torch.device | str) -> Self:
        # Call parent class to handle shifted_labels
        super().to(device)
        # Handle RL-specific fields
        self.advantages = self.advantages.to(device)
        if self.old_logprobs is not None:
            self.old_logprobs = self.old_logprobs.to(device)
        if self.ref_logprobs is not None:
            self.ref_logprobs = self.ref_logprobs.to(device)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = self.rollout_logprobs.to(device)
        if self.is_weights is not None:
            self.is_weights = self.is_weights.to(device)
        if self.global_grad_tokens is not None:
            self.global_grad_tokens = self.global_grad_tokens.to(device)
        if self.policy_loss_weight is not None:
            self.policy_loss_weight = self.policy_loss_weight.to(device)
        if self.kl_loss_weight is not None:
            self.kl_loss_weight = self.kl_loss_weight.to(device)
        if self.sampler_topk_ids is not None:
            self.sampler_topk_ids = self.sampler_topk_ids.to(device)
        if self.sampler_topk_logprobs is not None:
            self.sampler_topk_logprobs = self.sampler_topk_logprobs.to(device)
        return self


class ActorSnapshotLike(Protocol):
    """Structural contract of one pack's frozen actor loss inputs."""

    @property
    def shifted_labels(self) -> torch.Tensor: ...

    @property
    def old_logprobs(self) -> torch.Tensor: ...

    @property
    def is_weights(self) -> torch.Tensor | None: ...

    @property
    def ref_logprobs(self) -> torch.Tensor | None: ...

    @property
    def teacher_logprobs(self) -> torch.Tensor | None: ...

    @property
    def target_token_ids(self) -> torch.Tensor | None: ...


class BaseRLLossContext(CELossContext):
    loss_cfg: BaseRLLossConfig  # type: ignore[assignment]
    loss_kwargs: BaseRLLossKwargs  # type: ignore[assignment]

    def bind_actor_snapshot(self, snapshot: ActorSnapshotLike) -> None:
        """Bind one pack's frozen stage outputs to this local context.

        Snapshot tensors are already sequence-parallel local, so they are only
        moved to the context device and never split again.

        Args:
            snapshot (ActorSnapshotLike): Frozen labels, old/ref logprobs, IS
                weights and teacher targets of this pack.

        Returns:
            None
        """
        kwargs = self.loss_kwargs
        if snapshot.shifted_labels.shape != kwargs.shifted_labels.shape:
            raise ValueError("Snapshot belongs to a different local layout")
        device = kwargs.shifted_labels.device
        kwargs.shifted_labels = snapshot.shifted_labels.to(device)
        kwargs.old_logprobs = snapshot.old_logprobs.to(device)
        for field in ("is_weights", "ref_logprobs", "teacher_logprobs", "target_token_ids"):
            value = getattr(snapshot, field)
            if value is not None:
                setattr(kwargs, field, value.to(device))

    def compute_rollout_is(
        self, sp_mesh: DeviceMesh, num_tokens: torch.Tensor
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        shifted_labels = self.loss_kwargs.shifted_labels
        rollout_logprobs = self.loss_kwargs.rollout_logprobs
        mask = shifted_labels != self.loss_cfg.ignore_idx
        old_logprobs = self.loss_kwargs.old_logprobs

        assert rollout_logprobs is not None
        assert old_logprobs is not None

        if sp_mesh and sp_mesh.size() > 1:
            # Temporarily sp_gather old_logprobs here, but not modify loss_kwargs.old_logprobs(still in sp_split state)
            rollout_logprobs = sp_gather(rollout_logprobs, sp_mesh, dim=1)
            old_logprobs = sp_gather(old_logprobs, sp_mesh, dim=1)
            old_logprobs = old_logprobs[:, : rollout_logprobs.size(1)]  # type: ignore
            mask = sp_gather(mask, sp_mesh, dim=1)
            mask = mask[:, : rollout_logprobs.size(1)]  # type: ignore

        rollout_is_weights, rollout_is_mask, mismatch_metrics, rollout_is_metrics = (
            self.loss_cfg.rollout_is.compute_rollout_importance_weights_and_metrics(
                old_log_prob=old_logprobs,
                rollout_log_prob=rollout_logprobs,
                num_tokens=num_tokens,
                response_mask=mask,
            )
        )
        if sp_mesh and sp_mesh.size() > 1:
            rollout_is_mask = sp_split(rollout_is_mask, sp_mesh, 1, 0)
            assert rollout_is_mask.size(1) == shifted_labels.size(1), (
                f"rollout_is_mask {rollout_is_mask.size(1)} vs shifted_labels {shifted_labels.size(1)}"
            )

            if rollout_is_weights is not None:
                rollout_is_weights = sp_split(rollout_is_weights, sp_mesh, 1, 0)
                assert rollout_is_weights.size(1) == shifted_labels.size(1), (
                    f"rollout_is_weights {rollout_is_weights.size(1)} vs shifted_labels {shifted_labels.size(1)}"
                )

        # Out-of-place update: the full source labels handed to build() are never mutated.
        self.loss_kwargs.shifted_labels = shifted_labels.masked_fill(~rollout_is_mask.bool(), self.loss_cfg.ignore_idx)

        self.loss_kwargs.is_weights = rollout_is_weights
        return mismatch_metrics, rollout_is_metrics


def calibrate_token_mean_loss_weights(loss_ctx_list: Sequence[BaseRLLossContext]) -> None:
    """Set one optimizer step's policy and KL weights to a global token mean.

    Every supervised token of the step, across all ranks, gets weight ``1 / global_grad_tokens`` (times its rollout
    importance-sampling weight when present). Collective: every rank must call it once per step.

    Args:
        loss_ctx_list (Sequence[BaseRLLossContext]): Local contexts of one optimizer step.
    """
    assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"
    loss_cfg = loss_ctx_list[0].loss_cfg

    rank_grad_tokens = sum((ctx.loss_kwargs.shifted_labels != loss_cfg.ignore_idx).sum() for ctx in loss_ctx_list)
    global_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
    if dist.is_initialized():
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

    if global_grad_tokens == 0:
        logger.warning("Global gradient tokens is 0, which may lead to division by zero in loss weight calculation.")
        global_grad_tokens.add_(1)  # Avoid division by zero

    for loss_ctx in loss_ctx_list:
        loss_kwargs = loss_ctx.loss_kwargs
        shifted_labels = loss_kwargs.shifted_labels
        policy_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens
        policy_loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
        if loss_kwargs.is_weights is not None:
            policy_loss_weight = policy_loss_weight * loss_kwargs.is_weights
        if loss_cfg.use_kl_loss:
            assert loss_kwargs.ref_logprobs is not None, "ref_logprobs can not be None"
            kl_loss_weight = compute_kl_loss_weight(
                shifted_labels, global_grad_tokens, loss_cfg.kl_loss_coef, loss_cfg.ignore_idx
            )
        else:
            kl_loss_weight = None
        loss_kwargs.policy_loss_weight = policy_loss_weight
        loss_kwargs.kl_loss_weight = kl_loss_weight
        loss_kwargs.global_grad_tokens = global_grad_tokens


def finalize_train_policy_metrics(extra_info_dict: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    if "reduced_train_policy_valid_count" not in extra_info_dict:
        return extra_info_dict

    has_clip_count = (
        "reduced_train_policy_clip_low_count" in extra_info_dict
        and "reduced_train_policy_clip_high_count" in extra_info_dict
    )
    sum_keys = [
        "reduced_train_policy_ratio_abs_dev_sum",
        "reduced_train_policy_kl1_sum",
        "reduced_train_policy_kl3_sum",
        "reduced_train_policy_valid_count",
    ]
    if has_clip_count:
        sum_keys.extend(["reduced_train_policy_clip_low_count", "reduced_train_policy_clip_high_count"])
    max_keys = ("reduced_train_policy_ratio_max",)
    min_keys = ("reduced_train_policy_ratio_min",)

    def reduce_values(keys, op):
        values = torch.tensor([extra_info_dict.pop(key, 0.0) for key in keys], dtype=torch.float32, device=device)
        if dist.is_initialized():
            dist.all_reduce(values, op=op)
        return dict(zip(keys, values.tolist()))

    train_policy_values = {}
    train_policy_values.update(reduce_values(sum_keys, dist.ReduceOp.SUM))
    train_policy_values.update(reduce_values(max_keys, dist.ReduceOp.MAX))
    train_policy_values.update(reduce_values(min_keys, dist.ReduceOp.MIN))

    valid_count = train_policy_values["reduced_train_policy_valid_count"]
    output_keys = {
        "reduced_train_policy_ratio_abs_dev_mean": "reduced_train_policy_ratio_abs_dev_sum",
        "reduced_train_policy_kl1": "reduced_train_policy_kl1_sum",
        "reduced_train_policy_kl3": "reduced_train_policy_kl3_sum",
    }
    if has_clip_count:
        output_keys.update(
            {
                "reduced_train_policy_clip_frac_low": "reduced_train_policy_clip_low_count",
                "reduced_train_policy_clip_frac_high": "reduced_train_policy_clip_high_count",
            }
        )
    for output_key, sum_key in output_keys.items():
        extra_info_dict[output_key] = train_policy_values[sum_key] / valid_count if valid_count > 0 else 0.0

    ratio_max = train_policy_values["reduced_train_policy_ratio_max"] if valid_count > 0 else 0.0
    ratio_min = train_policy_values["reduced_train_policy_ratio_min"] if valid_count > 0 else 0.0
    extra_info_dict["reduced_train_policy_ratio_max"] = ratio_max
    extra_info_dict["reduced_train_policy_ratio_min"] = ratio_min
    # legacy metric，keep here
    extra_info_dict["max_ratio"] = ratio_max
    return extra_info_dict
