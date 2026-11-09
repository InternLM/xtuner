from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.optim.adam import adam as torch_adam

from xtuner.v1.utils import get_device, get_logger, get_torch_device_module
from xtuner.v1.utils.dtensor import dtensor_checkpoint_view
from xtuner.v1.utils.pinned_memory import allocate_pinned_like

from .adamw import AdamW


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class SwapAdamW(AdamW):
    """AdamW optimizer with optimizer-state swap between device and host.

    Optimizer states are kept canonically on pinned CPU memory. During optimizer step, state tensors are copied to
    temporary device buffers, updated by fused kernels, and copied back to CPU.

    For sharded (``DTensor``) parameters each state entry is a ``DTensor`` with the same mesh and placements as its
    parameter, so every rank checkpoints only its own shard.

    Moments share pinned slabs of at most 1 GiB (larger moments stay standalone) to avoid per-tensor host allocator
    rounding. Small slabs allocate only their used bytes. Save the complete state dict
    (torch.save deduplicates storages) or use DCP (its filesystem writer compacts individual views). Saving each
    moment separately with raw torch.save requires a compact clone at the serialization boundary, otherwise each
    write includes its whole slab. No checkpoint copies are made by state_dict() or during optimizer steps.
    """

    _state_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        swap_optimizer_times: int = 16,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self._swap_optimizer_times = swap_optimizer_times
        self._init_swap_states()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable checkpoint views without copying tensor storage.

        Run the standard pre/post hooks first. Copy the per-parameter dictionaries
        before replacing offset DTensor wrappers, since Optimizer.state_dict
        otherwise shares those dictionaries with the live optimizer state.

        Returns:
            dict[str, Any]: Optimizer checkpoint with the usual state/group layout.
        """
        result = super().state_dict()
        result["state"] = {
            key: {
                name: dtensor_checkpoint_view(value) if isinstance(value, torch.Tensor) else value
                for name, value in state.items()
            }
            for key, state in result["state"].items()
        }
        return result

    def put_state_to_device(self, device: torch.device | str) -> bool:
        """Keep the canonical optimizer state on the host."""
        return False

    @torch._disable_dynamo
    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load into the preallocated pinned host buffers in place.

        Follows the Optimizer.load_state_dict protocol (pre/post hooks, id map, in-place group update) but never
        delegates to torch.optim.AdamW: the official loader replaces state/group objects and casts moments to the
        parameter device, which would drop pin_memory. copy_ converts floating-point dtypes instead. An empty state
        explicitly resets history (the engine's args-only load). Pre/post hooks are replayed from the base class'
        private hook dicts (torch>=2.6), because calling ``super()`` would re-enter the device cast this method
        exists to avoid.

        NOTE: the engine only reaches this method through ``TrainEngine.load_dcp``, which fixes
        ``StateDictOptions(cpu_offload=True)``: the incoming moments are CPU ``DTensor``s (CPU local tensors before
        this PR), ``step`` is a plain CPU tensor, and they are the very objects held in ``self.state``
        (``.to("cpu")`` is a no-op on CPU tensors), so ``dcp.load`` has already written the checkpoint into these
        buffers. The ``copy_`` below is therefore a self-copy on the DCP path; it is kept so externally built
        ``state_dict``s load correctly as well.

        Args:
            state_dict (Mapping[str, Any]): Optimizer state dict in the ``torch.optim.Optimizer.load_state_dict``
                schema. ``param_groups`` parameter ids may be indices or FQNs; they are paired with this optimizer's
                parameters by position, as the official loader does.

        Returns:
            None
        """
        # 1. Pre-hooks first; a hook may return a replacement state dict.
        state_dict = dict(state_dict)
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = pre_hook(self, state_dict)
            if result is not None:
                state_dict = result

        # 2. Validate the group layout and pair saved ids with live parameters by position.
        id_map = self._validate_state_layout(state_dict)

        # 3. Copy every state entry into the preallocated pinned buffers; an empty state resets history.
        self._copy_state_in_place(state_dict["state"], id_map)

        # 4. Merge hyperparameters into the existing groups in place.
        self._update_param_groups(state_dict["param_groups"])

        # 5. Post-hooks run last, as in torch.optim.Optimizer.load_state_dict.
        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    def _validate_state_layout(self, state_dict: Mapping[str, Any]) -> dict[Any, torch.Tensor]:
        # 在改动任何 buffer 之前先校验 checkpoint 结构:保存的 group 数量、每组的参数数量、以及 saved state 的
        # id 集合都必须与本优化器完全一致。返回 saved-id -> 当前参数 的映射,供后续拷贝步骤使用。
        saved_groups: list[dict[str, Any]] = state_dict["param_groups"]
        saved_state: Mapping[Any, Mapping[str, Any]] = state_dict["state"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(
            len(saved_group["params"]) != len(group["params"])
            for saved_group, group in zip(saved_groups, self.param_groups)
        ):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
            )

        id_map = {
            old_id: param
            for saved_group, group in zip(saved_groups, self.param_groups)
            for old_id, param in zip(saved_group["params"], group["params"])
        }
        # Require an exact id match: the official loader silently resets the history of parameters without saved state
        # (and keeps orphans), so a mismatch would resume training from mixed moments without any error.
        if saved_state and set(saved_state) != set(id_map):
            raise ValueError("loaded state dict contains a parameter state that doesn't match any parameter")
        return id_map

    def _copy_state_in_place(
        self, saved_state: Mapping[Any, Mapping[str, Any]], id_map: dict[Any, torch.Tensor]
    ) -> None:
        # 用 copy_ 把 checkpoint 中的 moment 和 step 写入预分配的 pinned host buffer:不替换任何 state 条目或
        # tensor 对象,保留 pin_memory。对 state key 或 local shape 不匹配的情况显式报错,而不是静默加载出错
        # 乱的数据;saved state 为空时表示清零历史。
        for param_id, param in id_map.items():
            target_state = self.state[param]
            # amsgrad=False keeps None placeholders for max_exp_avg_sq; they are not
            # part of the state contract, so ignore None entries on both sides.
            expected_keys = {key for key, value in target_state.items() if value is not None}
            if not saved_state:
                # The engine's args-only load passes an empty state: reset history instead of keeping it.
                for key in expected_keys:
                    self._to_local_tensor(target_state[key]).zero_()
                continue

            source_state = saved_state[param_id]
            source_keys = {key for key, value in source_state.items() if value is not None}
            if source_keys != expected_keys:
                raise ValueError(
                    f"loaded state dict has incompatible optimizer state keys for parameter {param_id}: "
                    f"got {sorted(source_keys)}, expected {sorted(expected_keys)}"
                )
            for key in expected_keys:
                target = self._to_local_tensor(target_state[key])
                source = self._to_local_tensor(source_state[key])
                assert isinstance(source, torch.Tensor)
                # copy_ broadcasts on mismatch, so an old or differently sharded checkpoint would load scrambled
                # moments silently; reject it with the parameter id and both shapes instead.
                if target.shape != source.shape:
                    raise ValueError(
                        f"loaded state dict has incompatible optimizer state shape for parameter {param_id} "
                        f"key {key}: got {tuple(source.shape)}, expected {tuple(target.shape)}"
                    )
                target.copy_(source)

    def _update_param_groups(self, saved_groups: list[dict[str, Any]]) -> None:
        # 把 checkpoint 中的超参(lr、betas、weight decay 等)原地合并进现有的 param group,
        # 使 step() 持续读到更新后的值。
        # Update in place so external references to these group objects (and step() below) see the new
        # hyperparameters. Unlike torch, this merges into the existing groups, so a key absent from the
        # checkpoint keeps its local value.
        for group, saved_group in zip(self.param_groups, saved_groups):
            # Shallow-copy per group so param_groups never aliases mutable values of the incoming state dict.
            saved_group = dict(saved_group)
            group.update({key: value for key, value in saved_group.items() if key != "params"})
            # Fall back to AdamW semantics when the checkpoint predates this key.
            group["decoupled_weight_decay"] = bool(saved_group.get("decoupled_weight_decay", True))

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()  # type: ignore[no-any-return]
        return tensor

    def _init_swap_states(self) -> None:
        swap_num = sum(
            self._to_local_tensor(param).numel() for group in self.param_groups for param in group["params"]
        )
        self.swap_numel = swap_num // self._swap_optimizer_times
        swap_memory = swap_num * 8 / 1024 / 1024
        logger.info(
            f"[Rank {DEVICE_MODULE.current_device()}] swap optimizer param num: {swap_num},  "
            f"param size: {swap_memory}MB\n",
            end="",
        )

        # Plan from metadata only: never materialize per-parameter states before allocating the slabs.
        requests: list[tuple[torch.Tensor, str, torch.Tensor]] = []
        for group in self.param_groups:
            amsgrad = bool(group["amsgrad"])
            for param in group["params"]:
                state = self.state[param]
                local_param = self._to_local_tensor(param)
                for key in self._state_keys:
                    if key == "max_exp_avg_sq" and not amsgrad:
                        state[key] = None
                    else:
                        requests.append((param, key, local_param))
                # Keep per-parameter step semantics consistent with torch.optim.AdamW.
                state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        cpu_tensors = allocate_pinned_like([local_param for _, _, local_param in requests], zero=True)
        for (param, key, _), cpu_tensor in zip(requests, cpu_tensors):
            state_tensor = cpu_tensor
            if isinstance(param, DTensor):
                # from_local moves CPU tensors to the mesh device, undoing offload. Construct directly
                # with the existing global spec, including nonuniform/empty local shard metadata.
                state_tensor = DTensor(cpu_tensor, param._spec, requires_grad=False)
            self.state[param][key] = state_tensor

        DEVICE_MODULE.synchronize()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            amsgrad = bool(group["amsgrad"])
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                param_state = self.state[param]
                step_tensor = param_state["step"]
                assert isinstance(step_tensor, torch.Tensor)

                cpu_exp_avg = param_state["exp_avg"]
                cpu_exp_avg_sq = param_state["exp_avg_sq"]
                cpu_max_exp_avg_sq = param_state["max_exp_avg_sq"]
                assert isinstance(cpu_exp_avg, torch.Tensor)
                assert isinstance(cpu_exp_avg_sq, torch.Tensor)

                exp_avg = self._to_local_tensor(cpu_exp_avg).to(device=DEVICE, non_blocking=True)
                exp_avg_sq = self._to_local_tensor(cpu_exp_avg_sq).to(device=DEVICE, non_blocking=True)
                max_exp_avg_sq = None
                if amsgrad:
                    assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                    max_exp_avg_sq = self._to_local_tensor(cpu_max_exp_avg_sq).to(device=DEVICE, non_blocking=True)

                torch_adam(
                    [self._to_local_tensor(param)],
                    [self._to_local_tensor(param.grad)],
                    [exp_avg],
                    [exp_avg_sq],
                    [max_exp_avg_sq] if max_exp_avg_sq is not None else [],
                    [step_tensor],
                    amsgrad=amsgrad,
                    has_complex=torch.is_complex(param),
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                    decoupled_weight_decay=group["decoupled_weight_decay"],
                )

                # Write the updated moments back into the same pinned host buffers.
                self._to_local_tensor(cpu_exp_avg).copy_(exp_avg, non_blocking=True)
                self._to_local_tensor(cpu_exp_avg_sq).copy_(exp_avg_sq, non_blocking=True)
                if amsgrad and max_exp_avg_sq is not None:
                    self._to_local_tensor(cpu_max_exp_avg_sq).copy_(max_exp_avg_sq, non_blocking=True)

        DEVICE_MODULE.synchronize()
        return loss
