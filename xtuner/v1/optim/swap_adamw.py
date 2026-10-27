from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.optim.adam import adam as torch_adam

from xtuner.v1.utils import get_device, get_logger, get_torch_device_module

from .adamw import AdamW


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class SwapAdamW(AdamW):
    """AdamW optimizer with optimizer-state swap between device and host.

    Optimizer states are kept canonically on pinned CPU memory. During optimizer step, state tensors are copied to
    temporary device buffers, updated by fused kernels, and copied back to CPU.
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
        self._param_to_group_map: dict[torch.Tensor, dict] = {}
        self._param_to_cpu_states_map: dict[torch.Tensor, dict[str, torch.Tensor | None]] = {}
        self._init_swap_states()

    def put_state_to_device(self, device: torch.device | str) -> bool:
        """Keep the canonical optimizer state on the host."""
        return False

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load into the preallocated pinned host buffers in place.

        Follows the Optimizer.load_state_dict protocol (pre/post hooks, id map, in-place group update) but never
        delegates to torch.optim.AdamW: the official loader replaces state/group objects and casts moments to the
        parameter device, which would drop pin_memory and the aliases held by the swap maps. copy_ converts floating-
        point dtypes instead. An empty state explicitly resets history (the engine's args-only load).
        """
        state_dict = dict(state_dict)
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = pre_hook(self, state_dict)
            if result is not None:
                state_dict = result

        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
            )

        id_map = {
            old_id: param
            for saved_group, group in zip(saved_groups, self.param_groups)
            for old_id, param in zip(saved_group["params"], group["params"])
        }
        saved_state = state_dict["state"]
        if saved_state and set(saved_state) != set(id_map):
            raise ValueError("loaded state dict contains a parameter state that doesn't match any parameter")

        for param_id, param in id_map.items():
            target_state = self.state[param]
            # amsgrad=False keeps None placeholders for max_exp_avg_sq; they are not
            # part of the state contract, so ignore None entries on both sides.
            expected_keys = {key for key, value in target_state.items() if value is not None}
            if saved_state:
                source_state = saved_state[param_id]
                if {key for key, value in source_state.items() if value is not None} != expected_keys:
                    raise ValueError(
                        f"loaded state dict has incompatible optimizer state keys for parameter {param_id}: "
                        f"got {sorted(source_state)}, expected {sorted(expected_keys)}"
                    )
                for key in expected_keys:
                    target = self._to_local_tensor(target_state[key])
                    source = self._to_local_tensor(source_state[key])
                    assert isinstance(source, torch.Tensor)
                    target.copy_(source)
            else:
                for key in expected_keys:
                    self._to_local_tensor(target_state[key]).zero_()

        for group, saved_group in zip(self.param_groups, saved_groups):
            # In-place update keeps the group objects referenced by _param_to_group_map.
            group.update({key: value for key, value in saved_group.items() if key != "params"})
            # Fall back to AdamW semantics when the checkpoint predates this key.
            group["decoupled_weight_decay"] = bool(saved_group.get("decoupled_weight_decay", True))
        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()  # type: ignore[no-any-return]
        return tensor

    def _init_swap_states(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                self._param_to_group_map[param] = group

        swap_num = sum(self._to_local_tensor(main_param).numel() for main_param in self._param_to_group_map)
        self.swap_numel = swap_num // self._swap_optimizer_times
        swap_memory = swap_num * 8 / 1024 / 1024
        logger.info(
            f"[Rank {DEVICE_MODULE.current_device()}] swap optimizer param num: {swap_num},  "
            f"param size: {swap_memory}MB\n",
            end="",
        )

        for group in self.param_groups:
            for param in group["params"]:
                cpu_state_dtensor = self.state[param]
                cpu_state: dict[str, torch.Tensor | None] = {}
                amsgrad = bool(self._param_to_group_map[param]["amsgrad"])
                local_param = self._to_local_tensor(param)

                for key in self._state_keys:
                    if key == "max_exp_avg_sq" and not amsgrad:
                        cpu_state_dtensor[key] = None
                        cpu_state[key] = None
                    else:
                        # Preserve global metadata so DCP saves every rank's shard.
                        if isinstance(param, DTensor):
                            state_tensor = torch.empty_like(param, device="cpu", pin_memory=True)
                            assert isinstance(state_tensor, DTensor)
                            cpu_tensor = state_tensor.to_local()
                            cpu_tensor.zero_()
                            cpu_state_dtensor[key] = state_tensor
                        else:
                            cpu_tensor = torch.zeros_like(
                                local_param, device="cpu", memory_format=torch.preserve_format
                            ).pin_memory()
                            cpu_state_dtensor[key] = cpu_tensor
                        cpu_state[key] = cpu_tensor

                # Keep per-parameter step semantics consistent with torch.optim.AdamW.
                cpu_state_dtensor["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")

                self._param_to_cpu_states_map[param] = cpu_state

        DEVICE_MODULE.synchronize()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_list = list(self._param_to_group_map.keys())

        for param in params_list:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            group = self._param_to_group_map[param]
            param_state = self.state[param]
            step_tensor = param_state.get("step")
            if step_tensor is None:
                step_tensor = torch.tensor(0.0, dtype=torch.float32, device="cpu")
                param_state["step"] = step_tensor
            assert isinstance(step_tensor, torch.Tensor)

            amsgrad = bool(group["amsgrad"])
            beta1, beta2 = group["betas"]
            cpu_state = self._param_to_cpu_states_map[param]
            cpu_exp_avg = cpu_state["exp_avg"]
            cpu_exp_avg_sq = cpu_state["exp_avg_sq"]
            cpu_max_exp_avg_sq = cpu_state.get("max_exp_avg_sq", None)

            assert isinstance(cpu_exp_avg, torch.Tensor)
            assert isinstance(cpu_exp_avg_sq, torch.Tensor)
            exp_avg = cpu_exp_avg.to(device=DEVICE, non_blocking=True)
            exp_avg_sq = cpu_exp_avg_sq.to(device=DEVICE, non_blocking=True)

            max_exp_avg_sq: torch.Tensor | None = None
            if amsgrad:
                assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                max_exp_avg_sq = cpu_max_exp_avg_sq.to(device=DEVICE, non_blocking=True)

            local_param = self._to_local_tensor(param)
            local_grad = self._to_local_tensor(param.grad)
            local_exp_avg = self._to_local_tensor(exp_avg)
            local_exp_avg_sq = self._to_local_tensor(exp_avg_sq)
            local_max_exp_avg_sq = self._to_local_tensor(max_exp_avg_sq) if max_exp_avg_sq is not None else None

            torch_adam(
                [local_param],
                [local_grad],
                [local_exp_avg],
                [local_exp_avg_sq],
                [local_max_exp_avg_sq] if local_max_exp_avg_sq is not None else [],
                [step_tensor],
                amsgrad=amsgrad,
                has_complex=torch.is_complex(local_param),
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

            cpu_exp_avg.copy_(exp_avg, non_blocking=True)
            cpu_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)
            if amsgrad and max_exp_avg_sq is not None:
                assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                cpu_max_exp_avg_sq.copy_(max_exp_avg_sq, non_blocking=True)

        DEVICE_MODULE.synchronize()
        return loss
