from collections.abc import Iterable

import torch
from torch.optim.adam import adam as torch_adam

from xtuner.v1.utils import get_device, get_logger, get_torch_device_module


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class SwapAdamW(torch.optim.AdamW):
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
                        cpu_tensor = torch.zeros_like(local_param, memory_format=torch.preserve_format)
                        cpu_tensor = cpu_tensor.to(device="cpu", non_blocking=True)
                        cpu_tensor = cpu_tensor.pin_memory()
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
