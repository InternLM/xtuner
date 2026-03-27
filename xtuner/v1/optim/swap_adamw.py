from collections.abc import Iterable

import torch

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

                self._param_to_cpu_states_map[param] = cpu_state

        DEVICE_MODULE.synchronize()

    @staticmethod
    def _next_step(group: dict) -> int:
        step = group.get("step", 0)
        if isinstance(step, torch.Tensor):
            next_step = int(step.item()) + 1
        else:
            next_step = int(step) + 1
        group["step"] = next_step
        return next_step

    @torch.no_grad()
    def step(self, closure=None):
        def adamw_step(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            maximize,
        ):
            with torch.no_grad():
                """完全等价的非融合实现."""
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # 更新一阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新二阶动量
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # AMSGrad: 取历史最大值
                    if max_exp_avg_sq is None:
                        max_exp_avg_sq = exp_avg_sq.clone()
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2**0.5)) + eps
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)) + eps

                step_size = lr / bias_correction1

                # 参数更新
                if maximize:
                    param.addcdiv_(exp_avg, denom, value=step_size)
                else:
                    param.addcdiv_(exp_avg, denom, value=-step_size)

                # 权重衰减（解耦）
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_list = list(self._param_to_group_map.keys())
        step_tensors: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            step_value = self._next_step(group)
            step_tensors[id(group)] = torch.tensor(
                step_value, dtype=torch.int64, device=DEVICE_MODULE.current_device()
            )

        for param in params_list:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            group = self._param_to_group_map[param]
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

            adamw_step(
                self._to_local_tensor(param),
                self._to_local_tensor(param.grad),
                self._to_local_tensor(exp_avg),
                self._to_local_tensor(exp_avg_sq),
                self._to_local_tensor(max_exp_avg_sq),
                step_tensors[id(group)],
                amsgrad=amsgrad,
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

            cpu_exp_avg.copy_(exp_avg, non_blocking=True)
            cpu_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)
            if amsgrad and max_exp_avg_sq is not None:
                assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                cpu_max_exp_avg_sq.copy_(max_exp_avg_sq, non_blocking=True)

        DEVICE_MODULE.synchronize()
        return loss
