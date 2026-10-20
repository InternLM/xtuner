from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.distributed.tensor import DTensor
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

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load optimizer state without staging moments on the accelerator.

        AdamW.load_state_dict deep-copies and casts every saved moment to the
        parameter device before any post hook runs, so a resume would briefly
        hold the full optimizer state (2-3x parameter size) in accelerator
        memory. SwapAdamW exists because those moments do not fit there. The
        moments are extracted first so only the scalar ``step`` tensors go
        through the base cast, and the swap states are then rebuilt from
        host-side tensors without a device round-trip. Distributed checkpoint
        loads funnel through this method as well.
        """
        saved_state = {
            param_id: dict(per_param_state) for param_id, per_param_state in (state_dict.get("state") or {}).items()
        }
        moments: dict[int, dict[str, torch.Tensor]] = {
            param_id: {key: state.pop(key) for key in tuple(state) if key in self._state_keys}
            for param_id, state in saved_state.items()
        }
        super().load_state_dict({**state_dict, "state": saved_state})
        if any(moments.values()):
            saved_id_to_param = {
                param_id: param
                for saved_group, group in zip(state_dict["param_groups"], self.param_groups)
                for param_id, param in zip(saved_group["params"], group["params"])
            }
            for param_id, per_param_state in moments.items():
                param = saved_id_to_param.get(param_id)
                if param is not None:
                    self.state.setdefault(param, {}).update(per_param_state)
        self._init_swap_states()

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()  # type: ignore[no-any-return]
        return tensor

    def _init_swap_states(self) -> None:
        self._param_to_group_map.clear()
        self._param_to_cpu_states_map.clear()
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
                cpu_state_dtensor = self.state.setdefault(param, {})
                cpu_state: dict[str, torch.Tensor | None] = {}
                amsgrad = bool(self._param_to_group_map[param]["amsgrad"])
                local_param = self._to_local_tensor(param)

                for key in self._state_keys:
                    if key == "max_exp_avg_sq" and not amsgrad:
                        cpu_state_dtensor[key] = None
                        cpu_state[key] = None
                    else:
                        restored = cpu_state_dtensor.get(key)
                        if isinstance(param, DTensor):
                            if restored is not None and (
                                not isinstance(restored, DTensor) or restored.shape != param.shape
                            ):
                                raise ValueError("Sharded SwapAdamW requires moments with complete DTensor metadata")
                            # DCP treats plain local tensors as replicated and saves
                            # only one rank. Keep global shape/placements in state;
                            # the swap map below aliases its pinned local storage.
                            state_tensor = torch.empty_like(param, device="cpu", pin_memory=True)
                            assert isinstance(state_tensor, DTensor)
                            cpu_tensor = state_tensor.to_local()
                            if restored is None:
                                cpu_tensor.zero_()
                            else:
                                cpu_tensor.copy_(restored.to_local())
                            cpu_state_dtensor[key] = state_tensor
                        elif restored is None:
                            cpu_tensor = torch.zeros_like(
                                local_param, device="cpu", memory_format=torch.preserve_format
                            ).pin_memory()
                            cpu_state_dtensor[key] = cpu_tensor
                        else:
                            cpu_tensor = self._to_local_tensor(restored).to(device="cpu").pin_memory()
                            cpu_state_dtensor[key] = cpu_tensor
                        cpu_state[key] = cpu_tensor

                # Keep per-parameter step semantics consistent with torch.optim.AdamW.
                step = cpu_state_dtensor.get("step")
                cpu_state_dtensor["step"] = (
                    torch.tensor(0.0, dtype=torch.float32, device="cpu")
                    if step is None
                    else self._to_local_tensor(step).to(device="cpu", dtype=torch.float32)
                )

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
