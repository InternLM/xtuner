from collections.abc import Iterable

import torch
import torch.distributed as dist

from xtuner.v1.utils import get_device, get_logger, get_torch_device_module


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class SwapAdamW(torch.optim.AdamW):
    """AdamW optimizer with optimizer-state swap between device and host.

    Optimizer states are initialized once, mirrored to pinned host tensors, and can be swapped to device only when
    needed (e.g. during optimizer step).
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
        self._swap_to_device_events_map: dict[torch.Tensor, torch.Event | None] = {}
        self._swap_to_host_events_map: dict[torch.Tensor, torch.Event | None] = {}
        self._param_to_cpu_states_map: dict[torch.Tensor, dict[str, torch.Tensor | None]] = {}
        self._param_to_device_states_map: dict[torch.Tensor, dict[str, torch.Tensor | None]] = {}
        self._states_on_device = False
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
                device_state_dtensor = self.state[param]
                device_state_tensor: dict[str, torch.Tensor | None] = {}
                cpu_state: dict[str, torch.Tensor | None] = {}
                amsgrad = bool(self._param_to_group_map[param]["amsgrad"])

                for key in self._state_keys:
                    if key == "max_exp_avg_sq" and not amsgrad:
                        device_state_dtensor[key] = None
                        device_state_tensor[key] = None
                        cpu_state[key] = None
                    else:
                        device_state_dtensor[key] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        device_tensor = self._to_local_tensor(device_state_dtensor[key])
                        cpu_tensor = torch.empty_like(device_tensor, pin_memory=True, device="cpu")
                        cpu_tensor.copy_(device_tensor, non_blocking=True)
                        device_tensor.storage().resize_(0)
                        device_state_tensor[key] = device_tensor
                        cpu_state[key] = cpu_tensor

                self._param_to_device_states_map[param] = device_state_tensor
                self._param_to_cpu_states_map[param] = cpu_state

        DEVICE_MODULE.synchronize()

    def swap_all_to_host(self) -> None:
        for param in self._param_to_cpu_states_map:
            self._swap_tensors_to_host(param)
        for param in self._param_to_cpu_states_map:
            event = self._swap_to_host_events_map.get(param, None)
            if event is not None:
                DEVICE_MODULE.current_stream().wait_event(event)
                self._swap_to_host_events_map[param] = None
        self._states_on_device = False

    def swap_all_to_device(self) -> None:
        dist.barrier(dist.group.WORLD)
        for param in self._param_to_cpu_states_map:
            self._swap_tensors_to_device(param)
        for param in self._param_to_cpu_states_map:
            event = self._swap_to_device_events_map.get(param, None)
            if event is not None:
                DEVICE_MODULE.current_stream().wait_event(event)
                self._swap_to_device_events_map[param] = None
        self._states_on_device = True

    def _ensure_states_on_device(self) -> None:
        if self._states_on_device:
            return
        self.swap_all_to_device()

    def _ensure_states_on_host(self) -> None:
        if not self._states_on_device:
            return
        self.swap_all_to_host()

    def prepare_for_checkpoint_save(self) -> None:
        self._ensure_states_on_device()

    def finalize_after_checkpoint_save(self) -> None:
        self._ensure_states_on_host()

    def prepare_for_checkpoint_load(self) -> None:
        self._ensure_states_on_device()

    def finalize_after_checkpoint_load(self) -> None:
        self._ensure_states_on_host()

    def _swap_tensors_to_device(self, param: torch.Tensor) -> None:
        cpu_state = self._param_to_cpu_states_map[param]
        device_state = self._param_to_device_states_map.get(param, None)
        if device_state is None:
            return
        for key in self._state_keys:
            device_tensor = device_state.get(key, None)
            cpu_tensor = cpu_state.get(key, None)
            if device_tensor is None or cpu_tensor is None:
                continue
            if device_tensor.storage().size() == 0:
                device_tensor.storage().resize_(cpu_tensor.storage().size())
                device_tensor.copy_(cpu_tensor, non_blocking=True)

        self._swap_to_device_events_map[param] = DEVICE_MODULE.current_stream().record_event()

    def _swap_tensors_to_host(self, param: torch.Tensor) -> None:
        cpu_state = self._param_to_cpu_states_map[param]
        device_state = self._param_to_device_states_map.get(param, None)
        if device_state is None:
            return
        for key in self._state_keys:
            device_tensor = device_state.get(key, None)
            cpu_tensor = cpu_state.get(key, None)
            if device_tensor is None or cpu_tensor is None:
                continue
            if device_tensor.storage().size() != 0:
                cpu_tensor.copy_(device_tensor, non_blocking=True)
                device_tensor.storage().resize_(0)

        self._swap_to_host_events_map[param] = DEVICE_MODULE.current_stream().record_event()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
                if group["step"].is_cpu:
                    group["step"] = group["step"].to(DEVICE)
            else:
                group["step"] = torch.tensor(1, dtype=torch.int64, device=DEVICE_MODULE.current_device())

        params_list = list(self._param_to_group_map.keys())
        self._ensure_states_on_device()

        for param in params_list:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            group = self._param_to_group_map[param]
            amsgrad = bool(group["amsgrad"])
            beta1, beta2 = group["betas"]
            state = self.state[param]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            max_exp_avg_sq = state.get("max_exp_avg_sq", None)
            assert isinstance(exp_avg, torch.Tensor)
            assert isinstance(exp_avg_sq, torch.Tensor)
            if amsgrad:
                assert isinstance(max_exp_avg_sq, torch.Tensor)

            torch._fused_adamw_(
                [self._to_local_tensor(param)],
                [self._to_local_tensor(param.grad)],
                [self._to_local_tensor(exp_avg)],
                [self._to_local_tensor(exp_avg_sq)],
                [self._to_local_tensor(max_exp_avg_sq)] if amsgrad else [],
                [group["step"]],
                amsgrad=amsgrad,
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

        self._ensure_states_on_host()
        DEVICE_MODULE.synchronize()
        return loss
