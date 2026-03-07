from abc import abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated

from xtuner.v1.optim import Muon
from xtuner.v1.utils import get_logger
import types

logger = get_logger()

class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[float, Parameter(help="Maximum gradient norm for gradient clipping")] = 1.0
    skip_grad_norm_threshold: Annotated[
        float | None, Parameter(help="Gradient norm threshold for skipping optimizer step.")
    ] = None

    @abstractmethod
    def build(self, params):
        pass


class AdamWConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.01
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for Adam optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Adam optimizer")] = 1e-8
    foreach: Annotated[Optional[bool], Parameter(help="Use foreach implementation for AdamW")] = None
    swap_optimizer: Annotated[Optional[bool], Parameter(help="Swap optimizer states to host memory.")] = False

    def build(self, model):
        params = [p for p in model.parameters() if p.requires_grad]

        trainable_parameters_names = model.trainable_parameters()
        trainable_names = [name for name, _ in trainable_parameters_names]
        untrainable_names = []
        num_total_requires_grad = 0
        num_total = 0
        for name, params_ in model.named_parameters():
            num_total += params_.numel()
            num_total_requires_grad += params_.numel() if name in trainable_names else 0
            if name not in trainable_names:
                untrainable_names.append(name)

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad // 1e6}M, total parameters: {num_total // 1e6}M"
            )
            logger.info(f"Untrainable parameters names: {untrainable_names}")
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, foreach=self.foreach
            )
        if self.swap_optimizer:
            SwapOptimizerOperate(optimizer).opt_states_initialization()
            optimizer.step = types.MethodType(swap_adamw_step, optimizer)
        return optimizer

class MuonConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.1
    momentum: Annotated[float, Parameter(help="Momentum coefficients for Muon optimizer")] = 0.95
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for AdamW optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Muon optimizer")] = 1e-8

    def build(self, model):
        trainable_parameters_names = model.trainable_parameters()
        trainable_names = {name for name, _ in trainable_parameters_names}

        untrainable_names = []
        num_total = 0
        num_total_requires_grad = 0
        num_muon = 0
        num_adamw = 0

        for name, p in model.named_parameters():
            n = p.numel()
            num_total += n
            if name in trainable_names:
                num_total_requires_grad += n
                is_muon_tensor = p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
                if is_muon_tensor:
                    num_muon += n
                else:
                    num_adamw += n
            else:
                untrainable_names.append(name)

        muon_params = [
            p
            for name, p in model.named_parameters()
            if name in trainable_names and p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if name in trainable_names and not (p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name)
        ]
        param_groups = [
            dict(params=muon_params),
            dict(params=adamw_params, algorithm="adamw"),
        ]

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad // 1e6}M, total parameters: {num_total // 1e6}M"
            )
            logger.info(f"Muon params: {num_muon // 1e6}M, AdamW params: {num_adamw // 1e6}M (counts by numel)")
            logger.info(f"Untrainable parameters names: {untrainable_names}")
            logger.info(
                f"using Muon optimizer distributed_mesh_size: {model.fsdp_mesh.size()}, "
                f"distributed_mesh: {model.fsdp_mesh}"
            )

        optimizer = Muon(
            param_groups,
            distributed_mesh=model.fsdp_mesh,  # TODO: 暂不支持 EP>1
            lr=self.lr,
            mu=self.momentum,
            betas=self.betas,
            weight_decay=self.weight_decay,
            nesterov=True,
            adjust_lr="rms_norm",
            use_triton=False,
            epsilon=self.eps,
        )
        return optimizer


class LRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr_type: Annotated[Literal["cosine", "linear", "constant"], Parameter(help="Type of learning rate schedule")] = (
        "constant"
    )
    warmup_ratio: Annotated[float, Parameter(help="Ratio of warmup steps to total training steps")] = 0.03
    lr_min: Annotated[float, Parameter(help="Minimum learning rate for optimization")] = 1e-6

class SwapOptimizerOperate():

    swap_to_device_stream = None
    swap_to_host_stream = None

    swap_to_device_events_map = {}
    swap_to_host_events_map = {}

    param_to_cpu_states_map = {}
    param_to_device_states_map = {}

    state_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']

    def __init__(self, optimizer, swap_optimizer_times=16):
        self.optimizer = optimizer
        self.swap_optimizer_times = swap_optimizer_times
        if SwapOptimizerOperate.swap_to_device_stream is None:
            SwapOptimizerOperate.swap_to_device_stream = torch.npu.Stream()
            SwapOptimizerOperate.swap_to_host_stream = torch.npu.Stream()

        # create all parameters list for step
        self.optimizer.param_to_group_map = {}

        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.optimizer.param_to_group_map[p] = group

        # print swap param num and size
        swap_num = sum([main_param.to_local().numel() for main_param in self.optimizer.param_to_group_map])
        swap_numel = swap_num // self.swap_optimizer_times
        self.optimizer.swap_numel = swap_numel
    
        swap_memory = swap_num * 8 / 1024 / 1024
        print('[Rank {}] swap optimizer param num: {},  param size: {}MB\n'.format(torch.npu.current_device(), swap_num, swap_memory), end='')

    def opt_states_initialization(self):
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                device_state_dtensor = self.optimizer.state[param]
                device_state_tensor = {}
                cpu_state = {}
                
                amsgrad = self.optimizer.param_to_group_map[param]['amsgrad']

                for key in self.state_keys:
                    if key == 'max_exp_avg_sq' and not amsgrad:
                        device_state_dtensor[key] = None
                        device_state_tensor[key] = None
                        cpu_state[key] = None
                    else:
                        device_state_dtensor[key] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # convert dtensor to tensor
                        device_state_tensor[key] = device_state_dtensor[key].to_local()
                        
                        cpu_state[key] = torch.empty_like(device_state_tensor[key], pin_memory=True, device='cpu')
                        cpu_state[key].copy_(device_state_tensor[key], non_blocking=True)

                        device_state_tensor[key].storage().resize_(0)

                self.param_to_device_states_map[param] = device_state_tensor
                self.param_to_cpu_states_map[param] = cpu_state
        torch.npu.synchronize()

    @classmethod
    def swap_all_to_host(cls):
        for param in cls.param_to_cpu_states_map.keys():
            cls.swap_tensors_to_host(param)
        for param in cls.param_to_cpu_states_map.keys():
            event = cls.swap_to_host_events_map.get(param, None)
            if event is not None:
                torch.npu.current_stream().wait_event(event)
                cls.swap_to_host_events_map[param] = None     

    @classmethod
    def swap_all_to_device(cls):
        for param in cls.param_to_cpu_states_map.keys():
            cls.swap_tensors_to_device(param)
        for param in cls.param_to_cpu_states_map.keys():
            event = cls.swap_to_device_events_map.get(param, None)
            if event is not None:
                torch.npu.current_stream().wait_event(event)
                cls.swap_to_device_events_map[param] = None     

    @classmethod
    def swap_tensors_to_device(cls, param):
        cpu_state = cls.param_to_cpu_states_map[param]

        if param in cls.param_to_device_states_map:
            device_state = cls.param_to_device_states_map[param]
            for key in cls.state_keys:
                if device_state[key] is not None and device_state[key].storage().size() == 0:
                    device_state[key].storage().resize_(cpu_state[key].storage().size())
                    device_state[key].copy_(cpu_state[key], non_blocking=True)

        cls.swap_to_device_events_map[param] =  torch.npu.current_stream().record_event()

    @classmethod
    def wait_swap_to_device_event(cls, param):
        event = cls.swap_to_device_events_map.get(param, None)
        if event is not None:
            torch.npu.current_stream().wait_event(event)
            cls.swap_to_device_events_map[param] = None

    @classmethod
    def swap_tensors_to_host(cls, param):
        cpu_state = cls.param_to_cpu_states_map[param]

        if param in cls.param_to_device_states_map:
            device_state = cls.param_to_device_states_map[param]
            for key in cls.state_keys:
                if key in device_state and device_state[key] is not None and device_state[key].storage().size() != 0:
                    cpu_state[key].copy_(device_state[key], non_blocking=True)
                    device_state[key].storage().resize_(0)

        cls.swap_to_host_events_map[param] = torch.npu.current_stream().record_event()

def swap_adamw_step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        if 'step' in group:
            group['step'] += 1
            if group['step'].is_cpu:
                group['step'] = group['step'].npu()
        else:
            group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.npu.current_device())
    
    params_list = list(self.param_to_group_map.keys())

    SwapOptimizerOperate.swap_all_to_device()

    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError('AdamW does not support sparse gradients')

        group = self.param_to_group_map[param]
        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']
        state = self.state[param]

        torch._fused_adamw_([param.to_local()], [param.grad.to_local()], [state['exp_avg'].to_local()], [state['exp_avg_sq'].to_local()], [state['max_exp_avg_sq']] if amsgrad else [],
                                 [group['step']], amsgrad=amsgrad, lr=group['lr'], beta1=beta1, beta2=beta2, weight_decay=group['weight_decay'],
                                 eps=group['eps'], maximize=group['maximize'])

    # it maybe removed 
    torch.npu.synchronize()
    return loss