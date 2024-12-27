import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, distribute_tensor

from xtuner._lite import get_logger, get_torch_device_module

logger = get_logger()

DEVICE_MODULE = get_torch_device_module()


@torch.no_grad
def dp_lazy_init(module, module_map, dp_mesh):
    device = DEVICE_MODULE.current_device()
    module.to_empty(device=DEVICE_MODULE.current_device(), recurse=False)

    if dp_mesh.get_local_rank() == 0:
        master_module = module_map[module]
        master_params = {
            name: param
            for name, param in master_module.named_parameters(recurse=False)
        }
        master_buffers = {
            name: buffer
            for name, buffer in master_module.named_buffers(recurse=False)
        }

        for name, param in module.named_parameters(recurse=False):

            p_copy = master_params[name].to(device).to(param.dtype)
            # if param.requires_grad:
            #     p_copy = p_copy.to(device).to(param.dtype)
            # else:
            #     p_copy = p_copy.to(device)
            param.data.copy_(p_copy)

        for name, buffer in module.named_buffers(recurse=False):

            b_copy = master_buffers[name].to(device).to(buffer.dtype)
            # b_copy = b_copy.to(device)
            buffer.data.copy_(b_copy)


@torch.no_grad
def dp_sp_lazy_init(module, module_map, dp_mesh, sp_mesh):
    device = DEVICE_MODULE.current_device()
    module.to_empty(device=DEVICE_MODULE.current_device(), recurse=False)

    if dp_mesh.get_local_rank() == 0 and sp_mesh.get_local_rank() == 0:
        master_module = module_map[module]
        master_params = {
            name: param
            for name, param in master_module.named_parameters(recurse=False)
        }
        master_buffers = {
            name: buffer
            for name, buffer in master_module.named_buffers(recurse=False)
        }

        for name, param in module.named_parameters(recurse=False):
            p_copy = master_params[name].to(device).to(param.dtype)
            param.data.copy_(p_copy)

        for name, buffer in module.named_buffers(recurse=False):
            b_copy = master_buffers[name].to(device).to(buffer.dtype)
            buffer.data.copy_(b_copy)


@torch.no_grad
def lazy_init_megatron(module, rank0_map, dp_mesh, tp_mesh=None, pp_mesh=None):
    device = DEVICE_MODULE.current_device()

    if dp_mesh.get_rank() == 0:
        rank0_module = rank0_map[module]
        rank0_params = {
            name: param
            for name, param in rank0_module.named_parameters(recurse=False)
        }
        rank0_buffers = {
            name: buffer
            for name, buffer in rank0_module.named_buffers(recurse=False)
        }
    else:
        rank0_params = None
        rank0_buffers = None

    param_shapes = {
        name: param.full_tensor().shape
        if isinstance(param, DTensor) else param.shape
        for name, param in module.named_parameters(recurse=False)
    }

    module.to_empty(device=DEVICE_MODULE.current_device(), recurse=False)

    for name, param in module.named_parameters(recurse=False):
        dtype = param.dtype
        if dp_mesh.get_rank() == 0:
            rank0_param = rank0_params[name].to(device).to(dtype)
        else:
            full_shape = param_shapes[name]
            rank0_param = torch.zeros(full_shape, dtype=dtype, device=device)

        dist.broadcast(rank0_param, src=0)

        if isinstance(param, DTensor):
            mesh = param.device_mesh
            assert mesh == tp_mesh
            placements = param.placements
            rank0_param = distribute_tensor(rank0_param, mesh, placements)

        param.data.copy_(rank0_param)
        dist.barrier()

    # TP does not shard buffers
    for name, buffer in module.named_buffers(recurse=False):
        if dp_mesh.get_rank() == 0:
            rank0_buffer = rank0_buffers[name].to(device).to(buffer.dtype)
        else:
            rank0_buffer = torch.empty_like(buffer).to(device)

        dist.broadcast(rank0_buffer, src=0)
        buffer.data.copy_(rank0_buffer)


class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_
