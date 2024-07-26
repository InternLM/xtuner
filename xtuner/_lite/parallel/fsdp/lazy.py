import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, distribute_tensor


@torch.no_grad
def dp_lazy_init(module, module_map, dp_mesh):
    device = torch.cuda.current_device()
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

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
    device = torch.cuda.current_device()
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

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
def dp_tp_lazy_init(module, module_map, dp_mesh, tp_mesh):
    device = torch.cuda.current_device()
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

    if dp_mesh.get_local_rank() != 0:
        return

    if tp_mesh.get_local_rank() == 0:
        master_module = module_map[module]
        master_params = {
            name: param
            for name, param in master_module.named_parameters(recurse=False)
        }
        master_buffers = {
            name: buffer
            for name, buffer in master_module.named_buffers(recurse=False)
        }
    else:
        master_params = None
        master_buffers = None

    for name, param in module.named_parameters(recurse=False):
        if isinstance(param, DTensor):

            p_full = param.full_tensor()
            if tp_mesh.get_local_rank() == 0:
                p_copy = master_params[name]
                p_copy = p_copy.to(device).to(param.dtype)
            else:
                p_copy = torch.empty_like(p_full)

            mesh = param.device_mesh
            placements = param.placements

            p_dtensor = distribute_tensor(p_copy, mesh, placements)
            param.data.copy_(p_dtensor)

        else:
            if tp_mesh.get_local_rank() == 0:
                p_copy = master_params[name]
                p_copy = p_copy.to(device).to(param.dtype)
            else:
                p_copy = torch.empty_like(param)

            tp_group = tp_mesh.get_group()
            dist.broadcast(p_copy, 0, tp_group)
            param.data.copy_(p_copy)

    for name, buffer in module.named_buffers(recurse=False):

        if isinstance(buffer, DTensor):

            b_full = buffer.full_tensor()
            if tp_mesh.get_local_rank() == 0:
                b_copy = master_buffers[name]
                b_copy = b_copy.to(device).to(buffer.dtype)
            else:
                b_copy = torch.empty_like(b_full)

            mesh = buffer.device_mesh
            placements = buffer.placements

            b_dtensor = distribute_tensor(b_copy, mesh, placements)
            buffer.data.copy_(b_dtensor)

        else:
            if tp_mesh.get_local_rank() == 0:
                b_copy = master_buffers[name]
                b_copy = b_copy.to(device).to(buffer.dtype)
            else:
                b_copy = torch.empty_like(buffer)

            tp_group = tp_mesh.get_group()
            dist.broadcast(b_copy, 0, tp_group)
            buffer.data.copy_(b_copy)


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
