# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import torch
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import Tensor
from torch import distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from transformers import PreTrainedModel

from xtuner._lite import get_logger, get_torch_device_module

logger = get_logger()

DEVICE_MODULE = get_torch_device_module()


def _group_tensors_by_mesh(
    tensors: List[Tensor],
) -> Dict[Optional[DeviceMesh], List[Tensor]]:
    ret: Dict[Optional[DeviceMesh], List[Tensor]] = {}
    for tensor in tensors:
        if isinstance(tensor, DTensor):
            device_mesh = cast(DTensor, tensor).device_mesh
        else:
            device_mesh = None
        if device_mesh in ret.keys():
            ret[device_mesh].append(tensor)
        else:
            ret[device_mesh] = [tensor]
    return ret


@_no_grad
def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach=None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device

    grouped_grads: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [grads]
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            # If model has applied different parallel strategies for its modules, e.g.
            # VLM apply pure FSDP for vision part and FSDP+TP for language part, grads
            # will have different device meshes. However, for_each operations doesn't
            # support multiple meshes as of torch 2.5.1. We group them manually.
            mesh_grouped_grads = _group_tensors_by_mesh(device_grads)
            for _, mesh_grads in mesh_grouped_grads.items():
                norms.extend(torch._foreach_norm(mesh_grads, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    # torch.stack doesn't support DTensors with different device meshes as of
    # torch 2.5.1. we manually group tensors by meshes, calculate mesh-wise norms
    # and then do reduction
    total_norms: List[Tensor] = []
    mesh_grouped_norms = _group_tensors_by_mesh(norms)
    for _, mesh_norms in mesh_grouped_norms.items():
        total_norm = torch.linalg.vector_norm(
            torch.stack([norm.to(first_device) for norm in mesh_norms]),
            norm_type,
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        total_norms.append(total_norm)
    if len(total_norms) == 1:
        total_norm = total_norms[0]
    else:
        total_norm = torch.linalg.vector_norm(torch.stack(total_norms), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            # If model has applied different parallel strategies for its modules, e.g.
            # VLM apply pure FSDP for vision part and FSDP+TP for language part, grads
            # will have different device meshes. However, for_each operations doesn't
            # support multiple meshes as of torch 2.5.1. We group them manually.
            mesh_grouped_grads = _group_tensors_by_mesh(device_grads)
            for _, mesh_grads in mesh_grouped_grads.items():
                torch._foreach_mul_(mesh_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device.to(g.dtype))

    return total_norm


def download_model_from_hub(
    model_name_or_path: str,
    from_hub: Literal["huggingface", "modelscope"] = "huggingface",
    cache_dir: Optional[str] = None,
) -> str:
    """Automatically download model from the HUB.

    Note:
        If `model_name_or_path` is a local path, it will return the path
        directly without downloading it again.

    Args:
        model_name_or_path (str): The model name, model path or repo id.
        config (str | None): The config path. Default is None.
        from_hub (str): The model hosting hub, modelscope, or huggingface.
            Default is huggingface.
        cache_dir (str | None):
            The save path when downloading the model. If it is None, it
            will be stored in the default location of the HUB. For
            Huggingface, it's ~/.cache/huggingface/hub, for ModelScope,
            it's ~/.cache/modelscope/hub.
    Returns:
        str: The local path of the model.
    """
    if os.path.isdir(model_name_or_path):
        model_path = model_name_or_path
    elif from_hub == "huggingface":
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=model_name_or_path, cache_dir=cache_dir)
    elif from_hub == "modelscope":
        from modelscope import snapshot_download

        model_path = snapshot_download(model_id=model_name_or_path, cache_dir=cache_dir)
    else:
        # TODO support openxlab
        raise NotImplementedError(
            "The model does not support downloading "
            f"from {from_hub}, it only supports "
            "`huggingface` and `modelscope`."
        )

    return model_path


class HFCheckpointLoader:
    def __init__(self, model_path, cache_dir=None, from_hub="huggingface"):
        self.model_path = download_model_from_hub(model_path, from_hub, cache_dir)

        if "model.safetensors.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.safetensors.index.json")
            self.weight_map = json.load(open(index_json))["weight_map"]
            self.use_safetensors = True
        elif "model.bin.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.bin.index.json")
            self.weight_map = json.load(open(index_json))["weight_map"]
            self.use_safetensors = False
        elif "model.safetensors" in os.listdir(self.model_path):
            with safe_open(
                os.path.join(self.model_path, "model.safetensors"), framework="pt"
            ) as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
            self.use_safetensors = True
        else:
            raise FileNotFoundError

        self.current_file = None
        self.buffer = None

    def load(self, key):
        if key not in self.weight_map:
            logger.warning(f"{key} not in checkpoint.")
            return

        _file = self.weight_map[key]

        if self.use_safetensors:
            if self.current_file is None:
                self.buffer = safe_open(
                    os.path.join(self.model_path, _file), framework="pt"
                )
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = safe_open(
                    os.path.join(self.model_path, _file), framework="pt"
                )
                self.current_file = _file
            weight = self.buffer.get_tensor(key)

        else:
            if self.current_file is None:
                self.buffer = torch.load(os.path.join(self.model_path, _file))
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = torch.load(os.path.join(self.model_path, _file))

            weight = self.buffer[key]

        return weight


@torch.no_grad()
def lazy_init_fn(
    module, module2name, checkpoint_loader, enable_fp8=False, ep_mesh=None
):
    device = DEVICE_MODULE.current_device()

    module_name = module2name[module]

    if ".mlp.experts" in module_name:
        params = {}
        ep_rank = ep_mesh.get_local_rank()
        ep_size = ep_mesh.size()
        for name, _ in module.named_parameters(recurse=False):
            if "w1w3" in f"{module_name}.{name}" or "w2" in f"{module_name}.{name}":
                assert "weight" in f"{module_name}.{name}"
                key = f"{module_name}.{name}"
                key = key.replace(".weight", "")
                values = checkpoint_loader.load(key)
                values = values.cuda()
                assert values is not None, key
                values = values.view(module.num_routed_experts, -1, values.shape[-1])
                div_scale = values.shape[0] // ep_size
                values = values[ep_rank * div_scale : (ep_rank + 1) * div_scale]
                values = values.transpose(1, 2)
                values = values.reshape(-1, values.shape[-1])
            else:
                values = checkpoint_loader.load(f"{module_name}.{name}")
                values = values.cuda()
                div_scale = values.shape[0] // ep_size
                values = values[ep_rank * div_scale : (ep_rank + 1) * div_scale]

            params[name] = values
    else:
        params = {}
        for name, _ in module.named_parameters(recurse=False):
            key = f"{module_name}.{name}"
            if "moe_pre_layer" in key:
                key = key.replace(".moe_pre_layer", "")
            params[name] = checkpoint_loader.load(key)

    buffers = {
        name: checkpoint_loader.load(f"{module_name}.{name}")
        for name, _ in module.named_buffers(recurse=False)
        if f"{module_name}.{name}" in checkpoint_loader.weight_map
    }

    module.to_empty(device=DEVICE_MODULE.current_device(), recurse=False)

    for name, param in module.named_parameters(recurse=False):
        dtype = param.dtype

        if params[name] is None:
            continue

        _param = params[name].to(device).to(dtype)

        if isinstance(param, DTensor):
            param = param._local_tensor

        if param.shape == _param.shape:
            param.data.copy_(_param)
        elif enable_fp8 and param.numel() == _param.numel():
            # we flatten the linear weights to handle cases where ngpus > out_features
            param.data.copy_(_param.view(*param.shape))
        else:
            logger.warning(
                f"The shape of {module_name}.{name}({param.shape}) "
                f"is inconsistent with that in the checkpoint({_param.shape}), "
                "it is initialized to 0 by default."
            )
            param.data.zero_()

    for name, buffer in module.named_buffers(recurse=False):
        if name in buffers:
            _buffer = buffers[name].to(device).to(buffer.dtype)

            if buffer.shape == _buffer.shape:
                buffer.data.copy_(_buffer)
            else:
                logger.warning(
                    f"The shape of {module_name}.{name}({buffer.shape}) "
                    f"is inconsistent with that in the checkpoint({_buffer.shape}), "
                    "it is initialized to 0 by default."
                )
                buffer.data.zero_()


@torch.no_grad()
def dense_model_init_weights(module, config=None):
    device = DEVICE_MODULE.current_device()
    module.to_empty(device=device, recurse=False)

    if config is None:
        logger.info("config is None, set default initializer_range=0.02")
        std = 0.02  # default value
    else:
        std = config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    # TODO： more elegant way to handle other modules
    if "RMSNorm" in type(module).__name__:
        module.weight.data.copy_(torch.ones_like(module.weight.data))


@dataclass
class FSDPConfig:
    tp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    reshard_after_forward: bool = True
    recompute_ratio: float = 1.0
    cpu_offload: bool = True
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.bfloat16
    torch_compile: torch.dtype = False
    max_length: Optional[int] = None
    mesh_prefix: str = "default"
    # add fp8-related to FSDPConfig temporarily
    enable_fp8: bool = False
    scaling_granularity_gemm: str = "tensorwise"
    scaling_granularity_grouped_gemm: str = "tilewise"


@dataclass
class ModelConfig:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int


class PatchedCausalLM(ABC, nn.Module):
    def __init__(self, model: PreTrainedModel, fsdp_config: FSDPConfig):
        super().__init__()

    @property
    @abstractmethod
    def rank0_model(self) -> Optional[PreTrainedModel]:
        pass

    @property
    @abstractmethod
    def patched_model(self) -> PreTrainedModel:
        pass

    @property
    @abstractmethod
    def fsdp_config(self) -> FSDPConfig:
        pass

    @property
    @abstractmethod
    def model_config(self) -> ModelConfig:
        pass

    @property
    @abstractmethod
    def data_parallel_mesh(self):
        pass

    @property
    @abstractmethod
    def data_mesh(self):
        pass

    @property
    @abstractmethod
    def sequence_parallel_mesh(self):
        pass

    @abstractmethod
    def dispatch_hf_code(self, model) -> PreTrainedModel:
        pass

    @abstractmethod
    def fully_shard(self):
        pass

    @abstractmethod
    def init_device_mesh(self, parallel_config: FSDPConfig):
        pass

    @abstractmethod
    def trainable_parameters(self) -> List[Dict[str, List[nn.Parameter]]]:
        pass

    @abstractmethod
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        if dist.is_initialized() and dist.is_available():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            self.rank0_model.to(torch.bfloat16)

        from torch.distributed._tensor import DTensor

        dtype = self.patched_model.config.torch_dtype
        for name, param in self.patched_model.state_dict().items():
            if self.fsdp_config.torch_compile and "_orig_mod." in name:
                name = name.replace("_orig_mod.", "")
            if isinstance(param, DTensor):
                full_param = param.to(dtype).full_tensor().cpu()
            else:
                full_param = param.to(dtype).cpu()

            if rank == 0:
                set_module_tensor_to_device(self.rank0_model, name, "cpu", full_param)

        if rank == 0:
            self.rank0_model.save_pretrained(
                save_directory,
                is_main_process,
                state_dict,
                save_function,
                push_to_hub,
                max_shard_size,
                safe_serialization,
                variant,
                token,
                save_peft_format,
                **kwargs,
            )

    # def save_checkpoint(self,
    #                     optimizer: Optional[torch.optim.Optimizer] = None):

    #     # FSDP cannot be saved via torch.save
    #     # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
    #     _options = StateDictOptions(
    #         cpu_offload=True, ignore_frozen_params=True)
    #     (shard_model_state_dict,
    #     shard_optimizer_state_dict) = get_state_dict(
    #         llm, optimizer, options=_options)

    #     state_dict = {
    #         'model': shard_model_state_dict,
    #         'optimizer': shard_optimizer_state_dict,
    #         'train_state': train_state.state_dict(),
    #         'warmup_scheduler': warmup_scheduler.state_dict(),
    #         'cosine_scheduler': cosine_scheduler.state_dict()
    #     }

    #     mkdir_or_exist(ckpt_dir)
    #     ckpt_handle = dcp.async_save(state_dict, checkpoint_id=ckpt_dir, process_group=gloo_group)

    # def load_checkpoint(self,
    #                     checkpoint_id: str,
    #                     optimizer: Optional[torch.optim.Optimizer] = None ):
    #     _options = StateDictOptions(
    #         cpu_offload=True, ignore_frozen_params=True)
    #     (shard_model_state_dict,
    #     shard_optimizer_state_dict) = get_state_dict(
    #         patched_llm.patched_model, optimizer, options=_options)
    #     state_dict = {
    #         'model': shard_model_state_dict,
    #         'optimizer': shard_optimizer_state_dict,
    #         'train_state': train_state,
    #         'warmup_scheduler': warmup_scheduler,
    #         'cosine_scheduler': cosine_scheduler
    #     }

    #     # inplace state_dict
    #     dcp.load(
    #         state_dict=state_dict,
    #         checkpoint_id=latest_checkpoint,
    #     )

    #     _options = StateDictOptions(
    #         cpu_offload=True, strict=False)
    #     set_state_dict(
    #         patched_llm.patched_model,
    #         optimizer,
    #         model_state_dict=state_dict["model"],
    #         optim_state_dict=state_dict["optimizer"],
    #         options=_options
    #     )
