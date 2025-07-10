from typing import Any, Optional, Tuple

import torch
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated


class FSDPConfig(BaseModel):
    tp_size: Annotated[int, Parameter(help="Tensor parallel size")] = 1
    sp_size: Annotated[int, Parameter(help="Sequence parallel size")] = 1
    ep_size: Annotated[int, Parameter(help="Expert parallel size")] = 1
    reshard_after_forward: Annotated[bool, Parameter(help="Reshard model parameters after forward pass")] = True
    recompute_ratio: Annotated[float, Parameter(help="Gradient checkpointing ratio for memory optimization")] = 1.0
    cpu_offload: Annotated[bool, Parameter(help="Enable CPU offloading for memory optimization")] = True
    param_dtype: Annotated[torch.dtype, Parameter(help="Data type for model parameters")] = torch.bfloat16
    reduce_dtype: Annotated[torch.dtype, Parameter(help="Data type for reduction operations")] = torch.bfloat16
    torch_compile: Annotated[bool, Parameter(help="Enable model compilation for faster inference")] = False
    compile_targets: Annotated[
        Optional[Tuple[str, ...]],
        Parameter(
            help="Specific targets for compilation, e.g. ('module.MyClass.method', 'module.function'). If None, all eligible functions will be compiled."
        ),
    ] = None
    mesh_prefix: Annotated[str, Parameter(help="Prefix for device mesh configuration in distributed training")] = (
        "default"
    )
    requires_grad: Annotated[bool, Parameter(help="Enable gradient computation for model parameters")] = True
    hsdp_sharding_size: Annotated[
        Optional[int], Parameter(help="Sharding size for HSDP (Hybrid Sharding Data Parallel)")
    ] = None

    # todo
    max_length: Annotated[Optional[int], Parameter(help="Maximum sequence length for input tokens")] = None

    # Unable to generate pydantic-core schema for <class 'torch.dtype'>.
    # Set `arbitrary_types_allowed=True` in the model_config to ignore this error
    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        if self.hsdp_sharding_size is not None:
            assert self.ep_size == 1, "Currently, HSDP requires expert parallel size to be 1"
