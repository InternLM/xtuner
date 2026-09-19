from typing import Optional

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator
from typing_extensions import Annotated


class FSDPConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        extra="forbid",
    )
    tp_size: Annotated[int, Parameter(help="Tensor parallel size")] = 1
    ep_size: Annotated[int, Parameter(help="Expert parallel size")] = 1
    reshard_after_forward: Annotated[bool, Parameter(help="Reshard model parameters after forward pass")] = True
    recompute_ratio: Annotated[float, Parameter(help="Gradient checkpointing ratio for memory optimization")] = 1.0
    vision_recompute_ratio: Annotated[float, Parameter(help="Recompute ratio for vision modules")] = 1.0
    checkpoint_preserve_rng_state: Annotated[bool, Parameter(help="Preserve RNG state during checkpointing")] = True
    # Training-time FSDP CPU offload is version-sensitive for XTuner model configs
    # that keep selected fp32 trainable parameters outside FSDP via
    # fp32_keys_pattern. The Qwen3.5-VL MoE RL path was verified to run on Torch
    # 2.9, but on PyTorch 2.12.x the combination of ignored fp32 params,
    # checkpointed forward/backward, and CPU offload can trigger autograd's
    # internal stream assertion:
    #   opt_ready_stream && opt_parent_stream
    # Keep this disabled unless the target model/config has explicitly validated
    # the full-offload path on the target PyTorch version.
    cpu_offload: Annotated[bool, Parameter(help="Enable CPU offloading for memory optimization")] = False
    # TODO: (caoweihan) Convert `torch.dtype` to `Annotated` for compatibility with cyclopts
    param_dtype: Annotated[torch.dtype, Parameter(help="Data type for model parameters")] = torch.bfloat16
    reduce_dtype: Annotated[torch.dtype, Parameter(help="Data type for reduction operations")] = torch.bfloat16
    fp32_lm_head: Annotated[bool, Parameter(help="Use float32 for language model head")] = False
    # TODO: deprecate `torch_compile` in favor of `compile_cfg` in XTunerBaseModelConfig
    torch_compile: Annotated[bool, Parameter(help="Enable model compilation for faster inference")] = True
    mesh_prefix: Annotated[str, Parameter(help="Prefix for device mesh configuration in distributed training")] = (
        "default"
    )
    requires_grad: Annotated[bool, Parameter(help="Enable gradient computation for model parameters")] = True
    hsdp_sharding_size: Annotated[
        Optional[int], Parameter(help="Sharding size for HSDP (Hybrid Sharding Data Parallel)")
    ] = None
    # Decoupled EP/FSDP ("dp2ep") layout. When enabled, `ep_size` is a sub-dimension of the
    # FSDP shard dimension instead of being orthogonal to it: routed experts are sharded
    # `dp_shard / ep_size` ways on top of EP, while every other parameter is sharded over the
    # full `dp_shard` (= `hsdp_sharding_size` or world size) without being replicated across EP
    # ranks. Requires `dp_shard % ep_size == 0`. When disabled, the legacy layout is untouched.
    decouple_ep_fsdp: Annotated[
        bool, Parameter(help="Decouple expert parallel from FSDP: shard dense params over the full FSDP mesh")
    ] = False

    @model_validator(mode="after")
    def _validate_ep_fsdp_topology(self) -> "FSDPConfig":
        # Explicit `ValueError`s instead of `assert`: the topology checks must survive `python -O`,
        # and pydantic reports them as a `ValidationError` when the config is built.
        if self.ep_size < 1:
            raise ValueError(f"`ep_size` must be a positive integer, got {self.ep_size}")
        if self.hsdp_sharding_size is None:
            return self
        if self.hsdp_sharding_size < 1:
            raise ValueError(f"`hsdp_sharding_size` must be a positive integer, got {self.hsdp_sharding_size}")
        if self.decouple_ep_fsdp:
            if self.hsdp_sharding_size % self.ep_size != 0:
                raise ValueError(
                    "`decouple_ep_fsdp` requires `hsdp_sharding_size` to be divisible by `ep_size`, "
                    f"got hsdp_sharding_size={self.hsdp_sharding_size}, ep_size={self.ep_size}"
                )
        elif self.ep_size != 1:
            raise ValueError("Currently, HSDP requires expert parallel size to be 1")
        return self

    @field_serializer("param_dtype", "reduce_dtype")
    def serialize_param_dtype(self, value: torch.dtype) -> str:
        return str(value)

    @field_validator("param_dtype", "reduce_dtype", mode="before")
    @classmethod
    def deserialize_param_dtype(cls, value: str | torch.dtype) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        elif isinstance(value, str):
            if "bfloat16" in value:
                return torch.bfloat16
            elif "float16" in value or "half" in value:
                return torch.float16
            elif "float32" in value or "float" in value:
                return torch.float32
            else:
                raise ValueError()
        else:
            return value
