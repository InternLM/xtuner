import importlib
import json
import multiprocessing as py_mp
import os
import pydoc
import re
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from itertools import chain
from pathlib import Path
from shutil import copy, copytree, rmtree
from typing import Annotated, Any, Generator, Iterable, Literal, Mapping, NamedTuple, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from cyclopts import Parameter
from more_itertools import consume
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, computed_field, model_validator
from pydantic.fields import FieldInfo
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from torch.utils import _pytree
from typing_extensions import NotRequired, Self, TypedDict, overload

from transformers.configuration_utils import PretrainedConfig
from xtuner.v1.config import FSDPConfig, GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTensorWiseFloat8CastTensor,
    WeightWithDynamicTilewiseFloat8CastTensor,
)
from xtuner.v1.loss import BaseLossConfig, BaseLossContext, CELossConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig, MLAConfig
from xtuner.v1.module.rope import RopeParametersConfig, RopeScalingConfig
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module, log_rank0, profile_time_and_memory
from xtuner.v1.utils.compile import MaybeCompile, is_compiled_function, maybe_compile
from xtuner.v1.utils.load_spec import (
    HFSavePlan,
    LoadSpec,
    unshard_tensors_for_hf_save,
)
from xtuner.v1.utils.loader import HFCheckpointLoader
from xtuner.v1.utils.misc import FunctionEnum, FunctionType, get_function_full_qualname, get_function_type
from xtuner.v1.utils.process import (
    get_async_hf_save_file_lock_slots,
    get_async_hf_writer_join_timeout,
    set_async_save_process_qos,
)

from .utils import ModelForwardExtraLogInfo


logger = get_logger()

DEVICE_MODULE = get_torch_device_module()
DEVICE = get_device()


class DataBatchInfo(TypedDict):
    step_consumed_tokens: int
    step_seqlen_tokens: int
    step_consumed_img_tokens: float
    efficient_attn_ratio: float
    img_efficient_attn_ratio: float


class BatchForwardInfo(TypedDict):
    logs_info: dict[str, float]
    extra_info: ModelForwardExtraLogInfo


class _HFSavePlan(TypedDict):
    hf_dir: Path
    save_tasks: list[tuple[str, dict[str, torch.Tensor]]]


@dataclass
class AsyncHFSaveHandle:
    process: Any
    hf_dir: Path
    tmp_hf_dir: Path


@dataclass
class AsyncHFResources:
    finalize_pg: dist.ProcessGroup | None
    commit_executor: ThreadPoolExecutor


class TorchCompileOption(TypedDict):
    fullgraph: NotRequired[bool]
    dynamic: NotRequired[bool | None]
    mode: NotRequired[str | None]
    options: NotRequired[dict[str, int | bool | str] | None]


class HFSaveCfg(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_per_rank: Annotated[int, Parameter(group="model")] = 16
    max_save_rank: Annotated[int, Parameter(group="model")] = 16
    # Max bytes per generated safetensors shard.
    bucket_size: Annotated[int, Parameter(group="model")] = 1024**3 * 4
    # TODO: `XTunerBaseModel` should also be able to specify which parameters to be trained in fp32,
    # currently it could only be specified in HFSaveCfg
    # Each entry is a **regex** pattern (passed to `re.search`) matched against the HF parameter name.
    # Remember to escape literal dots, e.g. use r"model\.layers\.\d+\.weight" instead of
    # r"model.layers.\d+.weight" to avoid unintended wildcard matches.
    fp32_keys_pattern: Annotated[list[str] | None, Parameter(group="model")] = None


class XTunerBaseModelConfig(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")
    hf_save_cfg: HFSaveCfg = HFSaveCfg()
    float8_cfg: Float8Config | None = None
    compile_cfg: Annotated[
        dict[str, TorchCompileOption] | None | bool,
        Parameter(
            group="model",
            help="The compile config of model. "
            "`None` | `True`: Use default compile config defined in model, "
            "`False`: Disable the compile"
            "`dict[str, TorchCompileOption]`: Customize the compile option",
        ),
    ] = None
    hf_key_mapping: Annotated[dict[str, str] | None, "Remapping hf key based on the `to_hf_key_list`"] = None
    dcp_ignore_frozen_params: bool = True
    lm_loss_cfg: BaseLossConfig = CELossConfig()

    @property
    def hf_config(self) -> PretrainedConfig | None:
        return None

    def save_hf(self, hf_path: str | Path):
        if self.hf_config is None:
            raise NotImplementedError("The `hf_config` property must be implemented to save in HuggingFace format.")

        self.hf_config.save_pretrained(hf_path)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        """Build a `TransformerConfig` from a pre-trained HuggingFace model.

        This method creates a configuration object based on a `PretrainedConfig` loaded from the specified HuggingFace model path.
        If you want to use this method, you must implement it in a subclass to correctly extract and map configuration parameters.

        Note:
            The `hf_config` field needs to be set to the `PretrainedConfig` object loaded from `hf_path`,
            otherwise it cannot be saved in HuggingFace format.

        Args:
            hf_path (str | Path): Path to the HuggingFace model.

        Returns:
            TransformerConfig: A configuration object populated with values from the pre-trained model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def build(self):
        raise NotImplementedError


DEFAULT_FLOAT8_CFG = {
    "xtuner.v1.float8.fsdp_utils.tensor_to_per_block_fp8_scales": TorchCompileOption(fullgraph=True),
    "xtuner.v1.float8.fsdp_utils.cast_to_per_block_fp8_with_scales": TorchCompileOption(fullgraph=True),
    "xtuner.v1.float8.triton_kernels.per_block_quant_gemm.per_block_quant_torch": TorchCompileOption(fullgraph=True),
    "xtuner.v1.float8.fsdp_utils.cast_to_per_tensor_fp8_with_scales": TorchCompileOption(fullgraph=True),
    "xtuner.v1.float8.float8_linear_tensor_wise.per_tensor_fp8_quant": TorchCompileOption(fullgraph=True),
}


class TransformerConfig(XTunerBaseModelConfig):
    """Base transformer configuration with unified RoPE parameters.

    This config uses `rope_parameters_cfg` as the primary source of truth for all RoPE-related
    settings. The legacy fields `rope_theta` and `rope_scaling_cfg` are kept for backward
    compatibility and are synchronized with `rope_parameters_cfg` via model validator.

    For new code, use `rope_parameters_cfg` directly. For loading old configs or HF models,
    use `RopeParametersConfig.from_legacy_cfg()` or `RopeParametersConfig.from_hf_config()`.
    """

    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
        protected_namespaces=(),
    )
    vocab_size: Annotated[int, Parameter(group="model")]
    max_position_embeddings: Annotated[int, Parameter(group="model")]
    eos_token_id: Annotated[int, Parameter(group="model")]
    pad_token_id: Annotated[int | None, Parameter(group="model")] = None
    num_hidden_layers: Annotated[int, Parameter(group="model")]
    hidden_size: Annotated[int, Parameter(group="model")]
    intermediate_size: Annotated[int, Parameter(group="model")]
    rms_norm_eps: Annotated[float, Parameter(group="model")]
    rms_norm_type: Annotated[Literal["default", "zero_centered"], Parameter(group="model")] = "default"
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    attention: MLAConfig | MHAConfig
    linear_attention: Annotated[GatedDeltaNetConfig | None, Parameter(group="model")] = None
    mlp_bias: Annotated[bool, Parameter(group="model")] = False
    tie_word_embeddings: Annotated[bool, Parameter(group="model")] = False
    model_type: Annotated[str | None, Parameter(group="model")] = None  # TODO: yehaochen maybe should be removed
    generate_config: GenerateConfig | None = None
    return_hidden_states: Annotated[bool, Parameter(group="model")] = False
    use_sliding_window: Annotated[bool, Parameter(group="model")] = False
    max_window_layers: Annotated[int | None, Parameter(group="model")] = None
    rope_parameters_cfg: Annotated[RopeParametersConfig | None, Parameter(group="model")] = Field(
        default_factory=RopeParametersConfig
    )
    mesh_prefix: Annotated[str, Parameter(help="Prefix for device mesh configuration in distributed training")] = (
        "default"
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_rope_params(cls, data: Any) -> Any:
        """Handle legacy rope_theta and rope_scaling_cfg construction
        parameters.

        Converts rope_theta and rope_scaling_cfg into rope_parameters_cfg before validation.
        """
        if not isinstance(data, dict):
            return data

        # Make a copy to avoid modifying the input
        data = dict(data)

        # Extract legacy parameters
        legacy_rope_theta = data.pop("rope_theta", None)
        legacy_rope_scaling_cfg = data.pop("rope_scaling_cfg", None)

        rope_params_field = cls.model_fields.get("rope_parameters_cfg")
        if isinstance(rope_params_field, FieldInfo):
            default_rope_params_cfg = rope_params_field.get_default(call_default_factory=True)
        else:
            default_rope_params_cfg = None
        default_params_data = default_rope_params_cfg.model_dump() if default_rope_params_cfg is not None else {}

        # Get existing rope_parameters_cfg if any
        rope_params_data = data.get("rope_parameters_cfg")
        if isinstance(rope_params_data, RopeParametersConfig):
            rope_params_data = rope_params_data.model_dump()
        elif isinstance(rope_params_data, dict):
            rope_params_data = dict(rope_params_data)
        else:
            # legacy case
            rope_params_data = {}

            # Apply legacy rope_theta if provided
            if legacy_rope_theta is not None:
                rope_params_data["rope_theta"] = legacy_rope_theta

            # Apply legacy rope_scaling_cfg if provided
            if legacy_rope_scaling_cfg is not None:
                # Convert dict to RopeScalingConfig if needed
                if isinstance(legacy_rope_scaling_cfg, dict):
                    legacy_rope_scaling_cfg = RopeScalingConfig(**legacy_rope_scaling_cfg)

                for src_field, dst_field in RopeParametersConfig._get_rope_scaling_to_parameters_mapping().items():
                    if hasattr(legacy_rope_scaling_cfg, src_field):
                        value = getattr(legacy_rope_scaling_cfg, src_field)
                        if value is not None:
                            rope_params_data[dst_field] = value

        # Replace rope_parameters_cfg by the updated default_params_data with new values
        default_params_data.update(rope_params_data)
        data["rope_parameters_cfg"] = RopeParametersConfig(**default_params_data)

        return data

    @property
    def rope_theta(self) -> float | None:
        """Get rope_theta from rope_parameters_cfg (backward compatibility)."""
        return self.rope_parameters_cfg.rope_theta if self.rope_parameters_cfg is not None else None

    @rope_theta.setter
    def rope_theta(self, value: float | None) -> None:
        """Set rope_theta and update rope_parameters_cfg (backward
        compatibility)."""
        params_dict = self.rope_parameters_cfg.model_dump() if self.rope_parameters_cfg is not None else {}
        params_dict["rope_theta"] = value
        self.rope_parameters_cfg = RopeParametersConfig(**params_dict)

    @property
    def rope_scaling_cfg(self) -> RopeScalingConfig | None:
        """Get RopeScalingConfig from rope_parameters_cfg (backward
        compatibility).

        Returns None if rope_type is default and no FoPE is used.
        """
        if self.rope_parameters_cfg is None or (
            self.rope_parameters_cfg.rope_type == "default" and not self.rope_parameters_cfg.use_fope
        ):
            return None

        # Build RopeScalingConfig from rope_parameters_cfg dynamically
        kwargs = {"type": self.rope_parameters_cfg.rope_type}

        for field_name in RopeParametersConfig.get_rope_scaling_field_names():
            value = getattr(self.rope_parameters_cfg, field_name)
            if value is not None:
                kwargs[field_name] = value

        return RopeScalingConfig(**kwargs)

    @rope_scaling_cfg.setter
    def rope_scaling_cfg(self, value: RopeScalingConfig | None) -> None:
        """Set rope_scaling_cfg and update rope_parameters_cfg (backward
        compatibility)."""
        params_dict = self.rope_parameters_cfg.model_dump() if self.rope_parameters_cfg is not None else {}

        if value is None:
            # Reset to default rope_type and clear scaling parameters dynamically
            params_dict["rope_type"] = "default"
            for field_name in RopeParametersConfig.get_rope_scaling_field_names():
                params_dict[field_name] = None
            params_dict["partial_rotary_factor"] = 1.0
            params_dict["truncate"] = False
        else:
            for src_field, dst_field in RopeParametersConfig._get_rope_scaling_to_parameters_mapping().items():
                if hasattr(value, src_field):
                    field_value = getattr(value, src_field)
                    if field_value is not None:
                        params_dict[dst_field] = field_value

        self.rope_parameters_cfg = RopeParametersConfig(**params_dict)

    @computed_field  # type: ignore[misc]
    @property
    def rope_scaling(self) -> dict | None:
        """Get rope_scaling dict for HF compatibility."""
        return self.rope_parameters_cfg.to_rope_scaling_dict() if self.rope_parameters_cfg is not None else None

    def standardize_rope_params(self):
        # This method is for compatibility with transformers 5.x rope_utils
        # XTuner now use rope_parameters_cfg as the single source of truth for all RoPE-related settings.
        # No need to standardize rope parameters

        # function call stack(typical path):
        # xtuner.v1.model.base.build_rotary_embedding ->
        # xtuner.v1.module.rope.get_rope_embedding ->
        # xtuner.v1.module.rope.RotaryEmbedding.__init__ ->
        # self.rope_type != "default", self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type] ->
        # transformers.modeling_rope_utils._compute_yarn_parameters ->
        # config.standardize_rope_params(), here config is a TransformerConfig in xtuner
        pass

    @computed_field  # type: ignore[misc]
    @property
    def rope_parameters(self) -> dict | None:
        """Get rope_parameters for HF compatibility."""
        return self.rope_parameters_cfg.to_rope_parameters_dict() if self.rope_parameters_cfg is not None else None

    @computed_field
    def num_attention_heads(self) -> int:
        return self.attention.num_attention_heads

    @computed_field
    def num_key_value_heads(self) -> int:
        return self.attention.num_key_value_heads

    @computed_field
    def head_dim(self) -> int:
        return self.attention.head_dim

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention", "linear_attention"]]:
        if not self.use_sliding_window:
            return ["full_attention"] * self.num_hidden_layers
        else:
            if self.max_window_layers is None:
                return ["sliding_attention"] * self.num_hidden_layers
            return [
                "sliding_attention" if i >= self.max_window_layers else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


class ModelOutputs(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    hidden_states: list[torch.Tensor] | None = None
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None  # TODO: `forward_only` mode for RL
    extra_info: ModelForwardExtraLogInfo | dict | None = None  # TODO: `forward_only` mode for RL

    def free_nongrad_feature(self):
        """Release large intermediate tensors not needed for backward or
        logging.

        This method is called immediately after forward() in the micro-batch loop.
        It releases large tensors (logits, hidden_states) while keeping:
        - loss: needed for backward pass
        - extra_info: lightweight logging info needed by post_micro_batch_forward()
        """
        self.hidden_states = None
        self.logits = None

    # TODO: Only for avoid BC. Should be removed later.
    def __getitem__(self, key):
        return getattr(self, key)

    # TODO: Only for avoid BC. Should be removed later.
    def __contains__(self, key):
        return key in self.model_fields_set

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Automatically register every subclass as a pytree node so that
        # FSDP can traverse the output tensors and insert pre_backward_hooks.
        super().__init_subclass__(**kwargs)
        cls._register_pytree_node()

    @staticmethod
    def _model_field_names(model_type: type[PydanticBaseModel]) -> list[str]:
        return list(model_type.model_fields)

    @staticmethod
    def _flatten_pydantic_model(
        model: PydanticBaseModel,
    ) -> tuple[list[Any], tuple[type[PydanticBaseModel], list[str]]]:
        # Flatten the model into a list of field values (the "leaves") plus a
        # context tuple that carries enough information to reconstruct it.
        field_names = ModelOutputs._model_field_names(type(model))
        children = [getattr(model, field_name) for field_name in field_names]
        return children, (type(model), field_names)

    @staticmethod
    def _unflatten_pydantic_model(
        children: Iterable[Any],
        context: tuple[type[PydanticBaseModel], list[str]],
    ) -> PydanticBaseModel:
        # Reconstruct the model from the (possibly transformed) leaf values.
        # model_construct is used to bypass Pydantic validation, which is safe
        # here because the values were produced by the flatten step above.
        model_type, field_names = context
        values = dict(zip(field_names, children, strict=True))
        return model_type.model_construct(**values)

    @staticmethod
    def _flatten_pydantic_model_with_keys(
        model: PydanticBaseModel,
    ) -> tuple[list[tuple[_pytree.KeyEntry, Any]], tuple[type[PydanticBaseModel], list[str]]]:
        # Same as _flatten_pydantic_model but pairs each leaf with a KeyEntry
        # so that pytree-aware tools (e.g. torch.export) can emit human-readable
        # paths like "logits" instead of bare integer indices.
        field_names = ModelOutputs._model_field_names(type(model))
        key_children: list[tuple[_pytree.KeyEntry, Any]] = [
            (_pytree.GetAttrKey(field_name), getattr(model, field_name)) for field_name in field_names
        ]
        return key_children, (type(model), field_names)

    @staticmethod
    def _to_dumpable_context(context: tuple[type[PydanticBaseModel], list[str]]) -> dict[str, Any]:
        # Serialize the context to a JSON-compatible dict so that the pytree
        # structure can be saved (e.g. for torch.export / torch.compile cache).
        model_type, field_names = context
        return {
            "module": model_type.__module__,
            "qualname": model_type.__qualname__,
            "field_names": field_names,
        }

    @staticmethod
    def _from_dumpable_context(context: dict[str, Any]) -> tuple[type[PydanticBaseModel], list[str]]:
        # Deserialize the context produced by _to_dumpable_context by
        # dynamically importing the model class from its module + qualname.
        module = importlib.import_module(context["module"])
        model_type: Any = module
        for attr in context["qualname"].split("."):
            model_type = getattr(model_type, attr)
        return model_type, list(context["field_names"])

    @classmethod
    def _register_pytree_node(cls) -> None:
        # Guard against double-registration (e.g. when the module is reloaded).
        if cls in _pytree.SUPPORTED_NODES:
            return

        _pytree.register_pytree_node(
            cls,
            cls._flatten_pydantic_model,
            cls._unflatten_pydantic_model,
            serialized_type_name=f"{cls.__module__}.{cls.__qualname__}",
            to_dumpable_context=cls._to_dumpable_context,
            from_dumpable_context=cls._from_dumpable_context,
            flatten_with_keys_fn=cls._flatten_pydantic_model_with_keys,
        )


# Register the base class itself; subclasses are handled by __init_subclass__.
ModelOutputs._register_pytree_node()


def _is_float8_available():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return DEVICE == "cuda" and DEVICE_MODULE.is_available() and DEVICE_MODULE.get_device_capability() >= (8, 9)


class ModelItem(TypedDict):
    seq_ctx: SequenceContext
    loss_ctx: dict[str, BaseLossContext] | None


def is_float8_weight(tensor):
    return isinstance(tensor, (WeightWithDynamicTilewiseFloat8CastTensor, WeightWithDynamicTensorWiseFloat8CastTensor))


def _save_file(
    tensors: dict[str, torch.Tensor],
    filename,
    metadata=None,
):
    if not tensors:
        return
    save_file(tensors, filename, metadata=metadata)


class _HFSaveBucketItem(NamedTuple):
    tensor: torch.Tensor
    save_plan: HFSavePlan
    runtime_is_float8: bool
    byte_size: int


class BaseModel(nn.Module):
    """Base class for all xtuner training models with HF checkpoint I/O
    support.

    Subclass ``__init__`` **must** call ``self._init_load_spec()`` at the end,
    once every parameter and submodule has been constructed (including any
    ``__init__``-time sharding such as MoE EP via ``distribute_tensor``). This
    populates ``self.load_spec_mapping`` so that ``from_hf`` / ``save_hf`` and
    the RL weight-sync path can translate between local params and HF
    checkpoint keys. ``fully_shard`` and ``Float8Handler.pad_for_fsdp`` may
    re-invoke ``_init_load_spec`` afterwards to keep the mapping in sync with
    the current layout. See ``docs/design/load_spec_refactor.md`` §5.2 for the
    full contract.
    """

    load_spec_mapping: dict[str, LoadSpec] = {}
    fsdp_mesh: DeviceMesh | None = None
    hsdp_mesh: DeviceMesh | None = None
    fsdp_config: FSDPConfig | None = None
    config: XTunerBaseModelConfig

    FSDP_SHARD_DIM = 0

    def __init__(self, config: XTunerBaseModelConfig):
        super().__init__()
        self.config = config

        self._hf_path: Path | None = None  # type: ignore
        self._async_hf_tensor_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._pending_async_hf: AsyncHFSaveHandle | None = None
        self._async_hf_resources: AsyncHFResources | None = None

        self._compile_cfg = self._resolve_compile_cfg(self.config)
        self._float8_handler: Float8Handler | None = None

    def set_hf(self, hf_path: str | Path):
        self._hf_path = Path(hf_path)

    def from_hf(
        self, hf_path: str | Path, strict: bool = True
    ) -> tuple[
        Annotated[set[str], "loaded keys"], Annotated[set[str], "unloaded keys"], Annotated[set[str], "missing keys"]
    ]:
        # Recompute from the complete HF key list and the current runtime layout.
        # `__init__` still initializes the mapping for consumers that read it before checkpoint I/O.
        self._init_load_spec()
        self._assert_load_spec_initialized()
        self._hf_path = Path(hf_path)

        if isinstance(hf_path, Path):
            hf_path = str(hf_path)

        hf_loader = HFCheckpointLoader(hf_path)
        loaded_keys, unloaded_keys, missing_keys = self._load_params(hf_loader, strict=strict)
        return loaded_keys, unloaded_keys, missing_keys

    def scale_and_reduce_grad(self):
        return

    def cal_grad_norm(self, grads: list[DTensor], dtype=torch.float32):
        from xtuner.v1.utils.grad_norm import cal_grad_norm

        return cal_grad_norm(grads, dtype=dtype)

    def to_hf_key_list(self, key: str) -> list[str]:
        raise NotImplementedError()

    def trainable_parameters(self):
        params = [(name, param) for name, param in self.named_parameters() if param.requires_grad]
        return params

    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
    ) -> Self:
        """Fully shard the model parameters."""
        self.fsdp_config = fsdp_config
        self.fsdp_mesh = self._init_world_mesh()
        self._world_mesh = self.fsdp_mesh

        if self.fsdp_config.requires_grad:
            for name, module in self.named_modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype)

        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, BaseModel):
                module.fully_shard(fsdp_config)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        self._fully_shard(
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None,
        )
        self._init_load_spec()
        return self

    def _fully_shard(
        self,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy,
        reshard_after_forward: bool,
        offload_policy: CPUOffloadPolicy | None,
        module: nn.Module | None = None,
    ) -> None:
        def traverse(module):
            for name, param in module.named_parameters(recurse=False):
                full_name = full_param_name_mapping[id(param)]
                full_name = self._clean_param_name(full_name)
                # Match fp32 patterns against the post-mapping HF keys (the names that actually land
                # in the saved checkpoint), so this FSDP-ignore decision stays consistent with the
                # save path (`_split_ignored_params` / `_get_save_dtype`). `load_spec_mapping` is
                # built in `_init_load_spec` during __init__ (and rebuilt after fp8 padding), both
                # of which run before `fully_shard`, so the lookup is always populated here.
                load_spec = self.load_spec_mapping.get(full_name)
                if load_spec is None:
                    raise ValueError(f"Internal Error. Parameter {full_name} not found in load_spec_mapping.")
                hf_name_list = load_spec.global_hf_keys

                for hf_name in hf_name_list:
                    if any(re.search(p, hf_name) for p in patterns):  # type: ignore
                        if not isinstance(param, DTensor):
                            dist_param = nn.Parameter(
                                distribute_tensor(
                                    param, self.world_mesh, [Replicate() for _ in range(self.world_mesh.ndim)]
                                ),
                                requires_grad=param.requires_grad,
                            )
                            module.register_parameter(name, dist_param)
                            ignored_params.add(dist_param)
                        else:
                            # param is already a DTensor (e.g. distributed by
                            # MoE._replicate_other_params on ep_mesh before _fully_shard
                            # is called). We skip re-distributing on world_mesh and just
                            # add it to ignored_params so FSDP leaves it alone.
                            # ASSUMPTION: fp32 distribution always happens AFTER any
                            # prior EP distribution, so the existing placement is correct.
                            ignored_params.add(param)
                        break

            for child in module.children():
                traverse(child)

        # Collect the parameters of `target` that match any fp32 pattern so they can be
        # excluded from FSDP sharding (passed as `ignored_params`).
        #
        # We intentionally iterate over `self.named_parameters()` rather than
        # `target.named_parameters()` so that `name` is always relative to the root model
        # (`self`). This matters when `target` is a sub-module (e.g. `self.embed_tokens`):
        # `target.named_parameters()` would yield bare names like `"weight"`, which
        # `to_hf_key_list` cannot resolve correctly. By iterating from `self` we get the
        # full path (e.g. `"embed_tokens.weight"`) and filter to `target`'s parameters
        # using identity comparison.
        full_param_name_mapping = {id(param): name for name, param in self.named_parameters()}
        ignored_params: set[nn.Parameter] = set()
        patterns = self.config.hf_save_cfg.fp32_keys_pattern

        target = module or self
        if patterns:
            traverse(target)

        fully_shard(
            target,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            ignored_params=ignored_params if ignored_params else None,
        )

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16, safetensors_prefix: str = "model"):
        # Save may be called without `fully_shard`; refresh from the current runtime layout.
        self._init_load_spec()
        self._assert_load_spec_initialized()
        with profile_time_and_memory(f"[Saving HF to [{safetensors_prefix}]{hf_dir} cost]"):
            self._save_hf(hf_dir=hf_dir, save_dtype=save_dtype, safetensors_prefix=safetensors_prefix)

    def _init_async_hf_resources(self) -> None:
        if self._async_hf_resources is not None:
            return

        finalize_pg: dist.ProcessGroup | None = None
        try:
            if dist.is_initialized():
                finalize_pg = dist.new_group(backend="gloo")
                dist.barrier(group=finalize_pg)
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                gathered: list[Any] = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, {"rank": rank, "ok": True}, group=finalize_pg)

            self._async_hf_resources = AsyncHFResources(
                finalize_pg=finalize_pg,
                commit_executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-hf-commit"),
            )
        except BaseException as exc:
            if finalize_pg is not None and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group(finalize_pg)
            raise RuntimeError("Failed to initialize async HF resources") from exc

    def destroy_async_hf_resources(self) -> None:
        resources = self._async_hf_resources
        self._async_hf_resources = None
        for module in self.modules():
            if isinstance(module, BaseModel):
                module._async_hf_tensor_cache.clear()
        if resources is None:
            return

        resources.commit_executor.shutdown(wait=True)
        if resources.finalize_pg is not None and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group(resources.finalize_pg)

    def _get_async_hf_resources(self) -> AsyncHFResources:
        if self._async_hf_resources is None:
            self._init_async_hf_resources()
        assert self._async_hf_resources is not None
        return self._async_hf_resources

    def _all_gather_async_hf_object(self, local_obj: Any) -> list[Any]:
        gathered: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_obj, group=self._get_async_hf_resources().finalize_pg)
        return gathered

    def _barrier_async_hf(self) -> None:
        dist.barrier(group=self._get_async_hf_resources().finalize_pg)

    def async_save_hf(
        self,
        hf_dir: Path | str,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
    ) -> Future[Path]:
        resources = self._get_async_hf_resources()
        if self._hf_path is None and self.config.hf_config is None:
            raise NotImplementedError(
                "The model is not loaded from Huggingface, and the `hf_config` property is not implemented, so it cannot be saved in Huggingface format."
            )
        # Async HF stages tensors in CPU memory before the writer process flushes them to disk.
        # Allowing multiple in-flight HF saves would make these staged tensors accumulate quickly
        # when background I/O cannot keep up with the training loop.
        if self._pending_async_hf is not None:
            raise RuntimeError(
                "Previous async HF save is still pending. Wait for the returned async HF handle before launching a new one."
            )
        rank = dist.get_rank() if dist.is_initialized() else 0

        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)
        tmp_hf_dir = hf_dir.with_name(f".{hf_dir.name}.incomplete")
        if rank == 0:
            if tmp_hf_dir.exists():
                rmtree(tmp_hf_dir)
            tmp_hf_dir.mkdir(parents=True, exist_ok=True)

        file_to_names, weight_map = self._prepare_async_hf_snapshot(
            save_dtype=save_dtype,
            safetensors_prefix=safetensors_prefix,
            device=DEVICE,
        )
        merged_weight_map: dict[str, str] = {}
        for rank_weight_map in self._all_gather_async_hf_object(weight_map):
            merged_weight_map.update(cast(dict[str, str], rank_weight_map))
        if rank == 0:
            self._write_hf_index_and_config(hf_dir=tmp_hf_dir, weight_map=merged_weight_map)
        self._barrier_async_hf()

        if hasattr(DEVICE_MODULE, "synchronize"):
            DEVICE_MODULE.synchronize()

        mp_ctx = py_mp.get_context("fork")
        process = mp_ctx.Process(
            target=self._run_async_hf_writer,
            args=(
                tmp_hf_dir,
                file_to_names,
            ),
            daemon=False,
        )
        process.start()

        handle = AsyncHFSaveHandle(
            process=process,
            hf_dir=hf_dir,
            tmp_hf_dir=tmp_hf_dir,
        )
        commit_future = resources.commit_executor.submit(
            self._commit_async_hf_save,
            handle,
        )
        self._pending_async_hf = handle

        def clear_pending_async_hf(_: Future[Path]) -> None:
            self._clear_pending_async_hf(handle)

        commit_future.add_done_callback(clear_pending_async_hf)
        return commit_future

    def _run_async_hf_writer(
        self,
        tmp_hf_dir: Path,
        file_to_names: list[tuple[str, list[str]]],
    ) -> None:
        log_rank0.info(f"[Async saving HF to {tmp_hf_dir} writer] started")
        try:
            set_async_save_process_qos()
            self._write_async_hf_snapshot(
                hf_dir=tmp_hf_dir,
                file_to_names=file_to_names,
            )
            log_rank0.info(f"[Async saving HF to {tmp_hf_dir} writer] finished")
        except Exception as exc:
            log_rank0.error(f"[Async saving HF to {tmp_hf_dir} writer] failed: {exc}")
            raise

    def _prepare_async_hf_snapshot(
        self,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
        device: torch.device | str = DEVICE,
    ) -> tuple[list[tuple[str, list[str]]], dict[str, str]]:
        # Compose models enter async save through their child modules directly.
        # Refresh here so every snapshot is planned from the post-FSDP runtime layout,
        # matching the public synchronous save path.
        self._init_load_spec()
        self._assert_load_spec_initialized()
        file_to_names: list[tuple[str, list[str]]] = []
        weight_map: dict[str, str] = {}
        for safetensor_name, name_list, hf_tensor_list in self._iter_hf_save_chunks(
            save_dtype=save_dtype,
            safetensors_prefix=safetensors_prefix,
            device=device,
        ):
            cached_names: list[str] = []
            for name, hf_tensor in zip(name_list, hf_tensor_list):
                cache_key = (("root", "hf"), ("name", name))
                self._get_or_update_async_hf_cpu_tensor(
                    hf_tensor,
                    cache=self._async_hf_tensor_cache,
                    path=cache_key,
                )
                cached_names.append(name)
                weight_map[name] = safetensor_name
            if cached_names:
                file_to_names.append((safetensor_name, cached_names))
            del hf_tensor_list
        return file_to_names, weight_map

    def _clear_pending_async_hf(self, handle: AsyncHFSaveHandle) -> None:
        if self._pending_async_hf is handle:
            self._pending_async_hf = None

    def _commit_async_hf_save(
        self,
        handle: AsyncHFSaveHandle,
    ) -> Path:
        process = handle.process
        hf_dir = handle.hf_dir

        local_ok = True
        local_error = ""

        rank = dist.get_rank() if dist.is_initialized() else 0
        join_timeout = get_async_hf_writer_join_timeout()
        process.join(timeout=join_timeout)
        if process.is_alive():
            process.terminate()
            process.join()

            local_ok = False
            local_error = f"child_timeout={join_timeout}"
        else:
            exit_code = process.exitcode
            if exit_code is None:
                local_ok = False
                local_error = "async_hf_writer_exitcode_missing"
            elif exit_code != 0:
                local_ok = False
                local_error = f"child_exit_code={exit_code}"

        local_status = {"rank": rank, "ok": local_ok, "error": local_error}
        all_status = cast(list[dict[str, Any]], self._all_gather_async_hf_object(local_status))

        if not all(status["ok"] for status in all_status):
            failed = ", ".join(
                f"rank={status['rank']}({status['error']})" for status in all_status if not status["ok"]
            )
            raise RuntimeError(f"Async HF save global consistency check failed: {failed}")

        if rank == 0:
            if hf_dir.exists():
                rmtree(hf_dir)
            handle.tmp_hf_dir.rename(hf_dir)
        self._barrier_async_hf()
        log_rank0.info(f"[Async saving HF to {hf_dir}] finalized")
        return hf_dir

    def hf_tensor_to_canonical(self, name: str, loaded_tensor: torch.Tensor) -> torch.Tensor:
        """Convert one checkpoint tensor to XTuner's unsharded canonical
        layout.

        ``HFLoadPlan`` owns HF-key concatenation, rank-local slicing, interleaved
        copies, and runtime padding. Model subclasses override only this format
        adapter when their HF tensor shape/order differs from XTuner's layout.
        """
        return loaded_tensor

    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        return safetensor

    @property
    def device(self) -> torch.device:
        if self.fsdp_config is not None and self.fsdp_config.cpu_offload:
            return torch.device("cpu")
        return torch.device(DEVICE)

    @property
    def world_mesh(self) -> DeviceMesh | None:
        if not hasattr(self, "_world_mesh"):
            self._world_mesh = self._init_world_mesh()
        return self._world_mesh

    @property
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return {}

    @property
    def compile_cfg(self) -> dict[str, TorchCompileOption]:
        _compile_cfg = self._compile_cfg.copy()
        for module in self.modules():
            if isinstance(module, BaseModel) and module is not self:
                sub_custom_cfg = module.compile_cfg
                _compile_cfg |= sub_custom_cfg

        return _compile_cfg

    @property
    def float8_handler(self):
        if (
            self.config.float8_cfg is not None
            and self.config.float8_cfg.enable_float8
            and self._float8_handler is None
        ):
            self._float8_handler = self.config.float8_cfg.build()

            if self.fsdp_mesh is not None:
                self._float8_handler.build_reduce_mesh(self, self.fsdp_mesh)
        return self._float8_handler

    @torch.no_grad()
    def init_weights(self):
        # TODO: HardCode here. The initialization method should be module specific. All module in model
        # in model should be defined in `xtuner.module`
        from xtuner.v1.utils import default_init_weights

        initialized_params = default_init_weights(self)
        if missing := {self._clean_param_name(name) for name, _ in self.named_parameters()} - initialized_params:
            raise RuntimeError(f"{missing} is not initialized")

    def build_rotary_embedding(self, config):
        # NOTE: XTuner initializes the entire model on meta device to avoid the overhead of allocating and
        # initializing real tensors upfront — weights will either be loaded from a HuggingFace checkpoint or
        # initialized from scratch afterward. However, rotary embedding must be initialized on CPU even when the
        # rest of the model is on meta device, for the following reasons:
        #
        # 1. Its buffers (e.g. `inv_freq`) require real arithmetic and cannot be computed on meta device.
        # 2. Its buffers are not model parameters, so the HuggingFace weight-loading path does not populate them.
        #    After `.to_empty()`, these buffers remain garbage-initialized, which would silently corrupt training.
        #
        # CPU is chosen specifically (rather than CUDA) to keep the computation numerically aligned with inference
        # engines (e.g. lmdeploy) during RL-phase training.
        #
        # To avoid repeating this error-prone logic in every subclass, the construction is encapsulated in
        # get_rope_embedding, and this default build_rotary_embedding in BaseModel enforces CPU initialization.
        from xtuner.v1.module.rope import get_rope_embedding

        with torch.device("cpu"):
            return get_rope_embedding(config=config)

    def _init_load_spec(self) -> None:
        if self.__class__.to_hf_key_list is BaseModel.to_hf_key_list:
            self.load_spec_mapping = {}
            return

        load_spec_mapping: dict[str, LoadSpec] = {}
        hf_key_mapping_missing: set[str] = set()

        for name, param in self.state_dict().items():
            name = self._clean_param_name(name)
            _hf_keys = self.to_hf_key_list(name)

            if not self.config.hf_key_mapping:
                hf_keys = _hf_keys
            else:
                hf_keys = []
                for key in _hf_keys:
                    max_matched_pattern = None
                    max_match_len = -1
                    for pattern in self.config.hf_key_mapping:
                        if (matched := re.search(pattern, key)) is not None:
                            matched_len = matched.end() - matched.start()

                            if matched_len > max_match_len:
                                max_match_len = matched_len
                                max_matched_pattern = pattern

                    if max_matched_pattern is None:
                        hf_key_mapping_missing.add(key)
                        hf_keys.append(key)
                    else:
                        repl = self.config.hf_key_mapping[max_matched_pattern]
                        hf_keys.append(re.sub(max_matched_pattern, repl, key))

            runtime_tensor = param._local_tensor if isinstance(param, DTensor) else param
            runtime_is_float8 = is_float8_weight(runtime_tensor)
            origin_shape = tuple(runtime_tensor._ori_shape) if runtime_is_float8 else None  # type: ignore[attr-defined]
            load_spec = LoadSpec.from_tensor(
                name=name,
                hf_keys=hf_keys,
                tensor=param,
                origin_shape=origin_shape,
            )
            load_spec_mapping[name] = load_spec

        if hf_key_mapping_missing:
            log_rank0.info("These hf keys will not be influenced by `hf_key_mapping`:")
            log_rank0.info(json.dumps(list(hf_key_mapping_missing), indent=2))

        self.load_spec_mapping = load_spec_mapping

    def _assert_load_spec_initialized(self) -> None:
        # `load_spec_mapping` defaults to the class-level empty dict; `_init_load_spec`
        # always assigns an instance attribute (possibly empty), so presence on
        # `self.__dict__` is the reliable signal that the subclass contract was honored.
        assert "load_spec_mapping" in self.__dict__, (
            f"{type(self).__name__}.__init__ must call self._init_load_spec() at the end. "
            "See docs/design/load_spec_refactor.md §5.2."
        )

    def _to_float8(
        self,
        gathered_tensor_list: list[torch.Tensor],
        name_list: list[str],
        runtime_is_float8_list: list[bool],
        dtype: torch.dtype,
    ) -> tuple[list[torch.Tensor], list[str]]:
        assert len(gathered_tensor_list) == len(name_list) == len(runtime_is_float8_list), (
            "Internal error: float8 conversion metadata length does not match tensor list"
        )
        gathered_tensor_list_new, name_list_new = [], []
        for gathered_tensor, name, runtime_is_float8 in zip(gathered_tensor_list, name_list, runtime_is_float8_list):
            if not runtime_is_float8:
                gathered_tensor_list_new.append(gathered_tensor)
                name_list_new.append(name)
                continue

            from xtuner.v1.float8.triton_kernels.per_block_quant_gemm import per_block_quant_torch

            gathered_tensor_fp8, scale = per_block_quant_torch(gathered_tensor, block_size=128, float8_dtype=dtype)
            gathered_tensor_list_new.extend([gathered_tensor_fp8, scale])
            name_list_new.extend([name, f"{name}_scale_inv"])
        return gathered_tensor_list_new, name_list_new

    def build_loss_ctx_batch(
        self,
        data_batch: list[dict],
        sp_mesh: DeviceMesh | None = None,
    ) -> list[dict[str, dict]]:
        """Build and calibrate loss contexts for the entire batch.

        For Dense model, only LM loss is needed.

        Args:
            data_batch (list[dict]): All microbatch data
            sp_mesh (DeviceMesh | None): Sequence parallel mesh
            cu_seq_lens_list (list[torch.IntTensor] | None): For calibration

        Returns:
            list[dict[str, BaseLossContext]]: Loss context dict for each microbatch
        """
        cu_seq_lens_list = [data["seq_ctx"].cu_seq_lens_k for data in data_batch]
        res: list[dict] = [{} for _ in range(len(data_batch))]

        lm_loss_ctx_list = self._build_loss_ctx(self.config.lm_loss_cfg, data_batch, sp_mesh)

        if lm_loss_ctx_list is not None:
            loss_ctx_cls = lm_loss_ctx_list[0].__class__
            lm_loss_ctx_list = loss_ctx_cls.build_batches(
                lm_loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=sp_mesh
            )

            if lm_loss_ctx_list is not None:
                for i, lm_loss_ctx in enumerate(lm_loss_ctx_list):
                    res[i]["lm"] = lm_loss_ctx

        return res

    def _add_auxiliary_loss(
        self,
        loss_name: str,
        loss_cfg: Any,
        data_batch: list[dict],
        res: list[dict],
    ) -> None:
        """Add auxiliary loss contexts to result.

        This helper builds loss contexts, calibrates them across the batch,
        and adds them to the result dictionary. If loss_cfg is None, does nothing.

        Args:
            loss_name (str): Name of the loss (e.g., "balancing", "z_loss").
            loss_cfg (Any): Loss configuration with a build() method. If None, skipped.
            data_batch (list[dict]): Batch data.
            res (list[dict]): Result dictionary to populate. Modified in-place.

        Example:
            def build_loss_ctx_batch(self, data_batch, sp_mesh):
                res = super().build_loss_ctx_batch(data_batch, sp_mesh)

                # One line per auxiliary loss
                self._add_auxiliary_loss("balancing", self.config.balancing_loss_cfg, data_batch, res)
                self._add_auxiliary_loss("z_loss", self.config.z_loss_cfg, data_batch, res)

                return res
        """
        if loss_cfg is None:
            return

        # Build loss contexts for all microbatches
        ctx_list = [loss_cfg.build() for _ in data_batch]

        # Calibrate across batch
        ctx_cls = ctx_list[0].__class__
        ctx_list = ctx_cls.build_batches(ctx_list)

        # Add to result
        for i, ctx in enumerate(ctx_list):
            res[i][loss_name] = ctx  # type: ignore

    def pre_micro_batch_forward(self, data_batches: Sequence[ModelItem]) -> DataBatchInfo:
        step_consumed_tokens = torch.tensor(0, device=DEVICE)
        step_seqlen_tokens = 0
        step_consumed_img_tokens = torch.tensor(0.0, device=DEVICE)
        efficient_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        total_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        img_efficient_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        img_total_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)

        for data in data_batches:
            seq_ctx = data["seq_ctx"]
            mask = seq_ctx.mask
            step_consumed_tokens += mask.sum()
            step_seqlen_tokens += mask.numel()
            num_tokens = seq_ctx.cu_seq_lens_k[1:] - seq_ctx.cu_seq_lens_k[:-1]
            efficient_forward_tokens += (num_tokens.long() ** 2).sum()
            total_forward_tokens += (num_tokens.long().sum()) ** 2

            if seq_ctx.num_img_tokens is not None:
                total_num_img_tokens = torch.tensor(0, dtype=torch.long)
                for num_img_token in seq_ctx.num_img_tokens:  # list[list]
                    step_consumed_img_tokens += sum(num_img_token)
                    num_img_tokens_ = torch.tensor(num_img_token, dtype=torch.long)  # list[int]
                    total_num_img_tokens += num_img_tokens_.sum()
                    img_efficient_forward_tokens += (num_img_tokens_**2).sum()
                img_total_forward_tokens += total_num_img_tokens**2

        efficient_attn_ratio = efficient_forward_tokens.float() / total_forward_tokens.float()
        img_efficient_attn_ratio = img_efficient_forward_tokens.float() / (img_total_forward_tokens.float() + 1e-8)

        if len(data_batches) > 0 and seq_ctx.sequence_parallel_mesh:
            step_consumed_img_tokens /= seq_ctx.sequence_parallel_mesh.size()

        batch_info: DataBatchInfo = {
            "step_consumed_tokens": cast(int, step_consumed_tokens.item()),
            "step_seqlen_tokens": step_seqlen_tokens,
            "step_consumed_img_tokens": cast(float, step_consumed_img_tokens.item()),
            "efficient_attn_ratio": cast(float, efficient_attn_ratio.item()),
            "img_efficient_attn_ratio": cast(float, img_efficient_attn_ratio.item()),
        }
        return batch_info

    def post_micro_batch_forward(self, batch_outputs: Sequence[ModelOutputs]) -> BatchForwardInfo:
        train_engine_extra_info = ModelForwardExtraLogInfo()

        local_total_loss = torch.tensor(0.0, device=DEVICE)
        reduced_other_losses: dict[str, float] = {}

        for output in batch_outputs:
            output_copy = output.model_copy()
            for name in output_copy.model_fields:
                obj = getattr(output_copy, name)
                if "loss" in name and isinstance(obj, torch.Tensor):
                    loss_item = obj.item()
                    local_total_loss += loss_item
                    reduced_name = f"reduced_{name}"

                    if reduced_name not in reduced_other_losses:
                        reduced_other_losses[reduced_name] = loss_item
                    else:
                        reduced_other_losses[reduced_name] += loss_item

            if "extra_info" in output_copy:
                extra_info = output["extra_info"]
                train_engine_extra_info.append(extra_info)

        for name, loss in reduced_other_losses.items():
            tensor_loss = torch.tensor(loss, device=DEVICE)
            dist.all_reduce(tensor_loss.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
            reduced_other_losses[name] = tensor_loss.item()

        if "reduced_loss" in reduced_other_losses:
            reduced_other_losses["reduced_llm_loss"] = reduced_other_losses.pop("reduced_loss")

        ret = BatchForwardInfo(
            logs_info=reduced_other_losses,
            extra_info=train_engine_extra_info,
        )
        return ret

    def _get_save_dtype(self, name: str, dtype: torch.dtype) -> torch.dtype:
        patterns = self.config.hf_save_cfg.fp32_keys_pattern
        if patterns and any(re.search(p, name) for p in patterns):
            return torch.float32
        return dtype

    def _get_hf_param(
        self,
        params: list[tuple[torch.Tensor, LoadSpec]],
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        bucket_size: int | None = None,
        distributed_save: bool = False,
        preserved_fused_shard_group: dist.ProcessGroup | None = None,
        target_fused_key_partition: tuple[int, int] | None = None,
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        """Yield HF checkpoint tensors for the given runtime params.

        Args:
            params (list[tuple[torch.Tensor, LoadSpec]]): Runtime tensors and their new-schema LoadSpecs.
            dtype (torch.dtype): Target checkpoint dtype, currently bfloat16 or float8_e4m3fn.
            device (torch.device | str): Device to move yielded tensors to.
            bucket_size (int | None): Approximate bucket size in bytes.
            distributed_save (bool): Whether to apply the HF save write policy. When enabled, non-fused tensors are
                yielded only on rank0 and fused HF keys are divided across save ranks.
            preserved_fused_shard_group (dist.ProcessGroup | None): Communication group whose fused-dim shard should
                stay local instead of being all-gathered. RL weight sync uses this to stream EP-local expert slices.
            target_fused_key_partition (tuple[int, int] | None): Rollout expert-parallel ``(rank, size)`` used to
                select fused HF keys after gathering the training layout.

        Returns:
            Generator[tuple[list[str], list[torch.Tensor]], None, None]: HF key names and tensors to save.
        """
        assert not (distributed_save and preserved_fused_shard_group is not None), (
            "distributed_save writes checkpoint files, while preserved_fused_shard_group streams local fused shards "
            "for RL."
        )
        assert not (preserved_fused_shard_group is not None and target_fused_key_partition is not None), (
            "preserved_fused_shard_group and target_fused_key_partition are mutually exclusive RL policies."
        )
        if not params:
            return

        if bucket_size is None:
            bucket_size = self.config.hf_save_cfg.bucket_size

        bucket_bytes = 0
        bucket: list[_HFSaveBucketItem] = []
        buffer_names = {self._clean_param_name(name) for name, _ in self.named_buffers()}

        for param, load_spec in params:
            save_item = self._make_hf_save_item(
                param,
                load_spec,
                checkpoint_dtype=dtype,
                is_buffer=load_spec.name in buffer_names,
                distributed_save=distributed_save,
                preserved_fused_shard_group=preserved_fused_shard_group,
                target_fused_key_partition=target_fused_key_partition,
            )
            if bucket_bytes + save_item.byte_size > bucket_size and bucket:
                yield self._build_hf_param_bucket(
                    bucket,
                    dtype=dtype,
                    device=device,
                )
                bucket_bytes = 0
                bucket = []

            bucket_bytes += save_item.byte_size
            bucket.append(save_item)

        if bucket:
            yield self._build_hf_param_bucket(
                bucket,
                dtype=dtype,
                device=device,
            )

    def _make_hf_save_item(
        self,
        param: torch.Tensor,
        load_spec: LoadSpec,
        *,
        checkpoint_dtype: torch.dtype,
        is_buffer: bool,
        distributed_save: bool,
        preserved_fused_shard_group: dist.ProcessGroup | None,
        target_fused_key_partition: tuple[int, int] | None,
    ) -> _HFSaveBucketItem:
        """Normalize one runtime parameter and compile its checkpoint save
        policy."""
        # LoadSpec records both continuous and interleaved shard history, so every
        # parameter enters the same SavePlan path from its runtime-local tensor.
        runtime_tensor = param._local_tensor if isinstance(param, DTensor) else param
        if runtime_tensor.is_floating_point() and not is_buffer:
            save_dtype = self._get_save_dtype(load_spec.global_hf_keys[0], torch.bfloat16)
            checkpoint_tensor = runtime_tensor.to(dtype=save_dtype)
        else:
            # Persistent buffers, e.g. FoPE rotary coefficients, keep their runtime dtype.
            checkpoint_tensor = runtime_tensor

        return _HFSaveBucketItem(
            tensor=checkpoint_tensor,
            save_plan=load_spec.plan_hf_save(
                distributed_save=distributed_save,
                preserve_process_group=preserved_fused_shard_group,
                target_fused_key_partition=target_fused_key_partition,
            ),
            runtime_is_float8=is_float8_weight(runtime_tensor),
            # Buckets bound the tensors reconstructed by HF save. DTensor.numel()
            # is the logical/global size; using the local shard here makes the
            # effective bucket grow with the FSDP world size.
            byte_size=self._get_tensor_size(param, checkpoint_dtype),
        )

    def _load_spec_params(self) -> list[tuple[torch.Tensor, LoadSpec]]:
        ret: list[tuple[torch.Tensor, LoadSpec]] = []
        for name, param in self.state_dict().items():
            name = self._clean_param_name(name)
            load_spec = self.load_spec_mapping.get(name)
            if load_spec is None:
                raise ValueError(f"Internal Error. Parameter {name} not found in load_spec_mapping.")
            ret.append((param, load_spec))
        return ret

    def _build_hf_param_bucket(
        self,
        bucket: list[_HFSaveBucketItem],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> tuple[list[str], list[torch.Tensor]]:
        name_list: list[str] = []
        tensor_list: list[torch.Tensor] = []
        runtime_is_float8_list: list[bool] = []

        full_tensor_list = unshard_tensors_for_hf_save(
            [item.tensor for item in bucket],
            [item.save_plan for item in bucket],
        )
        for full_tensor, save_item in zip(full_tensor_list, bucket, strict=True):
            hf_names, hf_tensors = self._split_hf_tensors_for_save(full_tensor, save_item.save_plan)
            name_list.extend(hf_names)
            tensor_list.extend(hf_tensors)
            runtime_is_float8_list.extend([save_item.runtime_is_float8] * len(hf_tensors))

        if dtype == torch.float8_e4m3fn:
            tensor_list, name_list = self._to_float8(tensor_list, name_list, runtime_is_float8_list, dtype)

        tensor_list = [tensor.to(device=device) for tensor in tensor_list]
        return name_list, tensor_list

    def _split_hf_tensors_for_save(
        self,
        full_tensor: torch.Tensor,
        save_plan: HFSavePlan,
    ) -> tuple[list[str], list[torch.Tensor]]:
        if not save_plan.hf_keys:
            return [], []

        if len(save_plan.hf_keys) == 1:
            if (
                not save_plan.preserves_shards
                and save_plan.distributed_save
                and dist.is_initialized()
                and dist.get_rank() != 0
            ):
                return [], []
            hf_name = save_plan.hf_keys[0]
            return [hf_name], [self.param_to_safetensor(full_tensor, hf_name)]

        dim = save_plan.fused_dim
        assert dim is not None, "fused_dim must be set when saving fused HF tensors"
        if save_plan.preserves_shards:
            hf_names = save_plan.hf_keys
            tensor_to_split = full_tensor
        else:
            hf_names = save_plan.hf_keys.copy()
            if save_plan.fused_key_range is not None:
                key_start, key_end = save_plan.fused_key_range
            elif save_plan.distributed_save:
                key_start, key_end = self._hf_save_key_range(save_plan)
            else:
                key_start, key_end = 0, len(hf_names)
            if key_start == key_end:
                return [], []
            hf_names = hf_names[key_start:key_end]
            key_size = full_tensor.shape[dim] / len(save_plan.hf_keys)
            assert key_size.is_integer(), (
                f"Fused dim size {full_tensor.shape[dim]} is not divisible by "
                f"{len(save_plan.hf_keys)} HF keys for {save_plan.name}"
            )
            key_size = int(key_size)
            # Keep the legacy save behavior here: fp8 per-block quant kernels have had correctness issues with
            # non-zero-storage-offset views, so materialize the save-rank slice before splitting HF keys.
            index = torch.arange(
                key_start * key_size,
                key_end * key_size,
                dtype=torch.int64,
                device=full_tensor.device,
            )
            tensor_to_split = torch.index_select(full_tensor, dim=dim, index=index)

        if not hf_names:
            return [], []

        hf_tensor_size = tensor_to_split.shape[dim] / len(hf_names)
        assert hf_tensor_size.is_integer(), (
            f"Fused dim size {tensor_to_split.shape[dim]} is not divisible by "
            f"{len(hf_names)} HF keys for {save_plan.name}"
        )
        split_size = int(hf_tensor_size)
        hf_tensors = tensor_to_split.split([split_size] * len(hf_names), dim=dim)
        return (
            hf_names,
            [self.param_to_safetensor(safetensor, name) for safetensor, name in zip(hf_tensors, hf_names)],
        )

    def _hf_save_key_range(self, save_plan: HFSavePlan) -> tuple[int, int]:
        if not dist.is_initialized():
            return 0, len(save_plan.hf_keys)

        current_rank = dist.get_rank()
        save_ranks = self._get_fused_save_ranks(len(save_plan.hf_keys))
        if current_rank not in save_ranks:
            return 0, 0

        key_per_rank = len(save_plan.hf_keys) // len(save_ranks)
        rank_index = save_ranks.index(current_rank)
        start = rank_index * key_per_rank
        return start, start + key_per_rank

    # TODO: Using `xtuenr.v1.utils.misc.clean_param_name`
    def _clean_param_name(self, name: str) -> str:
        if "_checkpoint_wrapped_module." in name:
            name = name.replace("_checkpoint_wrapped_module.", "")
        if "_orig_mod." in name:
            name = name.replace("_orig_mod.", "")
        return name

    def _get_tensor_size(self, tensor: torch.Tensor, dtype: torch.dtype) -> int:
        """Get the size of the tensor in bytes."""
        # return tensor.element_size() * tensor.numel()
        return dtype.itemsize * tensor.numel()

    def _iter_hf_save_chunks(
        self,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
        device: torch.device | str = "cpu",
    ) -> Generator[tuple[str, list[str], list[torch.Tensor]], None, None]:
        assert save_dtype in [torch.float8_e4m3fn, torch.bfloat16], f"save_dtype {save_dtype} is not supported"
        save_rank = dist.get_rank() if dist.is_initialized() else 0
        saved_names: set[str] = set()
        safetensor_index = 0

        param_gen = self._get_hf_param(
            self._load_spec_params(),
            dtype=save_dtype,
            device=device,
            distributed_save=True,
        )
        for name_list, hf_tensor_list in param_gen:
            if not name_list:
                continue

            unique_name_list: list[str] = []
            unique_hf_tensor_list: list[torch.Tensor] = []
            for name, hf_tensor in zip(name_list, hf_tensor_list):
                if name in saved_names:
                    continue
                saved_names.add(name)
                unique_name_list.append(name)
                unique_hf_tensor_list.append(hf_tensor)
            if unique_name_list:
                safetensor_index += 1
                safetensor_name = f"{safetensors_prefix}-{safetensor_index:04d}-save_rank{save_rank}.safetensors"
                yield safetensor_name, unique_name_list, unique_hf_tensor_list

    def _write_hf_save_plan(self, save_plan: _HFSavePlan) -> list[str]:
        written_files: list[str] = []
        for safetensor_name, tensors in save_plan["save_tasks"]:
            filename = save_plan["hf_dir"] / safetensor_name
            self._save_hf_safetensors_file(
                tensors=tensors,
                filename=filename,
                hf_dir=save_plan["hf_dir"],
            )
            if tensors:
                written_files.append(safetensor_name)
        return written_files

    def _save_hf_safetensors_file(self, tensors: dict[str, torch.Tensor], filename: Path, hf_dir: Path) -> None:
        if not tensors:
            return

        lock_slots = get_async_hf_save_file_lock_slots()
        if lock_slots <= 0:
            _save_file(tensors, filename)
            return

        import fcntl

        rank = dist.get_rank() if dist.is_initialized() else 0
        lock_slot = rank % lock_slots
        lock_path = self._hf_save_file_lock_path(hf_dir, lock_slot)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                _save_file(tensors, filename)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _hf_save_file_lock_path(self, hf_dir: Path, lock_slot: int) -> Path:
        lock_dir = hf_dir.parent / ".async-hf-save-file-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", os.uname().nodename)
        return lock_dir / f"async-hf-save-file-{hostname}-slot-{lock_slot:03d}.lock"

    @staticmethod
    def _allocate_async_hf_cpu_buffer_like(tensor: torch.Tensor) -> torch.Tensor:
        cpu_tensor = torch.empty_like(tensor, device="cpu")
        if tensor.is_cuda:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor

    def _get_or_update_async_hf_cpu_tensor(
        self,
        tensor: torch.Tensor,
        cache: dict[tuple[Any, ...], torch.Tensor],
        path: tuple[Any, ...],
    ) -> torch.Tensor:
        detached = tensor.detach()

        cached = cache.get(path)
        if cached is None or (
            cached.shape != detached.shape
            or cached.dtype != detached.dtype
            or cached.layout != detached.layout
            or cached.stride() != detached.stride()
        ):
            cached = self._allocate_async_hf_cpu_buffer_like(detached)
            cache[path] = cached

        cached.copy_(detached, non_blocking=detached.is_cuda)
        return cached

    def _write_async_hf_snapshot(
        self,
        hf_dir: Path,
        file_to_names: list[tuple[str, list[str]]],
    ) -> None:
        for filename, names in file_to_names:
            tensors: dict[str, torch.Tensor] = {}
            for name in names:
                cache_key = (("root", "hf"), ("name", name))
                cached_tensor = cast(torch.Tensor | None, self._async_hf_tensor_cache.get(cache_key))
                if cached_tensor is None:
                    raise RuntimeError(f"Missing cached async HF tensor for key: {name}")
                tensors[name] = cached_tensor
            self._write_hf_save_plan({"hf_dir": hf_dir, "save_tasks": [(filename, tensors)]})

    def _write_hf_index_and_config(self, hf_dir: Path | str, weight_map: Mapping[str, str]) -> None:
        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)

        if self.config.hf_config is not None:
            self.config.save_hf(hf_dir)
        elif self._hf_path is not None:
            for file in cast(Path, self._hf_path).iterdir():
                if file.suffix != ".safetensors":
                    target_path = hf_dir / file.name
                    if file.is_file():
                        copy(file, target_path)
                    else:
                        copytree(file, target_path, ignore_dangling_symlinks=True, dirs_exist_ok=True)
        else:
            raise RuntimeError("Internal Error, both self.config.hf_config and self._hf_path are None")

        with open(hf_dir / "model.safetensors.index.json", "w") as f:
            index = {"weight_map": dict(weight_map), "metadata": {}}
            json.dump(index, f, indent=2, ensure_ascii=False)

    def _save_hf(
        self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16, safetensors_prefix: str = "model"
    ):
        """Save the hf model to the given directory.

        Args:
            hf_dir (str): The directory to save the model.
            save_dtype (torch.dtype): The dtype to save the model parameters, bfloat16 or float8.
        """
        if self._hf_path is None and self.config.hf_config is None:
            raise NotImplementedError(
                "The model is not loaded from Huggingface, and the `hf_config` property is not implemented, so it cannot be saved in Huggingface format."
            )

        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)

        DEVICE_MODULE.empty_cache()
        assert save_dtype in [torch.float8_e4m3fn, torch.bfloat16], f"save_dtype {save_dtype} is not supported"

        param_gen = self._get_hf_param(self._load_spec_params(), dtype=save_dtype, distributed_save=True)

        # Tell me why! why! old cao! @HIT-cwh
        # mp_context = multiprocessing.get_context("fork")
        # save_executor = ProcessPoolExecutor(max_workers=16, mp_context=mp_context)
        save_executor = ThreadPoolExecutor(max_workers=self.config.hf_save_cfg.worker_per_rank)

        if dist.is_initialized():
            save_rank = dist.get_rank()
        else:
            save_rank = 0

        save_futures = []
        weight_map = {}
        safetensor_index = 0

        for name_list, hf_tensor_list in param_gen:
            if not name_list:
                continue

            # Tied weights may map multiple runtime tensors to the same HF key; keep the first one.
            unique_name_list = []
            unique_hf_tensor_list = []
            for name, hf_tensor in zip(name_list, hf_tensor_list):
                if name in weight_map:
                    continue
                unique_name_list.append(name)
                unique_hf_tensor_list.append(hf_tensor)

            if not unique_name_list:
                continue

            safetensor_index += 1
            safetensor_name = f"{safetensors_prefix}-{safetensor_index:04d}-save_rank{save_rank}.safetensors"
            weight_map.update(dict.fromkeys(unique_name_list, safetensor_name))
            assert save_executor is not None, "Internal Error, save_executor should not be None"
            future = save_executor.submit(
                _save_file,
                dict(zip(unique_name_list, unique_hf_tensor_list)),
                hf_dir / safetensor_name,
            )
            save_futures.append(future)
            self._wait_save_task(save_futures)

        if save_futures:
            wait(save_futures)
            for future in save_futures:
                if future.exception():
                    raise future.exception()  # type: ignore
        save_executor.shutdown()

        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        weight_map_list: list[dict] | list[None] = [None for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather_object(weight_map_list, weight_map)
        else:
            weight_map_list = [weight_map]
        weight_map_list = cast(list[dict], weight_map_list)
        weight_map = reduce(lambda x, y: x | y, weight_map_list)

        if not dist.is_initialized() or dist.get_rank() == 0:
            if self.config.hf_config is None and self._hf_path is None:
                raise RuntimeError("Internal Error, both self.config.hf_config and self._hf_path are None")

            if self.config.hf_config is not None:
                self.config.save_hf(hf_dir)
            else:  # if self._hf_path is not None:
                for file in cast(Path, self._hf_path).iterdir():
                    if file.suffix != ".safetensors":
                        # Copy the model config and tokenizer files to the save path
                        target_path = hf_dir / file.name
                        if file.is_file():
                            copy(file, target_path)
                        else:
                            copytree(file, target_path, ignore_dangling_symlinks=True, dirs_exist_ok=True)

            # write or overwrite `model.safetensors.index.json`
            with open(hf_dir / "model.safetensors.index.json", "w") as f:
                index = {"weight_map": weight_map, "metadata": {}}
                json.dump(index, f, indent=2, ensure_ascii=False)

        if dist.is_initialized():
            torch.distributed.barrier()

    def _load_params(self, checkpoint_loader: HFCheckpointLoader, strict=True) -> tuple:
        matched_hf_keys: set[str] = set(checkpoint_loader.weight_map)
        expected_hf_keys: set[str] = set(chain(*map(self.to_hf_key_list, self.state_dict())))
        expected_keys = set(self.state_dict())

        if strict and matched_hf_keys != expected_hf_keys:
            _missing_keys = expected_hf_keys - matched_hf_keys
            if _missing_keys:
                raise RuntimeError(f"Missing keys in checkpoint: {_missing_keys}. ")
            unexpected_keys = matched_hf_keys - expected_hf_keys
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}. ")

        missing_keys: set[str] = set()
        loaded_keys: set[str] = set()

        @torch.no_grad  # type: ignore
        def _load_params_from_module(module: nn.Module, module_prefix: str):
            if self._has_meta_param(module):
                module.to_empty(device=DEVICE, recurse=False)

            for name, param in chain(
                module.named_parameters(recurse=False, prefix=module_prefix),
                module.named_buffers(recurse=False, prefix=module_prefix),
            ):
                # Buffer like `rotary_emb.inv_freq` should not be loaded. However, it will be
                # transversed by `named_parameters` and `named_buffers`.
                name = self._clean_param_name(name)
                if name not in expected_keys:
                    continue
                load_spec = self.load_spec_mapping.get(name)
                if load_spec is None:
                    raise RuntimeError(f"Internal Error. Parameter {name} not found in load_spec_mapping.")

                _missing_keys = self._load_hf_param(param, load_spec, checkpoint_loader)
                missing_keys.update(_missing_keys)

                if not _missing_keys:
                    loaded_keys.add(name)

            for name, child in module.named_children():
                _prefix = f"{module_prefix}.{name}" if module_prefix else name
                _load_params_from_module(child, _prefix)  # type: ignore

        with profile_time_and_memory("[HF loading cost]"):
            _load_params_from_module(self, "")  # type: ignore
            torch.accelerator.synchronize()
            torch.cpu.synchronize()

        if missing_keys and strict:
            raise RuntimeError(f"Internal error, Missing keys in checkpoint: {missing_keys}")

        if (unloaded_keys := (expected_keys - loaded_keys)) and strict:
            raise RuntimeError(f"Internal error, Unloaded keys in model: {unloaded_keys}")

        return loaded_keys, unloaded_keys, missing_keys

    def _load_hf_param(
        self, param: torch.Tensor, load_spec: LoadSpec, checkpoint_loader: HFCheckpointLoader
    ) -> list[str]:
        """Unified HF load path for a single parameter / buffer.

        ``LoadSpec.plan_hf_load`` computes this rank's HF keys and canonical source-to-local copy regions. This
        method reads and dequantizes those keys, then lets the plan drive model canonicalization and local writes.

        Returns the list of hf_keys that were expected but missing from the
        checkpoint; callers aggregate these for strict-mode reporting.
        """
        local_tensor = param._local_tensor if isinstance(param, DTensor) else param
        load_plan = load_spec.plan_hf_load()
        loaded_tensors, missing_keys = self._read_hf_tensors(
            load_plan.hf_keys,
            checkpoint_loader,
            device=local_tensor.device,
        )
        if missing_keys:
            return missing_keys

        load_plan.load_into(
            loaded_tensors,
            local_tensor,
            canonicalize=self.hf_tensor_to_canonical,
        )
        return []

    def _read_hf_tensors(
        self,
        hf_keys: list[str],
        checkpoint_loader: HFCheckpointLoader,
        *,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[str]]:
        """Read one plan's HF keys, dequantizing checkpoint FP8 when
        present."""
        loaded_tensors: list[torch.Tensor] = []
        missing_keys: list[str] = []
        for hf_key in hf_keys:
            tensor = self._read_hf_tensor(hf_key, checkpoint_loader)
            if tensor is None:
                missing_keys.append(hf_key)
            else:
                loaded_tensors.append(tensor.to(device))
        return loaded_tensors, missing_keys

    def _read_hf_tensor(
        self,
        hf_key: str,
        checkpoint_loader: HFCheckpointLoader,
    ) -> torch.Tensor | None:
        scale_key = hf_key + "_scale_inv"
        is_checkpoint_fp8 = checkpoint_loader.is_key_exist(hf_key) and checkpoint_loader.is_key_exist(scale_key)
        if not is_checkpoint_fp8:
            return checkpoint_loader.load(hf_key)
        if not _is_float8_available():
            raise RuntimeError(
                f"Float8 is not available on {DEVICE}. Please convert the checkpoint from float8 "
                "to bfloat16 on SM89 or later (H100+ GPUs)."
            )
        return self._load_fp8(hf_key, checkpoint_loader)

    def _load_fp8(self, hf_key: str, checkpoint_loader: HFCheckpointLoader) -> torch.Tensor | None:
        loaded_tensor_fp8 = checkpoint_loader.load(hf_key)
        loaded_tensor_scales = checkpoint_loader.load(hf_key + "_scale_inv")
        if loaded_tensor_fp8 is None or loaded_tensor_scales is None:
            return None

        from xtuner.v1.float8.triton_kernels import per_block_dequant_gemm

        return per_block_dequant_gemm(
            loaded_tensor_fp8.to(DEVICE),
            loaded_tensor_scales.to(DEVICE),
        )

    def _has_meta_param(self, module: nn.Module, recurse: bool = False) -> bool:
        """Check if the module has meta parameters."""
        for data in chain(module.parameters(recurse=recurse), module.buffers(recurse=False)):
            if data.is_meta:
                return True
        return False

    def _fsdp_foreach_allgather(
        self, tensor_list: list[torch.Tensor], load_spec_list: list[LoadSpec]
    ) -> list[torch.Tensor]:
        if self.fsdp_mesh is None:
            return tensor_list

        fsdp_group = self.fsdp_mesh.get_group()
        save_plan_list = [load_spec.plan_hf_save(gather_process_group=fsdp_group) for load_spec in load_spec_list]
        return unshard_tensors_for_hf_save(list(tensor_list), save_plan_list)

    @staticmethod
    def _is_same_process_group(left: dist.ProcessGroup, right: dist.ProcessGroup) -> bool:
        if left is right:
            return True
        return dist.get_process_group_ranks(left) == dist.get_process_group_ranks(right)

    def _get_fused_save_ranks(self, hf_key_count: int) -> list[int]:
        world_size = dist.get_world_size()
        max_save_ranks = min(world_size, self.config.hf_save_cfg.max_save_rank, hf_key_count)
        for save_rank_count in range(max_save_ranks, 0, -1):
            if hf_key_count % save_rank_count == 0:
                return list(range(save_rank_count))
        raise RuntimeError(f"Unable to choose save ranks for {hf_key_count} fused HF keys")

    def _to_device_dtype(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        skip_buffers_dtype: bool = False,
    ) -> Self:
        if device is None and dtype is None:
            return self

        if dtype is None:
            self.to(device=device, non_blocking=non_blocking)
            return self

        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError(
                f"_to_device_dtype only accepts floating point or complex dtypes, but got desired dtype={dtype}"
            )

        buffer_ids = set()
        if skip_buffers_dtype:
            # `Module._apply()` only passes the tensor itself into `fn`, so we
            # detect buffers by identity and let `_apply()` keep handling the
            # recursion for us.
            buffer_ids = {
                id(buffer) for module in self.modules() for buffer in module._buffers.values() if buffer is not None
            }

        def _convert_tensor(tensor: torch.Tensor) -> torch.Tensor:
            try:
                if skip_buffers_dtype and id(tensor) in buffer_ids:
                    if device is None:
                        return tensor
                    return tensor.to(device=device, non_blocking=non_blocking)

                return tensor.to(
                    device=device,
                    dtype=dtype if tensor.is_floating_point() or tensor.is_complex() else None,
                    non_blocking=non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                raise

        self._apply(_convert_tensor)
        return self

    def to_device(self, device: torch.device | str):
        if self.fsdp_config is not None and self.fsdp_config.cpu_offload:
            return
        self.to(device, non_blocking=True)
        DEVICE_MODULE.synchronize()
        return

    def _to_empty_meta(self):
        for module in self.modules():
            if self._has_meta_param(module):
                module.to_empty(device=self.device, recurse=False)
        DEVICE_MODULE.synchronize()
        return

    def _wait_save_task(self, tasks: list[Future]):
        "Limit the number of concurrent save tasks to avoid OOM."
        # The older version of xtuner does not have hf_save_worker attributes, using `getattr` avoid from unpickling
        # the old config for backward compatibility.
        if len(tasks) >= getattr(self.config, "hf_save_worker", 16):
            done, pending = wait(tasks)
            for future in done:
                if (exception := future.exception()) is not None:
                    raise exception
            tasks.clear()
            tasks.extend(pending)
        else:
            return

    def _compile_overwrite(self, func_name: str, compile_options: TorchCompileOption | None = None):
        """Overwrite a function in a module with a new function.

        Args:
            func_name (str): The name of the function to overwrite.
            new_func (FunctionType): The new function to use.
            module: The module containing the function to overwrite.
        """
        compiled_function = cast(FunctionType | MaybeCompile, pydoc.locate(func_name))

        if compile_options is None:
            compile_options = {}

        if compiled_function is None:
            raise AttributeError(f"Compiling Error! Cannot locate the function: {func_name}")

        if isinstance(compiled_function, maybe_compile):
            maybe_compile.enable_compile(compiled_function, **compile_options)
        elif (function_type := get_function_type(compiled_function)) is FunctionEnum.TOP_LEVEL_FUNCTION:
            raise ValueError(
                f"Compiling config error! {func_name} is a `TOP LEVEL FUNCTION`, it must be wrapped with "
                "`@maybe_compile` decorator to enable `torch.compile`."
            )
        else:
            compiled_function = cast(FunctionType, compiled_function)

            if (function_type := get_function_type(compiled_function)) is FunctionEnum.LOCAL_FUNCTION:
                raise ValueError(
                    f"Compiling config error! {func_name} is a local function, which is not supported yet."
                )
            elif function_type is FunctionEnum.CLASS_LEVEL_FUNCTION:
                qualname_split = compiled_function.__qualname__.split(".")
                assert len(qualname_split) == 2, (
                    f"XTuner Internal Error! the name of {compiled_function} should be recognized as "
                    f"<class_name>.<method_name>, but got {qualname_split}"
                )
                class_name, method_name = qualname_split

                module_name = compiled_function.__module__
                cls = getattr(import_module(module_name), class_name)

                if not is_compiled_function(compiled_function):
                    setattr(cls, method_name, torch.compile(compiled_function, **compile_options))

        full_name = get_function_full_qualname(compiled_function)  # type: ignore[arg-type]
        logger.debug(f"Enabling torch.compile for function {full_name} with options: {compile_options}")

    def _resolve_compile_cfg(
        self,
        config: XTunerBaseModelConfig,
    ) -> dict[str, TorchCompileOption]:
        if not hasattr(config, "compile_cfg"):
            return {}

        custom_cfg = config.compile_cfg
        if custom_cfg is False:
            self._disable_compile_cfg(self.config)
            return {}

        # torch.compile is not supported on NPU
        if DEVICE == "npu":
            if custom_cfg is not False:
                log_rank0.warning("torch.compile is not supported on NPU, disabling torch.compile.")
            self._disable_compile_cfg(self.config)
            return {}

        if custom_cfg is True or custom_cfg is None:
            compile_cfg = self.default_compile_cfg
        else:
            compile_cfg = custom_cfg

        return compile_cfg

    def _disable_compile_cfg(self, obj):
        if isinstance(obj, PydanticBaseModel) and hasattr(obj, "compile_cfg"):
            obj.compile_cfg = False
            consume(self._disable_compile_cfg(getattr(obj, x)) for x in obj.__class__.model_fields)
        elif isinstance(obj, Mapping):
            consume(map(self._disable_compile_cfg, obj.values()))
        # str&bytes are special Iterable, need to exclude it, otherwise it will infinite loop
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            consume(map(self._disable_compile_cfg, obj))
        else:
            return

    def _maybe_enable_compile(self, compile_cfg: dict[str, TorchCompileOption]):
        if compile_cfg:
            torch._dynamo.config.cache_size_limit = 256

        for target, option in compile_cfg.items():
            self._compile_overwrite(target, option)

    def _mark_dynamic(self, seq_ctx: SequenceContext, dim=0):
        """`cu_seq_lens_q` and `cu_seq_lens_k` are dynamic shapes in each
        fwd/bwd pass.

        Mark them as dynamic explicitly to avoid recompilation.
        """
        torch._dynamo.mark_dynamic(seq_ctx.cu_seq_lens_q, dim)
        torch._dynamo.mark_dynamic(seq_ctx.cu_seq_lens_k, dim)

    def _init_world_mesh(self):
        device = DEVICE
        world_size = dist.get_world_size()

        # TODO: Support hsdp_sharding_size
        fsdp_mesh = init_device_mesh(device, (world_size,))
        return fsdp_mesh

    def _collect_full_state_dict(self, module: nn.Module):
        assert isinstance(module, (nn.Module, FSDPModule))

        ret = {}
        for name, param in module.state_dict().items():  # type: ignore[attr-defined]
            if isinstance(param, DTensor):
                from xtuner.v1.utils.interleaved_shard import (
                    RuntimeLayout,
                    reconstruct_full_tensor,
                )

                if RuntimeLayout.from_dtensor(param).is_interleaved:
                    # `(Shard, InterleavedShard)`-style placements can't be redistributed; use the
                    # explicit reconstruct path instead of `.full_tensor()`.
                    param = reconstruct_full_tensor(param)
                else:
                    param = param.full_tensor()
            ret[name] = param
        return ret

    def _build_loss_ctx(
        self, loss_ctx_cfg: BaseLossConfig | None, data_batch: list[dict], sp_mesh: DeviceMesh | None
    ) -> list[BaseLossContext] | None:
        if loss_ctx_cfg is None:
            return None

        first_loss_ctx = loss_ctx_cfg.build(data=data_batch[0], sp_mesh=sp_mesh)
        # If first build returns None, assume all data in the batch have the same schema
        # and will also return None (e.g., missing required fields like shifted_labels)
        if first_loss_ctx is None:
            return None
        else:
            ret = [first_loss_ctx] + [loss_ctx_cfg.build(data=data, sp_mesh=sp_mesh) for data in data_batch[1:]]
            return ret  # type: ignore[return-value]

    # NOTE: Add this overload for inferring the return type for easier type checking and using
    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: SequenceContext,
        loss_ctx: dict[str, BaseLossContext] | None,
    ) -> ModelOutputs: ...

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        seq_ctx: list[SequenceContext],
        loss_ctx: list[dict[str, BaseLossContext]],
    ) -> ModelOutputs: ...

    __call__ = nn.Module.__call__
