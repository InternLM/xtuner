import json
import math
import pydoc
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import reduce
from importlib import import_module
from itertools import chain
from pathlib import Path
from shutil import copy, copytree
from typing import Annotated, Generator, Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, computed_field
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from typing_extensions import NamedTuple, NotRequired, Self, TypedDict

from transformers.configuration_utils import PretrainedConfig
from xtuner.v1.config import FSDPConfig, GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTensorWiseFloat8CastTensor,
    WeightWithDynamicTilewiseFloat8CastTensor,
)
from xtuner.v1.loss import BaseLossContext
from xtuner.v1.module.attention import MHAConfig, MLAConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.comm.foreach_allgather import foreach_all_gather
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module, profile_time_and_memory
from xtuner.v1.utils.compile import is_compiled_function, maybe_compile
from xtuner.v1.utils.load_spec import LoadEnum, LoadSpec
from xtuner.v1.utils.loader import HFCheckpointLoader
from xtuner.v1.utils.misc import FunctionEnum, FunctionType, get_function_type

from .utils import ModelForwardExtraLogInfo


logger = get_logger()

DEVICE_MODULE = get_torch_device_module()
DEVICE = get_device()


class TorchCompileOption(TypedDict):
    fullgraph: NotRequired[bool]
    dynamic: NotRequired[bool | None]
    mode: NotRequired[str | None]
    options: NotRequired[dict[str, int | bool | str] | None]


class CompileTarget(NamedTuple):
    name: str
    option: TorchCompileOption


DEFAULT_FLOAT8_CFG = [
    CompileTarget("xtuner.v1.float8.fsdp_utils.tensor_to_per_block_fp8_scales", TorchCompileOption(fullgraph=True)),
    CompileTarget("xtuner.v1.float8.fsdp_utils.cast_to_per_block_fp8_with_scales", TorchCompileOption(fullgraph=True)),
    CompileTarget(
        "xtuner.v1.float8.triton_kernels.per_block_quant_gemm.per_block_quant_torch",
        TorchCompileOption(fullgraph=True),
    ),
    CompileTarget(
        "xtuner.v1.float8.fsdp_utils.cast_to_per_tensor_fp8_with_scales", TorchCompileOption(fullgraph=True)
    ),
    CompileTarget(
        "xtuner.v1.float8.float8_linear_tensor_wise.per_tensor_fp8_quant", TorchCompileOption(fullgraph=True)
    ),
]


class TransformerConfig(PydanticBaseModel):
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
    rope_theta: Annotated[float, Parameter(group="model")]  # required by transformers's build rope
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    attention: MLAConfig | MHAConfig
    mlp_bias: Annotated[bool, Parameter(group="model")] = False
    tie_word_embeddings: Annotated[bool, Parameter(group="model")] = False
    model_type: Annotated[str | None, Parameter(group="model")] = None  # TODO: yehaochen maybe should be removed
    generate_config: GenerateConfig | None = None
    float8_cfg: Float8Config | None = None
    return_hidden_states: Annotated[bool, Parameter(group="model")] = False
    use_sliding_window: Annotated[bool, Parameter(group="model")] = False
    max_window_layers: Annotated[int | None, Parameter(group="model")] = None
    rope_scaling_cfg: RopeScalingConfig | None = None
    hf_save_worker: Annotated[int, Parameter(group="model")] = 16
    compile_cfg: list[str | CompileTarget] | None | bool = (
        None  # None means use default compile option, False means disable compile
    )
    dcp_ignore_frozen_params: Annotated[bool, Parameter(group="model")] = False

    @computed_field
    def num_attention_heads(self) -> int:
        return self.attention.num_attention_heads

    @computed_field
    def head_dim(self) -> int:
        return self.attention.head_dim

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention"]]:
        if not self.use_sliding_window:
            return ["full_attention"] * self.num_hidden_layers
        else:
            if self.max_window_layers is None:
                return ["sliding_attention"] * self.num_hidden_layers
            return [
                "sliding_attention" if i >= self.max_window_layers else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    def build(self) -> "BaseModel":
        raise NotImplementedError

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

    @property
    def hf_config(self) -> PretrainedConfig | None:
        """HuggingFace configuration."""
        return None

    def save_hf(self, hf_path: str | Path):
        """Save the configuration to a HuggingFace-compatible format.

        Args:
            hf_path (str | Path): Path where the configuration should be saved.
        """

        if self.hf_config is None:
            raise NotImplementedError("The `hf_config` property must be implemented to save in HuggingFace format.")

        self.hf_config.save_pretrained(hf_path)


class ModelOutputs(TypedDict):
    hidden_states: NotRequired[list[torch.Tensor]]
    logits: NotRequired[torch.Tensor]
    loss: torch.Tensor
    extra_info: ModelForwardExtraLogInfo


def _is_float8_available():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return DEVICE == "cuda" and DEVICE_MODULE.is_available() and DEVICE_MODULE.get_device_capability() >= (8, 9)


class ModelItem(TypedDict):
    seq_ctx: SequenceContext
    loss_ctx: BaseLossContext


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


class BaseModel(nn.Module):
    load_spec_mapping: dict[str, LoadSpec] = {}
    fsdp_mesh: DeviceMesh | None = None
    hsdp_mesh: DeviceMesh | None = None
    fsdp_config: FSDPConfig | None = None
    config: TransformerConfig

    SAFETENSOR_SIZE = 1024**3 * 4  # 4GB
    FSDP_SHARD_DIM = 0

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self._hf_path: Path | None = None  # type: ignore
        self._compile_cfg = self._resolve_comile_cfg(self.config.compile_cfg)

    def set_hf(self, hf_path: str | Path):
        self._hf_path = Path(hf_path)

    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        self._hf_path = Path(hf_path)

        if isinstance(hf_path, Path):
            hf_path = str(hf_path)

        hf_loader = HFCheckpointLoader(hf_path)
        loaded_keys, unloaded_keys, missing_keys = self._load_params(hf_loader, strict=strict)
        return loaded_keys, unloaded_keys, missing_keys

    def scale_and_reduce_grad(self):
        return

    def to_hf_key_list(self, key: str) -> list[str]:
        raise NotImplementedError()

    def trainable_parameters(self):
        params = [(name, param) for name, param in self.named_parameters() if param.requires_grad]
        return params

    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ) -> "BaseModel":
        """Fully shard the model parameters."""
        raise NotImplementedError

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16):
        with profile_time_and_memory(f"[Saving HF to {hf_dir} cost]"):
            self._save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def safetensors_to_params(
        self,
        safetensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        param_name: str,
        start: int | None,
        end: int | None,
        dim: int | None,
    ):
        if len(safetensors) > 1:
            assert dim is not None, "Internal Error dim must not be None when len(safetensors) > 1"
            loaded_tensor = torch.cat(safetensors, dim=dim)
        else:
            loaded_tensor = safetensors[0]

        if start is not None and end is not None:
            assert self.fsdp_config is not None, (
                "Internal Error. fsdp_config must not be None when start and end is not None"
            )
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            loaded_tensor_slice = loaded_tensor.index_select(
                dim=self.FSDP_SHARD_DIM, index=torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            )
            non_pad_len = end - start
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)

            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        else:
            local_tensor.copy_(loaded_tensor)

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
    def default_compile_cfg(self) -> list[str | CompileTarget]:
        return []

    @property
    def compile_cfg(self) -> list[str | CompileTarget]:
        return self._compile_cfg

    @torch.no_grad()
    def init_weights(self):
        # TODO: HardCode here. The initialization method should be module specific. All module in model
        # in model should be defined in `xtuner.module`
        from xtuner.v1.utils import default_init_weights

        initialized_params = default_init_weights(self)

        if missing := {name for name, _ in self.named_parameters()} - initialized_params:
            raise RuntimeError(f"{missing} is not initialized")

    def _init_load_spec(self) -> None:
        # NOTE: (yehaochen) This is a workaround to distinguish between different parameter HF loading methods
        # and model partitioning methods. Although PyTorch provides Shard, Replicate and other Placements, in
        # MoE models, we need to handle both how to load HF weights and how to calculate gradients for partitioned
        # parameters during the backward phase, so a more complex ParallelParamSpec is defined to describe these:
        # Specifically:
        # - For model loading and saving:
        # From a computational efficiency perspective, we have to make the model parameter layout different from the
        # HF model, resulting in a one-to-one or many-to-many mapping relationship, and we need a specification to
        # describe this mapping.
        # - For gradient computation:
        # In MoE models, we need to divide the gradients of EP-partitioned parameters by ep_size (this is another
        # complex issue not elaborated here), and although ep and ep both belong to Shard, their processing logic
        # is different, so we need a specification to express the partitioning method in a more fine-grained way.

        def get_shard_placement(placements: tuple[Placement, ...]) -> Shard | None:
            ret = None
            for p in placements:
                if isinstance(p, Shard):
                    if ret is None:
                        ret = p
                    else:
                        raise RuntimeError("Multiple Shard placements found, please report this issue")
            return ret

        if self.__class__.to_hf_key_list is BaseModel.to_hf_key_list:
            self.load_spec_mapping = {}
            return

        load_spec_mapping: dict[str, LoadSpec] = {}

        for name, param in self.state_dict().items():
            name = self._clean_param_name(name)
            hf_keys = self.to_hf_key_list(name)
            if isinstance(param, DTensor) and (placement := get_shard_placement(param.placements)) is not None:
                dim = placement.dim
                _, _offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
                start = _offset[dim]
                end = _offset[dim] + param._local_tensor.shape[dim]
                local_shape = param._local_tensor.shape
                global_size = param.shape[dim]

                if len(hf_keys) > 1:
                    start_hf_key_idx = start / global_size * len(hf_keys)

                    assert start_hf_key_idx.is_integer(), "Internal xtuner error, please report this issue"
                    start_hf_key_idx = int(start_hf_key_idx)

                    end_hf_key_idx = end / global_size * len(hf_keys)
                    # TODO: (yehaochen) Support TP
                    assert end_hf_key_idx.is_integer(), "Internal xtuner error, please report this issue"
                    load_type = LoadEnum.FUSED
                    end_hf_key_idx = int(end_hf_key_idx)
                elif len(hf_keys) == 1:
                    start_hf_key_idx = start / global_size
                    end_hf_key_idx = end / global_size
                    if start_hf_key_idx == 0 and end_hf_key_idx == 1:
                        load_type = LoadEnum.SAME
                    else:
                        load_type = LoadEnum.SHARD
                else:
                    raise RuntimeError

                # TP shard
                if load_type is LoadEnum.SHARD:
                    load_spec = LoadSpec(
                        name=name,
                        hf_keys=hf_keys,
                        shape=local_shape,
                        dim=dim,
                        load_enum=LoadEnum.SHARD,
                        shard_start=start,
                        shard_end=end,
                        group=param.device_mesh.get_group(),
                    )
                # Replicate
                elif load_type == LoadEnum.SAME:
                    load_spec = LoadSpec(
                        name=name,
                        hf_keys=hf_keys,
                        shape=local_shape,
                        dim=dim,
                        load_enum=LoadEnum.SAME,
                        group=param.device_mesh.get_group(),
                    )
                # EPSHard
                else:
                    load_spec = LoadSpec(
                        name=name,
                        hf_keys=hf_keys[start_hf_key_idx:end_hf_key_idx],
                        shape=local_shape,
                        dim=dim,
                        load_enum=LoadEnum.FUSED,
                        group=param.device_mesh.get_group(),
                    )
            else:
                if len(hf_keys) == 1:
                    load_spec = LoadSpec(
                        name=name,
                        hf_keys=hf_keys,
                        shape=param.shape,
                        load_enum=LoadEnum.SAME,
                    )
                else:
                    load_spec = LoadSpec(
                        name=name,
                        hf_keys=hf_keys,
                        shape=param.shape,
                        load_enum=LoadEnum.FUSED,
                    )
            load_spec_mapping[name] = load_spec

        self.load_spec_mapping = load_spec_mapping

    def _to_float8(
        self,
        gathered_tensor_list: list[torch.Tensor],
        name_list: list[str],
        ori_tensor_list: list[torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[list[torch.Tensor], list[str]]:
        gathered_tensor_list_new, name_list_new = [], []
        for gathered_tensor, name, ori_tensor in zip(gathered_tensor_list, name_list, ori_tensor_list):
            if not is_float8_weight(ori_tensor):
                gathered_tensor_list_new.append(gathered_tensor)
                name_list_new.append(name)
                continue

            from xtuner.v1.float8.triton_kernels.per_block_quant_gemm import per_block_quant_torch

            gathered_tensor_fp8, scale = per_block_quant_torch(gathered_tensor, block_size=128, float8_dtype=dtype)
            gathered_tensor_list_new.extend([gathered_tensor_fp8, scale])
            name_list_new.extend([name, f"{name}_scale_inv"])
        return gathered_tensor_list_new, name_list_new

    def _get_shard_hf_param(
        self,
        params: list[tuple[torch.Tensor, LoadSpec]],
        dtype: torch.dtype,
        device="cpu",
        bucket_size=None,
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        if not params:
            return
        if dtype != torch.bfloat16:
            raise NotImplementedError

        load_spec0 = params[0][1]
        assert load_spec0.group is not None

        def _get_hf_params(fsdp_tensor_list: list[tuple[torch.Tensor, LoadSpec]]) -> list[torch.Tensor]:
            # Get fsdp unsharded params
            _tensor_list, _spec_list = list(zip(*fsdp_tensor_list))
            if self.fsdp_mesh is not None:
                fsdp_unsharded_tensor_list = self._fsdp_foreach_allgather(_tensor_list, _spec_list)  # type: ignore
            else:
                fsdp_unsharded_tensor_list = _tensor_list  # type: ignore

            # Get unsharded params
            _unsharded_tensor_list = foreach_all_gather(fsdp_unsharded_tensor_list, load_spec0.group)
            unsharded_tensor_list = [
                torch.cat([i.to(dtype) for i in tensors], dim=load_spec0.dim) for tensors in _unsharded_tensor_list
            ]
            name_list = [spec.hf_keys[0] for _, spec in fsdp_tensor_list]
            unsharded_tensor_list = [
                self.param_to_safetensor(safetensor, name)
                for safetensor, name in zip(unsharded_tensor_list, name_list)
            ]
            unsharded_tensor_list = [t.to(device) for t in unsharded_tensor_list]
            return unsharded_tensor_list

        if bucket_size is None:
            bucket_size = self.SAFETENSOR_SIZE
        safetensor_size = 0
        tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []
        name_list: list[str] = []

        for param, load_spec in params:
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.to(dtype=dtype)
            tensor_size = self._get_tensor_size(param, dtype)
            if safetensor_size + tensor_size > bucket_size and tensor_list:
                hf_params = _get_hf_params(tensor_list)

                yield name_list, hf_params
                safetensor_size = tensor_size
                name_list = load_spec.hf_keys.copy()
                tensor_list = [(local_tensor, load_spec)]
                continue
            safetensor_size += tensor_size
            tensor_list.append((local_tensor, load_spec))
            name_list.append(load_spec.hf_keys[0])

        if tensor_list:
            hf_params = _get_hf_params(tensor_list)
            yield name_list, hf_params

    def _get_fused_hf_param(
        self,
        params: list[tuple[torch.Tensor, LoadSpec]],
        dtype: torch.dtype,
        device="cpu",
        bucket_size=None,
        return_full_key_per_rank: bool = False,
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        if not params:
            return

        def _get_hf_params(
            fsdp_tensor_list: list[tuple[torch.Tensor, LoadSpec]],
            name_list: list[str],
        ) -> tuple[list[torch.Tensor], list[str]]:
            # Get fsdp unsharded params
            spec_list: list[LoadSpec]
            tensor_list: list[torch.Tensor]

            tensor_list, spec_list = list(zip(*fsdp_tensor_list))  # type: ignore[assignment]
            if self.fsdp_mesh is not None:
                fsdp_unshard_tensor_list = self._fsdp_foreach_allgather(tensor_list, spec_list)  # type: ignore
            else:
                fsdp_unshard_tensor_list = tensor_list  # type: ignore

            saved_fused_tensor_list: list[torch.Tensor] = []
            hf_keys_list: list[list[str]] = []

            for load_spec, fsdp_unshared_tensor in zip(spec_list, fsdp_unshard_tensor_list):
                hf_keys = load_spec.hf_keys

                if load_spec.group is not None:
                    all_hf_keys_list: list[None] | list[list[str]] = [None for _ in range(load_spec.group.size())]
                    dist.all_gather_object(all_hf_keys_list, hf_keys, group=load_spec.group)
                    all_hf_keys_list = cast(list[list[str]], all_hf_keys_list)
                    all_hf_keys = list(chain(*all_hf_keys_list))
                else:
                    all_hf_keys = hf_keys

                current_rank = dist.get_rank()
                fused_save_ranks = self._get_ranks_to_save_fused_tensor(len(all_hf_keys))
                key_per_rank = len(all_hf_keys) / len(fused_save_ranks)
                assert key_per_rank.is_integer(), (
                    f"XTuner Internal Error, size of all_hf_keys: {len(all_hf_keys)},  "
                    f"size of `fused_save_ranks` {len(fused_save_ranks)}"
                )

                # 1. When return_full_key_per_rank is False, we intends to save hf models across ranks,
                # each rank only saves part of hf keys and tensors
                # 2. When return_full_key_per_rank is True, we intends to generate full tensors on each
                # rank for ipc updating weights in RL training.
                if not return_full_key_per_rank:
                    start = int(current_rank * key_per_rank)
                    end = int(start + key_per_rank)
                else:
                    start = 0
                    end = len(all_hf_keys)

                _hf_key_list = all_hf_keys[start:end]

                if not _hf_key_list:
                    continue

                hf_keys_list.append(_hf_key_list)

                assert load_spec.dim is not None
                if load_spec.group is not None:
                    assert load_spec.dim is not None
                    _gathered_tensor_list = [
                        torch.zeros_like(fsdp_unshared_tensor) for _ in range(load_spec.group.size())
                    ]
                    dist.all_gather(_gathered_tensor_list, fsdp_unshared_tensor, group=load_spec.group)
                    _gathered_tensor = torch.cat(_gathered_tensor_list, dim=load_spec.dim)
                else:
                    _gathered_tensor = fsdp_unshared_tensor

                hf_tensor_size = _gathered_tensor.shape[load_spec.dim] / len(all_hf_keys)
                _saved_fused_tensor = torch.index_select(
                    _gathered_tensor,
                    dim=load_spec.dim,
                    index=torch.arange(
                        int(start * hf_tensor_size),
                        int(end * hf_tensor_size),
                        dtype=torch.int64,
                        device=_gathered_tensor.device,
                    ),
                )
                saved_fused_tensor_list.append(_saved_fused_tensor)

            # Split the fused tensor into hf tensors
            hf_tensor_list: list[torch.Tensor] = []
            # used in self._to_float8 to determine whether to convert a unshard hf_tensor to fp8
            fsdp_shard_tensor_list: list[torch.Tensor] = []
            # `origin_tensor_list` is only used to mark, which tensors are float8 weights for the
            # `_to_float8` function
            origin_tensor_list: list[torch.Tensor] = []

            for saved_tensor, load_spec, hf_keys, origin_tensor in zip(
                saved_fused_tensor_list, spec_list, hf_keys_list, tensor_list
            ):
                dim = cast(int, load_spec.dim)
                hf_tensor_size = saved_tensor.shape[dim] / len(hf_keys)
                assert hf_tensor_size.is_integer(), "Internal Error, hf_tensor_size is not integer"
                hf_tensor_size = int(hf_tensor_size)
                hf_tensor = saved_tensor.split([hf_tensor_size] * len(hf_keys), dim=dim)
                hf_tensor_list.extend(hf_tensor)
                fsdp_shard_tensor_list.extend([saved_tensor] * len(hf_tensor))
                origin_tensor_list.extend([origin_tensor] * len(hf_tensor))

            name_list = list(chain.from_iterable(hf_keys_list))
            hf_tensor_list = [
                self.param_to_safetensor(safetensor, name) for safetensor, name in zip(hf_tensor_list, name_list)
            ]

            if dtype == torch.float8_e4m3fn:
                hf_tensor_list_new, name_list_new = self._to_float8(
                    hf_tensor_list, name_list, origin_tensor_list, dtype
                )
                return hf_tensor_list_new, name_list_new

            hf_tensor_list = [t.to(device=device) for t in hf_tensor_list]

            return hf_tensor_list, name_list

        if bucket_size is None:
            bucket_size = self.SAFETENSOR_SIZE
        safetensor_size = 0
        tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []
        name_list: list[str] = []

        for param, load_spec in params:
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            tensor_size = self._get_tensor_size(param, dtype)
            if safetensor_size + tensor_size > bucket_size and tensor_list:
                hf_params, name_list = _get_hf_params(tensor_list, name_list)
                yield name_list, hf_params
                safetensor_size = tensor_size
                name_list = load_spec.hf_keys.copy()
                tensor_list = [(local_tensor, load_spec)]
                continue
            safetensor_size += tensor_size
            tensor_list.append((local_tensor, load_spec))
            name_list.extend(load_spec.hf_keys)

        if tensor_list:
            hf_params, name_list = _get_hf_params(tensor_list, name_list)
            yield name_list, hf_params

    def _get_same_hf_param(
        self,
        params: list[tuple[torch.Tensor, LoadSpec]],
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        bucket_size: int | None = None,
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        if not params:
            return
        if bucket_size is None:
            bucket_size = self.SAFETENSOR_SIZE
        safetensor_size = 0
        tensor_list: list[torch.Tensor] = []
        load_spec_list: list[LoadSpec] = []
        name_list: list[str] = []

        for param, load_spec in params:
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            tensor_size = self._get_tensor_size(param, dtype)
            if safetensor_size + tensor_size > bucket_size and tensor_list:
                if self.fsdp_mesh is not None:
                    gathered_tensor_list = self._fsdp_foreach_allgather(tensor_list, load_spec_list)
                else:
                    gathered_tensor_list = tensor_list
                gathered_tensor_list = [
                    self.param_to_safetensor(safetensor, name)
                    for safetensor, name in zip(gathered_tensor_list, name_list)
                ]
                if dtype == torch.float8_e4m3fn:
                    gathered_tensor_list, name_list = self._to_float8(
                        gathered_tensor_list, name_list, tensor_list, dtype
                    )
                gathered_tensor_list = [t.to(device=device) for t in gathered_tensor_list]
                yield name_list, gathered_tensor_list
                safetensor_size = tensor_size
                name_list = load_spec.hf_keys.copy()
                tensor_list = [local_tensor]
                load_spec_list = [load_spec]
                continue
            safetensor_size += tensor_size
            tensor_list.append(local_tensor)
            name_list.append(load_spec.hf_keys[0])
            load_spec_list.append(load_spec)

        if tensor_list:
            if self.fsdp_mesh is not None:
                gathered_tensor_list = self._fsdp_foreach_allgather(tensor_list, load_spec_list)
            else:
                gathered_tensor_list = tensor_list

            gathered_tensor_list = [
                self.param_to_safetensor(safetensor, name) for safetensor, name in zip(gathered_tensor_list, name_list)
            ]
            if dtype == torch.float8_e4m3fn:
                gathered_tensor_list, name_list = self._to_float8(gathered_tensor_list, name_list, tensor_list, dtype)
            gathered_tensor_list = [t.to(device=device) for t in gathered_tensor_list]
            yield name_list, gathered_tensor_list

    def _clean_param_name(self, name: str) -> str:
        if "._checkpoint_wrapped_module." in name:
            name = name.replace("._checkpoint_wrapped_module.", ".")
        if "._orig_mod." in name:
            name = name.replace("._orig_mod.", ".")
        return name

    def _group_param_by_load_spec(self, load_enum: LoadEnum):
        """Group the parameters by load spec."""
        ret = []
        for name, param in self.state_dict().items():
            load_spec = self.load_spec_mapping.get(name)
            if load_spec is None:
                raise ValueError(f"Internal Error. Parameter {name} not found in load_spec_mapping.")
            if load_spec.load_enum == load_enum:
                ret.append((param, load_spec))
            else:
                continue
        return ret

    def _get_tensor_size(self, tensor: torch.Tensor, dtype: torch.dtype) -> int:
        """Get the size of the tensor in bytes."""
        # return tensor.element_size() * tensor.numel()
        return dtype.itemsize * tensor.numel()

    def _get_safe_tensor_num(self, dtype: torch.dtype) -> int:
        """Get the size of the model in bytes."""
        shard_size = 0
        same_size = 0
        fused_size = 0
        for name, param in self.state_dict().items():
            load_spec = self.load_spec_mapping.get(name)
            if load_spec is None:
                raise ValueError(f"Internal Error. Parameter {name} not found in load_spec_mapping.")
            if load_spec.load_enum == LoadEnum.SHARD:
                shard_size += self._get_tensor_size(param, dtype)
            elif load_spec.load_enum == LoadEnum.SAME:
                same_size += self._get_tensor_size(param, dtype)
            elif load_spec.load_enum == LoadEnum.FUSED:
                fused_size += self._get_tensor_size(param, dtype)
        return (
            math.ceil(shard_size / self.SAFETENSOR_SIZE)
            + math.ceil(same_size / self.SAFETENSOR_SIZE)
            + math.ceil(fused_size / self.SAFETENSOR_SIZE)
        )

    def _save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16):
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

        # TODO: Support fp8 saving
        shard_gen = self._get_shard_hf_param(self._group_param_by_load_spec(LoadEnum.SHARD), dtype=save_dtype)
        same_gen = self._get_same_hf_param(self._group_param_by_load_spec(LoadEnum.SAME), dtype=save_dtype)
        fused_gen = self._get_fused_hf_param(self._group_param_by_load_spec(LoadEnum.FUSED), dtype=save_dtype)

        is_others_save_rank = not dist.is_initialized() or dist.get_rank() == 0

        # Tell me why! why! old cao! @HIT-cwh
        # mp_context = multiprocessing.get_context("fork")
        # save_executor = ProcessPoolExecutor(max_workers=16, mp_context=mp_context)
        save_executor = ThreadPoolExecutor(max_workers=16)

        if dist.is_initialized():
            save_rank = dist.get_rank()
        else:
            save_rank = 0

        # Sepreately save fused parameters and others to make sure each saving rank will not save
        # dupilicated keys
        #
        save_futures = []
        weight_map = {}
        safetensor_index = 0

        for name_list, hf_tensor_list in fused_gen:
            if not name_list:
                continue

            safetensor_index += 1
            safetensor_name = f"model-{safetensor_index:04d}-fused-save_rank{save_rank}.safetensors"
            weight_map.update({name: safetensor_name for name in name_list})
            assert save_executor is not None, "Internal Error, save_executor should not be None"
            future = save_executor.submit(
                _save_file,
                dict(zip(name_list, hf_tensor_list)),
                hf_dir / safetensor_name,
            )
            save_futures.append(future)
            self._wait_save_task(save_futures)

        safetensor_index = 0
        for name_list, hf_tensor_list in chain(same_gen, shard_gen):
            safetensor_index += 1
            safetensor_name = f"model-{safetensor_index:04d}-others-save_rank{save_rank}.safetensors"

            if is_others_save_rank:
                # for tie_word_embeddings, we need to make sure each key is only saved once
                unique_name_list = []
                unique_hf_tensor_list = []
                for name, hf_tensor in zip(name_list, hf_tensor_list):
                    if name not in weight_map:
                        unique_name_list.append(name)
                        unique_hf_tensor_list.append(hf_tensor)
                        weight_map[name] = safetensor_name

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
            if self.config.hf_config is not None:
                self.config.save_hf(hf_dir)
            elif self._hf_path is not None:
                for file in cast(Path, self._hf_path).iterdir():
                    if file.suffix != ".safetensors":
                        # Copy the model config and tokenizer files to the save path
                        target_path = hf_dir / file.name
                        if file.is_file():
                            copy(file, target_path)
                        else:
                            copytree(file, target_path)

            else:
                raise RuntimeError("Internal Error, both self.config.hf_config and self._hf_path are None")

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

                if load_spec.load_enum == LoadEnum.SAME:
                    _missing_keys = self._load_same_hf_param(param, load_spec, checkpoint_loader)
                elif load_spec.load_enum == LoadEnum.FUSED:
                    _missing_keys = self._load_fused_hf_param(param, load_spec, checkpoint_loader)
                elif load_spec.load_enum == LoadEnum.SHARD:
                    _missing_keys = self._load_shard_hf_param(param, load_spec, checkpoint_loader)
                else:
                    raise RuntimeError(f"Unsupported load_enum: {load_spec.load_enum}")
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

    def _is_loaded_param_fp8(self, hf_key: str, checkpoint_loader: HFCheckpointLoader) -> bool:
        hf_key_scale_inv = hf_key + "_scale_inv"
        return checkpoint_loader.is_key_exist(hf_key) and checkpoint_loader.is_key_exist(hf_key_scale_inv)

    def _load_fp8(self, hf_key: str, checkpoint_loader: HFCheckpointLoader) -> torch.Tensor | None:
        hf_key_scale_inv = hf_key + "_scale_inv"
        loaded_tensor_fp8 = checkpoint_loader.load(hf_key)
        loaded_tensor_scales = checkpoint_loader.load(hf_key_scale_inv)
        if loaded_tensor_fp8 is None or loaded_tensor_scales is None:
            return None

        from xtuner.v1.float8.triton_kernels import per_block_dequant_gemm

        loaded_tensor = per_block_dequant_gemm(
            loaded_tensor_fp8.to(DEVICE),
            loaded_tensor_scales.to(DEVICE),
        )
        return loaded_tensor

    def _load_same_hf_param(
        self, param: torch.Tensor, load_spec: LoadSpec, checkpoint_loader: HFCheckpointLoader
    ) -> list[str]:  # return missing key
        local_tensor = param._local_tensor if isinstance(param, DTensor) else param
        hf_key = load_spec.hf_keys[0]

        if self._is_loaded_param_fp8(hf_key, checkpoint_loader):
            if not _is_float8_available():
                raise RuntimeError(
                    f"Float8 is not available on {DEVICE}. Please convert the checkpoint from float8 to bfloat16 on SM89 or later (H100+ GPUs)."
                )
            loaded_tensor = self._load_fp8(hf_key, checkpoint_loader)
        else:
            loaded_tensor = checkpoint_loader.load(hf_key)
        if loaded_tensor is None:
            return [hf_key]

        loaded_tensor = loaded_tensor.to(local_tensor.device)

        if self.fsdp_mesh is not None and isinstance(param, nn.Parameter):
            shape_before_fsdp = load_spec.shape
            _, _offset = compute_local_shape_and_global_offset(
                shape_before_fsdp, self.fsdp_mesh, [Shard(self.FSDP_SHARD_DIM)]
            )
            fsdp_start = _offset[self.FSDP_SHARD_DIM]
            fsdp_end = fsdp_start + local_tensor.shape[self.FSDP_SHARD_DIM]

            start = fsdp_start
            end = fsdp_end
        else:
            start = None
            end = None

        self.safetensors_to_params(
            [loaded_tensor], local_tensor, param_name=load_spec.name, start=start, end=end, dim=load_spec.dim
        )
        return []

    def _load_fused_hf_param(
        self, param: torch.Tensor, load_spec: LoadSpec, checkpoint_loader: HFCheckpointLoader
    ) -> list[str]:
        # For expert parallel
        # NOTE:
        # 1. Get `hf-keys` required by sharded param (sharded by ep group)
        # 2. Asumming FSDP sharding the tensor at the same dim as ep group, Get the twice sharded
        #    `hf-keys`. For example, if we have 128 experts with ep-size 8 and fsdp-size 16. The
        #    the param sharded by ep group will have 128/8 = 16 `hf-keys`, and the param further sharded
        #    by FSDP will only have 128/8/16 = 1 `hf-keys`
        # 3. Calculating the `offset` and `size` of FSDP param base on the ep sharded params, and fill
        #    the FSDP param with the loaded tensor.
        hf_keys = load_spec.hf_keys
        local_tensor = param._local_tensor if isinstance(param, DTensor) else param

        assert load_spec.dim == self.FSDP_SHARD_DIM, "Only support FSDP and model parallel sharding at the same dim!"
        if self.fsdp_mesh is not None:
            shape_before_fsdp = load_spec.shape
            if is_float8_weight(local_tensor):
                # fp8 weights may be padded, so we need to calculate the hf_key_size base on local_tensor._ori_shape
                if load_spec.group is None:
                    hf_key_size = local_tensor._ori_shape[self.FSDP_SHARD_DIM] / len(hf_keys)  # type: ignore
                else:
                    hf_key_size = (
                        local_tensor._ori_shape[self.FSDP_SHARD_DIM]  # type: ignore
                        / dist.get_world_size(group=load_spec.group)
                        / len(hf_keys)
                    )
            else:
                # shape_before_fsdp[self.FSDP_SHARD_DIM] == local_tensor.shape[self.FSDP_SHARD_DIM] / dist.get_world_size(group=load_spec.group)
                hf_key_size = shape_before_fsdp[self.FSDP_SHARD_DIM] / len(hf_keys)
            assert hf_key_size.is_integer(), (
                "Model parallel sharding size should be divisible by fused huggingface tensors!"
            )
            hf_key_size = int(hf_key_size)
            _, _offset = compute_local_shape_and_global_offset(
                shape_before_fsdp, self.fsdp_mesh, [Shard(self.FSDP_SHARD_DIM)]
            )
            fsdp_start = _offset[self.FSDP_SHARD_DIM]
            fsdp_end = fsdp_start + local_tensor.shape[self.FSDP_SHARD_DIM]

            hf_keys_start = int(fsdp_start / hf_key_size)
            hf_keys_end = math.ceil(fsdp_end / hf_key_size)

            # Empty pad by fsdp
            if hf_keys_start == hf_keys_end:
                return []

            hf_keys = hf_keys[hf_keys_start:hf_keys_end]

            start = fsdp_start % hf_key_size
            end = start + local_tensor.shape[self.FSDP_SHARD_DIM]
        else:
            start = None
            end = None

        # dist.breakpoint(1)
        missing_keys: list[str] = []
        _loaded_tensor: list[torch.Tensor] = []
        for hf_key in hf_keys:
            weight = self._load_fp8(hf_key, checkpoint_loader)
            if weight is None:
                weight = checkpoint_loader.load(hf_key)
            if weight is None:
                missing_keys.append(hf_key)
                continue
            _loaded_tensor.append(weight.to(local_tensor.device))

        if not hf_keys:
            # fp8 pad
            assert self.config.float8_cfg is not None
            # assert self.fsdp_config is not None and self.fsdp_config.ep_size == 1, (
            #     "Only support fp8 pad for MoE with ep_size == 1"
            # )
            local_tensor.zero_()  # type: ignore  # padded part must be set to 0
            return missing_keys

        self.safetensors_to_params(
            _loaded_tensor, local_tensor, param_name=load_spec.name, start=start, end=end, dim=load_spec.dim
        )
        return missing_keys

    def _load_shard_hf_param(
        self, param: torch.Tensor, load_spec: LoadSpec, checkpoint_loader: HFCheckpointLoader
    ) -> list[str]:
        # For tensor parallel
        # NOTE:
        # 1. Get `hf-keys` required by sharded param (sharded by tp group, only 1 key)
        # 2. all gather the sharded param across tp group
        # 3 Fill the sharded param with the sliced gathered tensor.
        hf_key = load_spec.hf_keys[0]
        local_tensor = param._local_tensor if isinstance(param, DTensor) else param

        loaded_tensor = checkpoint_loader.load(hf_key)
        if loaded_tensor is None:
            return [hf_key]

        loaded_tensor = loaded_tensor.to(local_tensor.device)

        assert load_spec.shard_start is not None and load_spec.shard_end is not None, (
            "load_spec.shard_start and load_spec.shard_end should not be None for sharded params"
        )

        if self.fsdp_mesh is not None:
            shape_before_fsdp = load_spec.shape
            _, _offset = compute_local_shape_and_global_offset(
                shape_before_fsdp, self.fsdp_mesh, [Shard(self.FSDP_SHARD_DIM)]
            )
            fsdp_start = _offset[self.FSDP_SHARD_DIM]
            fsdp_end = fsdp_start + local_tensor.shape[self.FSDP_SHARD_DIM]

            start = fsdp_start + load_spec.shard_start
            end = fsdp_end + load_spec.shard_start
        else:
            start = load_spec.shard_start
            end = load_spec.shard_end

        self.safetensors_to_params(
            safetensors=[loaded_tensor],
            local_tensor=local_tensor,
            param_name=load_spec.name,
            start=start,
            end=end,
            dim=load_spec.dim,
        )
        return []

    def _has_meta_param(self, module: nn.Module, recurse: bool = False) -> bool:
        """Check if the module has meta parameters."""
        for data in chain(module.parameters(recurse=recurse), module.buffers(recurse=False)):
            if data.is_meta:
                return True
        return False

    def _fsdp_foreach_allgather(
        self, tensor_list: list[torch.Tensor], load_spec_list: list[LoadSpec]
    ) -> list[torch.Tensor]:
        assert self.fsdp_mesh is not None, "Internal Error, fsdp_mesh should not be None"
        origin_fsdp_size = []
        padded_tensor_list = []

        for param, load_spec in zip(tensor_list, load_spec_list):
            shape_before_fsdp = load_spec.shape[self.FSDP_SHARD_DIM]
            padded_size = math.ceil(shape_before_fsdp / self.fsdp_mesh.size())
            pad_list = [0] * (2 * param.dim())
            pad_idx = 2 * (param.dim() - 1 - self.FSDP_SHARD_DIM)
            pad_list[pad_idx + 1] = padded_size - param.shape[self.FSDP_SHARD_DIM]
            padded_tensor = F.pad(param, pad_list)
            padded_tensor_list.append(padded_tensor)
            if is_float8_weight(param):
                dim_before_fsdp: int
                if load_spec.group is None:
                    dim_before_fsdp = param._ori_shape[self.FSDP_SHARD_DIM]  # type: ignore
                else:
                    dim_before_fsdp = param._ori_shape[self.FSDP_SHARD_DIM] / dist.get_world_size(  # type: ignore
                        group=load_spec.group
                    )
                origin_fsdp_size.append(dim_before_fsdp)
            else:
                origin_fsdp_size.append(load_spec.shape[self.FSDP_SHARD_DIM])

        _fsdp_unsharded_tensor_list = foreach_all_gather(padded_tensor_list, self.fsdp_mesh.get_group())
        fsdp_unsharded_tensor_list = []

        # Concatenate the tensors along the FSDP shard dim
        for tensors, size in zip(_fsdp_unsharded_tensor_list, origin_fsdp_size):
            tensor = torch.cat(tensors, dim=self.FSDP_SHARD_DIM)
            cat_tensor = torch.index_select(
                tensor,
                dim=self.FSDP_SHARD_DIM,
                index=torch.arange(0, size, dtype=torch.int64, device=tensors[0].device),
            )
            pad_tensor = torch.index_select(
                tensor,
                dim=self.FSDP_SHARD_DIM,
                index=torch.arange(size, tensor.shape[0], dtype=torch.int64, device=tensors[0].device),
            )
            assert (pad_tensor == 0).all(), f"Internal Error, padded tensor is not zero {pad_tensor}!"
            fsdp_unsharded_tensor_list.append(cat_tensor)

        return fsdp_unsharded_tensor_list

    def _get_ranks_to_save_fused_tensor(self, fused_size: int) -> list[int]:
        # Goal: decide how many ranks are used to store model/expert parameters.
        # Policy: choose d such that:
        #   1) d is a positive divisor of world_size,
        #   2) d <= num_experts,
        #   3) d is as close to num_experts as possible under (1)(2).
        # This is equivalent to: pick the largest divisor of world_size that does not exceed num_experts.
        # Rationale: ensures feasibility under expert count, maximizes utilization, and yields balanced groups.
        # Implementation hint: enumerate divisor pairs (i, world_size // i) for i up to sqrt(world_size) and keep the max d <= num_experts.
        # Complexity: O(sqrt(world_size)).
        world_size = dist.get_world_size()

        if world_size >= fused_size:
            return list(range(fused_size))

        num_ranks_to_save = None
        best_diff = None

        i = 1
        while i * i <= fused_size:
            if fused_size % i == 0:
                for d in (i, fused_size // i):
                    diff = abs(d - world_size)
                    if (
                        num_ranks_to_save is None
                        or (diff < best_diff)  # type: ignore
                        or (diff == best_diff and d < num_ranks_to_save)
                    ):
                        num_ranks_to_save, best_diff = d, diff
            i += 1
        return list(range(cast(int, num_ranks_to_save)))

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
        compiled_function = pydoc.locate(func_name)

        if compile_options is None:
            compile_options = {}

        if isinstance(compiled_function, maybe_compile):
            maybe_compile.enable_compile(compiled_function, **compile_options)
            return

        if compiled_function is not None:
            compiled_function = cast(FunctionType, compiled_function)

            if (function_type := get_function_type(compiled_function)) is FunctionEnum.LOCAL_FUNCTION:
                raise ValueError(
                    f"Compiling config error! {func_name} is a local function, which is not supported yet."
                )
            elif function_type is FunctionEnum.CLASS_LEVEL_FUNCTION:
                class_name = compiled_function.__qualname__.split(".")[0]
                module_name = compiled_function.__module__
                _, method_name = func_name.rsplit(".", 1)

                cls = getattr(import_module(module_name), class_name)
                if not is_compiled_function(compiled_function):
                    setattr(cls, method_name, torch.compile(compiled_function, **compile_options))
        else:
            raise AttributeError(f"Compiling Error! Cannot locate the function: {func_name}")

    def _resolve_comile_cfg(self, custom_cfg: list[str | CompileTarget] | bool | None) -> list[str | CompileTarget]:
        if custom_cfg is False:
            return []

        if custom_cfg is True or custom_cfg is None:
            compile_cfg = self.default_compile_cfg
        else:
            compile_cfg = custom_cfg

        return compile_cfg

    def _maybe_enable_compile(self, compile_cfg: list[str | CompileTarget]):
        if compile_cfg:
            torch._dynamo.config.cache_size_limit = 256

        def _compile_cfg_to_dict(compile_cfg: list[str | CompileTarget]) -> dict[str, TorchCompileOption]:
            return {
                (i.name if isinstance(i, CompileTarget) else i): (
                    i.option if isinstance(i, CompileTarget) else TorchCompileOption()
                )
                for i in compile_cfg
            }

        compile_cfg_dict = _compile_cfg_to_dict(compile_cfg)

        for target, option in compile_cfg_dict.items():
            self._compile_overwrite(target, option)
