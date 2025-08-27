import json
import math
from concurrent.futures import ProcessPoolExecutor, wait
from functools import reduce
from itertools import chain
from pathlib import Path
from shutil import copy, copytree
from typing import Generator, TypedDict, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from xtuner.v1.config import FSDPConfig
from xtuner.v1.config.base_model import MoEConfig, TransformerConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTensorWiseFloat8CastTensor,
    WeightWithDynamicTilewiseFloat8CastTensor,
)
from xtuner.v1.float8.triton_kernels import per_block_dequant_gemm
from xtuner.v1.float8.triton_kernels.per_block_quant_gemm import per_block_quant_torch
from xtuner.v1.loss import CELossContext
from xtuner.v1.ops.comm.foreach_allgather import foreach_all_gather
from xtuner.v1.utils import get_device, get_torch_device_module, profile_time_and_memory
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils.load_spec import LoadEnum, LoadSpec
from xtuner.v1.utils.loader import HFCheckpointLoader


DEVICE_MODULE = get_torch_device_module()
DEVICE = get_device()


def _is_float8_available():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return DEVICE == "cuda" and DEVICE_MODULE.is_available() and DEVICE_MODULE.get_device_capability() >= (8, 9)


class ModelItem(TypedDict):
    seq_ctx: SequenceContext
    loss_ctx: CELossContext


def is_float8_weight(tensor):
    return isinstance(tensor, (WeightWithDynamicTilewiseFloat8CastTensor, WeightWithDynamicTensorWiseFloat8CastTensor))


def _save_file(
    tensors: dict[str, torch.Tensor],
    filename,
    metadata=None,
):
    torch.cpu.synchronize()
    save_file(tensors, filename, metadata=metadata)


class BaseModel(nn.Module):
    load_spec_mapping: dict[str, LoadSpec] = {}
    fsdp_mesh: DeviceMesh | None = None
    hsdp_mesh: DeviceMesh | None = None
    fsdp_config: FSDPConfig | None = None
    config: TransformerConfig

    SAFETENSOR_SIZE = 1024**3 * 4  # 4GB
    FSDP_SHARD_DIM = 0

    def __init__(self):
        super().__init__()
        self._hf_path = None

        self._hf_path: Path | None = None

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

                start_hf_key_idx = start / global_size * len(hf_keys)
                assert start_hf_key_idx.is_integer(), "Internal xtuner error, please report this issue"
                start_hf_key_idx = int(start_hf_key_idx)

                end_hf_key_idx = end / global_size * len(hf_keys)
                # TODO: (yehaochen) Support TP
                assert end_hf_key_idx.is_integer(), "Internal xtuner error, please report this issue"
                end_hf_key_idx = int(end_hf_key_idx)

                # TP shard
                if start_hf_key_idx == end_hf_key_idx:
                    load_spec = LoadSpec(
                        hf_keys=hf_keys,
                        shape=local_shape,
                        dim=dim,
                        load_enum=LoadEnum.SHARD,
                        shard_start=start,
                        shard_end=end,
                        group=param.device_mesh.get_group(),
                    )
                # Replicate
                elif len(hf_keys) == 1:
                    load_spec = LoadSpec(
                        hf_keys=hf_keys,
                        shape=local_shape,
                        dim=dim,
                        load_enum=LoadEnum.SAME,
                        group=param.device_mesh.get_group(),
                    )
                # EPSHard
                else:
                    load_spec = LoadSpec(
                        hf_keys=hf_keys[start_hf_key_idx:end_hf_key_idx],
                        shape=local_shape,
                        dim=dim,
                        load_enum=LoadEnum.FUSED,
                        group=param.device_mesh.get_group(),
                    )
            else:
                if len(hf_keys) == 1:
                    load_spec = LoadSpec(
                        hf_keys=hf_keys,
                        shape=param.shape,
                        load_enum=LoadEnum.SAME,
                    )
                else:
                    load_spec = LoadSpec(
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
            gathered_tensor_fp8, scale = per_block_quant_torch(gathered_tensor, block_size=128, float8_dtype=dtype)
            gathered_tensor_list_new.extend([gathered_tensor_fp8, scale])
            name_list_new.extend([name, f"{name}_scale_inv"])
        return gathered_tensor_list_new, name_list_new

    def _get_shard_hf_param(
        self, params: list[tuple[torch.Tensor, LoadSpec]], dtype: torch.dtype = torch.bfloat16
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        if not params:
            return
        if dtype != torch.bfloat16:
            raise NotImplementedError

        load_spec0 = params[0][1]
        assert load_spec0.group is not None

        for _, load_spec in params[1:]:
            assert load_spec0 == load_spec, "All params should have the same load spec for sharded params"

        def _get_hf_params(fsdp_tensor_list: list[tuple[torch.Tensor, LoadSpec]]) -> list[torch.Tensor]:
            # Get fsdp unsharded params
            _tensor_list, _spec_list = list(zip(*fsdp_tensor_list))
            fsdp_unsharded_tensor_list = self._fsdp_foreach_allgather(_tensor_list, _spec_list)  # type: ignore

            # Get unsharded params
            _unsharded_tensor_list = foreach_all_gather(fsdp_unsharded_tensor_list, load_spec0.group)
            unsharded_tensor_list = [
                torch.cat([i.to(dtype) for i in tensors], dim=load_spec0.dim) for tensors in _unsharded_tensor_list
            ]
            return unsharded_tensor_list

        safetensor_size = 0
        tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []
        name_list: list[str] = []

        for param, load_spec in params:
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.to(dtype=dtype)
            if safetensor_size + self._get_tensor_size(param, dtype) > self.SAFETENSOR_SIZE:
                safetensor_size = 0
                yield name_list, _get_hf_params(tensor_list)
                name_list = load_spec.hf_keys.copy()
                tensor_list = [(local_tensor, load_spec)]
                continue
            safetensor_size += self._get_tensor_size(param, dtype)
            tensor_list.append((local_tensor, load_spec))
            name_list.append(load_spec.hf_keys[0])

        if tensor_list:
            yield name_list, _get_hf_params(tensor_list)

    def _get_fused_hf_param(
        self, params: list[tuple[torch.Tensor, LoadSpec]], dtype: torch.dtype
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        if not params:
            return

        def _get_hf_params(
            fsdp_tensor_list: list[tuple[torch.Tensor, LoadSpec]],
            name_list: list[str],
        ) -> tuple[list[torch.Tensor], list[str]]:
            # Get fsdp unsharded params
            _tensor_list, _spec_list = list(zip(*fsdp_tensor_list))
            fsdp_unshard_tensor_list = self._fsdp_foreach_allgather(_tensor_list, _spec_list)  # type: ignore

            # Split the fused tensor into hf tensors
            hf_tensor_list: list[torch.Tensor] = []
            # used in self._to_float8 to determine whether to convert a unshard hf_tensor to fp8
            fsdp_shard_tensor_list: list[torch.Tensor] = []
            for unshard_tensor, load_spec, shard_tensor in zip(fsdp_unshard_tensor_list, _spec_list, _tensor_list):
                dim = load_spec.dim
                hf_tensor_size = unshard_tensor.shape[dim] / len(load_spec.hf_keys)
                assert hf_tensor_size.is_integer(), "Internal Error, hf_tensor_size is not integer"
                hf_tensor_size = int(hf_tensor_size)
                hf_tensor = unshard_tensor.split([hf_tensor_size] * len(load_spec.hf_keys), dim=dim)
                hf_tensor_list.extend(hf_tensor)
                fsdp_shard_tensor_list.extend([shard_tensor] * len(hf_tensor))
            assert len(name_list) == len(hf_tensor_list)

            if dtype == torch.float8_e4m3fn:
                hf_tensor_list_new, name_list_new = self._to_float8(
                    hf_tensor_list, name_list, fsdp_shard_tensor_list, dtype
                )
                return hf_tensor_list_new, name_list_new

            return hf_tensor_list, name_list

        safetensor_size = 0
        tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []
        name_list: list[str] = []

        for param, load_spec in params:
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            if safetensor_size + self._get_tensor_size(param, dtype) > self.SAFETENSOR_SIZE:
                if self.fsdp_mesh is not None:
                    hf_params, name_list = _get_hf_params(tensor_list, name_list)
                else:
                    if dtype == torch.bfloat16:
                        hf_params = [t for t, _ in tensor_list]
                    elif dtype == torch.float8_e4m3fn:
                        hf_params, name_list = self._to_float8(
                            [t for t, _ in tensor_list], name_list, [t for t, _ in tensor_list], dtype
                        )
                    else:
                        raise NotImplementedError(f"dtype {dtype} is not supported for fused hf param")
                safetensor_size = 0
                hf_params = [t.cpu() for t in hf_params]
                yield name_list, hf_params
                name_list = load_spec.hf_keys.copy()
                tensor_list = [(local_tensor, load_spec)]
                continue
            safetensor_size += self._get_tensor_size(param, dtype)
            tensor_list.append((local_tensor, load_spec))
            name_list.extend(load_spec.hf_keys)

        if tensor_list:
            if self.fsdp_mesh is not None:
                hf_params, name_list = _get_hf_params(tensor_list, name_list)
            else:
                if dtype == torch.bfloat16:
                    hf_params = [t for t, _ in tensor_list]
                elif dtype == torch.float8_e4m3fn:
                    hf_params, name_list = self._to_float8(
                        [t for t, _ in tensor_list], name_list, [t for t, _ in tensor_list], dtype
                    )
                else:
                    raise NotImplementedError(f"dtype {dtype} is not supported for fused hf param")
            hf_params = [t.cpu() for t in hf_params]
            yield name_list, hf_params

    def _get_same_hf_param(
        self, params: list[tuple[torch.Tensor, LoadSpec]], dtype: torch.dtype
    ) -> Generator[tuple[list[str], list[torch.Tensor]], None, None]:
        if not params:
            return
        safetensor_size = 0
        tensor_list: list[torch.Tensor] = []
        load_spec_list: list[LoadSpec] = []
        name_list: list[str] = []

        for param, load_spec in params:
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            if safetensor_size + self._get_tensor_size(param, dtype) > self.SAFETENSOR_SIZE:
                if self.fsdp_mesh is not None:
                    gathered_tensor_list = self._fsdp_foreach_allgather(tensor_list, load_spec_list)
                else:
                    gathered_tensor_list = tensor_list
                safetensor_size = 0
                if dtype == torch.float8_e4m3fn:
                    gathered_tensor_list, name_list = self._to_float8(
                        gathered_tensor_list, name_list, tensor_list, dtype
                    )
                gathered_tensor_list = [t.cpu() for t in gathered_tensor_list]
                yield name_list, gathered_tensor_list
                name_list = load_spec.hf_keys.copy()
                tensor_list = [local_tensor]
                load_spec_list = [load_spec]
                continue
            safetensor_size += self._get_tensor_size(param, dtype)
            tensor_list.append(local_tensor)
            name_list.append(load_spec.hf_keys[0])
            load_spec_list.append(load_spec)

        if tensor_list:
            if self.fsdp_mesh is not None:
                gathered_tensor_list = self._fsdp_foreach_allgather(tensor_list, load_spec_list)
            else:
                gathered_tensor_list = tensor_list
            if dtype == torch.float8_e4m3fn:
                gathered_tensor_list, name_list = self._to_float8(gathered_tensor_list, name_list, tensor_list, dtype)
            gathered_tensor_list = [t.cpu() for t in gathered_tensor_list]
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
        if self._hf_path is None:
            raise RuntimeError("Please call from_hf() before save_hf().")

        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)

        DEVICE_MODULE.empty_cache()
        assert save_dtype in [torch.float8_e4m3fn, torch.bfloat16], f"save_dtype {save_dtype} is not supported"

        # TODO: Support fp8 saving

        shard_gen = self._get_shard_hf_param(self._group_param_by_load_spec(LoadEnum.SHARD), dtype=save_dtype)
        same_gen = self._get_same_hf_param(self._group_param_by_load_spec(LoadEnum.SAME), dtype=save_dtype)
        fused_gen = self._get_fused_hf_param(self._group_param_by_load_spec(LoadEnum.FUSED), dtype=save_dtype)

        # We do not save HF tensor with FSDP sharded tensor since if 1 HF tensor is sharded by 2 FSDP
        # rank, it's hard to determine which FSDP rank should save the HF tensor.
        if self.fsdp_mesh is not None:
            if self.hsdp_mesh:
                # is_fused_save_rank = (dist.get_rank(group=self.hsdp_mesh.get_group(0)) == 0 and dist.get_rank(group=self.hsdp_mesh.get_group(1)) == 0)
                is_fused_save_rank = self.hsdp_mesh.get_coordinate() == [0, 0]
            else:
                is_fused_save_rank = self.fsdp_mesh.get_local_rank() == 0
        else:
            is_fused_save_rank = dist.get_rank() == 0

        is_others_save_rank = not dist.is_initialized() or dist.get_rank() == 0

        if is_fused_save_rank or is_others_save_rank:
            # save_executor = ThreadPoolExecutor(max_workers=16)
            save_executor = ProcessPoolExecutor(max_workers=16)
        else:
            save_executor = None

        if dist.is_initialized():
            if self.fsdp_mesh is not None:
                if self.hsdp_mesh:
                    save_rank = dist.get_rank() % (dist.get_world_size() // self.hsdp_mesh.size())
                else:
                    save_rank = dist.get_rank() % (dist.get_world_size() // self.fsdp_mesh.size())
            else:
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
            safetensor_index += 1
            safetensor_name = f"model-{safetensor_index:04d}-fused-save_rank{save_rank}.safetensors"
            if is_fused_save_rank:
                weight_map.update({name: safetensor_name for name in name_list})
                assert save_executor is not None, "Internal Error, save_executor should not be None"
                future = save_executor.submit(
                    save_file,
                    dict(zip(name_list, hf_tensor_list)),
                    hf_dir / safetensor_name,
                )
                save_futures.append(future)

        safetensor_index = 0
        for name_list, hf_tensor_list in chain(same_gen, shard_gen):
            safetensor_index += 1
            safetensor_name = f"model-{safetensor_index:04d}-others-save_rank{save_rank}.safetensors"
            if is_others_save_rank:
                weight_map.update({name: safetensor_name for name in name_list})
                assert save_executor is not None, "Internal Error, save_executor should not be None"
                future = save_executor.submit(
                    save_file,
                    dict(zip(name_list, hf_tensor_list)),
                    hf_dir / safetensor_name,
                )
                save_futures.append(future)

        if save_executor is not None:
            wait(save_futures)
            save_executor.shutdown()

        weight_map_list: list[dict] | list[None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(weight_map_list, weight_map)
        weight_map_list = cast(list[dict], weight_map_list)
        weight_map = reduce(lambda x, y: x | y, weight_map_list)

        if dist.get_rank() == 0:
            for file in cast(Path, self._hf_path).iterdir():
                if file.suffix != ".safetensors":
                    # Copy the model config and tokenizer files to the save path
                    target_path = hf_dir / file.name
                    if file.is_file():
                        copy(file, target_path)
                    else:
                        copytree(file, target_path)

            with open(hf_dir / "model.safetensors.index.json", "w") as f:
                index = {"weight_map": weight_map}
                json.dump(index, f, indent=2, ensure_ascii=False)

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
            torch.cuda.synchronize()
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

        if self.fsdp_mesh is not None:
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
        if start is not None and end is not None:
            # fp8 pad
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            loaded_tensor_slice = loaded_tensor.index_select(
                dim=self.FSDP_SHARD_DIM, index=torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            )
            non_pad_len = loaded_tensor_slice.shape[self.FSDP_SHARD_DIM]
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)
            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        else:
            local_tensor.copy_(loaded_tensor)
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

        if _loaded_tensor:
            loaded_tensor = torch.cat(tuple(_loaded_tensor), dim=load_spec.dim)
        else:
            if not hf_keys:
                # fp8 pad
                assert self.config.float8_cfg is not None
                assert cast(MoEConfig, self.config).ep_size == 1, "Only support fp8 pad for MoE with ep_size == 1"
                local_tensor.zero_()  # type: ignore  # padded part must be set to 0
            return missing_keys

        if start is not None and end is not None:
            # dist.breakpoint(1)
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            loaded_tensor_slice = loaded_tensor.index_select(
                dim=self.FSDP_SHARD_DIM, index=torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            )
            non_pad_len = loaded_tensor_slice.shape[self.FSDP_SHARD_DIM]
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)
            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                ep_size = cast(MoEConfig, self.config).ep_size
                assert (ep_size is None) or (ep_size == 1), "Only support fp8 pad for MoE with ep_size == 1"
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        else:
            local_tensor.copy_(loaded_tensor)
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

        if self.fsdp_mesh is not None:
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

        assert load_spec.shard_start is not None and load_spec.shard_end is not None, (
            "load_spec.shard_start and load_spec.shard_end should not be None for sharded params"
        )

        # Get the sharded tensor before FSDP
        sharded_loaded_tensor = loaded_tensor.index_select(
            dim=load_spec.dim,
            index=torch.arange(load_spec.shard_start, load_spec.shard_end, dtype=torch.int64),
        )

        if start is not None and end is not None:
            local_tensor.copy_(
                sharded_loaded_tensor.index_select(
                    dim=self.FSDP_SHARD_DIM,
                    index=torch.arange(start, end, dtype=torch.int64, device=sharded_loaded_tensor.device),
                ),
            )
        else:
            local_tensor.copy_(sharded_loaded_tensor)
        return []

    def _has_meta_param(self, module: nn.Module) -> bool:
        """Check if the module has meta parameters."""
        for data in chain(module.parameters(recurse=False), module.buffers(recurse=False)):
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
            cat_tensor = torch.index_select(
                torch.cat(tensors, dim=self.FSDP_SHARD_DIM),
                dim=self.FSDP_SHARD_DIM,
                index=torch.arange(0, size, dtype=torch.int64, device=DEVICE),
            )
            fsdp_unsharded_tensor_list.append(cat_tensor)

        return fsdp_unsharded_tensor_list

    def _maybe_compile_layers(self):
        if self.fsdp_config is not None:
            if self.fsdp_config.torch_compile:
                torch._dynamo.config.cache_size_limit = 128
                if self.fsdp_config.compile_targets is not None:
                    maybe_compile.clear_compile_targets()
                    for target in self.fsdp_config.compile_targets:
                        maybe_compile.set_compile_target(target)
            else:
                maybe_compile.clear_compile_targets()

    def to_device(self, device: torch.device | str):
        if self.fsdp_config is not None and self.fsdp_config.cpu_offload:
            return
        self.to(device, non_blocking=True)
        DEVICE_MODULE.synchronize()
        return
