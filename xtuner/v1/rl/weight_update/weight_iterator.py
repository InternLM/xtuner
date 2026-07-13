from __future__ import annotations

from itertools import chain
from typing import Any, cast

import torch
import torch.distributed as dist
import tqdm
from torch.distributed.tensor import DTensor

from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLForConditionalGeneration
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.utils import get_device, get_torch_device_module
from xtuner.v1.utils.load_spec import LoadEnum, LoadSpec

from .data import RolloutWeightUpdateInfo, WeightUpdateBatch


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class WeightIterator:
    def __init__(
        self,
        *,
        config: Any,
        engine: Any,
        rollout_info: RolloutWeightUpdateInfo,
        global_hf_keys_mapping_cache: dict[str, list[str]],
    ):
        self.config = config
        self._engine = engine
        self.rollout_info = rollout_info
        self._global_hf_keys_mapping_cache = global_hf_keys_mapping_cache

    def iter_batch_groups(self):
        # Export path depends on rollout protocol: turbomind consumes layer-wise batches,
        # compose models update submodules in order, and plain models use HF-style batches.
        if self.rollout_info.transport_type == "ipc" and self.rollout_info.backend == "turbomind":
            yield self.iter_layer_batches()
            return

        if isinstance(self.config.model_cfg, BaseComposeConfig):
            # Only the last compose submodule sends the final update marker.
            submodules = (
                ("language_model", False),
                ("vision_tower", False),
                ("multi_modal_projector", True),
            )
            for submodule, final_update in submodules:
                yield self.iter_hf_batches(submodule=submodule, final_update=final_update)
            return

        yield self.iter_hf_batches(final_update=True)

    def _get_hf_params(
        self,
        model,
        model_ep_size: int,
        target_ep_size: int,
        target_ep_rank: int,
        fsdp_tensor_list: list[tuple[torch.Tensor, LoadSpec]],
        should_gather_train_ep_shards: bool,
    ) -> tuple[list[torch.Tensor], list[str]]:
        hf_keys_list: list[str] = []
        hf_tensor_list: list[torch.Tensor] = []

        for fsdp_tensor, load_spec in fsdp_tensor_list:
            hf_keys = load_spec.hf_keys
            if model_ep_size > 1 and model.ep_mesh is not None:
                # Each train EP rank owns only part of the HF key list; gather the global
                # mapping once so rollout EP ranks can receive the right slice.
                if load_spec.name not in self._global_hf_keys_mapping_cache:
                    global_hf_keys: list[list[str] | None] = [None] * model_ep_size
                    dist.all_gather_object(global_hf_keys, hf_keys, group=model.ep_mesh.get_group())
                    global_hf_keys_gathered = cast(list[list[str]], global_hf_keys)
                    self._global_hf_keys_mapping_cache[load_spec.name] = list(
                        chain.from_iterable(global_hf_keys_gathered)
                    )
                hf_keys = self._global_hf_keys_mapping_cache[load_spec.name]

            fused_full_tensor = fsdp_tensor.bfloat16()
            if isinstance(fused_full_tensor, DTensor):
                fused_full_tensor = fused_full_tensor.full_tensor()
            # FUSED load specs pack multiple HF tensors along load_spec.dim; split them
            # back into HF tensors before selecting the target rollout EP shard.
            dim = cast(int, load_spec.dim)

            if should_gather_train_ep_shards and model_ep_size > 1:
                assert model.ep_mesh is not None
                ep_group = model.ep_mesh.get_group()

                output = torch.empty(
                    *fused_full_tensor.shape[:dim],
                    fused_full_tensor.shape[dim] * model_ep_size,
                    *fused_full_tensor.shape[dim + 1 :],
                    dtype=fused_full_tensor.dtype,
                    device=fused_full_tensor.device,
                )
                dist.all_gather_into_tensor(output, fused_full_tensor.contiguous(), group=ep_group)
                fused_full_tensor = output

            num_split = len(hf_keys)
            hf_tensor_size = fused_full_tensor.shape[dim] / num_split
            assert hf_tensor_size.is_integer(), "Internal Error, hf_tensor_size is not integer"
            hf_tensor_size = int(hf_tensor_size)

            hf_tensor = fused_full_tensor.split([hf_tensor_size] * num_split, dim=dim)
            assert num_split % target_ep_size == 0, (
                f"len(hf_keys) of '{hf_keys}' is {num_split}, it must be divisible by target_ep_size {target_ep_size}"
            )
            start_idx = (num_split // target_ep_size) * target_ep_rank
            end_idx = (num_split // target_ep_size) * (target_ep_rank + 1)

            hf_keys_list.extend(hf_keys[start_idx:end_idx])
            hf_tensor_list.extend(hf_tensor[start_idx:end_idx])

        hf_tensor_list = [
            model.param_to_safetensor(safetensor, name) for safetensor, name in zip(hf_tensor_list, hf_keys_list)
        ]

        return hf_tensor_list, hf_keys_list

    def _rl_get_fused_ep_hf_param(
        self,
        model: MoE,
        target_ep_rank: int,
        target_ep_size: int,
        bucket_size: int,
        should_gather_train_ep_shards: bool,
    ):
        fused_param_groups: list[tuple[torch.Tensor, LoadSpec]] = model._group_param_by_load_spec(LoadEnum.FUSED)
        model_ep_size = 1 if model.fsdp_config is None else model.fsdp_config.ep_size
        if not fused_param_groups:
            return

        safetensor_size = 0
        dtype = torch.bfloat16
        tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []

        for param, load_spec in fused_param_groups:
            tensor_size = dtype.itemsize * param.numel() // target_ep_size
            if safetensor_size + tensor_size > bucket_size and tensor_list:
                hf_params, name_list = self._get_hf_params(
                    model,
                    model_ep_size=model_ep_size,
                    target_ep_size=target_ep_size,
                    target_ep_rank=target_ep_rank,
                    fsdp_tensor_list=tensor_list,
                    should_gather_train_ep_shards=should_gather_train_ep_shards,
                )
                yield name_list, hf_params
                safetensor_size = tensor_size
                # Kept to mirror the legacy generator layout; the next iteration rebuilds
                # name_list from tensor_list before yielding.
                name_list = load_spec.hf_keys.copy()
                tensor_list = [(param, load_spec)]
                continue
            safetensor_size += tensor_size
            tensor_list.append((param, load_spec))

        if tensor_list:
            hf_params, name_list = self._get_hf_params(
                model=model,
                model_ep_size=model_ep_size,
                target_ep_size=target_ep_size,
                target_ep_rank=target_ep_rank,
                fsdp_tensor_list=tensor_list,
                should_gather_train_ep_shards=should_gather_train_ep_shards,
            )
            yield name_list, hf_params

    @torch.no_grad()
    def iter_hf_batches(self, submodule=None, final_update=False):
        """Update the model weights."""

        model = self._engine.model
        if submodule:
            model = getattr(model, submodule)

        dtype = torch.bfloat16
        bucket_size = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        same_gen = model._get_same_hf_param(
            model._group_param_by_load_spec(LoadEnum.SAME),
            dtype=dtype,
            device=DEVICE,
            bucket_size=bucket_size,
        )

        train_enable_ep = model.fsdp_config is not None and model.fsdp_config.ep_size > 1
        should_gather_train_ep_shards = self.rollout_info.transport_type == "nccl" and train_enable_ep

        if train_enable_ep:
            if self.rollout_info.transport_type == "ipc" and self.rollout_info.ep > 1:
                target_ep_rank = self.rollout_info.ipc_engine_parallel_rank
                target_ep_size = self.rollout_info.ipc_engine_parallel_size
                assert target_ep_rank is not None, "IPC rollout target for current train rank is not resolved."
                assert target_ep_size is not None, "IPC rollout target size for current train rank is not resolved."
                # Colocated IPC can send only the expert slice needed by the local rollout
                # EP rank
                fused_gen = self._rl_get_fused_ep_hf_param(
                    model,
                    target_ep_rank=target_ep_rank,
                    target_ep_size=target_ep_size,
                    bucket_size=bucket_size,
                    should_gather_train_ep_shards=should_gather_train_ep_shards,
                )
            else:
                # Disaggregated NCCL uses one trainer-side broadcast for all rollout ranks.
                # Gather train EP shards first, then send the full expert tensor instead of
                # slicing by rollout EP rank.
                fused_gen = self._rl_get_fused_ep_hf_param(
                    model,
                    target_ep_rank=0,
                    target_ep_size=1,
                    bucket_size=bucket_size,
                    should_gather_train_ep_shards=should_gather_train_ep_shards,
                )
        else:
            fused_gen = model._get_fused_hf_param(
                model._group_param_by_load_spec(LoadEnum.FUSED),
                dtype=dtype,
                device=DEVICE,
                bucket_size=bucket_size,
                update_weights_for_rl=True,
            )
        shard_gen = model._get_shard_hf_param(
            model._group_param_by_load_spec(LoadEnum.SHARD),
            dtype=dtype,
            device=DEVICE,
            bucket_size=bucket_size,
        )

        for name_list, fused_param_list in fused_gen:
            state_dict = {name: param.detach() for name, param in zip(name_list, fused_param_list)}
            yield WeightUpdateBatch(state_dict, train_enable_ep=train_enable_ep, finished=False)
            del state_dict, name_list, fused_param_list

        for name_list, param_list in chain(same_gen, shard_gen):
            state_dict = {name: param.detach() for name, param in zip(name_list, param_list)}
            yield WeightUpdateBatch(state_dict, train_enable_ep=train_enable_ep, finished=False)
            del state_dict, name_list, param_list

        # pytorch and vLLM use an empty final update as an end marker; SGLang and
        # turbomind do not consume this marker.
        if self.rollout_info.backend in ("pytorch", "vllm") and final_update:
            yield WeightUpdateBatch({}, train_enable_ep=train_enable_ep, finished=True)

        DEVICE_MODULE.empty_cache()

    @torch.no_grad()
    def iter_layer_batches(self):
        """Update the model weights."""

        model = self._engine.model
        DEVICE_MODULE.empty_cache()

        if isinstance(model.config, BaseComposeConfig):
            # TODO: support float8 for vision compose model.
            dtype = torch.bfloat16
        else:
            if (model.config.float8_cfg is not None) and (model.config.float8_cfg.enable_float8):
                dtype = torch.float8_e4m3fn
            else:
                dtype = torch.bfloat16

        def get_params(tensor_list, name_list, save_dtype):
            _tensor_list, _spec_list = list(zip(*tensor_list))
            fsdp_unshard_tensor_list = model._fsdp_foreach_allgather(_tensor_list, _spec_list)
            if save_dtype == torch.float8_e4m3fn:
                fsdp_unshard_tensor_list, name_list = model._to_float8(
                    fsdp_unshard_tensor_list, name_list, _tensor_list, save_dtype
                )
            return fsdp_unshard_tensor_list, name_list

        saved_list = []
        is_qwen3vl = False
        if isinstance(model.config, BaseComposeConfig):
            language_model = model.language_model
            if isinstance(model, Qwen3VLForConditionalGeneration):
                is_qwen3vl = True
        else:
            language_model = model

        if is_qwen3vl:
            vision_hf_prefix = "model.visual."
            projector_hf_prefix = "model.visual."
        else:
            vision_hf_prefix = "model.vision_tower."
            projector_hf_prefix = "model.multi_modal_projector."

        for i, layer in tqdm.tqdm(language_model.layers.items(), desc="[gather weight]"):
            tensor_list = []
            name_list = []
            for sub_name, param in layer.state_dict().items():
                if isinstance(model.config, BaseComposeConfig):
                    saved_list.append(f"language_model.layers.{i}.{sub_name}")
                else:
                    saved_list.append(f"layers.{i}.{sub_name}")
                local_tensor = param._local_tensor if isinstance(param, DTensor) else param
                local_tensor = local_tensor.bfloat16()
                load_spec = language_model.load_spec_mapping.get(f"layers.{i}.{sub_name}")

                if isinstance(model.config, BaseComposeConfig):
                    name = f"model.language_model.layers.{i}.{sub_name}"
                else:
                    name = f"model.layers.{i}.{sub_name}"

                if ".experts." in name and ".mlp.experts." not in name:
                    name = name.replace(".experts.", ".mlp.experts.")
                if ".gate." in name and ".mlp.gate." not in name:
                    name = name.replace(".gate.", ".mlp.gate.")
                name_list.append(name)
                tensor_list.append((local_tensor, load_spec))
            fsdp_unshard_tensor_list, name_list = get_params(tensor_list, name_list, dtype)
            state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
            yield WeightUpdateBatch(state_dict)

        for name, param in model.state_dict().items():
            if name in saved_list:
                continue
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            load_spec = model.load_spec_mapping.get(name)

            if isinstance(model.config, BaseComposeConfig):
                if "vision_tower." in name:
                    name = name.replace("vision_tower.", vision_hf_prefix)
                elif "multi_modal_projector." in name:
                    name = name.replace("multi_modal_projector.", projector_hf_prefix)
                elif name == "language_model.norm.weight":
                    name = "model.language_model.norm.weight"
                elif name == "language_model.embed_tokens.weight":
                    name = "model.language_model.embed_tokens.weight"
                elif name == "language_model.lm_head.weight":
                    name = "lm_head.weight"
            else:
                if name == "norm.weight":
                    name = "model.norm.weight"
                elif name == "embed_tokens.weight":
                    name = "model.embed_tokens.weight"
            tensor_list = [(local_tensor, load_spec)]
            name_list = [name]
            fsdp_unshard_tensor_list, name_list = get_params(tensor_list, name_list, dtype)
            state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
            yield WeightUpdateBatch(state_dict)

        if self.rollout_info.backend in ("pytorch", "vllm"):
            yield WeightUpdateBatch({}, finished=True)

        DEVICE_MODULE.empty_cache()
