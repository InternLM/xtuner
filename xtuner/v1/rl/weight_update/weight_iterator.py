from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
import tqdm
from torch.distributed.tensor import DTensor

from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLForConditionalGeneration
from xtuner.v1.utils import get_device, get_torch_device_module

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

    @torch.no_grad()
    def iter_hf_batches(self, submodule=None, final_update=False):
        """Update the model weights."""

        model = self._engine.model
        if submodule:
            model = getattr(model, submodule)

        dtype = torch.bfloat16
        bucket_size = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        train_enable_ep = model.fsdp_config is not None and model.fsdp_config.ep_size > 1
        params = model._load_spec_params()
        ep_mesh = getattr(model, "ep_mesh", None)
        ep_group = ep_mesh.get_group() if ep_mesh is not None and ep_mesh.size() > 1 else None

        fused_params = []
        other_params = []
        for param, load_spec in params:
            (fused_params if load_spec.is_fused else other_params).append((param, load_spec))

        preserved_fused_shard_group = None
        target_fused_key_partition = None
        if fused_params and self.rollout_info.transport_type == "ipc" and self.rollout_info.ep > 1:
            target_rank = self.rollout_info.ipc_engine_parallel_rank
            target_size = self.rollout_info.ipc_engine_parallel_size
            assert target_rank is not None, "IPC rollout target for current train rank is not resolved."
            assert target_size is not None, "IPC rollout target size for current train rank is not resolved."

            # Preserve the train EP shard only when it is exactly the rollout
            # target shard. Other topologies reconstruct globally, then select
            # the rollout-owned expert key range in the HF save plan.
            if ep_group is not None and target_size == ep_mesh.size() and target_rank == dist.get_rank(ep_group):
                preserved_fused_shard_group = ep_group
            else:
                target_fused_key_partition = (target_rank, target_size)

        fused_gen = model._get_hf_param(
            fused_params,
            dtype=dtype,
            device=DEVICE,
            bucket_size=bucket_size,
            preserved_fused_shard_group=preserved_fused_shard_group,
            target_fused_key_partition=target_fused_key_partition,
        )
        other_gen = model._get_hf_param(
            other_params,
            dtype=dtype,
            device=DEVICE,
            bucket_size=bucket_size,
        )
        for name_list, param_list in chain(fused_gen, other_gen):
            # FlattenedTensorBucket stores one dtype per payload. Qwen3.5 keeps
            # selected norm and A_log weights in fp32, so split those from the
            # otherwise bf16 HF-save bucket before handing it to the transport.
            state_dicts: dict[torch.dtype, dict[str, torch.Tensor]] = {}
            for name, param in zip(name_list, param_list, strict=True):
                state_dicts.setdefault(param.dtype, {})[name] = param.detach()
            for state_dict in state_dicts.values():
                yield WeightUpdateBatch(state_dict, train_enable_ep=train_enable_ep, finished=False)

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
