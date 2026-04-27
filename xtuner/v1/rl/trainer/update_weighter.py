import os
from itertools import chain
from typing import Dict, List, TypeAlias, cast

import requests
import torch
import torch.distributed as dist
import tqdm
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLForConditionalGeneration
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.utils import (
    get_device,
    get_torch_device_module,
    monkey_unpatch_torch_reductions,
    ray_method,
)
from xtuner.v1.utils.load_spec import LoadEnum, LoadSpec


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping service names to their URLs
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class UpdateWeighter:
    def _init_update_weighter(self):
        self.rollout_device_mesh: DeviceMesh | None = None
        self.rollout_url: str | None = None
        self.rollout_cfg_info: dict = dict()
        self.endpoints: dict[str, str] = dict()
        self.endpoints["update_weights"] = "update_weights"
        self._global_hf_keys_mapping_cache: dict[str, list[str]] = dict()
        self._ipc_tensor_bytes: int = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        self._update_params_ipc_tensor = None
        self._update_params_ipc_event = None

    @ray_method
    def update_rollout_info(
        self,
        engine_rank_mesh_array: DeviceMeshRaw,
        server_url_dict: ServiceUrlMap,
        rollout_config: RolloutConfig,
        worker_server_urls_status: Dict[str, bool],
        api_server_url: str | None = None,
    ):
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        if self.rollout_device_mesh is None:
            self.rollout_device_mesh = DeviceMesh(
                "cpu", mesh=engine_rank_mesh_array, mesh_dim_names=("engine_instance", "engine_parallel")
            )
        rollout_server_url = server_url_dict.get(self.rank, "")
        if worker_server_urls_status.get(rollout_server_url, "False") is False:
            self.logger.error(f"Rollout server url {rollout_server_url} is not available.")
            self.rollout_url = None
        else:
            self.rollout_url = rollout_server_url
        self.rollout_cfg_info["tp"] = tp
        self.rollout_cfg_info["ep"] = ep
        self.rollout_cfg_info["api_key"] = rollout_config.api_key
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            self.rollout_cfg_info["backend"] = "sglang"
        else:
            self.rollout_cfg_info["backend"] = (rollout_config.extra_rollout_config or dict()).get(
                "lmdeploy_backend", "pytorch"
            )

    @ray_method
    def update_weights(self):
        DEVICE_MODULE.empty_cache()
        self._update_params_ipc_event = DEVICE_MODULE.Event(interprocess=True)
        if self.rollout_cfg_info.get("backend") == "turbomind":
            self._update_weights_by_layer()
        else:
            if isinstance(self.config.model_cfg, BaseComposeConfig):
                self._update_weights_hf_generator(submodule="language_model", final_update=False)
                self._update_weights_hf_generator(submodule="vision_tower", final_update=False)
                self._update_weights_hf_generator(submodule="multi_modal_projector", final_update=True)
            else:
                self._update_weights_hf_generator(final_update=True)
        self._update_params_ipc_tensor = None
        self._update_params_ipc_event = None
        DEVICE_MODULE.empty_cache()

    def _rl_get_fused_ep_hf_param(self, model: MoE, target_ep_rank: int, target_ep_size: int, bucket_size: int):
        fused_param_groups: list[tuple[torch.Tensor, LoadSpec]] = model._group_param_by_load_spec(LoadEnum.FUSED)
        model_ep_size = 1 if model.fsdp_config is None else model.fsdp_config.ep_size
        if not fused_param_groups:
            return

        def _get_hf_params(
            fsdp_tensor_list: list[tuple[torch.Tensor, LoadSpec]],
        ) -> tuple[list[torch.Tensor], list[str]]:
            hf_keys_list: list[str] = []
            hf_tensor_list: list[torch.Tensor] = []

            for fsdp_tensor, load_spec in fsdp_tensor_list:
                hf_keys = load_spec.hf_keys
                if model_ep_size > 1 and model.ep_mesh is not None:
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
                dim = cast(int, load_spec.dim)
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

        safetensor_size = 0
        dtype = torch.bfloat16
        tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []

        for param, load_spec in fused_param_groups:
            tensor_size = dtype.itemsize * param.numel() // target_ep_size
            if safetensor_size + tensor_size > bucket_size and tensor_list:
                hf_params, name_list = _get_hf_params(tensor_list)
                yield name_list, hf_params
                safetensor_size = tensor_size
                name_list = load_spec.hf_keys.copy()
                tensor_list = [(param, load_spec)]
                continue
            safetensor_size += tensor_size
            tensor_list.append((param, load_spec))

        if tensor_list:
            hf_params, name_list = _get_hf_params(tensor_list)
            yield name_list, hf_params

    @torch.no_grad()
    def _update_weights_hf_generator(self, submodule=None, final_update=False):
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self._engine.model
        if submodule:
            model = getattr(model, submodule)

        dtype = torch.bfloat16
        bucket_size = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        same_gen = model._get_same_hf_param(
            model._group_param_by_load_spec(LoadEnum.SAME), dtype=dtype, device=DEVICE, bucket_size=bucket_size
        )

        train_enable_ep = model.fsdp_config is not None and model.fsdp_config.ep_size > 1
        if train_enable_ep:
            if self.rollout_cfg_info["ep"] > 1:
                fused_gen = self._rl_get_fused_ep_hf_param(
                    model,
                    target_ep_rank=self.rollout_device_mesh["engine_parallel"].get_coordinate()[0],
                    target_ep_size=self.rollout_device_mesh["engine_parallel"].size(),
                    bucket_size=bucket_size,
                )
            else:
                fused_gen = self._rl_get_fused_ep_hf_param(
                    model,
                    target_ep_rank=0,
                    target_ep_size=1,
                    bucket_size=bucket_size,
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
            model._group_param_by_load_spec(LoadEnum.SHARD), dtype=dtype, device=DEVICE, bucket_size=bucket_size
        )

        for name_list, fused_param_list in fused_gen:
            state_dict = {name: param.detach() for name, param in zip(name_list, fused_param_list)}
            self.request_update_params(state_dict, train_enable_ep=train_enable_ep, finished=False)
            del state_dict, name_list, fused_param_list

        for name_list, param_list in chain(same_gen, shard_gen):
            state_dict = {name: param.detach() for name, param in zip(name_list, param_list)}
            self.request_update_params(state_dict, train_enable_ep=train_enable_ep, finished=False)
            del state_dict, name_list, param_list

        if self.rollout_cfg_info["backend"] == "pytorch" and final_update:
            self.request_update_params({}, train_enable_ep=train_enable_ep, finished=True)

        dist.barrier()
        return

    def _update_weights_by_layer(self):
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self._engine.model
        DEVICE_MODULE.empty_cache()

        if isinstance(model.config, BaseComposeConfig):
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
            self.request_update_params(state_dict)

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
            self.request_update_params(state_dict)

        if self.rollout_cfg_info["backend"] == "pytorch":
            self.request_update_params({}, finished=True)

        dist.barrier()
        DEVICE_MODULE.empty_cache()
        return

    @staticmethod
    def _compute_state_dict_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
        total_bytes = 0
        for tensor in state_dict.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

    @staticmethod
    def _create_ipc_tensor(size_in_bytes: int, dtype: torch.dtype):
        return torch.empty(size_in_bytes, dtype=torch.uint8, device=DEVICE).view(dtype)

    @ray_method
    def request_update_params(self, state_dict, train_enable_ep=False, finished=False):
        cpu_mesh = self.rollout_device_mesh["engine_parallel"]
        cpu_group = cpu_mesh.get_group()
        head_rank = cpu_mesh.mesh[0].item()
        if self.rollout_url is None:
            self.logger.error(f"rank {self.rank} url in None, cannot update weights and skip")
            return

        if self.rollout_cfg_info["backend"] == "pytorch":
            from lmdeploy.utils import serialize_state_dict

            try:
                from lmdeploy.utils import FlattenedTensorBucket

                use_flattened_tensor_bucket = True
            except Exception:
                use_flattened_tensor_bucket = False

            if self.rollout_cfg_info["backend"] == "pytorch" and self.rollout_cfg_info["tp"] > 1:
                serialized_data = [None] * self.rollout_cfg_info["tp"]
                if use_flattened_tensor_bucket and state_dict:
                    state_dict_bytes = self._compute_state_dict_bytes(state_dict)
                    send_ipc_tensor = (
                        state_dict_bytes > self._ipc_tensor_bytes or self._update_params_ipc_tensor is None
                    )
                    if send_ipc_tensor:
                        self._ipc_tensor_bytes = max(self._ipc_tensor_bytes, state_dict_bytes)
                        if self._update_params_ipc_tensor is not None:
                            self._update_params_ipc_event.wait()
                            torch.cuda.synchronize()
                        self._update_params_ipc_tensor = self._create_ipc_tensor(
                            self._ipc_tensor_bytes, state_dict[next(iter(state_dict))].dtype
                        )
                    else:
                        self._update_params_ipc_event.wait()

                    flattened_tensor_bucket = FlattenedTensorBucket(
                        named_tensors=list(state_dict.items()),
                        flattened_tensor=self._update_params_ipc_tensor,
                    )
                    metadata = flattened_tensor_bucket.get_metadata()
                    flattened_tensor_data = {
                        "metadata": metadata,
                        "require_clone": False,
                    }
                    self._update_params_ipc_event.record()

                    if send_ipc_tensor:
                        flattened_tensor_data["flattened_tensor"] = flattened_tensor_bucket.get_flattened_tensor()
                        flattened_tensor_data["event_ipc_handle"] = self._update_params_ipc_event.ipc_handle()
                    tp_serialized_data = serialize_state_dict(flattened_tensor_data)
                else:
                    tp_serialized_data = serialize_state_dict(state_dict)
                dist.gather_object(
                    tp_serialized_data,
                    serialized_data if dist.get_rank() == head_rank else None,
                    dst=head_rank,
                    group=cpu_group,
                )
            elif self.rollout_cfg_info["backend"] == "pytorch":
                if use_flattened_tensor_bucket and state_dict:
                    state_dict_bytes = self._compute_state_dict_bytes(state_dict)
                    send_ipc_tensor = (
                        state_dict_bytes > self._ipc_tensor_bytes or self._update_params_ipc_tensor is None
                    )
                    if send_ipc_tensor:
                        self._ipc_tensor_bytes = max(self._ipc_tensor_bytes, state_dict_bytes)
                        if self._update_params_ipc_tensor is not None:
                            self._update_params_ipc_event.wait()
                            torch.cuda.synchronize()
                        self._update_params_ipc_tensor = self._create_ipc_tensor(
                            self._ipc_tensor_bytes, state_dict[next(iter(state_dict))].dtype
                        )
                    else:
                        self._update_params_ipc_event.wait()

                    flattened_tensor_bucket = FlattenedTensorBucket(
                        named_tensors=list(state_dict.items()),
                        flattened_tensor=self._update_params_ipc_tensor,
                    )
                    metadata = flattened_tensor_bucket.get_metadata()
                    flattened_tensor_data = {
                        "metadata": metadata,
                        "require_clone": False,
                    }
                    self._update_params_ipc_event.record()

                    if send_ipc_tensor:
                        flattened_tensor_data["flattened_tensor"] = flattened_tensor_bucket.get_flattened_tensor()
                        flattened_tensor_data["event_ipc_handle"] = self._update_params_ipc_event.ipc_handle()
                    serialized_data = serialize_state_dict(flattened_tensor_data)
                else:
                    serialized_data = serialize_state_dict(state_dict)
            else:
                serialized_data = serialize_state_dict(state_dict) if dist.get_rank() == head_rank else None
        else:
            from sglang.srt.utils import MultiprocessingSerializer
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

            try:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

                use_flattened_tensor_bucket = True
            except Exception:
                use_flattened_tensor_bucket = False

            monkey_patch_torch_reductions()
            state_dict = state_dict.items()
            if self.rollout_cfg_info["tp"] == 1:
                if use_flattened_tensor_bucket:
                    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_dict)
                    metadata = flattened_tensor_bucket.get_metadata()

                    flattened_tensor_data = {
                        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                        "metadata": metadata,
                    }
                    serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                else:
                    serialized_data = MultiprocessingSerializer.serialize(state_dict, output_str=True)

                serialized_data = [serialized_data]
            else:
                serialized_data = [None] * self.rollout_cfg_info["tp"]
                if use_flattened_tensor_bucket:
                    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_dict)
                    metadata = flattened_tensor_bucket.get_metadata()

                    flattened_tensor_data = {
                        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                        "metadata": metadata,
                    }
                    tp_serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                    dist.gather_object(
                        tp_serialized_data,
                        serialized_data if dist.get_rank() == head_rank else None,
                        dst=head_rank,
                        group=cpu_group,
                    )
                else:
                    tp_serialized_data = MultiprocessingSerializer.serialize(state_dict, output_str=True)
                    dist.gather_object(
                        tp_serialized_data,
                        serialized_data if dist.get_rank() == head_rank else None,
                        dst=head_rank,
                        group=cpu_group,
                    )

        if dist.get_rank() == head_rank:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.rollout_cfg_info['api_key']}",
            }
            if self.rollout_cfg_info["backend"] == "sglang":
                payload = {
                    "serialized_named_tensors": serialized_data,
                    "flush_cache": False,
                }
                try:
                    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

                    use_flattened_tensor_bucket = True
                except Exception:
                    use_flattened_tensor_bucket = False
                if use_flattened_tensor_bucket:
                    payload["load_format"] = "flattened_bucket"

                url = f"{self.rollout_url}/update_weights_from_tensor"
                response = requests.post(url, json=payload or {})
                response.raise_for_status()
            else:
                data = dict(serialized_named_tensors=serialized_data, finished=finished)
                try:
                    from lmdeploy.utils import FlattenedTensorBucket

                    use_flattened_tensor_bucket = True
                except Exception:
                    use_flattened_tensor_bucket = False

                if use_flattened_tensor_bucket:
                    data["load_format"] = "flattened_bucket"
                response = requests.post(
                    f"{self.rollout_url}/{self.endpoints['update_weights']}", headers=headers, json=data
                )
            assert response.status_code == 200, f"response.status_code = {response.status_code}"

        if finished or (self.rollout_cfg_info["backend"] == "pytorch" and train_enable_ep and self.rollout_cfg_info["tp"] > 1):
            dist.barrier(group=cpu_group)

        monkey_unpatch_torch_reductions()
        return
