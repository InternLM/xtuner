import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from itertools import chain
from threading import Lock
from typing import Any, Dict, List, TypeAlias, cast

import requests
import torch
import torch.distributed as dist
import tqdm
from packaging.version import parse as parse_version
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
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
RolloutEngineInfo: TypeAlias = list[tuple[int, str, int]]  # (rollout rank, server url, engine gpu count)
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class UpdateWeighter:
    rank: int
    logger: Any
    config: Any

    def _init_update_weighter(self):
        # Used to update weight to rollout engine
        self.rollout_device_mesh: DeviceMesh | None = None
        self.rollout_url: str | None = None
        self.rollout_cfg_info: dict = dict()
        self.endpoints: dict[str, str] = dict()
        self.endpoints["update_weights"] = "update_weights"

        self.rollout_engine_rank_mesh_array: DeviceMeshRaw = []
        self.rollout_server_url_dict: ServiceUrlMap = {}
        self.worker_server_urls_status: dict[str, bool] = {}

        self._global_hf_keys_mapping_cache: dict[str, list[str]] = dict()
        self._default_ipc_tensor_bytes: int = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        self._ipc_tensor_bytes_dict_by_dtype: dict[torch.dtype, int] = {}
        self._update_params_ipc_tensor_dict_by_dtype: dict[torch.dtype, torch.Tensor] = {}
        self._last_update_params_ipc_tensor_dtype: torch.dtype | None = None
        self._update_params_ipc_event = None
        self._sglang_disagg_group: dist.ProcessGroup | None = None
        self._sglang_disagg_group_name: str | None = None
        self._sglang_disagg_engine_urls: list[str] = []
        self._sglang_disagg_executor: ThreadPoolExecutor | None = None
        self._train_update_sync_group: dist.ProcessGroup | None = None
        self._sglang_disagg_update_lock = Lock()
        self.use_fake_weight_update = (
            False  # 仅在 lmdeploy 后端的 disaggregated 模式下使用，表示是否使用 fake 接口进行权重更新
        )

    def _hook_compare_test_sent_and_received_weight_hash(
        self,
        result: dict[str, Any],
        *,
        bucket_idx: int | None = None,
        names: list[str] | None = None,
    ) -> None:
        """Test hook for comparing sent and received weight hashes.

        This hook is intentionally a no-op in production code and is expected to be overridden in unit tests that need
        to compare training-side sent hashes with rollout-side received hashes returned by SGLang.
        """
        return

    @ray_method
    def update_rollout_info(
        self,
        engine_rank_mesh_array: DeviceMeshRaw,
        server_url_dict: ServiceUrlMap,
        rollout_config: RolloutConfig,
        worker_server_urls_status: Dict[str, bool],
        api_server_url: str | None = None,
        worker_session_url_dict: ServiceUrlMap | None = None,
        worker_session_urls_status: Dict[str, bool] | None = None,
    ):
        """Update the rollout information for the training worker."""
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        if self.rollout_device_mesh is None:
            self.rollout_device_mesh = DeviceMesh(
                "cpu",
                mesh=engine_rank_mesh_array,
                mesh_dim_names=("engine_instance", "engine_parallel"),
            )
        rollout_server_url = server_url_dict.get(self.rank, "")
        if worker_server_urls_status.get(rollout_server_url, "False") is False:
            self.logger.error(f"Rollout server url {rollout_server_url} is not available.")
            self.rollout_url = None
        else:
            self.rollout_url = rollout_server_url

        self.rollout_engine_rank_mesh_array = [[int(rank) for rank in ranks] for ranks in engine_rank_mesh_array]
        self.rollout_server_url_dict = {int(rank): url for rank, url in server_url_dict.items()}
        self.worker_server_urls_status = worker_server_urls_status

        self.rollout_cfg_info["tp"] = tp
        self.rollout_cfg_info["ep"] = ep
        self.rollout_cfg_info["api_key"] = rollout_config.api_key
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            self.rollout_cfg_info["backend"] = "sglang"
        elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
            self.rollout_cfg_info["backend"] = "vllm"
        else:
            self.rollout_cfg_info["backend"] = (rollout_config.extra_rollout_config or dict()).get(
                "lmdeploy_backend", "pytorch"
            )

    @ray_method
    def set_train_rollout_mode(self, train_rollout_mode: str):
        mode = train_rollout_mode.lower()
        if mode == "colocate":
            self.is_train_rollout_colocated = True
        elif mode == "disaggregated":
            self.is_train_rollout_colocated = False

            backend = self.rollout_cfg_info.get("backend", "").lower()
            if backend == "vllm":
                raise NotImplementedError("Disaggregated train-rollout mode is not supported for vLLM backend.")

            elif backend == "pytorch" or backend == "turbomind":
                self.logger.warning(
                    "Disaggregated train-rollout mode for lmdeploy backend is not fully supported yet. "
                    "A fake no-op interface will be used temporarily.",
                )
                self.use_fake_weight_update = True  # 后续 fake 接口可根据这个标志跳过实际同步

            elif backend == "sglang":
                self.use_fake_weight_update = False
            else:
                raise ValueError(
                    f"Unsupported rollout backend for disaggregated mode: {backend!r}. "
                    "Expected 'vllm', 'pytorch', 'turbomind' or 'sglang'."
                )

        else:
            raise ValueError(
                f"Unsupported train_rollout_mode: {train_rollout_mode!r}. Expected 'colocate' or 'disaggregated'."
            )

        if self.is_train_rollout_colocated:
            self._reset_sglang_disagg_group()

    def _reset_sglang_disagg_group(self):
        if self._sglang_disagg_executor is not None:
            self._sglang_disagg_executor.shutdown(wait=False, cancel_futures=True)
        try:
            if self._sglang_disagg_group is not None:
                dist.destroy_process_group(self._sglang_disagg_group)
        except Exception:
            pass
        self._sglang_disagg_group = None
        self._sglang_disagg_group_name = None
        self._sglang_disagg_engine_urls = []
        self._sglang_disagg_executor = None

    def _get_train_update_sync_group(self) -> dist.ProcessGroup:
        if self._train_update_sync_group is None:
            ranks = list(range(dist.get_world_size()))
            self._train_update_sync_group = dist.new_group(ranks=ranks, backend="gloo")
        return self._train_update_sync_group

    @ray_method
    def update_weights(self):
        """Update the model weights."""
        if not hasattr(self, "is_train_rollout_colocated"):
            raise RuntimeError(
                "train/rollout mode is not set. Please call set_train_rollout_mode() before update_weights()."
            )

        if self.is_train_rollout_colocated:
            self._update_weights_colocated()
        else:
            self._update_weights_disaggregated()

    def _update_weights_colocated(self):
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
        self._update_params_ipc_tensor_dict_by_dtype = {}
        self._last_update_params_ipc_tensor_dtype = None
        self._update_params_ipc_event = None

        DEVICE_MODULE.empty_cache()

    def _update_weights_disaggregated(self):
        if self.use_fake_weight_update:
            self.logger.warning(
                "Using fake weight update interface, no actual weight synchronization will happen. This is only for testing purposes and should not be used in production."
            )
            return

        DEVICE_MODULE.empty_cache()
        try:
            if isinstance(self.config.model_cfg, BaseComposeConfig):
                self._update_weights_hf_generator(submodule="language_model", final_update=False)
                self._update_weights_hf_generator(submodule="vision_tower", final_update=False)
                self._update_weights_hf_generator(submodule="multi_modal_projector", final_update=True)
            else:
                self._update_weights_hf_generator(final_update=True)
        finally:
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
        """Update the model weights."""
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

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
            model._group_param_by_load_spec(LoadEnum.SHARD),
            dtype=dtype,
            device=DEVICE,
            bucket_size=bucket_size,
        )

        for name_list, fused_param_list in fused_gen:
            state_dict = {name: param.detach() for name, param in zip(name_list, fused_param_list)}
            self.request_update_params(state_dict, train_enable_ep=train_enable_ep, finished=False)
            del state_dict, name_list, fused_param_list

        for name_list, param_list in chain(same_gen, shard_gen):
            state_dict = {name: param.detach() for name, param in zip(name_list, param_list)}
            self.request_update_params(state_dict, train_enable_ep=train_enable_ep, finished=False)
            del state_dict, name_list, param_list

        if self.rollout_cfg_info["backend"] in ("pytorch", "vllm") and final_update:
            self.request_update_params({}, train_enable_ep=train_enable_ep, finished=True)

        if self.is_train_rollout_colocated:
            dist.barrier()
        else:
            dist.barrier(group=self._get_train_update_sync_group())
        DEVICE_MODULE.empty_cache()
        return

    def _update_weights_by_layer(self):
        """Update the model weights."""
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self._engine.model
        DEVICE_MODULE.empty_cache()

        if isinstance(model.config, BaseComposeConfig):
            # TODO: support float8 for vision compose model
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

        if self.rollout_cfg_info["backend"] in ("pytorch", "vllm"):
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
    def _init_external_process_group(
        backend: str | Backend | None = None,
        init_method: str | None = None,
        timeout: timedelta | None = None,
        world_size: int = -1,
        rank: int = -1,
        store: Store | None = None,
        group_name: str | None = None,
        pg_options: Any | None = None,
    ) -> dist.ProcessGroup:
        assert (store is None) or (init_method is None), "Cannot specify both store and init_method."
        if store is not None:
            assert world_size > 0, "world_size must be positive if using store"
            assert rank >= 0, "rank must be non-negative if using store"
        elif init_method is None:
            init_method = "env://"

        backend = Backend(backend) if backend else Backend("undefined")
        if timeout is None:
            timeout = default_pg_timeout

        if store is None:
            assert init_method is not None
            rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timeout)
            if group_name is not None:
                store = PrefixStore(group_name, store)

        pg_options_param_name = (
            "backend_options" if parse_version(torch.__version__) >= parse_version("2.6") else "pg_options"
        )
        pg, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name=group_name,
            **{pg_options_param_name: pg_options},
            timeout=timeout,
        )
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        return pg

    @staticmethod
    def _create_ipc_tensor(size_in_bytes: int, dtype: torch.dtype):
        return torch.empty(size_in_bytes, dtype=torch.uint8, device=DEVICE).view(dtype)

    def _build_lmdeploy_flattened_tensor_data(self, state_dict: dict, flattened_tensor_bucket_cls) -> dict:
        # LMDeploy flattened buckets require all tensors in one bucket to share a dtype.
        state_dict_dtype = state_dict[next(iter(state_dict))].dtype
        update_params_ipc_tensor = self._update_params_ipc_tensor_dict_by_dtype.get(state_dict_dtype, None)
        state_dict_bytes = self._compute_state_dict_bytes(state_dict)
        ipc_tensor_bytes = self._ipc_tensor_bytes_dict_by_dtype.get(
            state_dict_dtype,
            self._default_ipc_tensor_bytes,
        )
        dtype_changed = (
            self._last_update_params_ipc_tensor_dtype is not None
            and state_dict_dtype != self._last_update_params_ipc_tensor_dtype
        )
        need_resize = state_dict_bytes > ipc_tensor_bytes
        send_ipc_tensor = dtype_changed or need_resize or update_params_ipc_tensor is None

        if update_params_ipc_tensor is not None:
            self._update_params_ipc_event.wait()
            if need_resize:
                torch.cuda.synchronize()

        if update_params_ipc_tensor is None or need_resize:
            ipc_tensor_bytes = max(ipc_tensor_bytes, state_dict_bytes)
            self._ipc_tensor_bytes_dict_by_dtype[state_dict_dtype] = ipc_tensor_bytes
            update_params_ipc_tensor = self._create_ipc_tensor(
                ipc_tensor_bytes,
                state_dict_dtype,
            )
            self._update_params_ipc_tensor_dict_by_dtype[state_dict_dtype] = update_params_ipc_tensor

        flattened_tensor_bucket = flattened_tensor_bucket_cls(
            named_tensors=list(state_dict.items()),
            flattened_tensor=update_params_ipc_tensor,
        )
        flattened_tensor_data = {
            "metadata": flattened_tensor_bucket.get_metadata(),
            "require_clone": False,
        }
        self._update_params_ipc_event.record()
        self._last_update_params_ipc_tensor_dtype = state_dict_dtype

        if send_ipc_tensor:
            flattened_tensor_data["flattened_tensor"] = flattened_tensor_bucket.get_flattened_tensor()
            flattened_tensor_data["event_ipc_handle"] = self._update_params_ipc_event.ipc_handle()
        return flattened_tensor_data

    def _get_sglang_disagg_engine_info(self) -> RolloutEngineInfo:
        engine_info: RolloutEngineInfo = []
        seen_urls: set[str] = set()
        rank_to_engine_size: dict[int, int] = {}
        for engine_ranks in self.rollout_engine_rank_mesh_array:
            engine_size = len(engine_ranks)
            for rank in engine_ranks:
                rank_to_engine_size[int(rank)] = engine_size

        for rank, url in sorted(self.rollout_server_url_dict.items(), key=lambda item: int(item[0])):
            rank = int(rank)
            if not url or url in seen_urls:
                continue
            if self.worker_server_urls_status.get(url, False) is False:
                continue
            seen_urls.add(url)
            engine_info.append(
                (
                    rank,
                    url,
                    rank_to_engine_size.get(
                        rank,
                        max(self.rollout_cfg_info["tp"], self.rollout_cfg_info["ep"]),
                    ),
                )
            )
        return engine_info

    def _ensure_sglang_disagg_group(self):
        if self._sglang_disagg_group is not None:
            return
        engine_info = self._get_sglang_disagg_engine_info()
        if not engine_info:
            self.logger.error("No active rollout engine url, cannot init sglang weight update group")
            return

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        backend = "nccl"

        master_address = None
        master_port = None
        # get address and port for weight-update
        try:
            import ray

            master_address = ray.util.get_node_ip_address()
        except Exception:
            master_address = socket.gethostbyname(socket.gethostname())

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            master_port = int(sock.getsockname()[1])

        group_name = f"xtuner_sglang_weight_update_{self.rank}"
        world_size = sum(engine_size for _, _, engine_size in engine_info) + 1

        self._sglang_disagg_executor = ThreadPoolExecutor(max_workers=max(1, len(engine_info)))
        init_futures = []
        rank_offset = 1
        for _, url, engine_size in engine_info:
            payload = {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            }
            init_futures.append(
                self._sglang_disagg_executor.submit(
                    requests.post,
                    f"{url}/init_weights_update_group",
                    json=payload,
                )
            )
            rank_offset += engine_size

        self._sglang_disagg_group = self._init_external_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )

        for init_future in init_futures:
            response = init_future.result()
            response.raise_for_status()
            result = response.json()
            assert result.get("success", True), (
                f"SGLang init_weights_update_group failed: {result.get('message', result)}"
            )

        self._sglang_disagg_group_name = group_name
        self._sglang_disagg_engine_urls = [url for _, url, _ in engine_info]

    def _request_update_params_sglang_disaggregated(self, state_dict):
        if not state_dict:
            return

        train_sync_group = self._get_train_update_sync_group()
        head_rank = 0
        if dist.get_rank() != head_rank:
            dist.barrier(group=train_sync_group)
            return

        self._ensure_sglang_disagg_group()
        if self._sglang_disagg_group is None:
            dist.barrier(group=train_sync_group)
            return

        assert self._sglang_disagg_executor is not None
        assert self._sglang_disagg_group_name is not None
        with self._sglang_disagg_update_lock:
            try:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
            except Exception as e:
                raise RuntimeError(
                    "Disaggregated update_weights currently only supports sglang builds "
                    "that provide `sglang.srt.model_executor.model_runner.FlattenedTensorBucket`."
                ) from e

            names = list(state_dict.keys())
            tensors = [
                tensor.detach().to(device=DEVICE, non_blocking=True).contiguous() for tensor in state_dict.values()
            ]
            payload = {
                "names": names,
                "dtypes": [str(tensor.dtype).replace("torch.", "") for tensor in tensors],
                "shapes": [list(tensor.shape) for tensor in tensors],
                "group_name": self._sglang_disagg_group_name,
                "load_format": "flattened_bucket",
            }
            update_futures = [
                self._sglang_disagg_executor.submit(
                    requests.post,
                    f"{url}/update_weights_from_distributed",
                    json=payload,
                )
                for url in self._sglang_disagg_engine_urls
            ]
            assert self._sglang_disagg_group is not None
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(zip(names, tensors)))
            flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

            dist.broadcast(flattened_tensor, src=0, group=self._sglang_disagg_group)
            DEVICE_MODULE.synchronize()
            for update_future in update_futures:
                response = update_future.result()
                response.raise_for_status()
                result = response.json()
                self._hook_compare_test_sent_and_received_weight_hash(
                    result,
                    names=names,
                )
                assert result.get("success", True), (
                    f"SGLang update_weights_from_distributed failed: {result.get('message', result)}"
                )
        dist.barrier(group=train_sync_group)

    @ray_method
    def request_update_params(self, state_dict, train_enable_ep=False, finished=False):
        """Send a request to update the parameters on the rollout workers.

        This method serializes the state dictionary and sends it to the
        appropriate rollout worker via an HTTP request.

        Args:
            state_dict (dict | list): The state dictionary containing the model
                parameters to update.
            train_enable_ep (bool): Whether the training engine enables expert parallelism.
                Defaults to False.
            finished (bool): A flag indicating whether this is the final
                batch of updates. Defaults to False.
        """

        if self.rollout_cfg_info["backend"] == "sglang" and not self.is_train_rollout_colocated:
            self._request_update_params_sglang_disaggregated(state_dict)
            return

        cpu_mesh = self.rollout_device_mesh["engine_parallel"]
        cpu_group = cpu_mesh.get_group()
        head_rank = cpu_mesh.mesh[0].item()
        if self.rollout_url is None:
            self.logger.error(f"rank {self.rank} url in None, cannot update weights and skip")
            return

        if self.rollout_cfg_info["backend"] == "vllm":

            def serialize_state_dict(state_dict: dict) -> str:
                import base64
                from io import BytesIO
                from multiprocessing.reduction import ForkingPickler

                from torch.multiprocessing.reductions import reduce_tensor

                data = [(k, reduce_tensor(v)) for k, v in state_dict.items()]
                buf = BytesIO()
                ForkingPickler(buf).dump(data)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode("utf-8")

            serialized_data = [None] * self.rollout_cfg_info["tp"]
            dist.gather_object(
                serialize_state_dict(state_dict),
                serialized_data if dist.get_rank() == head_rank else None,
                dst=head_rank,
                group=cpu_group,
            )
            if dist.get_rank() == head_rank:
                headers = {
                    "Content-Type": "application/json",
                }
                data_ = json.dumps(dict(serialized_named_tensors=serialized_data, finished=finished))
                data = dict(method="update_weight_npu_ipc", args=[data_])
                response = requests.post(f"{self.rollout_url}/collective_rpc", headers=headers, json=data)
                assert response.status_code == 200, f"response.status_code = {response.status_code}"

            if finished:
                dist.barrier(group=cpu_group)
            return

        if self.rollout_cfg_info["backend"] == "pytorch":
            # TODO(chenchiyu): remove lmdeploy related code
            from lmdeploy.utils import serialize_state_dict

            try:
                from lmdeploy.utils import FlattenedTensorBucket

                use_flattened_tensor_bucket = True
            except Exception:
                use_flattened_tensor_bucket = False

            if self.rollout_cfg_info["backend"] == "pytorch" and self.rollout_cfg_info["tp"] > 1:
                serialized_data = [None] * self.rollout_cfg_info["tp"]
                if use_flattened_tensor_bucket and state_dict:
                    flattened_tensor_data = self._build_lmdeploy_flattened_tensor_data(
                        state_dict,
                        FlattenedTensorBucket,
                    )
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
                    flattened_tensor_data = self._build_lmdeploy_flattened_tensor_data(
                        state_dict,
                        FlattenedTensorBucket,
                    )
                    serialized_data = serialize_state_dict(flattened_tensor_data)
                else:
                    serialized_data = serialize_state_dict(state_dict)
            else:
                # for turbomind backend, only head_rank should serialize data
                serialized_data = serialize_state_dict(state_dict) if dist.get_rank() == head_rank else None
        else:
            # sglang
            from sglang.srt.utils import MultiprocessingSerializer
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

            try:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

                use_flattened_tensor_bucket = True
            except Exception:
                use_flattened_tensor_bucket = False

            # NOTE: xtuner目前去掉sglang的patch也不会出问题，但为了保险起见，还是保留patch逻辑，并且在update_weights结束后unpatch
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

                if use_flattened_tensor_bucket and state_dict:
                    data["load_format"] = "flattened_bucket"
                response = requests.post(
                    f"{self.rollout_url}/{self.endpoints['update_weights']}", headers=headers, json=data
                )
            assert response.status_code == 200, f"response.status_code = {response.status_code}"

        # TODO(chenchiyu): narrow this condition
        if finished or (
            self.rollout_cfg_info["backend"] == "pytorch" and train_enable_ep and self.rollout_cfg_info["tp"] > 1
        ):
            # This barrier is aim to make each tp head rank sync with other ranks in engine_parallel group
            # which could not be barrier by `fsdp_foreach_allgather` of the next state dict. (Happens in same_gen, shard not tested)
            # Without barrier, some ranks in engine_parallel group would not wait for current iter data ipc event recording in lmdeploy.
            # They would write next iter state_dict into the ipc tensor before lmdeploy load current iter weight.
            dist.barrier(group=cpu_group)

        monkey_unpatch_torch_reductions()
        return
