from __future__ import annotations

import importlib
import json
import os
import socket
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Protocol, cast

import requests
import torch
import torch.distributed as dist
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

from xtuner.v1.utils import (
    get_device,
    get_torch_device_module,
    monkey_unpatch_torch_reductions,
)

from .data import RolloutWeightUpdateInfo, WeightUpdateBatch


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


@dataclass
class WeightUpdateRequest:
    # HTTP endpoint on the rollout server that should receive this update.
    endpoint: str
    # JSON body sent to the rollout backend adapter endpoint.
    body: dict[str, Any]


class WeightTransportAdapter(Protocol):
    def before_update(self) -> None: ...

    def after_update_all_groups(self) -> None: ...


class WeightTransport(ABC):
    def __init__(self, *, rollout_info: RolloutWeightUpdateInfo, logger: Any, rank: int):
        self.rollout_info = rollout_info
        self.logger = logger
        self.rank = rank
        self.backend = self.rollout_info.backend
        self.rollout_ep = self.rollout_info.ep
        self.rollout_tp = self.rollout_info.tp
        self._adapter: WeightTransportAdapter | None = None

        self.rollout_url = self.rollout_info.rollout_url

    @staticmethod
    def post_json(url: str, endpoint: str, payload: dict, *, api_key=None) -> dict:
        headers = {"Content-Type": "application/json"}
        # TODO move api key to init
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(f"{url}/{endpoint}", headers=headers, json=payload)
        assert response.status_code == 200, f"response.status_code = {response.status_code}"
        response.raise_for_status()
        return response.json()

    def update(self, weight_iterator: Any) -> None:
        assert self._adapter is not None
        self._adapter.before_update()
        DEVICE_MODULE.empty_cache()

        try:
            for batches in weight_iterator.iter_batch_groups():
                for batch in batches:
                    self.send(batch)
                self.after_update_per_group()
                DEVICE_MODULE.empty_cache()
        finally:
            self.after_update_all_groups()
            DEVICE_MODULE.empty_cache()

    @abstractmethod
    def send(self, batch: WeightUpdateBatch) -> None:
        raise NotImplementedError

    def after_update_all_groups(self) -> None:
        return

    def after_update_per_group(self) -> None:
        return

    def teardown(self) -> None:
        return


class IPCBackendAdapter:
    # def __init__(self, *, rollout_info: RolloutWeightUpdateInfo):
    def __init__(self, *, rollout_tp: int):
        self.rollout_tp = rollout_tp
        # self.rollout_info = rollout_info

    def before_update(self) -> None:
        return

    def after_update(self) -> None:
        return

    def build_request(
        self,
        batch: WeightUpdateBatch,
        serialized_data: Any,
    ) -> WeightUpdateRequest:
        raise NotImplementedError

    def serialize(
        self,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> list[Any]:
        raise NotImplementedError

    def after_update_per_batch(
        self, finished: bool, cpu_group: dist.ProcessGroup, train_enable_ep: bool = False
    ) -> None:
        return

    def after_update_all_groups(self) -> None:
        return


class VLLMIPCBackendAdapter(IPCBackendAdapter):
    @staticmethod
    def _serialize_state_dict(state_dict: dict) -> str:
        import base64
        from io import BytesIO
        from multiprocessing.reduction import ForkingPickler

        from torch.multiprocessing.reductions import reduce_tensor

        data = [(k, reduce_tensor(v)) for k, v in state_dict.items()]
        buf = BytesIO()
        ForkingPickler(buf).dump(data)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def serialize(
        self,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> list[Any]:
        serialized_data = [None] * self.rollout_tp
        dist.gather_object(
            self._serialize_state_dict(batch.state_dict),
            serialized_data if dist.get_rank() == head_rank else None,
            dst=head_rank,
            group=cpu_group,
        )
        return serialized_data

    def build_request(
        self,
        batch: WeightUpdateBatch,
        serialized_data: list[Any],
    ) -> WeightUpdateRequest:
        data_ = json.dumps(dict(serialized_named_tensors=serialized_data, finished=batch.finished))
        data = dict(method="update_weight_npu_ipc", args=[data_])
        return WeightUpdateRequest(endpoint="collective_rpc", body=data)

    def after_update_per_batch(
        self, finished: bool, cpu_group: dist.ProcessGroup, train_enable_ep: bool = False
    ) -> None:
        if finished:
            dist.barrier(group=cpu_group)


class LMDeployIPCBackendAdapter(IPCBackendAdapter):
    def __init__(self, *, rollout_tp: int, backend: str, default_ipc_tensor_bytes: int):
        super().__init__(rollout_tp=rollout_tp)
        self._default_ipc_tensor_bytes = default_ipc_tensor_bytes
        self._ipc_tensor_bytes_by_dtype: dict[torch.dtype, int] = {}
        self._update_params_ipc_tensor_by_dtype: dict[torch.dtype, torch.Tensor] = {}
        self._last_update_params_ipc_tensor_dtype: torch.dtype | None = None
        self._update_params_ipc_event = None
        self.backend = backend
        self.endpoints: dict[str, str] = dict()
        self.endpoints["update_weights"] = "update_weights"

        try:
            model_runner = importlib.import_module("lmdeploy.utils")
            getattr(model_runner, "FlattenedTensorBucket")
            self.use_flattened_tensor_bucket = True
        except Exception:
            self.use_flattened_tensor_bucket = False

    def before_update(self) -> None:
        self._update_params_ipc_event = DEVICE_MODULE.Event(interprocess=True)

    def after_update_all_groups(self) -> None:
        self._ipc_tensor_bytes_by_dtype = {}
        self._update_params_ipc_tensor_by_dtype = {}
        self._last_update_params_ipc_tensor_dtype = None
        self._update_params_ipc_event = None

    @staticmethod
    def _compute_state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int:
        total_bytes = 0
        for tensor in state_dict.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

    @staticmethod
    def _create_ipc_tensor(size_in_bytes: int, dtype: torch.dtype):
        return torch.empty(size_in_bytes, dtype=torch.uint8, device=DEVICE).view(dtype)

    def build_flattened_tensor_data(
        self,
        state_dict: dict,
        flattened_tensor_bucket_cls,
    ) -> dict:
        assert self._update_params_ipc_event is not None
        # LMDeploy flattened buckets require all tensors in one bucket to share a dtype.
        state_dict_dtype = state_dict[next(iter(state_dict))].dtype
        # LMDeploy can reuse the same IPC tensor across batches. A new handle is
        # sent only when dtype changes, capacity is insufficient, or this is the first batch.
        update_params_ipc_tensor = self._update_params_ipc_tensor_by_dtype.get(state_dict_dtype, None)
        state_dict_bytes = self._compute_state_dict_bytes(state_dict)
        ipc_tensor_bytes = self._ipc_tensor_bytes_by_dtype.get(
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
            # Wait until rollout has consumed the previous IPC tensor before reusing it.
            self._update_params_ipc_event.wait()
            if need_resize:
                # Synchronize before replacing a too-small IPC tensor to avoid freeing
                # storage that may still be referenced by the rollout process.
                DEVICE_MODULE.synchronize()

        if update_params_ipc_tensor is None or need_resize:
            ipc_tensor_bytes = max(ipc_tensor_bytes, state_dict_bytes)
            self._ipc_tensor_bytes_by_dtype[state_dict_dtype] = ipc_tensor_bytes
            update_params_ipc_tensor = self._create_ipc_tensor(
                ipc_tensor_bytes,
                state_dict_dtype,
            )
            self._update_params_ipc_tensor_by_dtype[state_dict_dtype] = update_params_ipc_tensor

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
            # Subsequent batches with the same cached IPC tensor only need metadata; the
            # tensor handle and event handle are resent only when the cached buffer changes.
            flattened_tensor_data["flattened_tensor"] = flattened_tensor_bucket.get_flattened_tensor()
            flattened_tensor_data["event_ipc_handle"] = self._update_params_ipc_event.ipc_handle()
        return flattened_tensor_data

    def serialize(
        self,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> list[Any]:
        from lmdeploy.utils import serialize_state_dict

        state_dict = batch.state_dict

        if self.use_flattened_tensor_bucket and state_dict:
            from lmdeploy.utils import FlattenedTensorBucket

            flattened_tensor_data = self.build_flattened_tensor_data(
                state_dict,
                FlattenedTensorBucket,
            )
            serialized_data = serialize_state_dict(flattened_tensor_data)
        else:
            serialized_data = serialize_state_dict(state_dict)

        if self.rollout_tp == 1:
            return serialized_data
        else:
            all_serialized_data = [None] * self.rollout_tp
            dist.gather_object(
                serialized_data,
                all_serialized_data if dist.get_rank() == head_rank else None,
                dst=head_rank,
                group=cpu_group,
            )
            return all_serialized_data

    def build_request(
        self,
        batch: WeightUpdateBatch,
        serialized_data: tuple[Any, bool],
    ) -> WeightUpdateRequest:
        state_dict = batch.state_dict
        data = dict(serialized_named_tensors=serialized_data, finished=batch.finished)
        if self.use_flattened_tensor_bucket and state_dict:
            data["load_format"] = "flattened_bucket"
        return WeightUpdateRequest(endpoint=self.endpoints["update_weights"], body=data)

    def after_update_per_batch(
        self, finished: bool, cpu_group: dist.ProcessGroup, train_enable_ep: bool = False
    ) -> None:
        # TODO(chenchiyu): narrow this condition.
        if finished or (train_enable_ep and self.rollout_tp > 1):
            # Make each TP head rank sync with other ranks in engine_parallel group.
            # FSDP all-gather of the next state_dict cannot cover this case, so without
            # this barrier some ranks could overwrite the IPC tensor before LMDeploy loads it.
            dist.barrier(group=cpu_group)


class SGLangIPCBackendAdapter(IPCBackendAdapter):
    def __init__(self, *, rollout_tp):
        super().__init__(rollout_tp=rollout_tp)

        try:
            model_runner = importlib.import_module("sglang.srt.model_executor.model_runner")
            getattr(model_runner, "FlattenedTensorBucket")

            self.use_flattened_tensor_bucket = True
        except Exception:
            self.use_flattened_tensor_bucket = False

    def serialize(
        self,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> list[Any]:
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        # NOTE: XTuner currently also works without the SGLang patch in some cases,
        # but keep the patch/unpatch pair for compatibility with SGLang serialization.
        # SGLang overrides torch tensor reduction for multiprocessing serialization.
        monkey_patch_torch_reductions()

        from sglang.srt.utils import MultiprocessingSerializer

        state_dict = batch.state_dict

        state_items = state_dict.items()

        if self.use_flattened_tensor_bucket:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_items)
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": flattened_tensor_bucket.get_metadata(),
            }
            serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
        else:
            serialized_data = MultiprocessingSerializer.serialize(state_items, output_str=True)

        if self.rollout_tp == 1:
            return [serialized_data]
        else:
            all_serialized_data = [None] * self.rollout_tp
            dist.gather_object(
                serialized_data,
                all_serialized_data if dist.get_rank() == head_rank else None,
                dst=head_rank,
                group=cpu_group,
            )
            return all_serialized_data

    def build_request(
        self,
        batch: WeightUpdateBatch,
        serialized_data: tuple[list[Any], bool],
    ) -> WeightUpdateRequest:
        payload = {
            "serialized_named_tensors": serialized_data,
            "flush_cache": False,
        }
        if self.use_flattened_tensor_bucket:
            payload["load_format"] = "flattened_bucket"
        return WeightUpdateRequest(endpoint="update_weights_from_tensor", body=payload)

    def after_update_per_batch(
        self, finished: bool, cpu_group: dist.ProcessGroup, train_enable_ep: bool = False
    ) -> None:
        # TODO(chenchiyu): narrow this condition.
        if finished:
            # Make each TP head rank sync with other ranks in engine_parallel group.
            # FSDP all-gather of the next state_dict cannot cover this case, so without
            # this barrier some ranks could overwrite the IPC tensor before LMDeploy loads it.
            dist.barrier(group=cpu_group)


class IPCWeightTransport(WeightTransport):
    _adapter: IPCBackendAdapter

    def __init__(
        self,
        *,
        rank: int,
        logger: Any,
        config: Any,
        rollout_info: RolloutWeightUpdateInfo,
    ):
        super().__init__(rank=rank, logger=logger, rollout_info=rollout_info)
        self.config = config
        self._adapter = self._build_adapter()

        self.ipc_update_device_mesh = DeviceMesh(
            "cpu",
            mesh=[list(ranks) for ranks in self.rollout_info.ipc_rank_mesh],
            mesh_dim_names=("engine_instance", "engine_parallel"),
        )
        self.cpu_mesh = self.ipc_update_device_mesh["engine_parallel"]
        self.cpu_group = self.cpu_mesh.get_group()
        self.head_rank = int(self.cpu_mesh.mesh[0].item())

    def _build_adapter(self) -> IPCBackendAdapter:
        if self.backend == "vllm":
            return VLLMIPCBackendAdapter(rollout_tp=self.rollout_info.tp)
        elif self.backend == "sglang":
            return SGLangIPCBackendAdapter(rollout_tp=self.rollout_info.tp)
        elif self.backend == "pytorch" or self.backend == "turbomind":
            return LMDeployIPCBackendAdapter(
                rollout_tp=self.rollout_info.tp,
                backend=self.backend,
                default_ipc_tensor_bytes=int(self.config.update_weight_bucket_size_in_gb * 1024**3),
            )
        else:
            raise ValueError(
                f"Unsupported IPC weight update backend: {self.backend!r}. Expected 'vllm', 'sglang', 'pytorch' or 'turbomind'."
            )

    def after_update_all_groups(self) -> None:
        self._adapter.after_update_all_groups()
        DEVICE_MODULE.empty_cache()

    def after_update_per_group(self) -> None:
        dist.barrier()

    def send(self, batch: WeightUpdateBatch) -> None:
        ipc_update_target = self.rollout_info._ipc_update_target
        assert ipc_update_target is not None, "IPC rollout target for current train rank is not resolved."
        if not ipc_update_target.is_active:
            return
        rollout_url = ipc_update_target.server_url

        DEVICE_MODULE.empty_cache()
        try:
            serialized_data = self._adapter.serialize(
                batch,
                self.cpu_group,
                self.head_rank,
            )
            if dist.get_rank() == self.head_rank:
                request = self._adapter.build_request(batch, serialized_data)
                self.post_json(
                    rollout_url,
                    request.endpoint,
                    request.body,
                    api_key=self.rollout_info.api_key,
                )

            self._adapter.after_update_per_batch(batch.finished, self.cpu_group, batch.train_enable_ep)

        finally:
            monkey_unpatch_torch_reductions()


class NCCLBackendAdapter:
    def __init__(self):
        pass

    def build_weight_update_payload(self, batch: WeightUpdateBatch, group_name: str):
        pass

    def build_request(
        self,
        payload: dict[str, Any],
    ) -> WeightUpdateRequest:
        raise NotImplementedError

    def before_update(self) -> None:
        return

    def after_update_all_groups(self) -> None:
        return


class SGLangNCCLBackendAdapter(NCCLBackendAdapter):
    def __init__(self):
        super().__init__()

    def build_weight_update_payload(self, batch: WeightUpdateBatch, group_name: str):
        try:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
        except Exception as e:
            raise RuntimeError(
                "Disaggregated update_weights currently only supports sglang builds "
                "that provide `sglang.srt.model_executor.model_runner.FlattenedTensorBucket`."
            ) from e

        state_dict = batch.state_dict
        finished = batch.finished
        if not finished:
            weight_names = list(state_dict.keys())
            weight_tensors = [
                tensor.detach().to(device=DEVICE, non_blocking=True).contiguous() for tensor in state_dict.values()
            ]
            payload = {
                "names": weight_names,
                "dtypes": [str(tensor.dtype).replace("torch.", "") for tensor in weight_tensors],
                "shapes": [list(tensor.shape) for tensor in weight_tensors],
                "group_name": group_name,
                "load_format": "flattened_bucket",
            }
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(zip(weight_names, weight_tensors)))
            flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

            return payload, flattened_tensor, weight_names
        else:
            return None, None, None

    def build_request(
        self,
        payload: dict[str, Any],
    ) -> WeightUpdateRequest:
        return WeightUpdateRequest(endpoint="update_weights_from_distributed", body=payload)


class LMDeployNCCLBackendAdapter(NCCLBackendAdapter):
    def __init__(self):
        super().__init__()

    def build_weight_update_payload(self, batch: WeightUpdateBatch, group_name: str):
        try:
            from lmdeploy.utils import FlattenedTensorBucket
        except Exception as e:
            raise RuntimeError(
                "Disaggregated update_weights for lmdeploy backend requires lmdeploy builds that provide "
                "`lmdeploy.utils.FlattenedTensorBucket`."
            ) from e

        state_dict = batch.state_dict
        finished = batch.finished
        # Pytorch backend will send empty state_dict when finished.
        if not finished:
            weight_names = list(state_dict.keys())
            weight_tensors = [
                tensor.detach().to(device=DEVICE, non_blocking=True).contiguous() for tensor in state_dict.values()
            ]
            payload = {
                "names": weight_names,
                "dtypes": [str(tensor.dtype).replace("torch.", "") for tensor in weight_tensors],
                "shapes": [list(tensor.shape) for tensor in weight_tensors],
                "group_name": group_name,
                "load_format": "flattened_bucket",
            }
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(zip(weight_names, weight_tensors)))
            flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

            return payload, flattened_tensor, weight_names
        else:
            # finalize-only request: no tensors to broadcast, just trigger the
            # rollout side's mod.update_weights() finalization hooks.
            payload = {
                "names": [],
                "dtypes": [],
                "shapes": [],
                "group_name": group_name,
                "load_format": "flattened_bucket",
                "finished": True,
            }
            return payload, None, None

    def build_request(
        self,
        payload: dict[str, Any],
    ) -> WeightUpdateRequest:
        return WeightUpdateRequest(endpoint="update_weights_from_distributed", body=payload)


class NCCLWeightTransport(WeightTransport):
    _adapter: NCCLBackendAdapter

    def __init__(self, *, rank: int, logger: Any, rollout_info: RolloutWeightUpdateInfo):
        super().__init__(rank=rank, logger=logger, rollout_info=rollout_info)
        self.group: dist.ProcessGroup | None = None
        self.group_name: str | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.train_update_sync_group: dist.ProcessGroup | None = None
        self.hook_compare_test_sent_and_received_weight_hash: Callable[..., None] = lambda result, **kwargs: None

        self.engine_urls: list[str] = []
        self.external_group_world_size: int | None = None

        self._adapter = self._build_adapter()

    def _build_adapter(self) -> NCCLBackendAdapter:
        if self.backend == "sglang":
            return SGLangNCCLBackendAdapter()
        elif self.backend == "pytorch":
            return LMDeployNCCLBackendAdapter()
        raise ValueError(f"Unsupported NCCL weight update backend: {self.backend!r}")

    def get_train_update_sync_group(self) -> dist.ProcessGroup:
        # Create a Gloo process group for synchronization during NCCL weight update.
        if self.train_update_sync_group is None:
            ranks = list(range(dist.get_world_size()))
            self.train_update_sync_group = dist.new_group(ranks=ranks, backend="gloo")
        return self.train_update_sync_group

    def get_weight_update_address(self) -> tuple[str, int]:
        # NCCL 会建立通信组 [train 0 + all rollout rank] 来进行broadcast，这里需要获得可用ip和port
        host = self.rollout_info.weight_update_host
        if not host:
            try:
                import ray

                host = ray.util.get_node_ip_address()
            except Exception:
                host = socket.gethostbyname(socket.gethostname())

        port = self.rollout_info.weight_update_port

        return cast(str, host), cast(int, port)

    def ensure_nccl_weight_update_group(self):
        """Create the NCCL weight update group if it has not been
        initialized."""

        if self.group is not None:
            return

        # RolloutWeightUpdateInfo owns the runtime target projection.
        engine_info = self.rollout_info.nccl_engine_infos

        if not engine_info:
            self.logger.error("No active rollout engine url, cannot init sglang weight update group")
            return

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        backend = "nccl"
        address, port = self.get_weight_update_address()

        group_name = f"xtuner_NCCL_weight_update_{self.rank}"
        # Train rank 0 is external group rank 0. Rollout engine ranks are assigned
        # contiguous offsets starting from rank 1.
        world_size = sum(engine_size for _, _, engine_size in engine_info) + 1

        self.external_group_world_size = world_size

        self.executor = ThreadPoolExecutor(max_workers=max(1, len(engine_info)))
        init_futures = []
        rank_offset = 1

        for _, url, engine_size in engine_info:
            payload = {
                "master_address": address,
                "master_port": port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            }
            init_futures.append(
                self.executor.submit(
                    self.post_json,
                    url,
                    "init_weights_update_group",
                    payload,
                    api_key=self.rollout_info.api_key,
                )
            )
            rank_offset += engine_size

        self.group = self._init_external_process_group(
            backend=backend,
            init_method=f"tcp://{address}:{port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )

        for init_future in init_futures:
            result = init_future.result()
            assert result.get("success", True), f"init_weights_update_group failed: {result.get('message', result)}"

        self.group_name = group_name
        self.engine_urls = [url for _, url, _ in engine_info]

    def send(self, batch: WeightUpdateBatch) -> None:
        state_dict = batch.state_dict
        if not state_dict:
            return

        train_sync_group = self.get_train_update_sync_group()
        head_rank = 0
        # Only train rank 0 drives the disaggregated NCCL update. Other train
        # ranks wait here so training and rollout steps remain aligned.
        if dist.get_rank() != head_rank:
            dist.barrier(group=train_sync_group)
            return

        self.ensure_nccl_weight_update_group()
        if self.group is None:
            # If the NCCL weight update group could not be initialized, release the
            # other training ranks waiting at the sync barrier and skip this update.
            dist.barrier(group=train_sync_group)
            return

        assert self.executor is not None
        assert self.group_name is not None
        payload, flattened_tensor, weight_names = self._adapter.build_weight_update_payload(batch, self.group_name)
        if payload is not None:
            request = self._adapter.build_request(payload)
            # Notify rollout engines first so they can join the external NCCL group and
            # prepare receive buffers described by names/dtypes/shapes.
            update_futures = [
                self.executor.submit(
                    self.post_json,
                    url,
                    request.endpoint,
                    request.body,
                    api_key=self.rollout_info.api_key,
                )
                for url in self.engine_urls
            ]
            if flattened_tensor is not None:
                # LMDeploy send empty payload finally.
                # Send the flattened weight tensor through the external NCCL group.
                dist.broadcast(flattened_tensor, src=0, group=self.group)
                DEVICE_MODULE.synchronize()
            # Wait for rollout engines to finish loading weights and validate
            # backend-specific update results.
            for update_future in update_futures:
                result = update_future.result()
                self.hook_compare_test_sent_and_received_weight_hash(
                    result,
                    names=weight_names,
                )
                assert result.get("success", True), (
                    f"update_weights_from_distributed failed: {result.get('message', result)}"
                )
            dist.barrier(group=train_sync_group)

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
        # Build a process group that includes external rollout processes, which
        # cannot be represented by dist.new_group over the current training world.
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

    def after_update_per_group(self) -> None:
        dist.barrier(group=self.get_train_update_sync_group())

    def teardown(self) -> None:
        # Reset only resources that depend on rollout metadata. The train-side sync group
        # is independent of rollout workers and should live until worker teardown.
        if self.group is not None:
            try:
                dist.destroy_process_group(self.group)
            except Exception as e:
                self.logger.warning(f"Failed to destroy NCCL weight update group: {e}")
            self.group = None

        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

        self.group_name = None
        self.engine_urls = []
        self.external_group_world_size = None


class DiskBackendAdapter:
    def build_weight_update_payload(self, hf_weight_path: str) -> dict[str, Any]:
        raise NotImplementedError

    def build_request(
        self,
        payload: dict[str, Any],
    ) -> WeightUpdateRequest:
        raise NotImplementedError

    def update(self, weight_iterator: Any) -> None:
        raise NotImplementedError

    def teardown(self) -> None:
        return


class SGLangDiskBackendAdapter(DiskBackendAdapter):
    def __init__(self, *, rank: int, rollout_info: RolloutWeightUpdateInfo):
        self.rank = rank
        self.rollout_info = rollout_info
        self.executor: ThreadPoolExecutor | None = None

    def build_weight_update_payload(self, hf_weight_path: str) -> dict[str, Any]:
        # SGLang already owns the disk reload path. XTuner only needs to pass
        # the HF checkpoint directory to the rollout server.
        return {
            "model_path": hf_weight_path,
            "load_format": "safetensors",
            "abort_all_requests": True,
            "flush_cache": True,
        }

    def build_request(
        self,
        payload: dict[str, Any],
    ) -> WeightUpdateRequest:
        return WeightUpdateRequest(endpoint="update_weights_from_disk", body=payload)

    def update(self, weight_iterator: Any) -> None:
        # SGLang consumes the checkpoint path on the rollout server side.
        del weight_iterator

        disk_weight_path = self.rollout_info.disk_weight_path
        if not disk_weight_path:
            raise RuntimeError("Disk weight update requires rollout_info.disk_weight_path from rollout_config.")

        try:
            if dist.get_rank() != 0:
                dist.barrier()
                return

            target_urls = [t.server_url for t in self.rollout_info.active_update_targets]
            if not target_urls:
                raise RuntimeError("Disk weight update requires at least one rollout server url.")
            payload = self.build_weight_update_payload(disk_weight_path)
            request = self.build_request(payload)
            self.executor = ThreadPoolExecutor(max_workers=max(1, len(target_urls)))
            futures = [
                self.executor.submit(
                    WeightTransport.post_json,
                    url,
                    request.endpoint,
                    request.body,
                    api_key=self.rollout_info.api_key,
                )
                for url in target_urls
            ]
            for future in futures:
                result = future.result()
                assert result.get("success", True), f"disk weight update failed: {result.get('message', result)}"
            dist.barrier()
        finally:
            self.teardown()
            DEVICE_MODULE.empty_cache()

    def teardown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None


class DiskWeightTransport(WeightTransport):
    _disk_adapter: DiskBackendAdapter

    def __init__(self, *, rank: int, logger: Any, rollout_info: RolloutWeightUpdateInfo, config: Any | None = None):
        super().__init__(rank=rank, logger=logger, rollout_info=rollout_info)
        self.config = config
        self._disk_adapter = self._build_adapter()

    def _build_adapter(self) -> DiskBackendAdapter:
        if self.backend == "sglang":
            return SGLangDiskBackendAdapter(rank=self.rank, rollout_info=self.rollout_info)
        raise ValueError(f"Unsupported disk weight update backend: {self.backend!r}")

    def update(self, weight_iterator: Any) -> None:
        self._disk_adapter.update(weight_iterator)

    def send(self, batch: WeightUpdateBatch) -> None:
        raise NotImplementedError("DiskWeightTransport bypasses WeightIterator batches.")
