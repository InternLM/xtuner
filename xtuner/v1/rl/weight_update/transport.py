from __future__ import annotations

import json
import os
import socket
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Lock
from typing import Any, Callable

import torch
import torch.distributed as dist
from packaging.version import parse as parse_version
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

from .client import RolloutWeightUpdateClient
from .data import RolloutEngineInfo, RolloutWeightUpdateInfo, WeightUpdateBatch


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class WeightTransport(ABC):
    def before_update(self) -> None:
        return

    @abstractmethod
    def send(self, batch: WeightUpdateBatch) -> None:
        raise NotImplementedError

    def after_update(self) -> None:
        return

    def teardown(self) -> None:
        return


class IPCBackendAdapter:
    def before_serialize(self, transport: IPCWeightTransport, batch: WeightUpdateBatch) -> None:
        return

    def serialize(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> Any:
        raise NotImplementedError

    def send_request(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        serialized_data: Any,
    ) -> None:
        raise NotImplementedError

    def postprocess(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
    ) -> None:
        return

    def after_serialize(self, transport: IPCWeightTransport, batch: WeightUpdateBatch) -> None:
        return

    def send(self, transport: IPCWeightTransport, batch: WeightUpdateBatch) -> None:
        """Send a request to update the parameters on the rollout workers.

        This method serializes the state dictionary and sends it to the appropriate rollout worker through the backend-
        specific IPC adapter.
        """
        cpu_mesh = transport.cpu_mesh
        cpu_group = cpu_mesh.get_group()
        head_rank = cpu_mesh.mesh[0].item()

        # Template method for IPC updates: all ranks serialize/gather, only the
        # engine-parallel head rank sends the rollout HTTP request.
        self.before_serialize(transport, batch)
        try:
            serialized_data = self.serialize(transport, batch, cpu_group, head_rank)
            if dist.get_rank() == head_rank:
                self.send_request(transport, batch, serialized_data)
            self.postprocess(transport, batch, cpu_group)
        finally:
            self.after_serialize(transport, batch)


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
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> list[Any]:
        info = transport.rollout_info
        serialized_data = [None] * info.tp
        dist.gather_object(
            self._serialize_state_dict(batch.state_dict),
            serialized_data if dist.get_rank() == head_rank else None,
            dst=head_rank,
            group=cpu_group,
        )
        return serialized_data

    def send_request(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        serialized_data: list[Any],
    ) -> None:
        info = transport.rollout_info
        data_ = json.dumps(dict(serialized_named_tensors=serialized_data, finished=batch.finished))
        data = dict(method="update_weight_npu_ipc", args=[data_])
        assert info.rollout_url is not None
        response = transport.client.collective_rpc(info.rollout_url, data)
        assert response.status_code == 200, f"response.status_code = {response.status_code}"

    def postprocess(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
    ) -> None:
        if batch.finished:
            dist.barrier(group=cpu_group)


class LMDeployIPCBackendAdapter(IPCBackendAdapter):
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
        transport: IPCWeightTransport,
        state_dict: dict,
        flattened_tensor_bucket_cls,
    ) -> dict:
        assert transport._update_params_ipc_event is not None
        # LMDeploy flattened buckets require all tensors in one bucket to share a dtype.
        state_dict_dtype = state_dict[next(iter(state_dict))].dtype
        # LMDeploy can reuse the same IPC tensor across batches. A new handle is
        # sent only when dtype changes, capacity is insufficient, or this is the first batch.
        update_params_ipc_tensor = transport._update_params_ipc_tensor_dict_by_dtype.get(state_dict_dtype, None)
        state_dict_bytes = self._compute_state_dict_bytes(state_dict)
        ipc_tensor_bytes = transport._ipc_tensor_bytes_dict_by_dtype.get(
            state_dict_dtype,
            transport._default_ipc_tensor_bytes,
        )
        dtype_changed = (
            transport._last_update_params_ipc_tensor_dtype is not None
            and state_dict_dtype != transport._last_update_params_ipc_tensor_dtype
        )
        need_resize = state_dict_bytes > ipc_tensor_bytes
        send_ipc_tensor = dtype_changed or need_resize or update_params_ipc_tensor is None

        if update_params_ipc_tensor is not None:
            # Wait until rollout has consumed the previous IPC tensor before reusing it.
            transport._update_params_ipc_event.wait()
            if need_resize:
                # Synchronize before replacing a too-small IPC tensor to avoid freeing
                # storage that may still be referenced by the rollout process.
                DEVICE_MODULE.synchronize()

        if update_params_ipc_tensor is None or need_resize:
            ipc_tensor_bytes = max(ipc_tensor_bytes, state_dict_bytes)
            transport._ipc_tensor_bytes_dict_by_dtype[state_dict_dtype] = ipc_tensor_bytes
            update_params_ipc_tensor = self._create_ipc_tensor(
                ipc_tensor_bytes,
                state_dict_dtype,
            )
            transport._update_params_ipc_tensor_dict_by_dtype[state_dict_dtype] = update_params_ipc_tensor

        flattened_tensor_bucket = flattened_tensor_bucket_cls(
            named_tensors=list(state_dict.items()),
            flattened_tensor=update_params_ipc_tensor,
        )
        flattened_tensor_data = {
            "metadata": flattened_tensor_bucket.get_metadata(),
            "require_clone": False,
        }
        transport._update_params_ipc_event.record()
        transport._last_update_params_ipc_tensor_dtype = state_dict_dtype

        if send_ipc_tensor:
            # Subsequent batches with the same cached IPC tensor only need metadata; the
            # tensor handle and event handle are resent only when the cached buffer changes.
            flattened_tensor_data["flattened_tensor"] = flattened_tensor_bucket.get_flattened_tensor()
            flattened_tensor_data["event_ipc_handle"] = transport._update_params_ipc_event.ipc_handle()
        return flattened_tensor_data

    def serialize(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> Any:
        from lmdeploy.utils import serialize_state_dict

        info = transport.rollout_info
        state_dict = batch.state_dict

        try:
            from lmdeploy.utils import FlattenedTensorBucket

            use_flattened_tensor_bucket = True
        except Exception:
            use_flattened_tensor_bucket = False
            FlattenedTensorBucket = None

        if info.tp > 1:
            serialized_data = [None] * info.tp
            if use_flattened_tensor_bucket and state_dict:
                flattened_tensor_data = self.build_flattened_tensor_data(
                    transport,
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
        else:
            if use_flattened_tensor_bucket and state_dict:
                flattened_tensor_data = self.build_flattened_tensor_data(
                    transport,
                    state_dict,
                    FlattenedTensorBucket,
                )
                serialized_data = serialize_state_dict(flattened_tensor_data)
            else:
                serialized_data = serialize_state_dict(state_dict)
        return serialized_data, use_flattened_tensor_bucket

    def send_request(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        serialized_data: tuple[Any, bool],
    ) -> None:
        info = transport.rollout_info
        state_dict = batch.state_dict
        serialized_named_tensors, use_flattened_tensor_bucket = serialized_data
        data = dict(serialized_named_tensors=serialized_named_tensors, finished=batch.finished)
        if use_flattened_tensor_bucket and state_dict:
            data["load_format"] = "flattened_bucket"
        assert info.rollout_url is not None
        response = transport.client.update_weights(info.rollout_url, info.endpoints["update_weights"], data)
        assert response.status_code == 200, f"response.status_code = {response.status_code}"

    def postprocess(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
    ) -> None:
        info = transport.rollout_info
        # TODO(chenchiyu): narrow this condition.
        if batch.finished or (batch.train_enable_ep and info.tp > 1):
            # Make each TP head rank sync with other ranks in engine_parallel group.
            # FSDP all-gather of the next state_dict cannot cover this case, so without
            # this barrier some ranks could overwrite the IPC tensor before LMDeploy loads it.
            dist.barrier(group=cpu_group)


class SGLangIPCBackendAdapter(IPCBackendAdapter):
    def before_serialize(self, transport: IPCWeightTransport, batch: WeightUpdateBatch) -> None:
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        # NOTE: XTuner currently also works without the SGLang patch in some cases,
        # but keep the patch/unpatch pair for compatibility with SGLang serialization.
        # SGLang overrides torch tensor reduction for multiprocessing serialization.
        monkey_patch_torch_reductions()

    def serialize(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        cpu_group: dist.ProcessGroup,
        head_rank: int,
    ) -> tuple[list[Any], bool]:
        from sglang.srt.utils import MultiprocessingSerializer

        info = transport.rollout_info
        state_dict = batch.state_dict

        try:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

            use_flattened_tensor_bucket = True
        except Exception:
            use_flattened_tensor_bucket = False
            FlattenedTensorBucket = None

        state_items = state_dict.items()
        if info.tp == 1:
            if use_flattened_tensor_bucket:
                flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_items)
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": flattened_tensor_bucket.get_metadata(),
                }
                serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
            else:
                serialized_data = MultiprocessingSerializer.serialize(state_items, output_str=True)
            serialized_data = [serialized_data]
        else:
            serialized_data = [None] * info.tp
            if use_flattened_tensor_bucket:
                flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_items)
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                    "metadata": flattened_tensor_bucket.get_metadata(),
                }
                tp_serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
            else:
                tp_serialized_data = MultiprocessingSerializer.serialize(state_items, output_str=True)
            dist.gather_object(
                tp_serialized_data,
                serialized_data if dist.get_rank() == head_rank else None,
                dst=head_rank,
                group=cpu_group,
            )
        return serialized_data, use_flattened_tensor_bucket

    def send_request(
        self,
        transport: IPCWeightTransport,
        batch: WeightUpdateBatch,
        serialized_data: tuple[list[Any], bool],
    ) -> None:
        info = transport.rollout_info
        serialized_named_tensors, use_flattened_tensor_bucket = serialized_data
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "flush_cache": False,
        }
        if use_flattened_tensor_bucket:
            payload["load_format"] = "flattened_bucket"

        assert info.rollout_url is not None
        response = transport.client.update_weights_from_tensor(info.rollout_url, payload)
        response.raise_for_status()
        assert response.status_code == 200, f"response.status_code = {response.status_code}"

    def after_serialize(self, transport: IPCWeightTransport, batch: WeightUpdateBatch) -> None:
        monkey_unpatch_torch_reductions()


class IPCWeightTransport(WeightTransport):
    def __init__(
        self,
        *,
        rank: int,
        logger: Any,
        config: Any,
        rollout_info: RolloutWeightUpdateInfo,
    ):
        self.rank = rank
        self.logger = logger
        self.config = config
        self.rollout_info = rollout_info
        self.client = RolloutWeightUpdateClient(rollout_info.api_key)
        self._default_ipc_tensor_bytes: int = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        self._ipc_tensor_bytes_dict_by_dtype: dict[torch.dtype, int] = {}
        self._update_params_ipc_tensor_dict_by_dtype: dict[torch.dtype, torch.Tensor] = {}
        self._last_update_params_ipc_tensor_dtype: torch.dtype | None = None
        self._update_params_ipc_event = None
        self._adapter = self._build_adapter()

    @property
    def cpu_mesh(self):
        assert self.rollout_info.rollout_device_mesh is not None
        return self.rollout_info.rollout_device_mesh["engine_parallel"]

    def _build_adapter(self) -> IPCBackendAdapter:
        backend = self.rollout_info.backend
        if backend == "vllm":
            return VLLMIPCBackendAdapter()
        if backend == "sglang":
            return SGLangIPCBackendAdapter()
        return LMDeployIPCBackendAdapter()

    def before_update(self) -> None:
        DEVICE_MODULE.empty_cache()
        self._update_params_ipc_event = DEVICE_MODULE.Event(interprocess=True)

    def after_update(self) -> None:
        self._update_params_ipc_tensor_dict_by_dtype = {}
        self._last_update_params_ipc_tensor_dtype = None
        self._update_params_ipc_event = None
        DEVICE_MODULE.empty_cache()

    def send(self, batch: WeightUpdateBatch) -> None:
        if self.rollout_info.rollout_url is None:
            self.logger.error(f"rank {self.rank} url in None, cannot update weights and skip")
            return
        self._adapter.send(self, batch)


class NCCLBackendAdapter:
    def send(self, transport: NCCLWeightTransport, batch: WeightUpdateBatch) -> None:
        raise NotImplementedError


class SGLangNCCLBackendAdapter(NCCLBackendAdapter):
    def send(self, transport: NCCLWeightTransport, batch: WeightUpdateBatch) -> None:
        state_dict = batch.state_dict
        if not state_dict:
            return

        train_sync_group = transport.get_train_update_sync_group()
        head_rank = 0
        # Disaggregated SGLang update is driven by train rank 0. Other train ranks
        # only wait so optimizer/rollout steps stay aligned.
        if dist.get_rank() != head_rank:
            dist.barrier(group=train_sync_group)
            return

        transport.ensure_group()
        if transport.group is None:
            dist.barrier(group=train_sync_group)
            return

        assert transport.executor is not None
        assert transport.group_name is not None
        with transport.update_lock:
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
                "group_name": transport.group_name,
                "load_format": "flattened_bucket",
            }
            # Notify rollout engines first so they can join the external NCCL group and
            # prepare receive buffers described by names/dtypes/shapes.
            update_futures = [
                transport.executor.submit(
                    transport.client.update_weights_from_distributed,
                    url,
                    payload,
                )
                for url in transport.engine_urls
            ]
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(zip(names, tensors)))
            flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

            dist.broadcast(flattened_tensor, src=0, group=transport.group)
            DEVICE_MODULE.synchronize()
            for update_future in update_futures:
                response = update_future.result()
                response.raise_for_status()
                result = response.json()
                transport.hook_compare_test_sent_and_received_weight_hash(
                    result,
                    names=names,
                )
                assert result.get("success", True), (
                    f"SGLang update_weights_from_distributed failed: {result.get('message', result)}"
                )
        dist.barrier(group=train_sync_group)


class NCCLWeightTransport(WeightTransport):
    def __init__(self, *, rank: int, logger: Any, rollout_info: RolloutWeightUpdateInfo):
        self.rank = rank
        self.logger = logger
        self.rollout_info = rollout_info
        self.client = RolloutWeightUpdateClient(rollout_info.api_key)
        self.group: dist.ProcessGroup | None = None
        self.group_name: str | None = None
        self.engine_urls: list[str] = []
        self.executor: ThreadPoolExecutor | None = None
        self.train_update_sync_group: dist.ProcessGroup | None = None
        self.update_lock = Lock()
        self.hook_compare_test_sent_and_received_weight_hash: Callable[..., None] = lambda result, **kwargs: None
        self._adapter = self._build_adapter()

    def _build_adapter(self) -> NCCLBackendAdapter:
        backend = self.rollout_info.backend
        if backend == "sglang":
            return SGLangNCCLBackendAdapter()
        raise ValueError(f"Unsupported NCCL weight update backend: {backend!r}")

    def get_train_update_sync_group(self) -> dist.ProcessGroup:
        if self.train_update_sync_group is None:
            ranks = list(range(dist.get_world_size()))
            self.train_update_sync_group = dist.new_group(ranks=ranks, backend="gloo")
        return self.train_update_sync_group

    def get_engine_info(self) -> RolloutEngineInfo:
        engine_info: RolloutEngineInfo = []
        seen_urls: set[str] = set()
        rank_to_engine_size: dict[int, int] = {}
        for engine_ranks in self.rollout_info.rollout_engine_rank_mesh_array:
            engine_size = len(engine_ranks)
            for rank in engine_ranks:
                rank_to_engine_size[int(rank)] = engine_size

        for rank, url in sorted(
            self.rollout_info.rollout_server_url_dict.items(),
            key=lambda item: int(item[0]),
        ):
            rank = int(rank)
            # Active server URLs are engine entrypoints, not one endpoint per rollout rank.
            # Deduplicate URLs and skip workers marked unhealthy by the rollout controller.
            if not url or url in seen_urls:
                continue
            if self.rollout_info.worker_server_urls_status.get(url, False) is False:
                continue
            seen_urls.add(url)
            engine_info.append(
                (
                    rank,
                    url,
                    rank_to_engine_size.get(
                        rank,
                        max(self.rollout_info.tp, self.rollout_info.ep),
                    ),
                )
            )
        return engine_info

    def ensure_group(self):
        if self.group is not None:
            return
        engine_info = self.get_engine_info()
        if not engine_info:
            self.logger.error("No active rollout engine url, cannot init sglang weight update group")
            return

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
        backend = "nccl"

        # Get address and port for the external weight-update process group.
        try:
            import ray

            master_address = ray.util.get_node_ip_address()
        except Exception:
            master_address = socket.gethostbyname(socket.gethostname())

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            master_port = int(sock.getsockname()[1])

        group_name = f"xtuner_sglang_weight_update_{self.rank}"
        # Train rank 0 is external group rank 0. Rollout engine ranks are assigned
        # contiguous offsets starting from rank 1.
        world_size = sum(engine_size for _, _, engine_size in engine_info) + 1

        self.executor = ThreadPoolExecutor(max_workers=max(1, len(engine_info)))
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
                self.executor.submit(
                    self.client.init_weights_update_group,
                    url,
                    payload,
                )
            )
            rank_offset += engine_size

        self.group = self._init_external_process_group(
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

        self.group_name = group_name
        self.engine_urls = [url for _, url, _ in engine_info]

    def send(self, batch: WeightUpdateBatch) -> None:
        self._adapter.send(self, batch)

    def teardown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        try:
            if self.group is not None:
                dist.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None
        self.group_name = None
        self.engine_urls = []
        self.executor = None

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
