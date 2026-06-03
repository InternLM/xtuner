import json
import multiprocessing as py_mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree
from typing import Callable, Mapping, Self, Sequence

import torch
import torch.distributed as dist
from pydantic import ConfigDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
)
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss import BaseLossContext
from xtuner.v1.model import BaseModel
from xtuner.v1.model.base import (
    DEVICE_MODULE,
    AsyncHFSaveHandle,
    BatchForwardInfo,
    ModelOutputs,
    XTunerBaseModelConfig,
)
from xtuner.v1.utils import get_device, get_logger, log_rank0
from xtuner.v1.utils.process import set_async_save_process_qos

from ..utils.misc import update_weight_map_from_safetensors_index


DEVICE = get_device()
logger = get_logger()


class BaseComposeConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    vision_config: XTunerBaseModelConfig
    projector_config: XTunerBaseModelConfig
    text_config: XTunerBaseModelConfig

    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


def to_hf_key_list_wrapper(fn: Callable[[str], list[str]], convertor: Callable[[str], str]):
    def wrapper(self, *args, **kwargs):
        return [convertor(i) for i in fn(*args, **kwargs)]

    return wrapper


class BaseComposeModel(BaseModel):
    def __init__(self, config: BaseComposeConfig):
        super().__init__(config)  # type: ignore[arg-type]
        self._hf_path: Path | None = None

        self.vision_tower = config.vision_config.build()
        self.multi_modal_projector = config.projector_config.build()
        self.language_model = config.text_config.build()

        self._maybe_enable_compile(self.compile_cfg)
        self._freeze_modules()

    def _freeze_modules(self):
        freeze_vision = self.config.freeze_vision
        if freeze_vision:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
            log_rank0.info("Freeze vision tower")
        freeze_projector = self.config.freeze_projector
        if freeze_projector:
            self.multi_modal_projector.requires_grad_(False)
            self.multi_modal_projector.eval()
            log_rank0.info("Freeze multi modal projector")
        freeze_language = self.config.freeze_language
        if freeze_language:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
            log_rank0.info("Freeze language model")

    @override
    def init_weights(self) -> None:
        self.vision_tower.init_weights()
        self.language_model.init_weights()
        self.multi_modal_projector.init_weights()

    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
    ) -> Self:
        self.fsdp_config = fsdp_config
        self.language_model.fully_shard(self.fsdp_config)
        self.vision_tower.fully_shard(self.fsdp_config)
        self.multi_modal_projector.fully_shard(self.fsdp_config)

        # TODO: 判断其余模块是否已经被 fsdp 切分了

        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype)

        self.fsdp_mesh = init_world_mesh()
        # Note: 非常关键，不能删除这个 assert
        assert self.fsdp_mesh is not None

        self._fully_shard(
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

        if isinstance(self.vision_tower.blocks[0], FSDPModule):
            self.language_model.embed_tokens.set_modules_to_forward_prefetch(  # type: ignore
                [self.vision_tower.blocks[0]]
            )
            self.vision_tower.blocks[-1].set_modules_to_forward_prefetch(  # type: ignore
                [self.multi_modal_projector]
            )
        if isinstance(self.multi_modal_projector, FSDPModule):
            self.multi_modal_projector.set_modules_to_forward_prefetch([self.language_model])  # type: ignore
        self.language_model.set_modules_to_forward_prefetch([self.language_model.layers["0"]])  # type: ignore

        self._to_empty_meta()
        return self

    def from_hf(self, hf_path: str | Path, strict=True):
        self._hf_path = Path(hf_path)

        if isinstance(hf_path, Path):
            hf_path = str(hf_path)

        _, _, missing_llm_keys = self.language_model.from_hf(hf_path, strict=False)
        _, _, missing_vision_keys = self.vision_tower.from_hf(hf_path, strict=False)
        _, _, missing_project_keys = self.multi_modal_projector.from_hf(hf_path, strict=False)

        missing = missing_llm_keys | missing_vision_keys | missing_project_keys
        if strict:
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)}. ")

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16, safetensors_prefix: str = "model"):
        hf_dir = Path(hf_dir)
        self.language_model.save_hf(hf_dir, save_dtype, "model-language")

        weight_map_dict: dict = {}
        update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

        self.vision_tower.save_hf(hf_dir, save_dtype, "model-vision")
        update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

        self.multi_modal_projector.save_hf(hf_dir, save_dtype, "model-projector")
        update_weight_map_from_safetensors_index(weight_map_dict, hf_dir)

        if dist.get_rank() == 0:
            with open(hf_dir / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map_dict, "metadata": {}}, f, indent=4)
        dist.barrier()

    def _write_hf_index_and_config(self, hf_dir: Path | str, weight_map: Mapping[str, str]) -> None:
        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)

        self._write_hf_non_weight_files(hf_dir)

        with open(hf_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": dict(weight_map), "metadata": {}}, f, indent=4)

    def destroy_async_hf_resources(self) -> None:
        super().destroy_async_hf_resources()
        for module in (self.language_model, self.vision_tower, self.multi_modal_projector):
            module._async_hf_tensor_cache.clear()

    def async_save_hf(
        self,
        hf_dir: Path | str,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
        file_finalize_callback: Callable[[Path], None] | None = None,
    ) -> AsyncHFSaveHandle:
        return self._async_save_hf(
            hf_dir=hf_dir,
            save_dtype=save_dtype,
            safetensors_prefix=safetensors_prefix,
            file_finalize_callback=file_finalize_callback,
        )

    def _async_save_hf(
        self,
        hf_dir: Path | str,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
        file_finalize_callback: Callable[[Path], None] | None = None,
        use_file_lock: bool = True,
    ) -> AsyncHFSaveHandle:
        self._get_async_hf_resources()
        if self._hf_path is None and self.config.hf_config is None:
            raise NotImplementedError(
                "The model is not loaded from Huggingface, and the `hf_config` property is not implemented, so it cannot be saved in Huggingface format."
            )
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

        module_file_to_names: list[tuple[BaseModel, list[tuple[str, list[str]]]]] = []
        merged_weight_map: dict[str, str] = {}
        for module, prefix in (
            (self.language_model, "model-language"),
            (self.vision_tower, "model-vision"),
            (self.multi_modal_projector, "model-projector"),
        ):
            file_to_names, weight_map = module._prepare_async_hf_snapshot(
                save_dtype=save_dtype,
                safetensors_prefix=prefix,
                device=DEVICE,
            )
            module_file_to_names.append((module, file_to_names))
            merged_weight_map.update(weight_map)

        global_weight_map = self._merge_async_hf_weight_map(merged_weight_map)
        self._prewrite_async_hf_metadata(tmp_hf_dir=tmp_hf_dir, weight_map=global_weight_map)

        if hasattr(DEVICE_MODULE, "synchronize"):
            DEVICE_MODULE.synchronize()

        mp_ctx = py_mp.get_context("fork")
        process = mp_ctx.Process(
            target=self._run_async_hf_compose_writer,
            args=(
                tmp_hf_dir,
                module_file_to_names,
                use_file_lock,
            ),
            daemon=False,
        )
        process.start()

        handle = AsyncHFSaveHandle(
            process=process,
            commit_future=None,
            hf_dir=hf_dir,
            tmp_hf_dir=tmp_hf_dir,
        )
        commit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-hf-commit")
        handle.commit_future = commit_executor.submit(
            self.commit_async_hf_save,
            handle,
            file_finalize_callback=file_finalize_callback,
        )
        self._pending_async_hf = handle
        handle.commit_future.add_done_callback(self._record_async_hf_commit_result)
        handle.commit_future.add_done_callback(lambda _: self._clear_pending_async_hf(handle))
        handle.commit_future.add_done_callback(lambda _: commit_executor.shutdown(wait=False))
        return handle

    def _async_hf_tensor_cache_stats(self) -> tuple[int, int]:
        modules = (self.language_model, self.vision_tower, self.multi_modal_projector)
        cache_tensors = 0
        cache_bytes = 0
        for module in modules:
            module_cache_tensors, module_cache_bytes = module._async_hf_tensor_cache_stats()
            cache_tensors += module_cache_tensors
            cache_bytes += module_cache_bytes
        return cache_tensors, cache_bytes

    def _run_async_hf_compose_writer(
        self,
        tmp_hf_dir: Path,
        module_file_to_names: list[tuple[BaseModel, list[tuple[str, list[str]]]]],
        use_file_lock: bool = True,
    ) -> None:
        log_rank0.info(f"[Async saving HF to {tmp_hf_dir} writer] started")
        try:
            set_async_save_process_qos()
            for module, file_to_names in module_file_to_names:
                self._write_async_hf_module_snapshot(
                    hf_dir=tmp_hf_dir,
                    module=module,
                    file_to_names=file_to_names,
                    use_file_lock=use_file_lock,
                )
            log_rank0.info(f"[Async saving HF to {tmp_hf_dir} writer] finished")
        except Exception as exc:
            log_rank0.error(f"[Async saving HF to {tmp_hf_dir} writer] failed: {exc}")
            raise

    def _write_async_hf_module_snapshot(
        self,
        hf_dir: Path,
        module: BaseModel,
        file_to_names: list[tuple[str, list[str]]],
        use_file_lock: bool = True,
    ) -> None:
        for filename, names in file_to_names:
            tensors: dict[str, torch.Tensor] = {}
            for name in names:
                cache_key = (("root", "hf"), ("name", name))
                cached_tensor = module._async_hf_tensor_cache.get(cache_key)
                if cached_tensor is None:
                    raise RuntimeError(f"Missing cached async HF tensor for key: {name}")
                tensors[name] = cached_tensor
            module._write_hf_save_plan(
                {"hf_dir": hf_dir, "save_tasks": [(filename, tensors)]},
                use_file_lock=use_file_lock,
            )

    def post_micro_batch_forward(self, batch_outputs: Sequence[ModelOutputs]) -> BatchForwardInfo:
        return self.language_model.post_micro_batch_forward(batch_outputs)

    def scale_and_reduce_grad(self):
        self.language_model.scale_and_reduce_grad()

    @override
    def build_loss_ctx_batch(  # type: ignore[override]
        self,
        data_batch: list[dict],
        sp_mesh: DeviceMesh | None = None,
    ) -> list[dict[str, BaseLossContext]]:
        """Delegate loss_ctx building to the language model."""
        # TODO: Maybe we need to consider the `loss_ctx` of vision model.
        return self.language_model.build_loss_ctx_batch(data_batch, sp_mesh=sp_mesh)
