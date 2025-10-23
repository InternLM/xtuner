import inspect
from typing import Any, Dict, List, Literal, Optional

from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig
from pydantic import BaseModel, ConfigDict

from xtuner.v1.utils import get_logger


logger = get_logger()


class LMDeployDefaultPytorchEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str = "auto"
    tp: int = 1
    dp: int = 1
    dp_rank: int = 0
    ep: int = 1
    session_len: int | None = None
    max_batch_size: int | None = None
    cache_max_entry_count: float = 0.8
    prefill_interval: int = 16
    block_size: int = 64
    num_cpu_blocks: int = 0
    num_gpu_blocks: int = 0
    adapters: Dict[str, str] | None = None
    max_prefill_token_num: int = 4096
    thread_safe: bool = False
    enable_prefix_caching: bool = False
    device_type: str = "cuda"
    eager_mode: bool = False
    custom_module_map: Dict[str, str] | None = None
    download_dir: str | None = None
    revision: str | None = None
    quant_policy: Literal[0, 4, 8] = 0
    distributed_executor_backend: str | None = None
    empty_init: bool = False
    enable_microbatch: bool = False
    enable_eplb: bool = False
    enable_mp_engine: bool = False
    mp_engine_backend: str = "mp"
    model_format: str | None = None
    enable_metrics: bool = False
    hf_overrides: Optional[Dict[str, Any]] | None = None
    disable_vision_encoder: bool = False
    logprobs_mode: str | None = None

    role: str = "Hybrid"
    migration_backend: str = "DLSlime"

    def to_lmdeploy_engine_config(self) -> PytorchEngineConfig:
        server_args_params = set(inspect.signature(PytorchEngineConfig).parameters.keys())
        default_server_args_fields = set(self.model_fields.keys())
        missing_params = server_args_params - default_server_args_fields
        if missing_params:
            logger.warning("Parameters in SGLang ServerArgs but not initialized in Xtuner DefaultServerArgs:")
            for param in sorted(missing_params):
                logger.info(f"- {param}")

        default_args_dict = self.model_dump()
        filtered_args = {key: value for key, value in default_args_dict.items() if key in server_args_params}
        from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend

        if filtered_args.get("role") == "Hybrid":
            filtered_args["role"] = EngineRole.Hybrid
        elif filtered_args.get("role") == "Prefill":
            filtered_args["role"] = EngineRole.Prefill
        elif filtered_args.get("role") == "Decode":
            filtered_args["role"] = EngineRole.Decode
        else:
            logger.warning(f"Unknown role {filtered_args.get('role')}, defaulting to Hybrid")
            filtered_args["role"] = EngineRole.Hybrid
        if filtered_args.get("migration_backend") == "DLSlime":
            filtered_args["migration_backend"] = MigrationBackend.DLSlime
        else:
            logger.warning(
                f"Unknown migration_backend {filtered_args.get('migration_backend')}, defaulting to DLSlime"
            )
            filtered_args["migration_backend"] = MigrationBackend.DLSlime
        return PytorchEngineConfig(**filtered_args)


class LMDeployDefaultTurbomindEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str = "auto"
    model_format: Optional[str] = None
    tp: int = 1
    dp: int = 1
    device_num: int | None = None
    attn_tp_size: int | None = None
    attn_dp_size: int | None = None
    mlp_tp_size: int | None = None
    mlp_dp_size: int | None = None
    outer_dp_size: int | None = None
    session_len: Optional[int] | None = None
    max_batch_size: int | None = None
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    cache_block_seq_len: int = 64
    enable_prefix_caching: bool = False
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: bool = False
    download_dir: Optional[str] | None = None
    revision: Optional[str] | None = None
    max_prefill_token_num: int = 8192
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    devices: Optional[List[int]] | None = None
    empty_init: bool = False
    communicator: str = "nccl"
    hf_overrides: Optional[Dict[str, Any]] | None = None
    enable_metrics: bool = False

    def to_lmdeploy_engine_config(self) -> TurbomindEngineConfig:
        server_args_params = set(inspect.signature(TurbomindEngineConfig).parameters.keys())
        default_server_args_fields = set(self.model_fields.keys())
        missing_params = server_args_params - default_server_args_fields
        if missing_params:
            logger.info("Parameters in SGLang ServerArgs but not initialized in Xtuner DefaultServerArgs:")
            for param in sorted(missing_params):
                logger.info(f"- {param}")

        default_args_dict = self.model_dump()
        filtered_args = {key: value for key, value in default_args_dict.items() if key in server_args_params}
        return TurbomindEngineConfig(**filtered_args)
