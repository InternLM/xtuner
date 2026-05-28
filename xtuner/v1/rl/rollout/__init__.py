import os

from ._generation.external_http_entry import ExternalRolloutHttpEntry, ExternalRolloutHttpEntryConfig
from ._generation.internal_http_entry import (
    InternalRolloutHttpEntry,
    InternalRolloutHttpEntryConfig,
    build_internal_rollout_http_entry_app,
    serve_internal_rollout_http_entry_in_thread,
)
from ._generation.session_worker_selector import RolloutWorkerHandle, SessionWorkerSelector
from .controller import RolloutController
from .rollout_generator import (
    LocalRolloutGenerator,
    LocalRolloutGeneratorConfig,
    RolloutGenerateHandle,
    RolloutGenerateHandleConfig,
)
from .rollout_worker_build import RolloutRuntime, RolloutWorkerBuilder, RolloutWorkerRuntime, build_rollout_runtime
from .worker import RolloutWorker


if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
    from .sglang import SGLangWorker
if os.environ.get("XTUNER_USE_VLLM", "0") == "1":
    from .vllm import vLLMWorker
if os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
    from .lmdeploy import LMDeployWorker
