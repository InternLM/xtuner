import os

from .controller import RolloutController
from .endpoint import RolloutEndpoint, RolloutEndpointConfig
from .health_manager import RolloutHealthManager, RolloutWorkerRouteInfo
from .worker_extern_router import (
    WorkerExternRouter,
    WorkerExternRouterConfig,
)
from .worker_http_router import (
    WorkerHttpRouter,
    WorkerHttpRouterConfig,
    build_worker_http_router_app,
    serve_worker_http_router_in_thread,
)
from .worker_local_router import (
    WorkerLocalRouter,
    WorkerLocalRouterConfig,
)
from .worker import RolloutWorker


if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
    from .sglang import SGLangWorker
if os.environ.get("XTUNER_USE_VLLM", "0") == "1":
    from .vllm import vLLMWorker
if os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
    from .lmdeploy import LMDeployWorker

from .utils import continue_generation, pause_generation
