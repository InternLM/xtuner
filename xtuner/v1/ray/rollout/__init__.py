import os
from .worker import RolloutWorker
if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
    from .sglang import SGLangWorker
if os.environ.get("XTUNER_USE_VLLM", "0") == "1":
    from .vllm import vLLMWorker
