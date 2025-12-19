import atexit
import signal
import subprocess
from typing import Any

import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel
from xtuner.v1.utils.logger import get_logger


def sp_split(
    tensor,
    sp_mesh: DeviceMesh,
    split_dim: int,
    padding_value: Any,
):
    tensor = pad_to_multiple_of(tensor, padding_value, sp_mesh.size(), split_dim)
    tensor = split_for_sequence_parallel(tensor, dim=split_dim, sp_mesh=sp_mesh)
    return tensor


def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs


logger = get_logger()


def close_ray():
    """Clean up the ray resource."""
    import ray

    # 1. Shutdown ray if initialized
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during ray.shutdown(): {e}")

    # 2. Stop ray launched by CLI
    try:
        result = subprocess.run(["ray", "stop", "--force"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(f"Ray stop failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error stopping ray cluster: {e}")


def register_cleanup():
    """Register cleanup handlers for Ray on exit and signals."""
    _cleaned = False

    def cleanup_once():
        nonlocal _cleaned
        if not _cleaned:
            _cleaned = True
            close_ray()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_once()
        import sys

        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(cleanup_once)
