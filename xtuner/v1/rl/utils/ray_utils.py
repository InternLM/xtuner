import atexit
import signal
import subprocess
from typing import TYPE_CHECKING, Optional, cast

import ray
from ray import ObjectRef

from xtuner.v1.utils.logger import get_logger

from .misc import find_free_ports


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RolloutState

    from .ray_worker import AcceleratorType


logger = get_logger()


@ray.remote
def find_master_addr_and_port(
    nums: int = 1, start_port: Optional[int] = None, end_port: Optional[int] = None
) -> tuple[str, int] | tuple[str, list[int]]:
    """Finds an available master address and a specified number of ports.

    This remote function gets the node's IP address and binds to one or more
    available ports, which can be used for distributed communication.

    Args:
        nums (int): The number of ports to find. Defaults to 1.
        start_port (Optional[int]): The starting port to search from.
            If None, random available ports will be used. Defaults to None.
        end_port (Optional[int]): The ending port to search to (exclusive).
            If start_port is None, this parameter is ignored. Defaults to None.

    Returns:
        A tuple containing the address and a single port if `nums` is 1,
        or a list of ports if `nums` is greater than 1.
    """
    addr = ray.util.get_node_ip_address()
    ports = find_free_ports(nums=nums, host="", start_port=start_port, end_port=end_port)

    if len(ports) == 1:
        return addr, ports[0]
    else:
        return addr, ports


@ray.remote
def get_accelerator_ids(accelerator: str) -> list:
    """Get the IDs of the available accelerators (GPUs, NPUs, etc.) in the Ray
    cluster."""
    return ray.get_runtime_context().get_accelerator_ids()[accelerator]


def get_ray_accelerator() -> "AcceleratorType":
    from xtuner.v1.utils.device import get_device

    """Get the type of accelerator available in the Ray environment.

    This function checks for the availability of CUDA and NPU devices and
    returns the corresponding accelerator type.

    Returns:
        AcceleratorType: The type of accelerator ("GPU" or "NPU").

    Raises:
        NotImplementedError: If neither CUDA nor NPU is available.
    """
    accelerator = None
    if get_device() == "cuda":
        accelerator = "GPU"
        return "GPU"
    else:
        try:
            import torch_npu  # noqa: F401

            accelerator = "NPU"
        except ImportError:
            pass

    if accelerator is None:
        raise NotImplementedError(
            "Supports only CUDA or NPU. If your device is CUDA or NPU, "
            "please make sure that your environmental settings are "
            "configured correctly."
        )

    return cast("AcceleratorType", accelerator)


def free_object_refs(refs: list[ObjectRef]) -> None:
    valid_refs = [ref for ref in refs if isinstance(ref, ObjectRef)]
    if not valid_refs:
        return

    try:
        ray._private.internal_api.free(valid_refs, local_only=False)
    except Exception:
        ray.internal.free(valid_refs, local_only=False)


def clear_rollout_response_for_rerun(rollout_state: "RolloutState") -> "RolloutState":
    routed_experts = getattr(rollout_state, "routed_experts", None)
    if isinstance(routed_experts, ObjectRef):
        free_object_refs([routed_experts])
    rollout_state.tokens = getattr(rollout_state, "prompt_ids", None)
    rollout_state.response = None
    rollout_state.response_ids = []
    rollout_state.logprobs = []
    rollout_state.routed_experts = None
    rollout_state.finish_reason = None
    rollout_state.response_mask = []
    rollout_state.response_model_steps = []
    rollout_state.reward = None
    rollout_state.error_msg = None
    return rollout_state


def close_ray():
    """Clean up the ray resource."""
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


def bind_train_rollout(
    train_workers,
    rollout_controller,
) -> None:
    """Bind the training and rollout workers for updating weights.

    This function retrieves rollout information from the rollout controller
    and distributes it to the training workers, enabling them to update the
    rollout models' weights.

    Args:
        train_workers: A list of training worker actors.
        rollout_controller: The rollout controller actor.
    """
    info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())  # type: ignore[attr-defined]
    ray.get([worker.update_rollout_info.remote(**info_dict) for worker in train_workers])  # type: ignore[attr-defined]
    return
