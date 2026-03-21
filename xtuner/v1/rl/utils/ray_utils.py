import atexit
import signal
import socket
import subprocess
from typing import TYPE_CHECKING, cast

import ray

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils.logger import get_logger

from .misc import _is_port_available


if TYPE_CHECKING:
    from .ray_worker import AcceleratorType


logger = get_logger()


@ray.remote
def find_master_addr_and_port(nums=1, start_port=None, end_port=None):
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
    ports: list[int] = []
    sockets: list[socket.socket] = []

    if start_port is None:
        for _ in range(nums):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # if the port is binded and listened by this socket and then we close it,
            # socket.SO_REUSEADDR would make the port be reusable even it's in TIME_WAIT state.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sockets.append(s)
            if _is_port_available(check_socket=s, port=0):
                ports.append(s.getsockname()[1])
    else:
        assert isinstance(start_port, int), "If start_port isn't None, it must be an integer."
        assert isinstance(end_port, int), "If start_port isn't None, end_port must be an integer."
        assert end_port - start_port >= nums, (
            "If start_port isn't None, the range between start_port and end_port must be at least nums."
        )

        for candidate_port in range(start_port, end_port):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # if the port is binded and listened by this socket and then we close it,
            # socket.SO_REUSEADDR would make the port be reusable even it's in TIME_WAIT state.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sockets.append(s)
            if _is_port_available(check_socket=s, port=candidate_port):
                ports.append(candidate_port)
            # enough ports found
            if len(ports) >= nums:
                break

        if len(ports) < nums:
            raise RuntimeError(f"Could not find {nums} available ports starting from port {start_port} to {end_port}.")

    # close all sockets, no matter available or not
    for s in sockets:
        s.close()

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
    info_dict = ray.get(rollout_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get([worker.update_rollout_info.remote(**info_dict) for worker in train_workers])  # type: ignore[attr-defined]
    return


def fake_collator(instances: list[RolloutState], **kwargs):
    for rollout_state in instances:
        if hasattr(rollout_state, "mm_info") and rollout_state.mm_info is not None:
            pixel_values = rollout_state.mm_info.get("pixel_values", None)
            if pixel_values is not None:
                rollout_state.mm_info["pixel_values"] = ray.put(pixel_values)
    return instances
