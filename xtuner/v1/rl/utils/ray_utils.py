import atexit
import os
import signal
import subprocess
from collections.abc import Callable, Iterable
from types import FrameType
from typing import TYPE_CHECKING, Optional, cast

import ray
from ray import ObjectRef

from xtuner.v1.utils.logger import get_logger

from .misc import find_free_ports


if TYPE_CHECKING:
    from .ray_accelerator_worker import AcceleratorType


logger = get_logger()
_cleanup_registration: tuple[int, Callable[[int, FrameType | None], None]] | None = None


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


def free_object_refs(refs: ObjectRef | Iterable[ObjectRef]) -> None:
    if isinstance(refs, ObjectRef):
        refs = [refs]

    seen: set[str] = set()
    valid_refs = []
    for ref in refs:
        if not isinstance(ref, ObjectRef):
            continue
        ref_key = ref.hex()
        if ref_key in seen:
            continue
        seen.add(ref_key)
        valid_refs.append(ref)
    if not valid_refs:
        return

    try:
        ray._private.internal_api.free(valid_refs, local_only=False)
    except Exception:
        ray.internal.free(valid_refs, local_only=False)


def close_ray(*, stop_local_cluster: bool = False) -> None:
    """Clean up the ray resource."""
    # 1. Shutdown ray if initialized
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during ray.shutdown(): {e}")

    # Stopping a shared cluster can kill unrelated jobs. Only an explicit
    # owner of a local cluster may request the CLI-wide stop.
    if not stop_local_cluster:
        return

    # 2. Stop ray launched by CLI
    try:
        result = subprocess.run(["ray", "stop", "--force"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(f"Ray stop failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error stopping ray cluster: {e}")


def register_cleanup() -> None:
    """Register once, and reinstall signal handlers after ``ray.init()``.

    Ray replaces SIGTERM during initialization. Repeated registration restores
    our handler while retaining a single cleanup/atexit owner for this process.
    """
    global _cleanup_registration
    if _cleanup_registration is not None and _cleanup_registration[0] == os.getpid():
        handler = _cleanup_registration[1]
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        return
    owner_pid = os.getpid()
    _cleaned = False

    def cleanup_once(outcome: str | None = None) -> None:
        nonlocal _cleaned
        if os.getpid() != owner_pid:
            return  # Inherited atexit callbacks do not own the parent run.
        if not _cleaned:
            _cleaned = True
            # Also protect the atexit path if a later library replaced our
            # handlers: repeated TERM must not interrupt Ray shutdown/reaping.
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            try:
                try:
                    if os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR"):
                        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.cleanup import request_stop_from_environment

                        if not request_stop_from_environment():
                            logger.error("Sandbox stop barrier unconfirmed; final recovery must retry it")
                finally:
                    if os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR"):
                        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.cleanup import guard_ray_shutdown

                        # Keep Ray shutdown on its original thread. An independent
                        # guard recovers abandoned ledgers if it never returns.
                        with guard_ray_shutdown(training_outcome=outcome):
                            close_ray()
                    else:
                        close_ray()
            finally:
                # Ray workers may die before their Trial finally completes.
                # Recover only journaled IDs whose owner locks are now free.
                try:
                    if os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY") or os.environ.get(
                        "XTUNER_SANDBOX_CLEANUP_DIR"
                    ):
                        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.cleanup import cleanup_from_environment

                        if not cleanup_from_environment(finalize=True, training_outcome=outcome):
                            logger.error("Sandbox cleanup incomplete; inspect the run's cleanup-report.json")
                except Exception as exc:
                    logger.error(f"Sandbox cleanup failed ({type(exc).__name__}); retain the run journal")

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        if os.getpid() != owner_pid:
            # A forked helper must still terminate, but must never stop its
            # parent's run or execute the parent's Ray shutdown callback.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        if _cleaned:
            # A repeated TERM/INT must not interrupt the bounded recovery child.
            logger.info(f"Deferring signal {signum} while shutdown cleanup runs")
            return
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_once("interrupted")
        import sys

        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    _cleanup_registration = (os.getpid(), signal_handler)

    atexit.register(cleanup_once)


def bind_train_rollout(
    train_workers,
    rollout_controller,
    rollout_config,
) -> None:
    """Bind the training and rollout workers for updating weights.

    This function retrieves rollout information from the rollout controller
    and distributes it to the training workers, enabling them to update the
    rollout models' weights.

    Args:
        train_workers: A list of training worker actors.
        rollout_controller: The rollout controller actor.
    """
    targets = ray.get(rollout_controller.get_weight_update_targets.remote())  # type: ignore[attr-defined]
    ray.get(
        [
            worker.bind_rollout_weight_update.remote(
                targets=targets,
                rollout_config=rollout_config,
            )
            for worker in train_workers
        ]
    )  # type: ignore[attr-defined]
    return
