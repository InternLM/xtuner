import asyncio
import importlib
import socket
from asyncio import AbstractEventLoop, Task
from typing import TYPE_CHECKING, Callable, Coroutine, List, Optional, cast

import ray


if TYPE_CHECKING:
    import ray.actor

    from xtuner.v1.ray.accelerator import AcceleratorType


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


def load_function(path):
    """Load a function from a module.

    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


@ray.remote
def find_master_addr_and_port(nums=1):
    """Finds an available master address and a specified number of ports.

    This remote function gets the node's IP address and binds to one or more
    available ports, which can be used for distributed communication.

    Args:
        nums (int): The number of ports to find. Defaults to 1.

    Returns:
        A tuple containing the address and a single port if `nums` is 1,
        or a list of ports if `nums` is greater than 1.
    """
    addr = ray.util.get_node_ip_address()
    ports = []
    sockets = []
    try:
        for _ in range(nums):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            s.listen(1)
            ports.append(s.getsockname()[1])
            sockets.append(s)
    finally:
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


def handle_task_exception(task: Task):
    """Handles exceptions from an asyncio Task.

    This function checks if a task has raised an exception and, if so,
    re-raises it. It ignores `asyncio.CancelledError`.

    Args:
        task (Task): The asyncio task to check for exceptions.

    Raises:
        Exception: The exception raised by the task.
    """
    try:
        exc = task.exception()
        if exc is not None:
            raise exc
    except asyncio.CancelledError:
        pass  # Task was cancelled, ignore


def create_task(
    coro: Coroutine,
    loop: Optional[AbstractEventLoop] = None,
    done_callbacks: Optional[List[Callable[[Task], object]]] = None,
) -> asyncio.tasks.Task:
    """Creates and configures an asyncio Task.

    This function creates a task from a coroutine and attaches specified
    done callbacks. By default, it includes a callback to handle exceptions.

    Args:
        coro (Coroutine): The coroutine to wrap in a task.
        loop (Optional[AbstractEventLoop], optional): The event loop to run
            the task in. If None, the current event loop is used.
            Defaults to None.
        done_callbacks (Optional[List[Callable[[Task], object]]], optional):
            A list of callbacks to add to the task. If None, a default
            exception handler is used. Defaults to None.

    Returns:
        asyncio.tasks.Task: The created asyncio task.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    if done_callbacks is None:
        done_callbacks = [handle_task_exception]
    task = loop.create_task(coro)
    for callback in done_callbacks:
        task.add_done_callback(callback)
    return task
