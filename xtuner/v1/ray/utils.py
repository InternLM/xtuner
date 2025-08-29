import asyncio
import importlib
import socket
from asyncio import AbstractEventLoop, Task
from typing import TYPE_CHECKING, Callable, Coroutine, List, Optional, cast

import ray
import torch


if TYPE_CHECKING:
    import ray.actor

    from xtuner.v1.ray.accelerator import AcceleratorType


def get_ray_accelerator() -> "AcceleratorType":
    accelerator = None
    if torch.cuda.is_available():
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
    """自动找到一个可用的端口号."""
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
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(rollout_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get([worker.update_rollout_info.remote(**info_dict) for worker in train_workers])  # type: ignore[attr-defined]
    return


def handle_task_exception(task: Task):
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
    if loop is None:
        loop = asyncio.get_event_loop()
    if done_callbacks is None:
        done_callbacks = [handle_task_exception]
    task = loop.create_task(coro)
    for callback in done_callbacks:
        task.add_done_callback(callback)
    return task
