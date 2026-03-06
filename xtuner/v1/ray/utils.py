import asyncio
import importlib
import socket
from asyncio import AbstractEventLoop, Task
from typing import TYPE_CHECKING, Any, Callable, Coroutine, List, Optional, cast

import ray


if TYPE_CHECKING:
    import ray.actor

    from xtuner.v1.ray.base.accelerator import AcceleratorType


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


def _is_port_available(check_socket: socket.socket, port: int) -> bool:
    try:
        check_socket.bind(("", port))
        check_socket.listen(1)
        return True
    except OSError:
        return False


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


_ASYNCIO_RUN_LOOP: AbstractEventLoop | None = None


def _get_default_asyncio_loop() -> AbstractEventLoop:
    """Get a module-level event loop reused by ``asyncio_run``."""
    global _ASYNCIO_RUN_LOOP
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    if _ASYNCIO_RUN_LOOP is None or _ASYNCIO_RUN_LOOP.is_closed() or _ASYNCIO_RUN_LOOP is not loop:
        _ASYNCIO_RUN_LOOP = loop
    return _ASYNCIO_RUN_LOOP


def asyncio_run(coro: Coroutine, loop: Optional[AbstractEventLoop] = None) -> Any:
    """Synchronously run a coroutine on a shared/explicit event loop.

    This helper is used by `RLColocateTrainer.fit` for rollout collection:
    1) Trainer runs in sync code and repeatedly calls:
       - self.eval_agent_loop_manager.produce_batch(...)
       - self.agent_loop_manager.produce_batch(...)
    2) `produce_batch` is async, and internally runs `ProduceStrategy.produce_batch`,
       which launches many nested async tasks (`create_task`) and ultimately calls
       `AgentLoop.generate_group -> generate_sample`.
    3) In `VerlToolAgentLoop`, `generate_sample` awaits `self.verl_tool_agent_loop.run()`,
       where the tool loop stays on the same loop.

    In this pattern, if sync code uses `asyncio.run` every call, each invocation
    creates/closes a fresh loop, but `VerlToolAgentLoop` keeps internal work
    on one loop, the wrapped `generate_sample -> run -> Ray futures` chain can see
    mismatched loop ownership and trigger:
    ``Future attached to a different loop``.

    `asyncio_run` keeps calls bound to a stable loop instance so nested task/future
    chains stay compatible across repeated rollout phases.

    This helper is for sync-to-async boundaries only and should not be used from
    within an already running event loop.
    """
    if loop is None:
        loop = _get_default_asyncio_loop()
    if loop.is_running():
        raise RuntimeError("asyncio_run does not support being called from a running event loop.")
    return loop.run_until_complete(coro)


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
