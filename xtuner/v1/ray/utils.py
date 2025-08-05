import importlib
import inspect
import multiprocessing
import socket
import time
from typing import TYPE_CHECKING, cast

import ray
import requests  # type: ignore[import-untyped]
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import create_model


if TYPE_CHECKING:
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


def openai_server_api(custom_function, server_addr, server_port, route="/compute_reward"):
    sig = inspect.signature(custom_function)
    fields = {}
    for k, v in sig.parameters.items():
        ann = v.annotation
        default = v.default if v.default is not inspect.Parameter.empty else ...
        fields[k] = (ann, default)
    InputModel = create_model("InputModel", **fields)
    OutputModel = create_model("OutputModel", result=(sig.return_annotation, ...))
    app = FastAPI()

    @app.post(route, response_model=OutputModel)
    def api(req: InputModel):
        args = [getattr(req, k) for k in sig.parameters.keys()]
        result = custom_function(*args)
        return {"result": result}

    def run_server():
        uvicorn.run(app, host=server_addr, port=server_port)

    p = multiprocessing.Process(target=run_server)
    p.start()

    timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
    start_time = time.perf_counter()

    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:
            try:
                response = session.get(f"http://{server_addr}:{server_port}/openapi.json")
                # 保证成功启动
                if response.status_code == 200:
                    return p
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)
    p.terminate()
    return None
