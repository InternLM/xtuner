import socket
from typing import TYPE_CHECKING, Tuple, cast

import ray
import torch


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


@ray.remote
def find_master_addr_and_port() -> Tuple[str, int]:
    """自动找到一个可用的端口号."""
    addr = ray.util.get_node_ip_address()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # 绑定到任意可用端口
        s.listen(1)
        port = s.getsockname()[1]  # 获取分配的端口号
    return addr, port


@ray.remote
def get_accelerator_ids(accelerator: str) -> list:
    """Get the IDs of the available accelerators (GPUs, NPUs, etc.) in the Ray
    cluster."""
    return ray.get_runtime_context().get_accelerator_ids()[accelerator]
