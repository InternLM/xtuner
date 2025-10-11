from typing import Literal

from .logger import get_logger


try:
    from petrel_client.client import Client
except ImportError:
    Client = None

oss_backends: dict = {"petrel": Client}
oss_backend_instances: dict = {}

logger = get_logger()


def get_oss_backend(backend: Literal["petrel"] = "petrel", **kwargs):
    global oss_backend_instances

    if backend not in oss_backends:
        raise ValueError(f"Unsupported OSS backend: {backend}")

    if backend not in oss_backend_instances:
        if oss_backends[backend] is None:
            raise ImportError(f"{backend} backend import failed, please install it first.")
        logger.info(f"Initializing OSS backend: {backend}, kwargs: {kwargs}")
        oss_backend_instances[backend] = oss_backends[backend](**kwargs)
    return oss_backend_instances[backend]
