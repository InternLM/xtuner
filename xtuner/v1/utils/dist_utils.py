# Copyright (c) OpenMMLab. All rights reserved.
"""Utilities for distributed training: local world size, local-rank checks, and node process groups."""

import datetime
import os
from threading import Lock
from typing import cast

from torch import distributed as dist

from xtuner.v1.utils.device import get_torch_device_module


_LOCK = Lock()
_LOCAL_PROCESS_GROUP: dist.ProcessGroup | None = None


def get_local_world_size() -> int:
    """Return how many parallel processes are assumed to run on this machine.

    Resolution order when ``torch.distributed`` is initialized: environment
    variable ``LOCAL_WORLD_SIZE``, then ``PROC_PER_NODE``, then the accelerator
    device count from :func:`~xtuner.v1.utils.device.get_torch_device_module`.
    When distributed is not initialized, returns ``1``.

    Returns:
        int: The local (per-node) world size used to map global ranks to nodes.
    """
    if dist.is_initialized():
        env = os.getenv("LOCAL_WORLD_SIZE")
        if env is not None:
            return int(env)
        env = os.getenv("PROC_PER_NODE")
        if env is not None:
            return int(env)
        return int(get_torch_device_module().device_count())
    return 1


def is_local_rank0() -> bool:
    """Return whether this process is local rank 0 within its node.

    When ``torch.distributed`` is initialized, this compares
    ``dist.get_rank()`` to :func:`get_local_world_size` using the same
    contiguous stride mapping as :func:`get_local_process_group`. When
    distributed is not initialized, falls back to the ``LOCAL_RANK`` environment
    variable: ``True`` if unset or equal to ``"0"``, ``False`` otherwise.

    Returns:
        bool: ``True`` if this process is the first rank on its node.
    """
    if not dist.is_initialized():
        return True
    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return dist.get_rank() % get_local_world_size() == 0


def get_local_process_group() -> dist.ProcessGroup:
    """Return the process group spanning ranks that belong to this node only.

    Global ranks are split into contiguous blocks of length
    :func:`get_local_world_size`; each block is one node's subgroup. The subgroup
    is created once per interpreter and cached so callers (for example multiple
    dataset instances) reuse the same communicator. Prefer this over a global
    barrier when coordinating node-local filesystem paths such as ``/tmp``.

    Returns:
        dist.ProcessGroup: The cached node-local process group for the current rank.
    """
    global _LOCAL_PROCESS_GROUP
    with _LOCK:
        if _LOCAL_PROCESS_GROUP is None:
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed is not initialized.")
            world_size = dist.get_world_size()
            local_ws = get_local_world_size()
            num_nodes = (world_size + local_ws - 1) // local_ws
            timeout = datetime.timedelta(seconds=1800)
            if "gloo" in (backend := dist.get_backend()):
                group_kwargs: dict = {"backend": backend, "timeout": timeout}
            else:
                group_kwargs = {"timeout": timeout}
            node_id = dist.get_rank() // local_ws
            local_group: dist.ProcessGroup | None = None
            for i in range(num_nodes):
                start = i * local_ws
                end = min(start + local_ws, world_size)
                ranks = list(range(start, end))
                g = dist.new_group(ranks=ranks, **group_kwargs)
                if i == node_id:
                    local_group = g
            _LOCAL_PROCESS_GROUP = local_group
    return cast(dist.ProcessGroup, _LOCAL_PROCESS_GROUP)
