from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xtuner.v1.utils import get_logger
from xtuner.v1.utils.device import get_device


logger = get_logger()

DEVICE = get_device()


def health_job(dtype, loop=10):
    x = torch.rand(128, 128, dtype=dtype, device=DEVICE)
    dist.broadcast(x, src=0)

    y = x
    for _ in range(loop):
        y = F.normalize(y, dim=0)
        torch.matmul(x, y, out=y)
    y = y.mean()
    return y


def check_health(loop=10):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    rtol = 1.6e-2
    atol = 1e-5
    # from torch.testing.assert_close:
    # +---------------------------+------------+----------+
    # | ``dtype``                 | ``rtol``   | ``atol`` |
    # +===========================+============+==========+
    # | :attr:`~torch.float16`    | ``1e-3``   | ``1e-5`` |
    # +---------------------------+------------+----------+
    # | :attr:`~torch.bfloat16`   | ``1.6e-2`` | ``1e-5`` |
    # +---------------------------+------------+----------+
    # | :attr:`~torch.float32`    | ``1.3e-6`` | ``1e-5`` |
    # +---------------------------+------------+----------+

    y = health_job(dtype, loop)

    # gather check
    y_list = [torch.tensor(0.0, dtype=dtype, device=DEVICE) for _ in range(world_size)] if rank == 0 else None
    dist.gather(y, y_list)
    gather_check = torch.tensor(1, dtype=torch.int32, device=DEVICE)
    if rank == 0:
        for i in range(world_size):
            if not torch.allclose(y, y_list[i], rtol=rtol, atol=atol):
                gather_check = torch.tensor(0, dtype=torch.int32, device=DEVICE)
                break
    dist.all_reduce(gather_check, op=dist.ReduceOp.MIN)

    # all reduce check
    z = y.clone()
    dist.all_reduce(z, op=dist.ReduceOp.AVG)
    all_reduce_check = (
        torch.tensor(1, dtype=torch.int32, device=DEVICE)
        if torch.allclose(y, z, rtol=rtol, atol=atol)
        else torch.tensor(0, dtype=torch.int32, device=DEVICE)
    )
    dist.all_reduce(all_reduce_check, op=dist.ReduceOp.MIN)

    if gather_check.item() == 1 and all_reduce_check.item() == 1:
        return True

    if rank == 0:  # log
        logger.error(
            f"Health check failed: gather_check={gather_check.item()}, all_reduce_check={all_reduce_check.item()}. rtol={rtol}, atol={atol}."
        )
        logger.error(f"All reduce check info: y: {y.item()}, z: {z.item()}")

        y2rank = defaultdict(list)
        for ranki, yi in enumerate(y_list):
            y2rank[yi.item()].append(ranki)
        for yi, ranks in y2rank.items():
            logger.error(f"Gather check info: rank {sorted(ranks)}: {yi}")

    return False
