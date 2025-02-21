# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
from mmengine.dist import infer_launcher, init_dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed.c10d_logger import _exception_logger

from xtuner._lite import get_device

origin_reduce_scatter_tensor = torch.distributed.reduce_scatter_tensor


# mlu's reduce_scatter_tensor do not support ReduceOp.AVG, use ReduceOp.SUM / group_world_size instead.
@_exception_logger
def mlu_reduce_scatter_tensor(
    output, input, op=ReduceOp.SUM, group=None, async_op=False
):
    if op == ReduceOp.AVG:
        result = origin_reduce_scatter_tensor(
            output, input, ReduceOp.SUM, group, async_op
        )
        output.div_(torch.distributed.get_world_size(group))
        return result
    else:
        return origin_reduce_scatter_tensor(output, input, op, group, async_op)


def setup_parallel():
    if not dist.is_initialized():
        dist_launcher = infer_launcher()
        init_dist(dist_launcher)

    device = get_device()

    if device == "mlu":
        torch.distributed.reduce_scatter_tensor = mlu_reduce_scatter_tensor
