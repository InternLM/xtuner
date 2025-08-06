import torch
from mmengine.dist import dist

from xtuner.utils.device import get_device


DEVICE = get_device()


def cal_global_grad_tokens(labels: list[torch.Tensor], sp_mesh=None):
    # calculate global token number which is used for loss scaling
    rank_grad_tokens = torch.tensor(0, dtype=torch.int64, device=DEVICE)
    for label in labels:
        rank_grad_tokens += (label >= 0).sum()
    dist.all_reduce(rank_grad_tokens)
    if sp_mesh:
        # data in different sp ranks are replicated
        global_grad_tokens = rank_grad_tokens / sp_mesh.size()
    else:
        global_grad_tokens = rank_grad_tokens
    return global_grad_tokens
