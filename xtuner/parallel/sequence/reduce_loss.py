import torch
import torch.distributed as dist

from .setup_distributed import get_sequence_parallel_group


class _ReduceLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mean_loss, loss_scale, process_group):
        ctx.mode = process_group
        if loss_scale == 0:
            # convert nan to 0 just for logging
            mean_loss = torch.nan_to_num(mean_loss)
        loss_sum = mean_loss * loss_scale
        dist.all_reduce(loss_sum, group=process_group)
        dist.all_reduce(loss_scale, group=process_group)
        loss = loss_sum / loss_scale
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def reduce_sequence_parallel_loss(mean_loss,
                                  loss_scale,
                                  sp_group: dist.ProcessGroup = None):
    if dist.get_world_size(sp_group) == 1:
        return mean_loss
    if sp_group is None:
        # avoid bc breaking
        sp_group = get_sequence_parallel_group()
    return _ReduceLoss.apply(mean_loss, loss_scale, sp_group)
