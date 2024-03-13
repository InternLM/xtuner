import torch
import torch.distributed as dist

from .setup_distributed import get_sequence_parallel_group


def reduce_sequence_parallel_loss(mean_loss, num_tokens_for_loss):
    sequence_parallel_group = get_sequence_parallel_group()
    if num_tokens_for_loss == 0:
        # convert nan to 0 just for logging
        mean_loss = torch.nan_to_num(mean_loss)
    loss_sum = mean_loss * num_tokens_for_loss
    dist.all_reduce(loss_sum, group=sequence_parallel_group)
    dist.all_reduce(num_tokens_for_loss, group=sequence_parallel_group)

    loss = loss_sum / num_tokens_for_loss
    return loss
