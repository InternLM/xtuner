import copy

import torch
import torch.distributed as dist

from .setup_distributed import get_sequence_parallel_group


def rescale_sp_loss(loss_per_sp_rank,
                    labels_per_sp_rank,
                    sp_group: dist.ProcessGroup = None,
                    ignore_index=-100):
    if sp_group is None:
        sp_group = get_sequence_parallel_group()

    if (sp_group is None) or (dist.get_world_size(sp_group) == 1):
        return loss_per_sp_rank

    shift_labels = labels_per_sp_rank[..., 1:].view(-1)
    active_tokens = (shift_labels != ignore_index).long().sum()
    global_active_tokens = copy.deepcopy(active_tokens)
    dist.all_reduce(global_active_tokens, group=sp_group)
    loss_weight = active_tokens / global_active_tokens * dist.get_world_size(
        group=sp_group)

    if active_tokens == 0:
        # convert nan to 0 just for logging
        loss_per_sp_rank = torch.nan_to_num(loss_per_sp_rank)

    return loss_per_sp_rank * loss_weight


def reduce_sp_loss_for_debug(loss_per_sp_rank,
                             labels_per_sp_rank,
                             sp_group: dist.ProcessGroup = None,
                             ignore_index=-100):
    # Reduce loss to check whether the training losses is different
    # when using sp. This function is only used for debugging
    if sp_group is None:
        sp_group = get_sequence_parallel_group()

    if (sp_group is None) or (dist.get_world_size(sp_group) == 1):
        return loss_per_sp_rank

    shift_labels = labels_per_sp_rank[..., 1:].view(-1)
    active_tokens = (shift_labels != ignore_index).long().sum()
    if active_tokens == 0:
        # convert nan to 0 just for logging
        loss_per_sp_rank = torch.nan_to_num(loss_per_sp_rank)

    loss_sum = loss_per_sp_rank * active_tokens
    global_active_tokens = copy.deepcopy(active_tokens)
    dist.all_reduce(loss_sum, group=sp_group)
    dist.all_reduce(global_active_tokens, group=sp_group)
    return loss_sum / global_active_tokens
