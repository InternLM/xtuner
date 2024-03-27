# Copyright (c) OpenMMLab. All rights reserved.
import torch.distributed as dist

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_WORLD_SIZE = None
_DATA_PARALLEL_RANK = None


def init_sequence_parallel(sequence_parallel_size: int = 1):
    assert dist.is_initialized()
    world_size: int = dist.get_world_size()

    # enable_ds_sequence_parallel = sequence_parallel_size > 1
    # if enable_ds_sequence_parallel:
    if world_size % sequence_parallel_size != 0:
        raise RuntimeError(f'world_size ({world_size}) is not divisible by '
                           f'sequence_parallel_size {sequence_parallel_size}')

    num_sequence_parallel_groups: int = world_size // sequence_parallel_size

    rank = dist.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, \
        'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    start_rank = 0
    end_rank = world_size
    for j in range(sequence_parallel_size):
        ranks = range(start_rank + j, end_rank, sequence_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
        group = dist.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    if not dist.is_initialized():
        _SEQUENCE_PARALLEL_WORLD_SIZE = 1
    else:
        _SEQUENCE_PARALLEL_WORLD_SIZE = dist.get_world_size(
            group=get_sequence_parallel_group())
    return _SEQUENCE_PARALLEL_WORLD_SIZE


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    if not dist.is_initialized():
        _SEQUENCE_PARALLEL_RANK = 0
    else:
        _SEQUENCE_PARALLEL_RANK = dist.get_rank(
            group=get_sequence_parallel_group())
    return _SEQUENCE_PARALLEL_RANK


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    global _DATA_PARALLEL_WORLD_SIZE
    if _DATA_PARALLEL_WORLD_SIZE is not None:
        return _DATA_PARALLEL_WORLD_SIZE
    if not dist.is_initialized():
        _DATA_PARALLEL_WORLD_SIZE = 1
    else:
        _DATA_PARALLEL_WORLD_SIZE = dist.get_world_size(
            group=get_data_parallel_group())
    return _DATA_PARALLEL_WORLD_SIZE


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    global _DATA_PARALLEL_RANK
    if _DATA_PARALLEL_RANK is not None:
        return _DATA_PARALLEL_RANK
    if not dist.is_initialized():
        _DATA_PARALLEL_RANK = 0
    else:
        _DATA_PARALLEL_RANK = dist.get_rank(group=get_data_parallel_group())
    return _DATA_PARALLEL_RANK
