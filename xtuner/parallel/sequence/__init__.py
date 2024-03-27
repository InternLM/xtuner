# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import init_dist

from .attention import sequence_parallel_wrapper
from .data_collate import (pad_for_sequence_parallel,
                           split_for_sequence_parallel)
from .reduce_loss import reduce_sequence_parallel_loss
from .sampler import SequenceParallelSampler
from .setup_distributed import (get_data_parallel_group,
                                get_data_parallel_rank,
                                get_data_parallel_world_size,
                                get_sequence_parallel_group,
                                get_sequence_parallel_rank,
                                get_sequence_parallel_world_size,
                                init_sequence_parallel)

__all__ = [
    'sequence_parallel_wrapper', 'pad_for_sequence_parallel',
    'split_for_sequence_parallel', 'SequenceParallelSampler',
    'init_sequence_parallel', 'get_sequence_parallel_group',
    'get_sequence_parallel_world_size', 'get_sequence_parallel_rank',
    'get_data_parallel_group', 'get_data_parallel_world_size',
    'get_data_parallel_rank', 'reduce_sequence_parallel_loss', 'init_dist'
]
