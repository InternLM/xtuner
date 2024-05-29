import os
import random
from typing import Optional

import numpy as np
import torch

DEFAULT_SEED_NUMBER = 1234


def set_seed(seed: int = DEFAULT_SEED_NUMBER):
    if seed is None or not isinstance(seed, int):
        seed = DEFAULT_SEED_NUMBER
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # refer to https://pytorch.org/docs/1.13/notes/randomness.html#reproducibility  # noqa: E501
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn_deterministic = True
        torch.backends.cudnn_benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # refer to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility  # noqa: E501
    os.putenv('CUBLAS_WORKSPACE_CONFIG',
              os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))


def expand_reward_token_id(reward_token_id: int,
                           input_ids: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           pad_token_id=0):
    assert len(input_ids.shape) == 2, \
        f'expand_reward_token_id error, len(input_ids.shape()) = {len(input_ids.shape())}'  # noqa: E501
    new_input_ids = torch.zeros((input_ids.shape[0], input_ids.shape[1] + 1),
                                dtype=input_ids.dtype).to(input_ids.device)
    new_attention_mask = torch.zeros_like(
        new_input_ids, dtype=torch.int64).to(input_ids.device)
    for i in range(input_ids.size(0)):
        row = input_ids[i]
        nonzero_index = (row != pad_token_id).nonzero(as_tuple=False)
        if nonzero_index.numel() > 0:
            nonzero_index = nonzero_index[-1] + 1
            new_input_ids[i] = torch.cat(
                (input_ids[i][:nonzero_index],
                 torch.tensor([reward_token_id], dtype=input_ids.dtype).to(
                     input_ids.device), input_ids[i][nonzero_index:]),
                0).to(input_ids.device)
            if attention_mask is not None:
                new_attention_mask[i] = torch.cat(
                    (attention_mask[i][:nonzero_index],
                     torch.tensor([1], dtype=torch.int64).to(
                         input_ids.device), attention_mask[i][nonzero_index:]),
                    0).to(input_ids.device)
        else:
            new_input_ids[i] = torch.cat(
                (input_ids[i][:],
                 torch.tensor([reward_token_id], dtype=input_ids.dtype).to(
                     input_ids.device)), 0).to(input_ids.device)
            if attention_mask is not None:
                new_attention_mask[i] = torch.cat(
                    (attention_mask[i][:], torch.tensor(
                        [1], dtype=torch.int64).to(input_ids.device)),
                    0).to(input_ids.device)

    return new_input_ids, new_attention_mask
