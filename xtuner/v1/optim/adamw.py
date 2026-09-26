import torch

from .optimizer import Optimizer


class AdamW(torch.optim.AdamW, Optimizer):
    """AdamW with the shared state-device interface.

    Training, constructor arguments, and checkpoint schema match ``torch.optim.AdamW``. The shared base contributes
    ``put_state_to_device`` and does not override initialization.
    """
