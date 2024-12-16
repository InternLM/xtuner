# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor

def layer_norm_forward(self, hidden_states):
    
    if isinstance(self.weight, DTensor):
        weight = self.weight.full_tensor()
    else:
        weight = self.weight

    if isinstance(self.bias, DTensor):
        bias = self.bias.full_tensor()
    else:
        bias = self.bias

    if isinstance(hidden_states, DTensor):
        hidden_states = hidden_states.full_tensor()
    else:
        hidden_states = hidden_states

    return F.layer_norm(
            hidden_states, self.normalized_shape, weight, bias, self.eps
        )