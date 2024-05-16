# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def layer_norm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = F.layer_norm(
        hidden_states, (hidden_states.shape[-1], ), eps=self.variance_epsilon)
    hidden_states = self.weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype)
