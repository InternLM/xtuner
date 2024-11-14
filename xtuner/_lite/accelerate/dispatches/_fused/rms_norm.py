# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from torch.distributed._tensor import DTensor
from xtuner._lite.accelerate import lmdeploy_is_available, npu_is_available


def rms_norm_forward(self, hidden_states):

    from torch.distributed._functional_collectives import AsyncCollectiveTensor
    if isinstance(hidden_states, AsyncCollectiveTensor):
        hidden_states = hidden_states.wait()
    if (hidden_states.device == torch.device('cpu')
            or self.weight.device == torch.device('cpu')):
        raise RuntimeError(
            'Can not use triton kernels on cpu. Please set `USE_TRITON_KERNEL`'
            ' environment variable to 0 before training.')
    
    if isinstance(self.weight, DTensor):
        weight = self.weight.full_tensor()
    else:
        weight = self.weight

    if lmdeploy_is_available() and not self.training:
        from lmdeploy.pytorch.kernels import rms_norm
        ret = rms_norm(hidden_states, weight, eps=self.variance_epsilon)
    elif is_flash_attn_2_available():
        # from flash_attn.ops.triton.layer_norm import rms_norm_fn
        try:
            from flash_attn.ops.triton.layernorm import rms_norm_fn
        except ImportError:
            try:
                from flash_attn.ops.triton.layer_norm import rms_norm_fn
            except ImportError:
                import flash_attn
                raise ImportError(f'flash_attn version {flash_attn.__version__}')
        ret = rms_norm_fn(
            hidden_states, weight, None, eps=self.variance_epsilon)

    elif npu_is_available():
        import torch_npu
        ret = torch_npu.npu_rms_norm(
            hidden_states, weight, epsilon=self.variance_epsilon)[0]
    return ret
