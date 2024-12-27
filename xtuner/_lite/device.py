import torch


def get_device():
    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        try:
            import torch_npu  # noqa: F401
            device = 'npu'
        except ImportError:
            pass

    if device is None:
        raise NotImplementedError(
            'Supports only CUDA or NPU. If your device is CUDA or NPU, '
            'please make sure that your environmental settings are '
            'configured correctly.')

    return device


def get_torch_device_module():

    device = get_device()
    if device == 'cuda':
        return torch.cuda
    elif device == 'npu':
        return torch.npu
    else:
        raise NotImplementedError
