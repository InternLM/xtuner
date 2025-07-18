import torch


# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):
    """Converts a tensor to a saturated fp8 tensor.

    Note:
        The default behavior in PyTorch for casting to `float8_e4m3fn`
        and `e5m2` is to not saturate. In this context, we should saturate.
        A common case where we want to saturate is when the history of a
        tensor has a maximum value of `amax1`, and the current amax value
        is `amax2`, where `amax1 < amax2`. This is common when using delayed
        scaling.
    """
    if float8_dtype in FP8_TYPES:
        max_value = torch.finfo(float8_dtype).max
        x = x.clamp(min=-max_value, max=max_value)
        return x.to(float8_dtype)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")
