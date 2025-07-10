import torch
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.utils import maybe_compile


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


@maybe_compile(fullgraph=True)
def cast_to_per_block_fp8(
    tensor: torch.Tensor, scales: torch.Tensor, block_size=128, float8_dtype=torch.float8_e4m3fn
):
    dout, din = tensor.shape

    if dout < block_size:
        tensor = tensor.view(dout, din // block_size, block_size).transpose(0, 1).reshape(din // block_size, -1)
    else:
        tensor = (
            tensor.view(dout // block_size, block_size, din // block_size, block_size)
            .transpose(1, 2)
            .reshape(-1, block_size * block_size)
        )
    scales = scales.view(-1, 1)

    # cast to fp8
    tensor_scaled = tensor.to(torch.float32) / scales
    tensor_bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)

    if dout < block_size:
        tensor_bits_fp8 = tensor_bits_fp8.view(din // block_size, dout, block_size).transpose(0, 1).reshape(dout, din)
    else:
        tensor_bits_fp8 = (
            tensor_bits_fp8.view(dout // block_size, din // block_size, block_size, block_size)
            .transpose(1, 2)
            .reshape(dout, din)
        )
    return tensor_bits_fp8


@maybe_compile(fullgraph=True)
def cast_to_per_block_fp8_devided_64(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    fsdp_mesh: DeviceMesh,
    block_size: int = 128,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
):
    dout, din = tensor.shape
    assert dout % block_size == 64
    dout_0 = dout // block_size * block_size
    # dout_1 = 64
    rank = fsdp_mesh.get_local_rank()

    if rank % 2 == 0:
        # 最后的 64 个 dim 属于一个 block
        tensor0 = tensor[:dout_0]
        tensor1 = tensor[dout_0:]
    else:
        # 前 64 个 dim 属于一个 block
        tensor0 = tensor[64:]
        tensor1 = tensor[:64]
    tensor0 = (
        tensor0.view(dout_0 // block_size, block_size, din // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    tensor1 = tensor1.view(64, din // block_size, block_size).transpose(0, 1).reshape(din // block_size, -1)

    if rank % 2 == 0:
        tensor0_scales = scales[:-1]
        tensor1_scales = scales[-1:]
    else:
        tensor0_scales = scales[1:]
        tensor1_scales = scales[:1]
    tensor0_scales = tensor0_scales.view(-1, 1)
    tensor1_scales = tensor1_scales.view(-1, 1)

    # cast to fp8
    tensor0_scaled = tensor0.to(torch.float32) / tensor0_scales
    tensor1_scaled = tensor1.to(torch.float32) / tensor1_scales
    tensor0_bits_fp8 = to_fp8_saturated(tensor0_scaled, float8_dtype)
    tensor1_bits_fp8 = to_fp8_saturated(tensor1_scaled, float8_dtype)

    tensor0_bits_fp8 = (
        tensor0_bits_fp8.view(dout_0 // block_size, din // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dout_0, din)
    )
    tensor1_bits_fp8 = tensor1_bits_fp8.view(din // block_size, 64, block_size).transpose(0, 1).reshape(64, din)

    if rank % 2 == 0:
        tensor_bits_fp8 = torch.cat([tensor0_bits_fp8, tensor1_bits_fp8], dim=0)
    else:
        tensor_bits_fp8 = torch.cat([tensor1_bits_fp8, tensor0_bits_fp8], dim=0)

    return tensor_bits_fp8


@maybe_compile(fullgraph=True)
def per_tensor_quant(
    tensor: torch.Tensor,
    float8_dtype=torch.float8_e4m3fn,
):
    amax = tensor.abs().max().to(torch.float64)
    scales = torch.clamp(amax, min=EPS) / torch.finfo(float8_dtype).max
    scales = scales.to(torch.float32)
    tensor_scaled = tensor.to(torch.float32) / scales
    tensor_bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
    return tensor_bits_fp8, scales


@maybe_compile(fullgraph=True)
def cast_to_per_tensor_fp8(tensor: torch.Tensor, scales: torch.Tensor, float8_dtype=torch.float8_e4m3fn):
    # Note: when the line below is compiled with `torch.compile`, `tensor` is automatically
    # upcasted to `float32` to multiply with the scale
    # In order to match numerics between eager and compile, we upcast manually here.
    tensor_scaled = tensor.to(torch.float32) / scales
    tensor_bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
    return tensor_bits_fp8
