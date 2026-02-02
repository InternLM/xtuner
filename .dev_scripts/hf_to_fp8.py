"""Example:

```console
python .dev_scripts/hf_to_fp8.py <bf16-path>  <fp8-path> '(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+.(gate_proj|down_proj|up_proj)|model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj))'
```
"""

import torch
from pathlib import Path
import re
import json
from safetensors import safe_open
from safetensors.torch import save_file

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil


import argparse


def get_args():
    parser = argparse.ArgumentParser(description='bf16 convert fp8')
    parser.add_argument("source", help="source hf model path", type=Path)
    parser.add_argument("target", help="target hf model path", type=Path)
    parser.add_argument('regex', help=r"regex expression to match fp8 weight. For example `model\.layers\.\d+\.mlp\.(down_proj|up_proj|gate_proj)`", type=str)

    args = parser.parse_args()
    return args


FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def _get_min_alignment(size: int, alignment_value: int) -> int:
    return (1 + ((size - 1) // alignment_value)) * alignment_value


def pad_tensor_for_matmul(tensor, dims) -> torch.Tensor:
    assert tensor.dim() == 2
    dim1, dim2 = tensor.shape

    if isinstance(dims, int):
        dims = (dims,)

    dim1_aligned = _get_min_alignment(dim1, 128) if 0 in dims else dim1
    dim2_aligned = _get_min_alignment(dim2, 128) if 1 in dims else dim2

    pad_dim1 = dim1_aligned - dim1
    pad_dim2 = dim2_aligned - dim2

    return torch.nn.functional.pad(tensor, (0, pad_dim2, 0, pad_dim1))


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


@torch.no_grad()
def per_block_quant_torch(tensor: torch.Tensor, block_size=128, float8_dtype=torch.float8_e4m3fn):
    dim0, dim1 = tensor.shape
    tensor_pad = pad_tensor_for_matmul(tensor, (0, 1))
    dim0_pad, dim1_pad = tensor_pad.shape
    tensor_pad = (
        tensor_pad.view(dim0_pad // block_size, block_size, dim1_pad // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    amax = tensor_pad.abs().amax(-1, True)
    amax = amax.to(torch.float64)
    scales = amax / torch.finfo(float8_dtype).max
    scales = scales.to(torch.float32)
    tensor_pad_scaled = tensor_pad.float() / scales
    tensor_pad_bits_fp8 = to_fp8_saturated(tensor_pad_scaled, float8_dtype)
    tensor_pad_bits_fp8 = (
        tensor_pad_bits_fp8.view(dim0_pad // block_size, dim1_pad // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dim0_pad, dim1_pad)
    )
    scales = scales.view(dim0_pad // block_size, dim1_pad // block_size)
    tensor_pad_bits_fp8 = tensor_pad_bits_fp8[:dim0, :dim1]
    return tensor_pad_bits_fp8, scales


def copy_others(source: Path, target: Path):
    for file in source.iterdir():
        if file.name.endswith("safetensors"):
            continue

        if file.name.startswith("."):
            continue

        target_path = target / (file.relative_to(source))
        shutil.copy(file, target_path)


def convert(source: Path, target: Path, pattern: re.Pattern):
    executor = ProcessPoolExecutor(max_workers=16)
    max_parallel_save_worker = 16

    with open(source / "model.safetensors.index.json") as f:
        index = json.load(f)

    target.mkdir(parents=True, exist_ok=True)

    weight_map = index["weight_map"]

    modules_to_not_convert = set()
    modules_to_convert = set()

    for param_name, _ in weight_map.items():
        module_name = param_name.rsplit(".", 1)[0]
        if pattern.search(param_name) is not None:
            modules_to_convert.add(module_name)
        else:
            modules_to_not_convert.add(module_name)

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": list(modules_to_not_convert),
    }

    origin_weigth_map = index.pop("weight_map")
    weight_map = {}
    save_queue = []

    for filename in tqdm(set(origin_weigth_map.values())):
        filepath = source / filename
        safetensor_fh = safe_open(filepath, framework="pt")
        new_safetensors_dict = {}

        for key in safetensor_fh.keys():
            weight_map[key] = filename
            if not pattern.search(key):
                new_safetensors_dict[key] = safetensor_fh.get_tensor(key) 
            else:
                origin_tensor = safetensor_fh.get_tensor(key)
                origin_tensor.cuda()

                fp8_tensor, scale = per_block_quant_torch(origin_tensor, block_size=128, float8_dtype=torch.float8_e4m3fn)

                scale_key = f"{key}_scale_inv"
                new_safetensors_dict[scale_key] = scale
                new_safetensors_dict[key] = fp8_tensor.cpu()

                weight_map[scale_key] = filename

        targetfile = target / filename
        if len(save_queue) <= max_parallel_save_worker:
            executor.submit(save_file, new_safetensors_dict, targetfile)
            # save_file(new_safetensors_dict, filename=targetfile)
    executor.shutdown()
    copy_others(source, target)
    index["weight_map"] = weight_map
    target_model_index_path = target / "model.safetensors.index.json"

    with open(target_model_index_path, "w") as f:
        json.dump(index, f, indent=2)

    with open(source / "config.json") as f:
        hf_config = json.load(f)
        hf_config["quantization_config"] = quantization_config

    target_hf_config_path = target / "config.json"
    with open(target_hf_config_path, "w") as f:
        json.dump(hf_config, f, indent=2)


def main():
    args = get_args()
    source = args.source
    target = args.target
    pattern = re.compile(args.regex)

    convert(source, target, pattern)



if __name__ == "__main__":
    main()
