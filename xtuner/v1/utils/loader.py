import json
import os
import re
from typing import Literal

import torch
from safetensors import safe_open


def download_model_from_hub(
    model_name_or_path: str,
    from_hub: Literal["huggingface", "modelscope"] = "huggingface",
    cache_dir: str | None = None,
) -> str:
    """Automatically download model from the HUB.

    Note:
        If `model_name_or_path` is a local path, it will return the path
        directly without downloading it again.

    Args:
        model_name_or_path (str): The model name, model path or repo id.
        config (str | None): The config path. Default is None.
        from_hub (str): The model hosting hub, modelscope, or huggingface.
            Default is huggingface.
        cache_dir (str | None):
            The save path when downloading the model. If it is None, it
            will be stored in the default location of the HUB. For
            Huggingface, it's ~/.cache/huggingface/hub, for ModelScope,
            it's ~/.cache/modelscope/hub.
    Returns:
        str: The local path of the model.
    """
    if os.path.isdir(model_name_or_path):
        model_path = model_name_or_path
    elif from_hub == "huggingface":
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=model_name_or_path, cache_dir=cache_dir)
    elif from_hub == "modelscope":
        from modelscope import snapshot_download

        model_path = snapshot_download(model_id=model_name_or_path, cache_dir=cache_dir)
    else:
        # TODO support openxlab
        raise NotImplementedError(
            f"The model does not support downloading from {from_hub}, it only supports `huggingface` and `modelscope`."
        )

    return model_path


class HFCheckpointLoader:
    def __init__(
        self,
        model_path: str,
        cache_dir: str | None = None,
        from_hub: Literal["huggingface", "modelscope"] = "huggingface",
    ):
        self.model_path = download_model_from_hub(model_path, from_hub, cache_dir)

        if "model.safetensors.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.safetensors.index.json")
            self.weight_map = json.load(open(index_json))["weight_map"]
            self.use_safetensors = True
        elif "model.bin.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.bin.index.json")
            self.weight_map = json.load(open(index_json))["weight_map"]
            self.use_safetensors = False
        elif "model.safetensors" in os.listdir(self.model_path):
            with safe_open(os.path.join(self.model_path, "model.safetensors"), framework="pt") as f:
                self.weight_map = dict.fromkeys(f.keys(), "model.safetensors")
            self.use_safetensors = True
        else:
            raise FileNotFoundError

        self._safetensor_cache: dict = {}
        self.current_file = None
        self.buffer = None

    def is_key_exist(self, key: str) -> bool:
        return key in self.weight_map or self._resolve_fused_mtp_expert(key) is not None

    def _resolve_fused_mtp_expert(self, key: str) -> tuple[str, int, str] | None:
        """Resolve only MTP expert aliases in unquantized safetensors sources.

        Qwen3.5 checkpoints can store [experts, 2*intermediate, hidden]
        gate/up and [experts, hidden, intermediate] down tensors while the
        native MTP load spec addresses per-expert matrices.
        """
        if not self.use_safetensors:
            return None
        match = re.fullmatch(r"(mtp\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight", key)
        if match is None:
            return None
        prefix, expert, part = match.groups()
        fused = prefix + (".down_proj" if part == "down" else ".gate_up_proj")
        return (fused, int(expert), part) if fused in self.weight_map else None

    def _load_fused_mtp_expert(self, key: str) -> torch.Tensor | None:
        resolved = self._resolve_fused_mtp_expert(key)
        if resolved is None:
            return None
        fused, expert, part = resolved
        path = os.path.join(self.model_path, self.weight_map[fused])
        if path not in self._safetensor_cache:
            self._safetensor_cache[path] = safe_open(path, framework="pt")
        view = self._safetensor_cache[path].get_slice(fused)
        shape = view.get_shape()
        if len(shape) != 3 or expert >= shape[0]:
            raise ValueError(f"Invalid fused MTP expert shape/index for {key}: {shape}")
        if view.get_dtype() not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Fused MTP aliases require unquantized floating weights: {fused}")
        if part == "down":
            return view[expert]
        if shape[1] % 2:
            raise ValueError(f"Fused MTP gate/up dimension must be even: {fused}")
        half = shape[1] // 2
        return view[expert, :half] if part == "gate" else view[expert, half:]

    def load(self, key) -> torch.Tensor | None:
        if key not in self.weight_map:
            return self._load_fused_mtp_expert(key)

        _file = self.weight_map[key]

        if self.use_safetensors:
            fh_cache_key = os.path.join(self.model_path, _file)
            if fh_cache_key not in self._safetensor_cache:
                self._safetensor_cache[fh_cache_key] = safe_open(fh_cache_key, framework="pt")
            fh = self._safetensor_cache[fh_cache_key]
            weight = fh.get_tensor(key)
        else:
            if self.current_file is None:
                self.buffer = torch.load(os.path.join(self.model_path, _file))
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = torch.load(os.path.join(self.model_path, _file))
            # TODO: missing typehint
            weight = self.buffer[key]  # type: ignore

        return weight
