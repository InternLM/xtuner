import json
import os
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
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
            self.use_safetensors = True
        else:
            raise FileNotFoundError

        self.current_file = None
        self.buffer = None

    def load(self, key):
        if key not in self.weight_map:
            return None

        _file = self.weight_map[key]

        if self.use_safetensors:
            if self.current_file is None:
                self.buffer = safe_open(os.path.join(self.model_path, _file), framework="pt")
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = safe_open(os.path.join(self.model_path, _file), framework="pt")
                self.current_file = _file
            weight = self.buffer.get_tensor(key)

        else:
            if self.current_file is None:
                self.buffer = torch.load(os.path.join(self.model_path, _file))
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = torch.load(os.path.join(self.model_path, _file))

            weight = self.buffer[key]

        return weight
