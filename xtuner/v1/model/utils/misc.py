import json
from pathlib import Path
from typing import Any

import torch
from mmengine.dist import master_only


def module_dict_repr(self):
    """Return a custom repr for ModuleList that compresses repeated module
    representations."""

    def _addindent(s_, numSpaces):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    list_of_reprs = [repr(item) for item in self.values()]
    if len(list_of_reprs) == 0:
        return self._get_name() + "()"

    start_end_indices = [[0, 0]]
    repeated_blocks = [list_of_reprs[0]]
    for i, r in enumerate(list_of_reprs[1:], 1):
        if r == repeated_blocks[-1]:
            start_end_indices[-1][1] += 1
            continue

        start_end_indices.append([i, i])
        repeated_blocks.append(r)

    lines = []
    main_str = self._get_name() + "("
    for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
        local_repr = f"({start_id}): {b}"  # default repr

        if start_id != end_id:
            n = end_id - start_id + 1
            local_repr = f"({start_id}-{end_id}): {n} x {b}"

        local_repr = _addindent(local_repr, 2)
        lines.append(local_repr)

    main_str += "\n  " + "\n  ".join(lines) + "\n"
    main_str += ")"
    return main_str


class ModelForwardExtraLogInfo(dict):
    """An extensible dictionary for carrying extra information in the model's
    output.

    In the Reinforcement Learning (RL) training process, this information will be processed by the `TrainingWorker`.
    In the SFT/Pretraining process, this information will be processed by the `Trainer`.
    """

    # Tensor to store the maximum model params update ratio.
    # Shape: `(n_chunk, intra_layer_micro_batch, 1)` if intra_layer_micro_batch > 1 else `(n_chunk, 1)`
    max_ratio: torch.Tensor
    # Tensor to store the ranking loss for logging.
    # Shape: `(intra_layer_micro_batch, 1)` if intra_layer_micro_batch > 1 else `(1,)`
    local_base_loss: torch.Tensor

    def __init__(self, init_dict: dict[str, Any] = {}):
        super().__init__()
        if init_dict:
            for k, v in init_dict.items():
                self[k] = v

    def append(self, input_info: dict[str, Any]):
        for key, tensor in input_info.items():
            # 统一处理为 2D 张量后在第 0 维拼接
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(0)
            if key in self:
                self[key] = torch.cat([self[key], tensor], dim=0)
            else:
                self[key] = tensor

    def get(self):
        return_dict = {}
        # 当增加新的字段时，需要在这里添加相应的处理逻辑
        if "max_ratio" in self:
            while self["max_ratio"].dim() >= 1:
                self["max_ratio"] = torch.max(self["max_ratio"], dim=-1).values
            max_ratio_value = self["max_ratio"].item()
            return_dict["max_ratio"] = max_ratio_value
        if "local_base_loss" in self:
            while self["local_base_loss"].dim() >= 1:
                self["local_base_loss"] = torch.sum(self["local_base_loss"], dim=-1)
            local_base_loss_value = self["local_base_loss"].item()
            # vague keys such as `loss` should be avoided in extra_log_info,
            # otherwise it may cause confusion in exp-track logs.
            return_dict["local_base_loss"] = local_base_loss_value
        return return_dict


@master_only
def update_weight_map_from_safetensors_index(weight_map: dict[str, str], hf_dir: Path | str):
    if not isinstance(hf_dir, Path):
        hf_dir = Path(hf_dir)
    with open(hf_dir / "model.safetensors.index.json") as f:
        weight_map.update(json.load(f)["weight_map"])
