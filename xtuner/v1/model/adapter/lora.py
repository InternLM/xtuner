import functools
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel as PydanticBaseModel
from torch.distributed.tensor import DTensor

from xtuner.v1.model.base import _save_file
from xtuner.v1.module.lora_linear.lora_linear import LoraLinear
from xtuner.v1.utils import get_device, get_torch_device_module, profile_time_and_memory
from xtuner.v1.utils.load_spec import LoadSpec


DEVICE_MODULE = get_torch_device_module()
DEVICE = get_device()


class LoraConfig(PydanticBaseModel):
    # 与 peft LoraConfig 对齐的字段
    r: int = 8
    target_modules: Optional[Union[List[str], str]] = None
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"  # "none" | "all" | "lora_only"
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: bool = True
    layers_to_transform: Optional[Union[List[int], int]] = None  # 暂不使用
    layers_pattern: Optional[str] = None  # 暂不使用
    base_model_name_or_path: Optional[str] = None

    def build(self, base_model):
        return LoraModel(base_model, self)

    def save_hf(self, hf_path: str | Path):
        """Save the configuration to a HuggingFace-compatible format.

        Args:
            hf_path (str | Path): Path where the configuration should be saved.
        """
        hf_config = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": self.base_model_name_or_path,
            "bias": self.bias,
            "corda_config": None,
            "eva_config": None,
            "exclude_modules": None,
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": self.init_lora_weights,
            "layer_replication": None,
            "layers_pattern": self.layers_pattern,
            "layers_to_transform": self.layers_to_transform,
            "loftq_config": {},
            "lora_alpha": self.lora_alpha,
            "lora_bias": False,
            "lora_dropout": self.lora_dropout,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": self.modules_to_save,
            "peft_type": "LORA",
            "qalora_group_size": 16,
            "r": self.r,
            "rank_pattern": {},
            "revision": None,
            "target_modules": self.target_modules,
            "target_parameters": None,
            "task_type": "CAUSAL_LM",
            "trainable_token_indices": None,
            "use_dora": False,
            "use_qalora": False,
            "use_rslora": False,
        }
        with open(os.path.join(hf_path, "adapter_config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)


def wrap_to_hf_key_list(obj):
    orig = getattr(obj, "to_hf_key_list")

    @functools.wraps(orig)
    def new_to_hf_key_list(*args, **kwargs):
        out = orig(*args, **kwargs)
        if ".base_layer." in out[0]:
            out[0] = out[0].replace(".base_layer.", ".")
        return out

    setattr(obj, "to_hf_key_list", new_to_hf_key_list)


class LoraModel(nn.Module):
    def __init__(self, model: nn.Module, lora_config: LoraConfig):
        super().__init__()
        self.base_model = model
        self.lora_config = lora_config

        # 1. 冻结整个原模型
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 2. 注入 LoRA
        self._replace_linear_layers(self.base_model, prefix="")

        # 3. 按 config.bias 设置 bias 的 requires_grad
        self._apply_bias_setting()

        # 4. 按 modules_to_save 让特定模块参数仍然可训练（例如 lm_head）
        self._apply_modules_to_save()

        # 5. 修改hf的key mapping
        wrap_to_hf_key_list(self.base_model)
        self.base_model._init_load_spec()

    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self.model, name)

    def _match_target(self, module_name: str) -> bool:
        """与 peft 类似的逻辑：

        - target_modules=None: 所有 nn.Linear 都加 LoRA
        - target_modules=str: 名字中包含该 substring 的模块
        - target_modules=List[str]: 名字中包含任一 substring 的模块
        """
        target = self.lora_config.target_modules
        if target is None:
            return True
        if isinstance(target, str):
            return target in module_name
        return any(t in module_name for t in target)

    def _replace_linear_layers(self, module: nn.Module, prefix: str):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, LoraLinear):
                # 避免重复 wrap
                continue

            if isinstance(child, nn.Linear) and self._match_target(full_name):
                lora_layer = LoraLinear(
                    base_layer=child,
                    rank=self.lora_config.r,
                    alpha=self.lora_config.lora_alpha,
                    lora_dropout=self.lora_config.lora_dropout,
                    init_lora_weights=self.lora_config.init_lora_weights,
                )
                setattr(module, name, lora_layer)
            else:
                self._replace_linear_layers(child, prefix=full_name)

    def _apply_bias_setting(self):
        """
        - "none": 所有 bias 冻结（默认）
        - "all":  所有 bias 可训练
        - "lora_only": 只有挂了 LoRA 的 Linear 的 bias 可训练
        """
        bias_mode = self.lora_config.bias

        if bias_mode == "none":
            # 已经在 init 时全部冻住，无需额外处理
            return

        if bias_mode == "all":
            for module in self.base_model.modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.requires_grad = True
            return

        if bias_mode == "lora_only":
            for module in self.base_model.modules():
                if isinstance(module, LoraLinear) and module.base_layer.bias is not None:
                    module.base_layer.bias.requires_grad = True
            return

        raise ValueError(f"Unknown bias mode: {bias_mode}, expected one of ['none', 'all', 'lora_only']")

    def _apply_modules_to_save(self):
        """modules_to_save 里的模块，即使用了 LoRA 也会直接训练 一般用来保留 lm_head、classification
        head 等。"""
        modules_to_save = self.lora_config.modules_to_save
        if not modules_to_save:
            return

        # 根据模块名匹配
        for module_name, module in self.base_model.named_modules():
            if module_name in modules_to_save:
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def merge_lora(self):
        for module in self.base_model.modules():
            if isinstance(module, LoraLinear):
                module.merge_lora()

    @torch.no_grad()
    def unmerge_lora(self):
        for module in self.base_model.modules():
            if isinstance(module, LoraLinear):
                module.unmerge_lora()

    def trainable_parameters(self):
        params = [(name, param) for name, param in self.named_parameters() if param.requires_grad]
        return params

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            num = param.numel()
            all_params += num
            if param.requires_grad:
                trainable_params += num

        print(
            f"trainable params: {trainable_params} || "
            f"all params: {all_params} || "
            f"trainable: {100 * trainable_params / all_params:.4f}%"
        )

    def fully_shard(self, *args, **kwargs):
        self.base_model = self.base_model.fully_shard(*args, **kwargs)
        return self

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    def to_device(self, device: torch.device | str):
        self.base_model.to_device(device)

    def scale_and_reduce_grad(self):
        self.base_model.scale_and_reduce_grad()

    def set_hf(self, hf_path: str | Path):
        self.lora_config.base_model_name_or_path = str(hf_path)
        self.base_model.set_hf(hf_path)

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16):
        with profile_time_and_memory(f"[Saving HF to {hf_dir} cost]"):
            self._save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def _save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16):
        """Save the hf model to the given directory.

        Args:
            hf_dir (str): The directory to save the model.
            save_dtype (torch.dtype): The dtype to save the model parameters, bfloat16 or float8.
        """
        # if self._hf_path is None and self.config.hf_config is None:
        #     raise NotImplementedError(
        #         "The model is not loaded from Huggingface, and the `hf_config` property is not implemented, so it cannot be saved in Huggingface format."
        #     )

        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)

        DEVICE_MODULE.empty_cache()
        assert save_dtype in [torch.float8_e4m3fn, torch.bfloat16], f"save_dtype {save_dtype} is not supported"

        lora_param = {}
        modules_to_save = set(self.lora_config.modules_to_save or [])

        for name, param in self.state_dict().items():
            if "lora_A." in name or "lora_B." in name:
                lora_param[name] = param
                continue

            if self.lora_config.bias != "none" and name.endswith("bias"):
                if ".base_layer.bias" in name:
                    lora_param[name] = param
                    continue

            if modules_to_save:
                for mod in modules_to_save:
                    if mod in name:
                        lora_param[name] = param

        tensor_list: list[torch.Tensor] = []
        load_spec_list: list[LoadSpec] = []
        name_list: list[str] = []
        for name, param in lora_param.items():
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            base_name = name[11:]  # remove "base_model." prefix
            if "lora_A." in name or "lora_B." in name:
                load_spec = self.base_model.load_spec_mapping.get(base_name)
                base_layer_name = base_name.replace("lora_A.", "base_layer.").replace("lora_B.", "base_layer.")
                base_layer_load_spec = self.base_model.load_spec_mapping.get(base_layer_name)
                hf_name = "base_model." + base_layer_load_spec.hf_keys[0][:-6] + "lora_" + name.split("lora_")[-1]
            else:
                load_spec = self.base_model.load_spec_mapping.get(base_name)
                hf_name = "base_model." + load_spec.hf_keys[0]
            tensor_list.append(local_tensor)
            name_list.append(hf_name)
            load_spec_list.append(load_spec)

        if self.base_model.fsdp_mesh is not None:
            gathered_tensor_list = self.base_model._fsdp_foreach_allgather(tensor_list, load_spec_list)
        else:
            gathered_tensor_list = tensor_list
        gathered_tensor_list = [
            self.base_model.param_to_safetensor(safetensor, name)
            for safetensor, name in zip(gathered_tensor_list, name_list)
        ]

        # Sepreately save fused parameters and others to make sure each saving rank will not save
        # dupilicated keys
        #
        weight_map = {}

        safetensor_name = "adapter_model.safetensors"

        if not dist.is_initialized() or dist.get_rank() == 0:
            # for tie_word_embeddings, we need to make sure each key is only saved once
            unique_name_list = []
            unique_hf_tensor_list = []
            for name, hf_tensor in zip(name_list, gathered_tensor_list):
                if name not in weight_map:
                    unique_name_list.append(name)
                    unique_hf_tensor_list.append(hf_tensor)
                    weight_map[name] = safetensor_name

            _save_file(
                dict(zip(unique_name_list, unique_hf_tensor_list)),
                hf_dir / safetensor_name,
            )
            self.lora_config.save_hf(hf_dir)

        if dist.is_initialized():
            torch.distributed.barrier()

    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        # TODO: load lora weight
        loaded_keys, unloaded_keys, missing_keys = self.base_model.from_hf(hf_path, strict=False)
        return loaded_keys, unloaded_keys, missing_keys
