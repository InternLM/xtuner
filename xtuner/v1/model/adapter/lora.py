import functools
import json
import os
import re
from concurrent.futures import Future
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field
from safetensors import safe_open
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.model.base import _save_file
from xtuner.v1.model.compose.base import BaseComposeModel
from xtuner.v1.module import LMHead
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.lora_linear.lora_grouped_linear import LoraGroupedLinear
from xtuner.v1.module.lora_linear.lora_linear import LoraLinear
from xtuner.v1.utils import get_torch_device_module, profile_time_and_memory
from xtuner.v1.utils.load_spec import LoadEnum, LoadSpec
from xtuner.v1.utils.loader import download_model_from_hub


DEVICE_MODULE = get_torch_device_module()


class LoraConfig(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=8, gt=0)
    target_modules: list[str] | str | None = None
    lora_alpha: int = Field(default=8, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    modules_to_save: list[str] | None = None
    init_lora_weights: bool = True
    layers_to_transform: list[int] | int | None = None  # 暂不使用
    layers_pattern: str | None = None  # 暂不使用
    base_model_name_or_path: str | None = None

    def build(self, base_model):
        return LoraModel(base_model, self)

    def save_hf(self, hf_path: str | Path):
        """Save the configuration to a HuggingFace-compatible format.

        Args:
            hf_path (str | Path): Path where the configuration should be saved.
        """
        hf_config: dict[str, Any] = {
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
            json.dump(hf_config, f, indent=2, ensure_ascii=False)


class _AdapterCheckpointLoader:
    def __init__(self, adapter_path: Path):
        self._file = safe_open(str(adapter_path), framework="pt")
        self.weight_map = dict.fromkeys(self._file.keys(), adapter_path.name)

    def is_key_exist(self, key: str) -> bool:
        return key in self.weight_map

    def load(self, key: str) -> torch.Tensor | None:
        if key not in self.weight_map:
            return None
        return self._file.get_tensor(key)


class _TensorCheckpointLoader:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors
        self.weight_map = dict.fromkeys(tensors, "adapter_model.safetensors")

    def is_key_exist(self, key: str) -> bool:
        return key in self.tensors

    def load(self, key: str) -> torch.Tensor | None:
        return self.tensors.get(key)


def wrap_to_hf_key_list(obj):
    orig = getattr(obj, "to_hf_key_list")

    @functools.wraps(orig)
    def new_to_hf_key_list(key: str):
        if "base_layer." in key:
            key = key.replace("base_layer.", "")
        out = orig(key)
        # if ".base_layer." in out[0]:
        #     out[0] = out[0].replace(".base_layer.", ".")
        return out

    setattr(obj, "to_hf_key_list", new_to_hf_key_list)


class LoraModel(nn.Module):
    _PEFT_PREFIX = "base_model.model."

    def __init__(self, model: nn.Module, lora_config: LoraConfig):
        super().__init__()
        if isinstance(model, BaseComposeModel):
            raise NotImplementedError("LoRA adapters for compose models are not supported yet")
        self.base_model = model
        self.lora_config = lora_config
        self._loaded_adapter_path: Path | None = None

        self._apply_lora()
        wrap_to_hf_key_list(self.base_model)
        self.base_model._init_load_spec()

    def _apply_lora(self):
        # 1. 冻结整个原模型
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 2. 注入 LoRA
        self._replace_linear_layers(self.base_model, prefix="")

        # 3. 按 config.bias 设置 bias 的 requires_grad
        self._apply_bias_setting()

        # 4. 按 modules_to_save 让特定模块参数仍然可训练（例如 lm_head）
        self._apply_modules_to_save()

        if not any(param.requires_grad for param in self.base_model.parameters()):
            raise ValueError(
                f"No trainable parameters matched target_modules={self.lora_config.target_modules} "
                f"or modules_to_save={self.lora_config.modules_to_save}"
            )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)

    @staticmethod
    def _matches_module_name(module_name: str, candidates: list[str]) -> bool:
        return any(module_name == candidate or module_name.endswith(f".{candidate}") for candidate in candidates)

    def _is_module_to_save(self, module_name: str) -> bool:
        modules_to_save = self.lora_config.modules_to_save or []
        return self._matches_module_name(module_name, modules_to_save)

    @staticmethod
    def _target_candidate_names(module_name: str) -> list[str]:
        candidates = [module_name]
        if module_name.endswith(".fused_w1w3"):
            prefix = module_name.removesuffix("fused_w1w3")
            candidates.extend([prefix + "gate_proj", prefix + "up_proj"])
        elif module_name.endswith(".fused_w2"):
            candidates.append(module_name.removesuffix("fused_w2") + "down_proj")
        return candidates

    def _raw_target_match(self, module_name: str) -> bool:
        target = self.lora_config.target_modules
        if target is None:
            return True
        if isinstance(target, str):
            if target == "all-linear":
                return True
            return re.fullmatch(target, module_name) is not None
        return self._matches_module_name(module_name, target)

    def _match_target(self, module_name: str) -> bool:
        """与 peft 类似的逻辑：

        - target_modules=None: 所有支持的 Linear（不包含 LMHead）
        - target_modules=str: 按正则表达式完整匹配
        - target_modules=List[str]: 匹配完整模块名或模块名后缀
        """
        candidate_names = self._target_candidate_names(module_name)
        target_matches = any(self._raw_target_match(candidate) for candidate in candidate_names)

        if not target_matches or self.lora_config.layers_to_transform is None:
            return target_matches

        layer_indices = self.lora_config.layers_to_transform
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        layer_pattern = self.lora_config.layers_pattern or "layers"
        match = re.search(rf"(?:^|\.){re.escape(layer_pattern)}\.(\d+)(?:\.|$)", module_name)
        return match is not None and int(match.group(1)) in layer_indices

    def _replace_linear_layers(self, module: nn.Module, prefix: str):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (LoraLinear, LoraGroupedLinear)):
                continue

            if self._is_module_to_save(full_name):
                continue

            if self._match_target(full_name) and isinstance(child, nn.Linear):
                if isinstance(child, LMHead):
                    if self.lora_config.target_modules is None:
                        continue
                    raise ValueError("LMHead cannot be wrapped by LoraLinear; add it to modules_to_save instead")
                lora_layer = LoraLinear(
                    base_layer=child,
                    rank=self.lora_config.r,
                    alpha=self.lora_config.lora_alpha,
                    lora_dropout=self.lora_config.lora_dropout,
                    init_lora_weights=self.lora_config.init_lora_weights,
                )
                setattr(module, name, lora_layer)
            elif self._match_target(full_name) and isinstance(child, (GroupedLinear, TileWiseFloat8GroupedLinear)):
                if full_name.endswith(".fused_w1w3") and not self._raw_target_match(full_name):
                    prefix = full_name.removesuffix("fused_w1w3")
                    gate_matched = self._raw_target_match(prefix + "gate_proj")
                    up_matched = self._raw_target_match(prefix + "up_proj")
                    if gate_matched != up_matched:
                        raise ValueError(
                            "fused_w1w3 uses a shared LoRA A matrix; gate_proj and up_proj must be targeted together"
                        )
                lora_layer = LoraGroupedLinear(
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
            for name, param in self.base_model.named_parameters():
                if name.endswith("bias"):
                    param.requires_grad = True
            return

        if bias_mode == "lora_only":
            for module in self.base_model.modules():
                if isinstance(module, (LoraLinear, LoraGroupedLinear)):
                    bias = getattr(module.base_layer, "bias", None)
                    if bias is not None:
                        bias.requires_grad = True
            return

    def _apply_modules_to_save(self):
        """modules_to_save 里的模块，即使用了 LoRA 也会直接训练 一般用来保留 lm_head、classification
        head 等。"""
        modules_to_save = self.lora_config.modules_to_save
        if not modules_to_save:
            return

        for module_name, module in self.base_model.named_modules():
            if self._matches_module_name(module_name, modules_to_save):
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def init_weights(self):
        self.base_model.init_weights()
        self._reset_lora_parameters()

    def _reset_lora_parameters(self):
        for module in self.base_model.modules():
            if isinstance(module, (LoraLinear, LoraGroupedLinear)):
                module.reset_parameters()

    def _lora_modules_by_fsdp_manager(
        self,
    ) -> tuple[list[tuple[FSDPModule, list[LoraLinear | LoraGroupedLinear]]], list[LoraLinear | LoraGroupedLinear]]:
        named_modules = dict(self.base_model.named_modules())
        managed: dict[int, tuple[FSDPModule, list[LoraLinear | LoraGroupedLinear]]] = {}
        standalone: list[LoraLinear | LoraGroupedLinear] = []

        for module_name, module in named_modules.items():
            if not isinstance(module, (LoraLinear, LoraGroupedLinear)):
                continue

            manager = None
            prefix = module_name
            while True:
                candidate = named_modules[prefix]
                if isinstance(candidate, FSDPModule):
                    manager = candidate
                    break
                if not prefix:
                    break
                prefix = prefix.rpartition(".")[0]

            if manager is None:
                standalone.append(module)
            else:
                managed.setdefault(id(manager), (manager, []))[1].append(module)

        return list(managed.values()), standalone

    def _apply_lora_weight_op(self, op_name: Literal["merge_lora", "unmerge_lora"]):
        managed, standalone = self._lora_modules_by_fsdp_manager()
        for manager, modules in managed:
            manager.unshard()
            try:
                for module in modules:
                    getattr(module, op_name)()
            finally:
                manager.reshard()

        for module in standalone:
            getattr(module, op_name)()

    @torch.no_grad()
    def merge_lora(self):
        """Merge LoRA weights into base layers.

        主要用于 RL 权重更新场景：推理引擎目前只支持接收全量参数（base weight + LoRA delta），
        不支持直接加载 LoRA 参数。因此在 weight update 前需要临时 merge，update 后再 unmerge。
        """
        try:
            self._apply_lora_weight_op("merge_lora")
        except BaseException:
            if any(
                module.merged
                for module in self.base_model.modules()
                if isinstance(module, (LoraLinear, LoraGroupedLinear))
            ):
                self._apply_lora_weight_op("unmerge_lora")
            raise

    @torch.no_grad()
    def unmerge_lora(self):
        """Unmerge LoRA weights from base layers.

        与 merge_lora() 配对使用，在 RL 权重更新完成后恢复 LoRA 状态，
        保证训练侧模型可以继续正常训练。
        """
        self._apply_lora_weight_op("unmerge_lora")

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
        if self.base_model.fsdp_config is None or self.base_model.fsdp_config.requires_grad:
            # Some model implementations re-tie or replace parameters while sharding.
            # Restore the adapter trainability policy before the optimizer is built.
            self._apply_bias_setting()
            self._apply_modules_to_save()
        return self

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    def to_device(self, device: torch.device | str):
        self.base_model.to_device(device)

    def scale_and_reduce_grad(self):
        self.base_model.scale_and_reduce_grad()

    def set_hf(self, hf_path: str | Path):
        hf_path_obj = Path(hf_path) if isinstance(hf_path, (str, Path)) else None
        if hf_path_obj is not None and hf_path_obj.is_dir() and (hf_path_obj / "adapter_config.json").is_file():
            with open(hf_path_obj / "adapter_config.json") as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            if base_model_path is None:
                raise ValueError(f"Missing base_model_name_or_path in {hf_path_obj / 'adapter_config.json'}")
        else:
            base_model_path = str(hf_path)

        self.lora_config.base_model_name_or_path = str(base_model_path)
        self.base_model.set_hf(base_model_path)

    def _get_load_spec(self, param_name: str) -> LoadSpec:
        load_spec = self.base_model.load_spec_mapping.get(param_name)
        if load_spec is None:
            raise RuntimeError(f"Parameter {param_name} is missing from the base model load_spec_mapping")
        return load_spec

    @staticmethod
    def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup, dim: int) -> torch.Tensor:
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def _global_hf_keys(load_spec: LoadSpec) -> list[str]:
        if load_spec.load_enum != LoadEnum.FUSED or load_spec.group is None or not dist.is_initialized():
            return load_spec.hf_keys.copy()

        gathered_keys: list[list[str] | None] = [None] * dist.get_world_size(group=load_spec.group)
        dist.all_gather_object(gathered_keys, load_spec.hf_keys, group=load_spec.group)
        return list(chain.from_iterable(keys for keys in gathered_keys if keys is not None))

    def _gather_param(self, param: torch.Tensor, load_spec: LoadSpec) -> torch.Tensor:
        local_tensor = param.to_local() if isinstance(param, DTensor) else param

        if self.base_model.fsdp_mesh is not None and tuple(local_tensor.shape) != tuple(load_spec.shape):
            gathered_tensor = self.base_model._fsdp_foreach_allgather([local_tensor], [load_spec])[0]
        else:
            gathered_tensor = local_tensor

        if (
            load_spec.group is not None
            and dist.is_initialized()
            and load_spec.load_enum in (LoadEnum.FUSED, LoadEnum.SHARD)
        ):
            gathered_tensor = self._all_gather_tensor(gathered_tensor, load_spec.group, load_spec.dim or 0)

        return gathered_tensor.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()

    @staticmethod
    def _split_fused_tensor(tensor: torch.Tensor, load_spec: LoadSpec, hf_keys: list[str]) -> list[torch.Tensor]:
        if len(hf_keys) == 1:
            return [tensor]
        dim = load_spec.dim or 0
        if tensor.shape[dim] % len(hf_keys) != 0:
            raise RuntimeError(
                f"Cannot split {load_spec.name} with shape {tuple(tensor.shape)} into {len(hf_keys)} HF tensors"
            )
        return list(torch.chunk(tensor, len(hf_keys), dim=dim))

    @classmethod
    def _adapter_key(cls, hf_weight_key: str, lora_name: Literal["lora_A", "lora_B"] | None = None) -> str:
        if lora_name is None:
            return cls._PEFT_PREFIX + hf_weight_key
        if not hf_weight_key.endswith("weight"):
            raise RuntimeError(f"LoRA can only be exported for weight tensors, got {hf_weight_key}")
        return cls._PEFT_PREFIX + hf_weight_key[:-6] + f"{lora_name}.weight"

    def _export_lora_module(
        self,
        module_name: str,
        module: LoraLinear | LoraGroupedLinear,
        adapter_state: dict[str, torch.Tensor],
    ) -> None:
        module_name = self.base_model._clean_param_name(module_name)
        a_spec = self._get_load_spec(f"{module_name}.lora_A.weight")
        b_spec = self._get_load_spec(f"{module_name}.lora_B.weight")
        base_spec = self._get_load_spec(f"{module_name}.base_layer.weight")
        base_hf_keys = self._global_hf_keys(base_spec)
        lora_a = self._gather_param(module.lora_A.weight, a_spec)
        lora_b = self._gather_param(module.lora_B.weight, b_spec)

        if isinstance(module, LoraLinear):
            if len(base_hf_keys) != 1:
                raise RuntimeError(f"Expected one HF key for {module_name}, got {base_hf_keys}")
            adapter_state[self._adapter_key(base_hf_keys[0], "lora_A")] = lora_a
            adapter_state[self._adapter_key(base_hf_keys[0], "lora_B")] = lora_b
            return

        num_experts = module.num_routed_experts
        if len(base_hf_keys) % num_experts != 0:
            raise RuntimeError(f"HF keys for {module_name} cannot be divided across {num_experts} experts")
        projections_per_expert = len(base_hf_keys) // num_experts
        if module.out_features % projections_per_expert != 0:
            raise RuntimeError(f"out_features of {module_name} cannot be split across its HF projection keys")

        a_view = lora_a.view(num_experts, module.rank, module.in_features)
        b_view = lora_b.view(num_experts, module.out_features, module.rank)
        b_chunks = b_view.chunk(projections_per_expert, dim=1)
        for expert_idx in range(num_experts):
            for projection_idx in range(projections_per_expert):
                hf_key = base_hf_keys[expert_idx * projections_per_expert + projection_idx]
                adapter_state[self._adapter_key(hf_key, "lora_A")] = a_view[expert_idx].contiguous()
                adapter_state[self._adapter_key(hf_key, "lora_B")] = b_chunks[projection_idx][expert_idx].contiguous()

    def _adapter_state_dict(self) -> dict[str, torch.Tensor]:
        adapter_state: dict[str, torch.Tensor] = {}
        for module_name, module in self.base_model.named_modules():
            if isinstance(module, (LoraLinear, LoraGroupedLinear)):
                self._export_lora_module(module_name, module, adapter_state)

        for param_name, param in self.base_model.named_parameters():
            param_name = self.base_model._clean_param_name(param_name)
            if "lora_A." in param_name or "lora_B." in param_name or not param.requires_grad:
                continue

            load_spec = self._get_load_spec(param_name)
            hf_keys = self._global_hf_keys(load_spec)
            gathered_param = self._gather_param(param, load_spec)
            for hf_key, tensor in zip(hf_keys, self._split_fused_tensor(gathered_param, load_spec, hf_keys)):
                adapter_state[self._adapter_key(hf_key)] = tensor.contiguous()

        return adapter_state

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16):
        with profile_time_and_memory(f"[Saving HF to {hf_dir} cost]"):
            self._save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def _save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16):
        """Save a PEFT-compatible adapter checkpoint."""
        if isinstance(hf_dir, str):
            hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(DEVICE_MODULE, "empty_cache"):
            DEVICE_MODULE.empty_cache()
        assert save_dtype in [torch.float8_e4m3fn, torch.bfloat16], f"save_dtype {save_dtype} is not supported"
        adapter_state = self._adapter_state_dict()

        if not dist.is_initialized() or dist.get_rank() == 0:
            _save_file(adapter_state, hf_dir / "adapter_model.safetensors")
            self.lora_config.save_hf(hf_dir)

        if dist.is_initialized():
            torch.distributed.barrier()

    def async_save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16, **_: Any) -> Future[Path]:
        future: Future[Path] = Future()
        try:
            self.save_hf(hf_dir=hf_dir, save_dtype=save_dtype)
            future.set_result(Path(hf_dir))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def destroy_async_hf_resources(self) -> None:
        return

    @staticmethod
    def _resolve_adapter_dir(hf_path: str | Path) -> Path | None:
        resolved_path = Path(download_model_from_hub(str(hf_path)))
        if (resolved_path / "adapter_config.json").is_file() and (
            resolved_path / "adapter_model.safetensors"
        ).is_file():
            return resolved_path
        return None

    def _load_base_hf(self, hf_path: str | Path, strict: bool) -> tuple[set[str], set[str], set[str]]:
        loaded_keys, unloaded_keys, missing_keys = self.base_model.from_hf(hf_path, strict=False)
        self._reset_lora_parameters()

        unexpected_unloaded = {key for key in unloaded_keys if "lora_A." not in key and "lora_B." not in key}
        unexpected_missing = {key for key in missing_keys if "lora_A." not in key and "lora_B." not in key}
        if strict and (unexpected_unloaded or unexpected_missing):
            raise RuntimeError(
                f"Failed to load base model. unloaded={unexpected_unloaded}, missing={unexpected_missing}"
            )

        self.lora_config.base_model_name_or_path = str(hf_path)
        return loaded_keys, unloaded_keys, missing_keys

    @classmethod
    def _load_adapter_tensor(
        cls, loader: _AdapterCheckpointLoader, hf_key: str, consumed: set[str]
    ) -> torch.Tensor | None:
        standard_key = cls._PEFT_PREFIX + hf_key
        legacy_key = "base_model." + hf_key
        for key in (standard_key, legacy_key):
            tensor = loader.load(key)
            if tensor is not None:
                consumed.add(key)
                return tensor
        return None

    def _load_lora_module_tensors(
        self,
        module_name: str,
        module: LoraLinear | LoraGroupedLinear,
        loader: _AdapterCheckpointLoader,
        consumed: set[str],
        missing: set[str],
    ) -> dict[str, torch.Tensor]:
        module_name = self.base_model._clean_param_name(module_name)
        a_spec = self._get_load_spec(f"{module_name}.lora_A.weight")
        b_spec = self._get_load_spec(f"{module_name}.lora_B.weight")
        base_spec = self._get_load_spec(f"{module_name}.base_layer.weight")
        base_hf_keys = self._global_hf_keys(base_spec)

        if isinstance(module, LoraLinear):
            if len(base_hf_keys) != 1:
                raise RuntimeError(f"Expected one HF key for {module_name}, got {base_hf_keys}")
            a_key = base_hf_keys[0][:-6] + "lora_A.weight"
            b_key = base_hf_keys[0][:-6] + "lora_B.weight"
            lora_a = self._load_adapter_tensor(loader, a_key, consumed)
            lora_b = self._load_adapter_tensor(loader, b_key, consumed)
            if lora_a is None:
                missing.add(self._PEFT_PREFIX + a_key)
            if lora_b is None:
                missing.add(self._PEFT_PREFIX + b_key)
            if lora_a is None or lora_b is None:
                return {}
            return {a_spec.hf_keys[0]: lora_a, b_spec.hf_keys[0]: lora_b}

        num_experts = module.num_routed_experts
        projections_per_expert = len(base_hf_keys) // num_experts
        expert_a: list[torch.Tensor] = []
        expert_b: list[torch.Tensor] = []
        for expert_idx in range(num_experts):
            projection_a: list[torch.Tensor] = []
            projection_b: list[torch.Tensor] = []
            for projection_idx in range(projections_per_expert):
                hf_key = base_hf_keys[expert_idx * projections_per_expert + projection_idx]
                a_key = hf_key[:-6] + "lora_A.weight"
                b_key = hf_key[:-6] + "lora_B.weight"
                lora_a = self._load_adapter_tensor(loader, a_key, consumed)
                lora_b = self._load_adapter_tensor(loader, b_key, consumed)
                if lora_a is None:
                    missing.add(self._PEFT_PREFIX + a_key)
                else:
                    projection_a.append(lora_a)
                if lora_b is None:
                    missing.add(self._PEFT_PREFIX + b_key)
                else:
                    projection_b.append(lora_b)

            if len(projection_a) != projections_per_expert or len(projection_b) != projections_per_expert:
                continue
            if any(not torch.equal(projection_a[0], tensor) for tensor in projection_a[1:]):
                raise RuntimeError(
                    f"{module_name} shares lora_A across fused projections, but the adapter contains different A tensors"
                )
            expert_a.append(projection_a[0])
            expert_b.append(torch.cat(projection_b, dim=0))

        if len(expert_a) != num_experts or len(expert_b) != num_experts:
            return {}
        lora_a = torch.stack(expert_a).reshape(num_experts * module.rank, module.in_features)
        lora_b = torch.stack(expert_b).reshape(num_experts * module.out_features, module.rank)
        return {a_spec.hf_keys[0]: lora_a, b_spec.hf_keys[0]: lora_b}

    def _load_param_from_tensors(
        self, param: torch.Tensor, load_spec: LoadSpec, tensor_loader: _TensorCheckpointLoader
    ) -> list[str]:
        if load_spec.load_enum == LoadEnum.SAME:
            return self.base_model._load_same_hf_param(param, load_spec, tensor_loader)
        if load_spec.load_enum == LoadEnum.FUSED:
            return self.base_model._load_fused_hf_param(param, load_spec, tensor_loader)
        if load_spec.load_enum == LoadEnum.SHARD:
            return self.base_model._load_shard_hf_param(param, load_spec, tensor_loader)
        raise RuntimeError(f"Unsupported load enum {load_spec.load_enum}")

    def _load_adapter(self, adapter_dir: Path, strict: bool) -> tuple[set[str], set[str], set[str]]:
        loader = _AdapterCheckpointLoader(adapter_dir / "adapter_model.safetensors")
        consumed: set[str] = set()
        missing_adapter_keys: set[str] = set()
        internal_tensors: dict[str, torch.Tensor] = {}
        targets: list[tuple[str, torch.Tensor, LoadSpec]] = []

        for module_name, module in self.base_model.named_modules():
            if not isinstance(module, (LoraLinear, LoraGroupedLinear)):
                continue
            internal_tensors.update(
                self._load_lora_module_tensors(
                    module_name, module, loader, consumed=consumed, missing=missing_adapter_keys
                )
            )
            clean_name = self.base_model._clean_param_name(module_name)
            targets.extend(
                [
                    (
                        f"{clean_name}.lora_A.weight",
                        module.lora_A.weight,
                        self._get_load_spec(f"{clean_name}.lora_A.weight"),
                    ),
                    (
                        f"{clean_name}.lora_B.weight",
                        module.lora_B.weight,
                        self._get_load_spec(f"{clean_name}.lora_B.weight"),
                    ),
                ]
            )

        for param_name, param in self.base_model.named_parameters():
            param_name = self.base_model._clean_param_name(param_name)
            if "lora_A." in param_name or "lora_B." in param_name or not param.requires_grad:
                continue
            load_spec = self._get_load_spec(param_name)
            for hf_key in self._global_hf_keys(load_spec):
                tensor = self._load_adapter_tensor(loader, hf_key, consumed)
                if tensor is None:
                    missing_adapter_keys.add(self._adapter_key(hf_key))
                else:
                    internal_tensors[hf_key] = tensor
            targets.append((param_name, param, load_spec))

        tensor_loader = _TensorCheckpointLoader(internal_tensors)
        loaded_params: set[str] = set()
        unloaded_params: set[str] = set()
        for param_name, param, load_spec in targets:
            missing_internal = self._load_param_from_tensors(param, load_spec, tensor_loader)
            if missing_internal:
                unloaded_params.add(param_name)
            else:
                loaded_params.add(param_name)

        unexpected_keys = set(loader.weight_map) - consumed
        if strict and (missing_adapter_keys or unloaded_params or unexpected_keys):
            raise RuntimeError(
                "Failed to load LoRA adapter. "
                f"missing={missing_adapter_keys}, unloaded={unloaded_params}, unexpected={unexpected_keys}"
            )

        self._loaded_adapter_path = adapter_dir
        return loaded_params, unloaded_params, missing_adapter_keys

    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        adapter_dir = self._resolve_adapter_dir(hf_path)
        if adapter_dir is None:
            return self._load_base_hf(hf_path, strict=strict)

        with open(adapter_dir / "adapter_config.json") as f:
            adapter_config = json.load(f)
        if adapter_config.get("r") != self.lora_config.r:
            raise ValueError(
                f"Adapter rank {adapter_config.get('r')} does not match configured rank {self.lora_config.r}"
            )
        if adapter_config.get("lora_alpha") != self.lora_config.lora_alpha:
            raise ValueError(
                "Adapter lora_alpha "
                f"{adapter_config.get('lora_alpha')} does not match configured value {self.lora_config.lora_alpha}"
            )
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_dir / 'adapter_config.json'}")

        relative_base_path = adapter_dir / base_model_path
        if not Path(base_model_path).is_absolute() and relative_base_path.exists():
            base_model_path = str(relative_base_path)

        self._load_base_hf(base_model_path, strict=strict)
        return self._load_adapter(adapter_dir, strict=strict)
