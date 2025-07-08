import re

import torch

from xtuner.v1.utils import HFCheckpointLoader, get_logger

from .moe import MoE


logger = get_logger()


class Qwen3MoE(MoE):
    def to_hf_key_list(self, key: str) -> list[str] | str:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

        ep_rank = self.ep_mesh.get_local_rank() if self.ep_mesh is not None else 0
        ep_size = self.ep_mesh.size() if self.ep_mesh is not None else 1

        n_experts_per_rank = self.config.n_routed_experts // ep_size

        if "fused_w1w3.weight" in key:
            w1_keys: list[str] = []
            w3_keys: list[str] = []
            for i in range(ep_rank * n_experts_per_rank, (ep_rank + 1) * n_experts_per_rank):
                w1_keys.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
                w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))

            return w1_keys + w3_keys

        elif "fused_w2.weight" in key:
            w2_keys: list[str] = []
            for i in range(ep_rank * n_experts_per_rank, (ep_rank + 1) * n_experts_per_rank):
                w2_keys.append(key.replace("fused_w2.weight", f"{i}.down_proj.weight"))
            return w2_keys

        elif key.startswith("norm."):
            return key.replace("norm.", "model.norm.")
        else:
            return key

    def from_hf(self, hf_path: str, device: torch.device | None = None, strict=True):
        hf_loader = HFCheckpointLoader(hf_path)

        if device is None:
            # TODO: NPU support (need `get_available_device`)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cur_device = next(iter(self.parameters())).device
        if cur_device == torch.device("meta"):
            self.to_empty(device=device)

        not_matched = []
        not_loaded = []
        loaded = []

        with torch.no_grad():
            for name, value in self.state_dict().items():
                hf_keys = self.to_hf_key_list(name)
                if isinstance(hf_keys, list):
                    hf_values = []
                    for hf_key in hf_keys:
                        _value = hf_loader.load(hf_key)
                        if _value is None:
                            not_loaded.append(f"{name}")
                            logger.warning(f"Parameter {f'{name}'} -> {hf_key} not found in HF checkpoint.")
                            break
                        hf_values.append(_value)
                    hf_value = torch.cat(hf_values, dim=0)

                    if hf_value.shape != value.shape:
                        torch.distributed.breakpoint()
                        not_matched.append(f"{f'{name}'} {hf_value.shape} != {value.shape}")
                        logger.warning(
                            f"Parameter {f'{name}'} shape mismatch: expected {value.shape}, got {hf_value.shape}."
                        )
                        continue
                    value.copy_(hf_value)
                    loaded.extend(hf_keys)
                else:
                    hf_value = hf_loader.load(hf_keys)
                    if hf_value is None:
                        not_loaded.append(f"{name}")
                        torch.distributed.breakpoint()
                        logger.warning(f"Parameter {f'{name}'} -> {hf_keys} not found in HF checkpoint.")
                        continue

                    if hf_value.shape != value.shape:
                        torch.distributed.breakpoint()
                        not_matched.append(
                            f"Parameter {f'{name}'} -> {hf_keys}: {f'{name}'} {hf_value.shape} != {value.shape}"
                        )
                        logger.warning(
                            f"Parameter {f'{name}'} shape mismatch: expected {value.shape}, got {hf_value.shape}."
                        )
                    value.copy_(hf_value)
                    loaded.append(hf_keys)

        missing = set(hf_loader.weight_map) - set(loaded)

        if strict:
            if not_matched:
                raise RuntimeError(f"Some parameters from {hf_path} do not match the model: {not_matched}. ")
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {missing}. ")
