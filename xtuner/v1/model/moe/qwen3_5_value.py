from pathlib import Path
from typing import Annotated, Literal

import torch
from typing_extensions import override

from xtuner.v1.model.moe.moe import MoEConfig
from xtuner.v1.module import LMHead
from xtuner.v1.utils import init_params, log_rank0

from .qwen3_5_text import Qwen3_5_VLTextMoE, Qwen3_5_VLTextMoE35BA3BConfig


class Qwen3_5_VLTextMoEValueModel(Qwen3_5_VLTextMoE):
    """Qwen3.5 MoE backbone with an unbounded scalar value head."""

    _LOCAL_VALUE_HEAD_KEY = "lm_head.weight"
    _HF_VALUE_HEAD_KEY = "value_head.weight"

    @override
    def build_head(self, config: MoEConfig) -> LMHead:
        """Build a bias-free scalar head.

        Args:
            config (MoEConfig): Model configuration.

        Returns:
            LMHead: Linear ``hidden_size -> 1`` head with no output activation.
        """
        return LMHead(config.hidden_size, 1, bias=False)

    @override
    def to_hf_key_list(self, key: str) -> list[str]:
        """Map XTuner parameters to Hugging Face checkpoint keys.

        Args:
            key (str): XTuner parameter key.

        Returns:
            list[str]: Corresponding checkpoint keys.
        """
        if key == self._LOCAL_VALUE_HEAD_KEY:
            return [self._HF_VALUE_HEAD_KEY]
        return super().to_hf_key_list(key)

    @override
    def from_hf(
        self, hf_path: str | Path, strict: bool = True
    ) -> tuple[
        Annotated[set[str], "loaded keys"],
        Annotated[set[str], "unloaded keys"],
        Annotated[set[str], "missing keys"],
    ]:
        """Load a Critic checkpoint or initialize from an Actor checkpoint.

        Actor checkpoints contain a vocabulary ``lm_head`` and no scalar value head. The value head maps to its own
        checkpoint key, so the incompatible vocabulary tensor is never loaded. When that key is absent, the scalar
        head is explicitly zero-initialized; trained Critic checkpoints load it normally.

        Args:
            hf_path (str | Path): Hugging Face checkpoint path.
            strict (bool): Whether missing backbone keys should raise an error.

        Returns:
            tuple[set[str], set[str], set[str]]: Loaded local keys, unloaded local keys, and missing checkpoint keys.
        """
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict=False)

        if self._HF_VALUE_HEAD_KEY in missing_keys:
            init_params(self.lm_head.weight, torch.nn.init.zeros_)
            unloaded_keys.discard(self._LOCAL_VALUE_HEAD_KEY)
            missing_keys.discard(self._HF_VALUE_HEAD_KEY)
            log_rank0.info("Initialized missing Critic value head with zeros")

        if strict and missing_keys:
            raise RuntimeError(f"Missing parameters from {hf_path}: {sorted(missing_keys)}")

        return loaded_keys, unloaded_keys, missing_keys


class Qwen3_5_VLTextMoE35BA3BValueConfig(Qwen3_5_VLTextMoE35BA3BConfig):
    """Qwen3.5-VL-MoE-35B-A3B text Critic configuration."""

    mtp_config: None = None
    mesh_prefix: Literal["critic"] = "critic"

    @override
    def build(self) -> Qwen3_5_VLTextMoEValueModel:
        """Build the scalar value model.

        Returns:
            Qwen3_5_VLTextMoEValueModel: Configured Critic language model.
        """
        return Qwen3_5_VLTextMoEValueModel(self)
