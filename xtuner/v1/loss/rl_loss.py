from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.rl.utils import gather_logprobs
from xtuner.v1.utils.device import get_device

from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs


DEVICE = get_device()


class LogProbConfig(BaseLossConfig):
    @property
    def loss_ctx_cls(self) -> type["LogProbContext"]:
        return LogProbContext

    def build(self, shifted_labels: torch.Tensor, sp_mesh: DeviceMesh | None = None) -> "LogProbContext":
        loss_kwargs = LogProbKwargs(shifted_labels=shifted_labels)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        return self.loss_ctx_cls(self, loss_kwargs)


class LogProbKwargs(BaseLossKwargs):
    shifted_labels: torch.Tensor


class LogProbContext(BaseLossContext):
    loss_cfg: LogProbConfig
    loss_kwargs: LogProbKwargs

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["LogProbContext"], *args: Any, **kwargs: Any
    ) -> list["LogProbContext"]:
        del args, kwargs
        batch_size = len(loss_ctx_list)
        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = batch_size
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: LogProbKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        logits = F.linear(hidden_states, head_weight, head_bias).float()
        logprobs = gather_logprobs(logits, loss_kwargs.shifted_labels)
        return logprobs, (None, {})

    def chunk_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: LogProbKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        assert self.loss_cfg.chunk_size is not None, "chunk_size must be set in chunk mode"

        bs, seq_len = loss_kwargs.shifted_labels.shape
        logprobs = torch.zeros((bs, seq_len), device=loss_kwargs.shifted_labels.device)
        for i in range(0, seq_len, self.loss_cfg.chunk_size):
            hidden_states_chunk = hidden_states[:, i : i + self.loss_cfg.chunk_size, :]
            logits = F.linear(hidden_states_chunk, head_weight, head_bias).float()
            chunked_labels = loss_kwargs.shifted_labels[:, i : i + self.loss_cfg.chunk_size]
            chunked_logprobs = gather_logprobs(logits, chunked_labels)
            logprobs[:, i : i + self.loss_cfg.chunk_size] = chunked_logprobs
        return logprobs, (None, {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        assert self.loss_kwargs is not None, "loss_kwargs must be set before calling forward"
        if self.loss_cfg.mode == "chunk":
            logprobs, _ = self.chunk_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)
        else:
            logprobs, _ = self.eager_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)
        return logprobs, (None, {})
