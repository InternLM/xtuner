from typing import Any

import torch
import torch.nn.functional as F
from pydantic import Field
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.rl.utils.misc import gather_logprobs
from xtuner.v1.utils.device import get_device

from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs


DEVICE = get_device()


class LogProbConfig(BaseLossConfig):
    @property
    def loss_ctx_cls(self) -> type["LogProbContext"]:
        return LogProbContext

    @property
    def _loss_kwargs_cls(self) -> type["LogProbKwargs"]:
        return LogProbKwargs

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "LogProbContext | None":
        if "shifted_labels" not in data:
            return None
        loss_kwargs = LogProbKwargs(shifted_labels=data["shifted_labels"])
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
            logprobs, _ = self.loss_fn(hidden_states, head_weight, head_bias, self.loss_kwargs)
        return logprobs, (None, {})


class TopKLogProbConfig(BaseLossConfig):
    """Select model Top-K tokens and compute their exact full-softmax log
    probabilities."""

    top_k: int = Field(gt=0)

    @property
    def loss_ctx_cls(self) -> type["TopKLogProbContext"]:
        return TopKLogProbContext

    @property
    def _loss_kwargs_cls(self) -> type[BaseLossKwargs]:
        return BaseLossKwargs

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "TopKLogProbContext":
        del data, sp_mesh
        return self.loss_ctx_cls(self, self._loss_kwargs_cls())


class TopKLogProbContext(BaseLossContext):
    """Select model Top-K IDs without storing full log-softmax."""

    loss_cfg: TopKLogProbConfig
    loss_kwargs: BaseLossKwargs

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        del loss_kwargs
        logits = F.linear(hidden_states, head_weight, head_bias).float()
        selected_logits, token_ids = torch.topk(logits, k=self.loss_cfg.top_k, dim=-1)
        selected_logprobs = selected_logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return selected_logprobs, (token_ids, {})

    def chunk_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        assert self.loss_cfg.chunk_size is not None, "chunk_size must be set in chunk mode"

        logprob_chunks = []
        token_id_chunks = []
        for hidden_states_chunk in torch.split(hidden_states, self.loss_cfg.chunk_size, dim=1):
            logprobs, (token_ids, _) = self.loss_fn(
                hidden_states_chunk,
                head_weight,
                head_bias,
                loss_kwargs,
            )
            logprob_chunks.append(logprobs)
            token_id_chunks.append(token_ids)
        return torch.cat(logprob_chunks, dim=1), (torch.cat(token_id_chunks, dim=1), {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        if self.loss_cfg.mode == "chunk":
            return self.chunk_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)
        else:
            return self.loss_fn(hidden_states, head_weight, head_bias, self.loss_kwargs)
