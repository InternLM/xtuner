import torch
import torch.nn.functional as F

from xtuner.v1.rl.utils import gather_logprobs


class LogProbContext:
    def __init__(self, chunk_size: int, shifted_labels: torch.Tensor):
        self.chunk_size = chunk_size
        self.shifted_labels = shifted_labels  # (bs,seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ):
        bs, seq_len = self.shifted_labels.shape
        loss = torch.zeros((bs, seq_len), device=self.shifted_labels.device)
        for i in range(0, seq_len, self.chunk_size):
            hidden_states_chunk = hidden_states[:, i : i + self.chunk_size, :]
            logits = F.linear(hidden_states_chunk, head_weight, head_bias).float()
            chunked_label = self.shifted_labels[:, i : i + self.chunk_size]
            chunked_loss = gather_logprobs(logits, chunked_label)
            loss[:, i : i + self.chunk_size] = chunked_loss
        return loss, (None, None)
