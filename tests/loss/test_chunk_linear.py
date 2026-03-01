import unittest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.loss.chunk_linear import FusedLinearForPPOFunction
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def create_forward_inputs(batch_size, num_tokens, hidden_size, vocab_size, dtype):
    hidden = (
        torch.empty((batch_size, num_tokens, hidden_size), dtype=dtype, device=DEVICE)
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    weight = (
        torch.empty((vocab_size, hidden_size), dtype=dtype, device=DEVICE)
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    labels = torch.randint(0, vocab_size, (batch_size, num_tokens), device=DEVICE)
    return hidden, weight, labels


def run_torch_linear(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, temperature: float, reduction="none"
) -> list[torch.Tensor]:
    orig_ndim = hidden.ndim
    assert orig_ndim in (2, 3), f"Invalid hidden shape, received {hidden.shape}"
    if orig_ndim == 3:
        orig_batch_size = hidden.shape[0]
        hidden = hidden.flatten(0, 1)
        labels = labels.flatten(0, 1)

    weight = weight.transpose(0, 1).to(torch.float32)
    logits = torch.matmul(hidden, weight)  # [num_tokens, vocab_size]
    logits /= temperature
    pd = torch.nn.functional.softmax(logits, dim=-1)  # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1)  # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1)  # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)  # [num_tokens]
    logprobs = torch.neg(logprobs)

    if orig_ndim == 3:
        logprobs = logprobs.view(orig_batch_size, -1)
        entropy = entropy.view(orig_batch_size, -1)
    return logprobs, entropy


def run_chunk_linear(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
):
    chunk_size = 512
    logprobs, entropy = FusedLinearForPPOFunction.apply(
        hidden,
        weight,
        labels,
        temperature,
        chunk_size,
    )
    return logprobs, entropy


def create_backward_inputs(batch_size, num_tokens, dtype):
    g_entropy = torch.empty((batch_size, num_tokens,), dtype=dtype, device=DEVICE).uniform_(-0.5, 0.5)
    g_logprobs = torch.empty((batch_size, num_tokens,), dtype=dtype, device=DEVICE).uniform_(-1, 1)
    return g_entropy, g_logprobs


# Adapted from https://github.com/verl-project/verl/blob/b178a3cd56f58b2d502255daba591e92833930a7/tests/utils/test_linear_cross_entropy.py#L164
class TestChunkLinear(DistributedTestBase):
    def test_chunk_linear(self):
        # 1. forward
        # create
        # B, T, D, V = 1, 1024, 896, 152000
        B, T, D, V = 2, 3688, 4193, 122065
        dtype = torch.float32
        temperature = 1.0
        #[B,T,D] [V,D] [B,T]
        hidden, weight, labels = create_forward_inputs(B, T, D, V, dtype)
        print(f"hidden: {hidden.shape}, weight: {weight.shape}, labels: {labels.shape}")
        (ref_logprobs, ref_entropy) = run_torch_linear(hidden, weight, labels, temperature)
        print(f"ref_logprobs: {ref_logprobs.shape}, ref_entropy: {ref_entropy.shape}")

        # operate
        chunk_logprobs, chunk_entropy = run_chunk_linear(hidden, weight, labels, temperature)
        print(f"chunk_logprobs: {chunk_logprobs.shape}, chunk_entropy: {chunk_entropy.shape}")

        # check
        torch.testing.assert_close(ref_logprobs, chunk_logprobs)  # , atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ref_entropy, chunk_entropy)  #, atol=1e-4, rtol=1e-4)
        print("test_chunk_linear::forward passed!")
    
        # 2. backward
        # create
        g_entropy, g_logprobs = create_backward_inputs(B, T, dtype)
        print(f"g_entropy: {g_entropy.shape}, g_logprobs: {g_logprobs.shape}")
        (d_ref_hidden, d_ref_weight) = torch.autograd.grad(
            (ref_entropy, ref_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
        )

        # operate
        (d_chunk_hidden, d_chunk_weight) = torch.autograd.grad(
            (chunk_entropy, chunk_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
        )

        # check
        torch.testing.assert_close(d_ref_hidden, d_chunk_hidden, atol=1e-2, rtol=1e-4)
        torch.testing.assert_close(d_ref_weight, d_chunk_weight, atol=1e-2, rtol=1e-4)
        print("test_chunk_linear::backward passed!")
    

if __name__ == "__main__":
    unittest.main()
