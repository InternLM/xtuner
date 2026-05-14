import os

os.environ.setdefault("FLA_CACHE_RESULTS", "1")
os.environ.setdefault("FLA_USE_CUDA_GRAPH", "0")
os.environ.setdefault("FLA_USE_TMA", "0")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import torch

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
from xtuner.v1.ops.gated_deltanet.chunk_gated_delta_rule import chunk_gated_delta_rule as xtuner_chunk_gated_delta_rule


SEED = 20260514
DEVICE = "cuda"
DTYPE = torch.bfloat16


def make_inputs(cu_dtype: torch.dtype):
    lens = [17, 31, 23]
    cu_seqlens = torch.tensor([0, 17, 48, 71], device=DEVICE, dtype=cu_dtype)
    batch_size, seq_len, num_heads, key_dim, value_dim = 1, sum(lens), 2, 32, 32
    num_states = len(lens)

    q = torch.randn(batch_size, seq_len, num_heads, key_dim, device=DEVICE, dtype=DTYPE).requires_grad_()
    k = torch.randn(batch_size, seq_len, num_heads, key_dim, device=DEVICE, dtype=DTYPE).requires_grad_()
    v = torch.randn(batch_size, seq_len, num_heads, value_dim, device=DEVICE, dtype=DTYPE).requires_grad_()
    g = torch.empty(batch_size, seq_len, num_heads, device=DEVICE, dtype=DTYPE).uniform_(-0.2, -0.01).requires_grad_()
    beta = torch.empty(batch_size, seq_len, num_heads, device=DEVICE, dtype=DTYPE).uniform_(0.05, 0.95)
    beta = beta.requires_grad_()
    # h0 = torch.randn(num_states, num_heads, key_dim, value_dim, device=DEVICE, dtype=DTYPE).requires_grad_()
    h0 = None
    return q, k, v, g, beta, h0, cu_seqlens


def clone_inputs(inputs):
    q, k, v, g, beta, h0, cu_seqlens = inputs
    return (
        q.detach().clone().requires_grad_(),
        k.detach().clone().requires_grad_(),
        v.detach().clone().requires_grad_(),
        g.detach().clone().requires_grad_(),
        beta.detach().clone().requires_grad_(),
        h0.detach().clone().requires_grad_() if h0 is not None else None,
        cu_seqlens.detach().clone(),
    )


def run_case(fn, inputs):
    q, k, v, g, beta, h0, cu_seqlens = inputs
    o, final_state = fn(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    loss = o.float().sum() + 0.125 * final_state.float().sum()
    loss.backward()
    return {
        "o": o.detach(),
        "final_state": final_state.detach(),
        "dq": q.grad.detach(),
        "dk": k.grad.detach(),
        "dv": v.grad.detach(),
        "dg": g.grad.detach(),
        "dbeta": beta.grad.detach(),
        "dh0": h0.grad.detach() if h0 is not None else None,
    }


def assert_close(name: str, expected: torch.Tensor, actual: torch.Tensor):
    if expected is None:
        assert actual is None, f"{name}: expected None but got a tensor"
        print(f"{name:>12}: both are None")
        return
    expected_f32 = expected.float()
    actual_f32 = actual.float()
    torch.testing.assert_close(actual_f32, expected_f32, atol=2e-2, rtol=2e-2)
    max_abs = (actual_f32 - expected_f32).abs().max().item()
    print(f"{name:>12}: max_abs={max_abs:.6g}")


def compare(cu_dtype: torch.dtype):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    inputs = make_inputs(cu_dtype)

    for fn in (xtuner_chunk_gated_delta_rule, fla_chunk_gated_delta_rule):
        run_case(fn, clone_inputs(inputs))
    torch.cuda.synchronize()

    actual = run_case(xtuner_chunk_gated_delta_rule, clone_inputs(inputs))
    expected = run_case(fla_chunk_gated_delta_rule, clone_inputs(inputs))
    torch.cuda.synchronize()

    print(f"\ncu_seqlens dtype: {cu_dtype}")
    for name, expected_tensor in expected.items():
        assert_close(name, expected_tensor, actual[name])


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the gated delta rule eager comparison.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)

    compare(torch.int32)
    compare(torch.int64)
    print("\nOK")


if __name__ == "__main__":
    main()
