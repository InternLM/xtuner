import torch
import random
from xtuner.v1.ops import grouped_gemm_triton


def grouped_gemm_torch(x, w, tokens_per_expert):
    """Grouped matrix multiplication (GMM) for expert models using PyTorch.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, din).
        w (Tensor): Weight tensor of shape (num_experts, dout, din).
        tokens_per_expert (Tensor): Number of tokens per expert.

    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, dout).
    """
    outs = []
    start = 0
    for i, tokens in enumerate(tokens_per_expert):
        end = start + tokens
        outs.append(torch.matmul(x[start:end], w[i].T))
        start = end
    return torch.cat(outs)

def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    lst = [0] * length
    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, 2 * int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return lst


def row_max_normalization(tensor):
    row_maxs = tensor.abs().max(dim=-1).values + 1e-9
    tensor_normalized = tensor / row_maxs.unsqueeze(dim=-1)
    return tensor_normalized


def test_grouped_gemm_triton():
    groups = 128
    tokens_per_expert = torch.Tensor(generate_random_list(groups, groups * 4096)).cuda().to(torch.int64).abs()
    seqlen = tokens_per_expert.sum().item()
    for dim0, dim1 in ((768 * 2, 2048), (2048, 768), (1536 * 2, 4096), (4096, 1536)):
        torch.cuda.empty_cache()
        x = torch.randn(seqlen, dim0, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(groups, dim1, dim0, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)
        w_ref = w.clone().detach().requires_grad_(True)
        out_ref = grouped_gemm_torch(x_ref, w_ref, tokens_per_expert)
        out = grouped_gemm_triton(x, w, tokens_per_expert)
        out.mean().backward()
        out_ref.mean().backward()
        assert torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2), "Output mismatch between Triton and PyTorch implementations"
        assert torch.allclose(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2), "Gradient mismatch for input tensor"
        assert torch.allclose(w.grad, w_ref.grad, rtol=1e-2, atol=1e-2), "Gradient mismatch for weight tensor"
