"""Chunked Triton implementation of ChunkLoss optimized for large vocabularies.

This implementation addresses the following issues:
1. Labels outside the valid range.
2. Unified Buffer (UB) overflow through chunked processing.
3. Missing mask handling.
4. Numerical instability.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function


# ============================================================================
# Chunked implementation for large vocabularies (production path)
# ============================================================================
@triton.jit
def fused_ce_loss_kernel_chunked(
    logits_ptr,
    weight_ptr,
    labels_ptr,
    out_ptr,
    N,
    C,
    VOCAB_CHUNK_SIZE: tl.constexpr,  # Number of classes processed per chunk
    ignore_index: tl.constexpr,
):
    """Compute cross-entropy in vocabulary chunks for vocabularies larger than
    8K.

    The kernel:
    1. Splits the vocabulary dimension into chunks.
    2. Keeps each chunk within the Unified Buffer (UB) capacity.
    3. Computes the softmax maximum and exponential sum chunk by chunk.
    4. Computes the final loss from the accumulated statistics.

    Unified Buffer capacity planning:
    - Each chunk uses ``VOCAB_CHUNK_SIZE * 2`` bytes for FP16 values.
    - A recommended size is 8192 elements, or 16 KiB, which fits within 85 KiB.
    """
    i = tl.program_id(0)
    if i >= N:
        return

    # Load the label and token weight.
    label = tl.load(labels_ptr + i)
    weight = tl.load(weight_ptr + i)

    # Skip tokens marked with ignore_index.
    if label == ignore_index:
        tl.store(out_ptr + i, 0.0)
        return

    # Compute softmax statistics in chunks.
    # Initialize the global maximum and exponential sum.
    global_max = float("-inf")
    global_sum = 0.0

    # Offsets within a vocabulary chunk.
    chunk_offsets = tl.arange(0, VOCAB_CHUNK_SIZE)

    # Process the vocabulary one chunk at a time.
    num_chunks = tl.cdiv(C, VOCAB_CHUNK_SIZE)
    for chunk_id in range(num_chunks):
        chunk_start = chunk_id * VOCAB_CHUNK_SIZE
        chunk_end = tl.minimum(chunk_start + VOCAB_CHUNK_SIZE, C)  # noqa: F841

        # Mask elements beyond the vocabulary size in the final chunk.
        chunk_mask = (chunk_start + chunk_offsets) < C

        # Load logits for the current chunk.
        chunk_logits = tl.load(logits_ptr + i * C + chunk_start + chunk_offsets, mask=chunk_mask, other=float("-inf"))

        # Compute the maximum for the current chunk.
        chunk_max = tl.max(chunk_logits, 0)

        # Update the global maximum.
        new_global_max = tl.maximum(global_max, chunk_max)

        # Update the exponential sum, accounting for a change in the maximum.
        if global_max == float("-inf"):
            # The first chunk has no accumulated sum to rescale.
            chunk_sum = tl.sum(tl.exp(chunk_logits - new_global_max), 0)
        else:
            # Rescale the accumulated sum before adding a later chunk.
            # sum_new = sum_old * exp(old_max - new_max) + sum_chunk
            adjust = tl.exp(global_max - new_global_max)
            chunk_sum = global_sum * adjust + tl.sum(tl.exp(chunk_logits - new_global_max), 0)

        global_max = new_global_max
        global_sum = chunk_sum

    # Compute log(sum(exp(x - max))).
    log_sum_exp = tl.log(global_sum)

    # Locate the chunk containing the target label.
    label_chunk_id = label // VOCAB_CHUNK_SIZE
    label_chunk_start = label_chunk_id * VOCAB_CHUNK_SIZE
    label_offset = label - label_chunk_start

    # Load logits from the chunk containing the target label.
    label_chunk_offsets = tl.arange(0, VOCAB_CHUNK_SIZE)
    label_chunk_mask = (label_chunk_start + label_chunk_offsets) < C

    label_chunk_logits = tl.load(
        logits_ptr + i * C + label_chunk_start + label_chunk_offsets, mask=label_chunk_mask, other=float("0.0")
    )

    # Extract the target logit.
    target_logit = tl.sum(label_chunk_logits * tl.where(label_chunk_offsets == label_offset, 1.0, 0.0), 0)

    # Compute the target log-probability.
    log_prob = (target_logit - global_max) - log_sum_exp

    # Compute the weighted loss.
    loss = -log_prob * weight
    tl.store(out_ptr + i, loss)


class FusedCrossEntropyLoss(Function):
    """Cross-entropy autograd function backed by Triton forward and backward
    kernels."""

    @staticmethod
    def forward(ctx, logits, weight, labels, ignore_index=-100, vocab_chunk_size=4096):
        """Run the fused cross-entropy forward pass.

        Args:
            logits: Logits tensor with shape ``[N, C]``.
            weight: Per-token weight tensor with shape ``[N]``.
            labels: Target label tensor with shape ``[N]``.
            ignore_index: Label value excluded from the loss.
            vocab_chunk_size: Number of vocabulary entries processed per chunk.

        Returns:
            Scalar sum of the weighted token losses.
        """
        N, C = logits.shape
        assert labels.shape == (N,)
        assert weight.shape == (N,)

        # Save tensors and parameters required by the backward pass.
        ctx.save_for_backward(logits, weight, labels)
        ctx.ignore_index = ignore_index
        ctx.vocab_chunk_size = vocab_chunk_size

        # Allocate one FP32 loss value per token.
        out = torch.empty(N, dtype=torch.float32, device=logits.device)

        # Use the chunked implementation for large vocabularies.
        # Triton requires the chunk size to be a power of two.
        VOCAB_CHUNK_SIZE = triton.next_power_of_2(vocab_chunk_size)
        grid = (N,)

        # Launch the Triton forward kernel.
        fused_ce_loss_kernel_chunked[grid](
            logits,
            weight,
            labels,
            out,
            N=N,
            C=C,
            VOCAB_CHUNK_SIZE=VOCAB_CHUNK_SIZE,
            ignore_index=ignore_index,
        )

        # Return the total weighted loss.
        return out.sum()

    @staticmethod
    def backward(ctx, grad_output):
        """Run the fused cross-entropy backward pass.

        Args:
            grad_output: Scalar gradient from the caller.

        Returns:
            Gradients for logits and weights, followed by ``None`` for labels,
            ``ignore_index``, and ``vocab_chunk_size``.
        """
        # Restore tensors saved by the forward pass.
        logits, weight, labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index

        # Allocate output gradients.
        grad_logits = torch.empty_like(logits, dtype=torch.bfloat16)
        grad_weight = torch.empty_like(weight, dtype=torch.bfloat16)

        # Read the token and vocabulary dimensions.
        N, V = logits.shape

        # Launch the Triton backward kernel.
        BLOCK_V = min(4096, V)
        grid = (N,)

        chunk_loss_bw_kernel[grid](
            logits, labels, weight, grad_output, grad_logits, grad_weight, N, V, ignore_index, BLOCK_V=BLOCK_V
        )

        # Return gradients in the same order as the forward arguments.
        return grad_logits, grad_weight, None, None, None


# Public convenience wrapper.
def fused_cross_entropy_loss(logits, weight, labels, ignore_index=-100, vocab_chunk_size=4096):
    """Compute fused cross-entropy with autograd support.

    Args:
        logits: Logits tensor with shape ``[N, C]``.
        weight: Per-token weight tensor with shape ``[N]``.
        labels: Target label tensor with shape ``[N]``.
        ignore_index: Label value excluded from the loss.
        vocab_chunk_size: Number of vocabulary entries processed per chunk.

    Returns:
        Scalar sum of the weighted token losses.
    """
    return FusedCrossEntropyLoss.apply(logits, weight, labels, ignore_index, vocab_chunk_size)


@triton.jit
def chunk_loss_bw_kernel(
    logits,
    labels,
    loss_weight,
    grad_output,  # Inputs
    grad_logits,
    grad_loss_weight,  # Outputs
    N,
    V,
    ignore_index,  # Shapes and configuration
    BLOCK_V: tl.constexpr,  # Vocabulary chunk size
):
    # Each program processes one row of the 2D logits tensor.
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the label and weight for the current row.
    label = tl.load(labels + pid)
    weight = tl.load(loss_weight + pid)
    go = tl.load(grad_output)  # Scalar gradient from the caller

    # --------------------------
    # Handle ignore_index.
    # --------------------------
    if label == ignore_index:
        # All gradients for an ignored token are zero.
        for off in range(0, V, BLOCK_V):
            idx_v = off + tl.arange(0, BLOCK_V)
            mask = idx_v < V
            tl.store(grad_logits + pid * V + idx_v, 0.0, mask=mask)
        tl.store(grad_loss_weight + pid, 0.0)
        return

    # --------------------------
    # 1. Compute softmax(logits).
    # --------------------------
    max_val = -float("inf")
    # First pass: find the maximum for numerical stability.
    for off in range(0, V, BLOCK_V):
        idx_v = off + tl.arange(0, BLOCK_V)
        mask = idx_v < V
        l = tl.load(logits + pid * V + idx_v, mask=mask, other=-float("inf"))
        current_max = tl.max(l)  # The reduction does not accept a mask argument.
        max_val = tl.maximum(max_val, current_max)

    sum_exp = 0.0
    # Second pass: compute the sum of exponentials.
    for off in range(0, V, BLOCK_V):
        idx_v = off + tl.arange(0, BLOCK_V)
        mask = idx_v < V
        l = tl.load(logits + pid * V + idx_v, mask=mask, other=-float("inf"))
        exp_l = tl.exp(l - max_val)
        sum_exp += tl.sum(exp_l)

    # --------------------------
    # 2. Compute grad_logits = (softmax - one_hot) * weight * grad_output.
    # --------------------------
    scale = weight * go
    for off in range(0, V, BLOCK_V):
        idx_v = off + tl.arange(0, BLOCK_V)
        mask = idx_v < V
        l = tl.load(logits + pid * V + idx_v, mask=mask, other=-float("inf"))
        prob = tl.exp(l - max_val) / sum_exp

        # Subtract one at the target-label position.
        prob = tl.where(idx_v == label, prob - 1.0, prob)
        g = prob * scale
        tl.store(grad_logits + pid * V + idx_v, g, mask=mask)

    # --------------------------
    # 3. Compute grad_loss_weight = ce_loss * grad_output.
    # --------------------------
    ce_loss = -tl.log(tl.exp(tl.load(logits + pid * V + label) - max_val) / sum_exp)
    tl.store(grad_loss_weight + pid, ce_loss * go)


# --------------------------
# Standalone helper using a Triton backward pass.
# --------------------------
def fused_cross_entropy_loss_back(logits, loss_weight, labels, ignore_index, grad_output):
    # Initialize output gradients.

    grad_logits = torch.zeros_like(logits, dtype=torch.float32)
    grad_loss_weight = torch.zeros_like(loss_weight, dtype=torch.float32)

    # Read the two-dimensional logits shape.
    N, V = logits.shape

    # Launch the Triton backward kernel.
    BLOCK_V = min(1024, V)
    grid = (N,)

    chunk_loss_bw_kernel[grid](
        logits, labels, loss_weight, grad_output, grad_logits, grad_loss_weight, N, V, ignore_index, BLOCK_V=BLOCK_V
    )

    return grad_logits, grad_loss_weight


class ChunkLoss(Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, loss_weight: torch.Tensor, shifted_labels: torch.Tensor):
        # Save tensors required by the backward pass.
        ctx.save_for_backward(logits, loss_weight, shifted_labels)
        ctx.ignore_index = -100

        # Compute the reference PyTorch loss.
        logits = logits.float()
        ce_loss = F.cross_entropy(logits, shifted_labels, reduction="none", ignore_index=ctx.ignore_index)
        chunk_loss = (ce_loss * loss_weight).sum()

        return chunk_loss

    @staticmethod
    def backward(ctx, grad_output):
        # Restore tensors in the order used by the forward pass.

        print("grad_output", grad_output)
        logits, loss_weight, shifted_labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index

        # ====================== 1. Gradient with respect to logits ======================
        probs = F.softmax(logits.float(), dim=-1)
        grad_logits = probs.clone()
        grad_logits.scatter_(dim=-1, index=shifted_labels.unsqueeze(-1), value=-1.0)

        mask = (shifted_labels != ignore_index).float()
        grad_logits *= loss_weight.unsqueeze(-1) * mask.unsqueeze(-1)
        grad_logits = grad_logits * grad_output

        # ====================== 2. Gradient with respect to loss_weight ======================
        # The weight gradient is the per-token CE loss times the mask and upstream gradient.
        ce_loss = F.cross_entropy(logits, shifted_labels, reduction="none", ignore_index=ignore_index)
        grad_loss_weight = ce_loss * mask * grad_output

        # Return gradients for logits, loss_weight, and shifted_labels, respectively.
        return grad_logits, grad_loss_weight, None


# ---------------------------
# Test whether fused_cross_entropy_loss produces a grad_fn.
# ---------------------------
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import torch_npu

    # Construct a small test batch.
    bs, seq_len, vocab_size = 1, 10, 1000
    logits = torch.randn(bs * seq_len, vocab_size, device="cpu", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, vocab_size, (bs * seq_len,), dtype=torch.int32, device="cpu")
    loss_weight = torch.randn(bs * seq_len, device="cpu", dtype=torch.float32)

    print("=== Test autograd support for fused_cross_entropy_loss ===")
    print(f"logits shape: {logits.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"loss_weight shape: {loss_weight.shape}")
    print(f"logits requires_grad: {logits.requires_grad}")

    try:
        # Run the fused cross-entropy implementation.
        loss = fused_cross_entropy_loss(logits, loss_weight, labels)

        print("\nLoss result:")
        print(f"loss: {loss}")
        print(f"loss.dtype: {loss.dtype}")
        print(f"loss.device: {loss.device}")
        print(f"loss.grad_fn: {loss.grad_fn}")

        if loss.grad_fn is not None:
            print("Success: loss has a grad_fn and supports backward propagation")

            # Test backward propagation.
            loss.backward()

            print("\nBackward result:")
            print(f"logits.grad: {logits.grad}")
            print(f"logits.grad.shape: {logits.grad.shape if logits.grad is not None else 'None'}")

            if logits.grad is not None:
                print("Success: backward propagation populated logits.grad")
                print(f"logits.grad.mean(): {logits.grad.mean()}")
                print(f"logits.grad.std(): {logits.grad.std()}")
            else:
                print("Failure: logits.grad is None after backward propagation")
        else:
            print("Failure: loss has no grad_fn and cannot run backward propagation")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
    # logits.requires_grad = True
    # loss_weight.requires_grad = True
    ignore_index = -100

    chunkloss_input = torch.load("../../../chunkloss.pt")
    logits = chunkloss_input["logits"].npu()
    labels = chunkloss_input["labels"].npu()
    loss_weight = chunkloss_input["weight"].npu()
    # logits.requires_grad = True
    # loss_weight.requires_grad = True
    breakpoint()
    fused_cross_entropy_loss(logits, loss_weight, labels)

    # # Triton fused operator
    # triton_loss = fused_cross_entropy_loss(logits, loss_weight, labels)
    # grad_output = torch.ones_like(triton_loss, dtype=torch.float32)
    # grad_logits, grad_loss_weight = fused_cross_entropy_loss_back(logits, loss_weight, labels, ignore_index, grad_output)

    # pt_loss = ChunkLoss.apply(logits, loss_weight, labels)
    # pt_loss.backward()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        record_shapes=True,  # Record input shapes and dtypes for Torch operators.
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./perf_part"),
    ) as prof:
        # Triton fused operator.
        triton_loss = fused_cross_entropy_loss(logits, loss_weight, labels)
        grad_output = torch.ones_like(triton_loss, dtype=torch.float32)
        grad_logits, grad_loss_weight = fused_cross_entropy_loss_back(
            logits, loss_weight, labels, ignore_index, grad_output
        )
        # PyTorch operator.
        pt_loss = ChunkLoss.apply(logits, loss_weight, labels)
        pt_loss.backward()

        prof.step()

    # Compare results; an error below 1e-6 indicates numerical agreement.
    print("PyTorch loss:", pt_loss.item())
    print("Triton loss:", triton_loss.item())
    print("Forward error:", torch.abs(pt_loss - triton_loss).sum())

    print("PyTorch grad_logits:", grad_logits)
    print("Triton grad_logits:", grad_logits)
    print("PyTorch grad_loss_weight:", grad_loss_weight)
    print("Triton grad_loss_weight:", grad_loss_weight)

    print("grad_logits error:", grad_logits.shape, torch.max(torch.abs(grad_logits - logits.grad)))
    print("grad_loss_weight:", grad_loss_weight.shape, torch.max(torch.abs(grad_loss_weight - loss_weight.grad)))
