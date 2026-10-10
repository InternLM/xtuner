"""Diagnostics for sampled, full-vocabulary end-to-end logits."""

import torch
import torch.nn.functional as F


def check_logits(actual, expected, *, max_relative_l2=0.05, min_cosine=0.998):
    """Check each sample separately; use FP64 only for offline reductions.

    Inputs are [sampled_positions, vocab], not hidden states or target-token logits.
    Callers must select the same globally indexed, non-padding positions on both sides.
    The 5% L2 bound is independent of the existing mean loss-curve tolerance.
    """
    assert actual.shape == expected.shape and actual.ndim == 2 and actual.numel() > 0
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "non-finite logits"
    actual = actual.detach().to(device="cpu", dtype=torch.float64).flatten()
    expected = expected.detach().to(device="cpu", dtype=torch.float64).flatten()
    norm = expected.norm()
    assert norm > 0, "zero reference norm"
    delta = actual - expected
    metrics = {
        "relative_l2": (delta.norm() / norm).item(),
        "cosine": F.cosine_similarity(actual, expected, dim=0).item(),
        "max_abs": delta.abs().max().item(),
    }
    assert metrics["relative_l2"] < max_relative_l2 and metrics["cosine"] > min_cosine, metrics
    return metrics
