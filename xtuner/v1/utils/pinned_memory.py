"""Bounded pinned allocations with ordinary views that retain CUDA event
ownership."""

from bisect import bisect_left, insort

import torch


def allocate_pinned_like(
    tensors: list[torch.Tensor], *, zero: bool = False, copy_non_blocking: bool = False
) -> list[torch.Tensor]:
    """Allocate compatible dense layouts in bounded pinned slabs.

    Ordinary views retain the host allocator's stream-event context. Full-state
    torch.save deduplicates their storage; DCP compacts individual views. A raw
    torch.save of a single view includes its whole slab unless cloned first.

    Args:
        tensors (list[torch.Tensor]): Plain strided tensors supplying shape, dtype and layout.
        zero (bool): Initialize each slab to zero before exposing its views.
        copy_non_blocking (bool): Copy each slab's sources without autograd as soon as its views are allocated,
            overlapping transfers with subsequent allocations. The caller must synchronize before CPU use.

    Returns:
        list[torch.Tensor]: CPU views in input order, with empty_like preserve-format layouts.
    """
    if zero and copy_non_blocking:
        raise ValueError("Choose either zero initialization or source copies")
    layouts = [
        tensor
        if tensor.is_contiguous()
        else torch.empty_like(tensor, device="meta", memory_format=torch.preserve_format)
        for tensor in tensors
    ]
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for index, tensor in enumerate(tensors):
        groups.setdefault((tensor.device, tensor.dtype), []).append(index)
    result: list[torch.Tensor | None] = [None] * len(tensors)
    for (_, dtype), indices in groups.items():
        sizes = [layouts[index].numel() * layouts[index].element_size() for index in indices]
        for members in plan_pinned_slabs(sizes):
            element_size = layouts[indices[members[0]]].element_size()
            slab = torch.empty(
                sum(sizes[i] for i in members) // element_size, dtype=dtype, device="cpu", pin_memory=True
            )
            if zero:
                slab.zero_()
            offset = 0
            copy_targets: list[torch.Tensor] = []
            copy_sources: list[torch.Tensor] = []
            for member in members:
                index = indices[member]
                layout = layouts[index]
                view = slab.as_strided(layout.shape, layout.stride(), offset)
                if copy_non_blocking:
                    copy_targets.append(view)
                    copy_sources.append(tensors[index])
                result[index] = view
                offset += layout.numel()
            if copy_non_blocking:
                with torch.no_grad():
                    torch._foreach_copy_(copy_targets, copy_sources, non_blocking=True)
    assert all(tensor is not None for tensor in result)
    return [tensor for tensor in result if tensor is not None]


def plan_pinned_slabs(sizes: list[int], cap_bytes: int = 1 << 30) -> list[list[int]]:
    """Plan deterministic best-fit-decreasing slabs bounded by a byte cap.

    Args:
        sizes (list[int]): Nonnegative allocation sizes in bytes.
        cap_bytes (int): Maximum slab size; larger individual requests stay standalone.

    Returns:
        list[list[int]]: Input indices grouped into slabs; small slabs allocate only used bytes.
    """
    if cap_bytes <= 0 or any(size < 0 for size in sizes):
        raise ValueError("Slab cap must be positive and request sizes nonnegative")
    bins: list[list[int]] = []
    free: list[tuple[int, int]] = []
    for index in sorted(range(len(sizes)), key=lambda index: (-sizes[index], index)):
        size = sizes[index]
        # Sort by remaining capacity then slab index to retain deterministic BFD ties.
        position = bisect_left(free, (size, -1))
        if position < len(free):
            remaining, best = free.pop(position)
            bins[best].append(index)
        else:
            best = len(bins)
            bins.append([index])
            remaining = cap_bytes
        remaining -= size
        if remaining >= 0:
            insort(free, (remaining, best))
    return bins
