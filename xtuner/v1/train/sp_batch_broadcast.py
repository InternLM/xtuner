"""Broadcast a complete batch: tensor bytes directly, ordinary values as metadata.

NCCL uses the existing SP group; Gloo is supported for CPU tests. No field schema
or coordinated error protocol: exceptions propagate and the launcher tears down peers.
"""

import torch
import torch.distributed as dist

from xtuner.v1.data_proto.sequence_context import SequenceContext


# 生产复用原 SP NCCL 组并使用当前 GPU；CPU/Gloo 仅用于已有测试。
# Production reuses the SP NCCL group on the local GPU; CPU/Gloo supports existing tests.
def _device(group):
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else "cpu"


def broadcast_control(value, group):
    """Small metadata, also used to share loader length during initialization."""
    # 这里只传小元信息；tensor 内容在 broadcast_batch 中直接传输。
    # Send small metadata here; broadcast_batch transfers tensor contents directly.
    values = [value]
    dist.broadcast_object_list(values, src=dist.get_process_group_ranks(group)[0], group=group, device=_device(group))
    return values[0]


def broadcast_batch(batch, group, *, epoch=0, seed_draw=False):
    """Return (complete CPU batch, leader epoch, seed_draw).

    Traverse dict/list/tuple and SequenceContext attributes dynamically. Preserve
    repeated tensor references, including q/k offsets and their CPU cache. Runtime
    mesh/device attributes are local; no source tensor is serialized as metadata.

    seed_draw is not batch data: the leader reports whether this fetch consumed a fresh
    main-process base_seed, so followers can mirror that RNG consumption and stay aligned
    with the BASE topology where every rank drew its own seed. epoch travels the same way
    (followers cannot observe StopIteration on a loader they do not own).
    """
    # 完整 batch 按实际容器/属性动态遍历；普通值复制，tensor 直接广播，重复引用只传一次。
    # Traverse actual containers/attributes; copy ordinary values and broadcast tensors once per identity.
    # epoch 与 seed_draw 是供数控制信息：同步 epoch，并通知 follower 是否补消耗一次随机数。
    # epoch and seed_draw synchronize fetch state; seed_draw signals one compensating RNG draw.
    leader = dist.get_rank(group) == 0
    tensors, descriptors, seen = [], [], {}

    def extract(value):
        if isinstance(value, torch.Tensor):
            key = id(value)
            if key not in seen:
                seen[key] = len(tensors)
                tensors.append(value)
                descriptors.append((tuple(value.shape), value.dtype))
            return ("tensor", seen[key])
        if isinstance(value, SequenceContext):
            # mesh/device 属于各 rank 本地运行状态，不把 leader 的进程组对象复制给 follower。
            # Runtime mesh/device are rank-local; do not copy the leader's process-group objects.
            fields = {k: v for k, v in vars(value).items() if k not in ("sequence_parallel_mesh", "device")}
            return ("context", extract(fields))
        if isinstance(value, dict):
            return ("dict", [(extract(k), extract(v)) for k, v in value.items()])
        if isinstance(value, (list, tuple)):
            return ("list" if isinstance(value, list) else "tuple", [extract(v) for v in value])
        if value is None or isinstance(value, (bool, int, float, str, bytes, torch.dtype, torch.device)):
            return ("value", value)
        raise TypeError(f"Unsupported batch value type: {type(value).__name__}")

    def restore(node):
        kind, value = node
        if kind == "tensor":
            return tensors[value]
        if kind == "context":
            ctx = SequenceContext.__new__(SequenceContext)
            # 按原属性恢复，不重新计算 position/offset，保留收到的 tensor 引用关系。
            # Restore the saved attributes as-is; never recompute positions/offsets.
            ctx.__dict__.update(restore(value))
            # 完整 CPU batch 接回原 prepare 路径；本地 SP mesh 稍后由 split 绑定。
            # Runtime mesh/device are local properties of the original rank: the mesh is
            # re-bound later by the SP split, and "cpu" matches what the collator handed
            # to _prepare_model_input in BASE (whose .to(DEVICE) then does the H2D).
            ctx.sequence_parallel_mesh, ctx.device = None, "cpu"
            return ctx
        if kind == "dict":
            return {restore(k): restore(v) for k, v in value}
        if kind in ("list", "tuple"):
            values = [restore(v) for v in value]
            return values if kind == "list" else tuple(values)
        return value

    metadata = None
    if leader:
        tree = extract(batch)
        metadata = (tree, descriptors, epoch, seed_draw)
    tree, descriptors, epoch, seed_draw = broadcast_control(metadata, group)
    src, device = dist.get_process_group_ranks(group)[0], _device(group)
    for i, (shape, dtype) in enumerate(descriptors):
        wire = tensors[i].to(device).contiguous() if leader else torch.empty(shape, dtype=dtype, device=device)
        # 空 tensor 已由 shape/dtype 元信息表达，无内容需传；各 rank 一致跳过 collective。
        # Zero-element tensors are described by metadata; all ranks skip the empty payload.
        if wire.numel():
            dist.broadcast(wire, src=src, group=group)
        if not leader:
            # 当前版本回 CPU 以兼容原输入驻留/消费方式；同步 D2H 完成后再传下一个 tensor。
            # Restore the original CPU input residency; finish synchronous D2H before the next tensor.
            tensors.append(wire.cpu())
    return (batch if leader else restore(tree)), epoch, seed_draw
