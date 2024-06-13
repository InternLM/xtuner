from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    'attention.wqkv': ColwiseParallel(),
    'attention.wo': RowwiseParallel(),
    'feed_forward.w1': ColwiseParallel(),
    'feed_forward.w2': RowwiseParallel(),
    'feed_forward.w3': ColwiseParallel(),
}
