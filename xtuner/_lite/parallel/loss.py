from torch import distributed as dist
from torch.distributed.nn.functional import all_reduce


def dist_softmax(logits, mesh, dim=-1, temperature=1):

    logits = logits / temperature

    max_values = logits.max(dim, keepdim=True)
    all_reduce(max_values, dist.ReduceOp.MAX, mesh.group())

    exps = (logits - max_values).exp()
    sums = exps.sum(dim, keepdim=True)
    all_reduce(sums, dist.ReduceOp.SUM, mesh.group())

    probs = exps / sums

    return probs
