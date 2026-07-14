from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner.v1.utils import init_params


@contextmanager
def _single_process_gloo_group() -> Iterator[None]:
    """Create an isolated CPU process group without reserving a TCP port."""
    assert not dist.is_initialized()
    with TemporaryDirectory() as tmp_dir:
        rendezvous = Path(tmp_dir, "gloo_rendezvous").resolve().as_uri()
        dist.init_process_group("gloo", init_method=rendezvous, rank=0, world_size=1)
        try:
            yield
        finally:
            dist.destroy_process_group()


def test_init_params_disables_autograd_for_regular_parameter() -> None:
    parameter = nn.Parameter(torch.ones(2, 3))

    # Use a raw in-place operation rather than torch.nn.init.*: the latter
    # already disables autograd internally and would not test init_params.
    init_params(parameter, lambda tensor: tensor.fill_(4.0))

    assert parameter.requires_grad
    assert parameter.is_leaf
    assert torch.equal(parameter, torch.full_like(parameter, 4.0))


def test_init_params_disables_autograd_for_dtensor_copy() -> None:
    with _single_process_gloo_group():
        mesh = init_device_mesh("cpu", (1,))
        parameter = distribute_tensor(
            torch.ones(2, 3, requires_grad=True),
            mesh,
            [Shard(0)],
        )

        assert isinstance(parameter, DTensor)
        assert parameter.requires_grad
        assert parameter.is_leaf

        # nn.init.zeros_ protects its own write, but not the subsequent
        # DTensor copy performed by init_params. That copy was the regression.
        init_params(parameter, torch.nn.init.zeros_)

        assert parameter.requires_grad
        assert parameter.is_leaf
        assert torch.count_nonzero(parameter.full_tensor()).item() == 0
