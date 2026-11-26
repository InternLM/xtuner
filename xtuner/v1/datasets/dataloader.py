from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.datasets.collator import ColateItem
from xtuner.v1.datasets.packing import ExpandSoftPackDataset, _LegacySoftPackDataset
from xtuner.v1.datasets.preset_sampler import PresetSampler
from xtuner.v1.datasets.sampler import LengthGroupedSampler, ParallelSampler
from xtuner.v1.utils import get_logger, log_rank0


logger = get_logger()


def reduce_sum_across_dp_group(dp_mesh: DeviceMesh | None, local_value: int) -> int:
    """Sum ``local_value`` over the DP process group (one contribution per
    data-parallel replica).

    Ranks that only differ in SP/TP see identical data batches and must not be summed with the global world group; see
    Training notes for SP+DP.
    """
    if dp_mesh is None or dp_mesh.size() <= 1:
        return int(local_value)
    if not dist.is_available() or not dist.is_initialized():
        return int(local_value)
    # The device must match the backend the group was created with.  The mesh
    # cannot be asked: ``Trainer._init_data_mesh`` builds it on ``"cpu"`` when
    # the parameters are not offloaded, and a CPU tensor in an ``nccl``/``hccl``
    # collective raises "No backend type associated with device type cpu".
    backend = str(dist.get_backend(dp_mesh.get_group())).lower()
    device_type = {
        "nccl": "cuda",
        "hccl": "npu",
        "xccl": "xpu",
        "cncl": "mlu",
    }.get(backend, "cpu")
    device = torch.device(device_type)
    tensor = torch.tensor([local_value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
    return int(tensor.item())


class BaseDataloader(ABC):
    """BaseDataloader represents the whole data module to interact with the
    training process.

    It contains all the detailed implementations such as sampler, dataset, collator, etc. But only exposes a few
    interfaces to the training process.
    """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None: ...

    @abstractmethod
    def get_state_dict(self) -> dict: ...

    @abstractmethod
    def __iter__(self) -> Iterator[list[ColateItem]]: ...


class Dataloader(torch.utils.data.DataLoader, BaseDataloader):
    """This Dataloader and torch's Dataloader are logically parallel concepts.

    It supplements the load_state_dict, get_state_dict and other interfaces to the training process. We can implement
    it by inheriting torch Dataloader or composing torch Dataloader. Both are ok, just choose one that's easier to
    implement.
    """

    def __init__(self, *args, **kwargs) -> None:
        dp_mesh: DeviceMesh | None = kwargs.pop("dp_mesh", None)
        super().__init__(*args, **kwargs)
        self._dp_mesh = dp_mesh
        self._init_total_samples = 0
        self._local_samples = 0

    def will_consume_base_seed(self) -> bool:
        """Report whether the next underlying iterator creation draws a base
        seed.

        XTuner's generator-based __iter__ defers PyTorch iterator creation until
        next(). Check before that fetch. Persistent workers reuse the underlying
        iterator across epochs without drawing another base seed.

        Returns:
            bool: Whether a new underlying iterator will consume a base seed.
        """
        # 首次创建或非 persistent 重建会抽取；persistent 跨 epoch 复用不重抽。
        # Keep the dependency on PyTorch's private iterator state inside the loader.
        return not self.persistent_workers or self._iterator is None

    def load_state_dict(self, state_dict: dict) -> None:
        sampler: ParallelSampler | LengthGroupedSampler | PresetSampler = self.sampler  # type: ignore[assignment]
        dataset = self.dataset
        sampler_state = state_dict["sampler"]

        if not hasattr(sampler, "load_state_dict"):
            log_rank0.warning(f"Resuming from {type(sampler)} is risky.")
        else:
            sampler.load_state_dict(sampler_state)

        self._init_total_samples = int(state_dict["total_consumed_samples"])
        self._local_samples = 0

        if hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict["dataset"])

    def get_state_dict(self) -> dict:
        total_steps = self._init_total_samples + reduce_sum_across_dp_group(self._dp_mesh, self._local_samples)
        sampler: ParallelSampler | LengthGroupedSampler | PresetSampler = self.sampler  # type: ignore[assignment]
        dataset: ExpandSoftPackDataset | _LegacySoftPackDataset = self.dataset  # type: ignore[assignment]
        dataloader_state: dict = {
            "sampler": {},
            "dataset": {},
            "total_consumed_samples": total_steps,
        }

        if not hasattr(sampler, "load_state_dict") or not hasattr(sampler, "get_state_dict"):
            log_rank0.warning(f"Resuming from {type(sampler)} is risky.")
        else:
            dataloader_state["sampler"].update(sampler.get_state_dict(total_steps))

        if not hasattr(dataset, "load_state_dict") or not hasattr(dataset, "get_state_dict"):
            log_rank0.warning(f"Resuming from {type(dataset)} is risky.")
        else:
            dataloader_state["dataset"].update(dataset.get_state_dict())

        return dataloader_state

    def __iter__(self) -> Iterator[list[ColateItem]]:  # type: ignore[override]
        # Override to count delivered batches, not prefetched indices.
        # With num_workers > 0 the sampler is iterated ahead by DataLoader's prefetch queue,
        # so recording inside sampler.__iter__ would count too many samples. Instead we
        # increment local consumed exactly once per batch that reaches the caller.
        for batch in super().__iter__():
            self._local_samples += len(batch)
            yield batch

    # Streaming dataloader may not have `set_epoch` and `__len__` method, so we add here.
    def set_epoch(self, epoch: int) -> None:
        assert hasattr(self.sampler, "set_epoch"), "Sampler must have `set_epoch` method"
        self.sampler.set_epoch(epoch)
