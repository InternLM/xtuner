from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
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
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
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
        self._explicit_consumption = False
        self._issued_samples = 0
        self._epoch_local_samples = 0
        self._pending_envelopes = deque()

    def enable_explicit_consumption(self):
        if self._local_samples or self._pending_envelopes:
            raise RuntimeError("explicit consumption must be enabled before reading batches")
        self._explicit_consumption = True

    def commit_batch(self, sample_count):
        if not self._explicit_consumption or not self._pending_envelopes:
            raise RuntimeError("no issued media envelope to commit")
        if sample_count != self._pending_envelopes[0]:
            raise ValueError("committed sample count differs from the oldest issued envelope")
        self._pending_envelopes.popleft()
        self._local_samples += sample_count
        self._epoch_local_samples += sample_count

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
        self._issued_samples = 0
        self._epoch_local_samples = 0
        self._pending_envelopes.clear()

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
        if self._explicit_consumption:
            dataloader_state["media_consumption"] = {
                "version": 1,
                "cursor": "committed",
                "uncommitted_envelopes": len(self._pending_envelopes),
            }

        if not hasattr(sampler, "load_state_dict") or not hasattr(sampler, "get_state_dict"):
            log_rank0.warning(f"Resuming from {type(sampler)} is risky.")
        else:
            dataloader_state["sampler"].update(sampler.get_state_dict(total_steps))
            if self._explicit_consumption:
                total_size = dataloader_state["sampler"].get("total_size", 0)
                dataloader_state["media_consumption"]["at_epoch_boundary"] = bool(
                    self._epoch_local_samples and total_size > 0 and total_steps % total_size == 0
                )

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
            if self._explicit_consumption:
                self._issued_samples += len(batch)
                self._pending_envelopes.append(len(batch))
            else:
                self._local_samples += len(batch)
            yield batch

    # Streaming dataloader may not have `set_epoch` and `__len__` method, so we add here.
    def set_epoch(self, epoch: int) -> None:
        assert hasattr(self.sampler, "set_epoch"), "Sampler must have `set_epoch` method"
        if self._explicit_consumption and self._pending_envelopes:
            raise RuntimeError("cannot advance the epoch with uncommitted media envelopes")
        self.sampler.set_epoch(epoch)
        self._epoch_local_samples = 0
