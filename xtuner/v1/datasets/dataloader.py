from abc import ABC, abstractmethod
from typing import Iterator, cast

import torch

from xtuner.v1.datasets.collator import ColateItem
from xtuner.v1.datasets.resume import get_dataloader_state, load_dataloader_state
from xtuner.v1.utils import get_logger


logger = get_logger()


class BaseDataloader(ABC):
    """BaseDataloader represents the whole data module to interact with the
    training process.

    It contains all the detailed implementations such as sampler, dataset, collator, etc. But only exposes a few
    interfaces to the training process.
    """

    @abstractmethod
    def load_state_dict(self, state_dict: dict, train_state_total_consumed_samples: int | None = None) -> None: ...

    @abstractmethod
    def get_state_dict(self, consumed_samples: int = -1) -> dict: ...

    @abstractmethod
    def __iter__(self) -> Iterator[list[ColateItem]]: ...


class Dataloader(torch.utils.data.DataLoader, BaseDataloader):
    """This Dataloader and torch's Dataloader are logically parallel concepts.

    It supplements the load_state_dict, get_state_dict and other interfaces to the training process. We can implement
    it by inheriting torch Dataloader or composing torch Dataloader. Both are ok, just choose one that's easier to
    implement.
    """

    def load_state_dict(
        self,
        state_dict: dict,
        train_state_total_consumed_samples: int | None = None,
    ) -> None:
        load_dataloader_state(
            self,
            state_dict,
            train_state_total_consumed_samples=train_state_total_consumed_samples,
        )

    def get_state_dict(self, consumed_samples: int = -1) -> dict:
        if consumed_samples != -1:
            logger.warning(
                "Dataloader.get_state_dict(consumed_samples=...) is deprecated; use the default (-1). "
                "Consumed samples are tracked on the sampler."
            )
        dataloader_state = get_dataloader_state(self, consumed_samples)
        return cast(dict, dataloader_state)

    def get_total_consumed_samples(self) -> int:
        sampler = self.sampler
        if hasattr(sampler, "get_total_consumed_steps"):
            return int(sampler.get_total_consumed_steps())
        return 0

    # __iter__ is inherited from torch.utils.data.DataLoader

    # Streaming dataloader may not have `set_epoch` and `__len__` method, so we add here.
    def set_epoch(self, epoch: int) -> None:
        assert hasattr(self.sampler, "set_epoch"), "Sampler must have `set_epoch` method"
        self.sampler.set_epoch(epoch)
