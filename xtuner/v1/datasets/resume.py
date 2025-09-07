from torch.utils.data import DataLoader
from typing_extensions import TypedDict

from xtuner.v1.utils import get_logger

from .packing import ExpandSoftPackDataset, _LegacySoftPackDataset
from .sampler import LengthGroupedSampler, ParallelSampler


logger = get_logger()


class DataloaderState(TypedDict):
    sampler: dict
    dataset: dict


def get_dataloader_state(dataloader: DataLoader, consumed_samples: int) -> DataloaderState:
    sampler: ParallelSampler | LengthGroupedSampler = dataloader.sampler  # type: ignore[assignment]
    dataset: ExpandSoftPackDataset | _LegacySoftPackDataset = dataloader.dataset  # type: ignore[assignment]
    dataloader_state = DataloaderState(sampler={}, dataset={})

    if not hasattr(sampler, "load_state_dict") or not hasattr(sampler, "get_state_dict"):
        logger.warning(f"Resuming from {type(sampler)} is risky.")
    else:
        dataloader_state["sampler"].update(sampler.get_state_dict(step=consumed_samples))

    if not hasattr(dataset, "load_state_dict") or not hasattr(dataset, "get_state_dict"):
        logger.warning(f"Resuming from {type(dataset)} is risky.")
    else:
        dataloader_state["dataset"].update(dataset.get_state_dict())

    return dataloader_state


def load_dataloader_state(dataloader: DataLoader, state: dict):
    sampler = dataloader.sampler
    dataset = dataloader.dataset

    # Sampler require `load_state_dict` to restore the training progress since the sampler state will
    # record the consumed samples.
    if not hasattr(sampler, "load_state_dict"):
        logger.warning(f"Resuming from {type(sampler)} is risky.")

    if hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(state["sampler"])

    # If the dataset records the training progress, we also restore it.
    if hasattr(dataset, "load_state_dict"):
        dataset.load_state_dict(state["dataset"])
