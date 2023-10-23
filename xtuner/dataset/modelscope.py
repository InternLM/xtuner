# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import Config, ConfigDict

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset


def process_ms_dataset(dataset, split='train', *args, **kwargs):
    """Post-process the dataset loaded from the ModelScope Hub."""

    if isinstance(dataset, (Config, ConfigDict)):
        dataset = BUILDER.build(dataset)
    if isinstance(dataset, dict):
        dataset = dataset[split]
    dataset = dataset.to_hf_dataset()
    return process_hf_dataset(dataset, *args, **kwargs)
