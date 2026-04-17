from collections import defaultdict
from typing import Callable, Tuple

import numpy as np
from scipy import stats

from .logger import get_logger


logger = get_logger()


def compute_metric(samples, source_normalizer: dict | Callable[[str], str | Tuple[str]] | None = None):
    # group by data source
    group_by_data_source = defaultdict(list)
    for sample in samples:
        data_source: str | Tuple[str] = str(sample.data.extra_info.get("origin_data_source", "unknown"))
        if isinstance(source_normalizer, dict):
            data_source = source_normalizer.get(data_source, data_source)  # type: ignore[arg-type]  # noqa
            assert isinstance(data_source, (str, tuple))
        elif callable(source_normalizer):
            data_source = source_normalizer(data_source)  # type: ignore[arg-type]
        if isinstance(data_source, str):
            data_source = (data_source,)
        for ds in data_source:
            group_by_data_source[ds].append(sample)

    metrics = dict()
    for data_source, _samples in group_by_data_source.items():
        metrics.update(compute_single_dataset_metric(_samples, data_source))

    print(metrics)
    return metrics


def compute_single_dataset_metric(samples, data_source):
    # group by message
    group_by_message = defaultdict(list)
    for sample in samples:
        group_by_message[str(sample.data.messages)].append(sample)

    num_samples_per_message = [len(s) for s in group_by_message.values()]
    max_samples_per_message = max(num_samples_per_message)
    min_samples_per_message = min(num_samples_per_message)
    # 众数
    mode_samples_per_message = stats.mode(num_samples_per_message)[0]
    if max_samples_per_message != mode_samples_per_message or min_samples_per_message != mode_samples_per_message:
        logger.warning(
            f"Data source {data_source} has non-uniform number of samples per message, "
            f"max samples per message: {max_samples_per_message}, "
            f"min samples per message: {min_samples_per_message}, "
            f"mode samples per message: {mode_samples_per_message}"
        )
    correctness = defaultdict(list)
    for message, _samples in group_by_message.items():
        for sample in _samples:
            correctness[message].append(sample.env.judger.reward["score"] > 0)
        if len(correctness[message]) < mode_samples_per_message:
            correctness[message].extend([False] * (mode_samples_per_message - len(correctness[message])))
        elif len(correctness[message]) > mode_samples_per_message:
            correctness[message] = correctness[message][:mode_samples_per_message]
    # average acc
    average_acc = np.mean(list(correctness.values()))
    # pass at k
    all_k = [mode_samples_per_message]
    while True:
        current_k = all_k[-1] // 2
        if current_k <= 1:
            break
        all_k.append(current_k)
    pass_at_k = dict()
    for k in all_k[::-1]:
        current_correctness = []
        for message, _correctness in correctness.items():
            current_correctness.append(any(_correctness[:k]))
        pass_at_k[f"{data_source}_Pass@{k}"] = np.mean(current_correctness)
    return {
        f"{data_source}_Acc_Mean@{mode_samples_per_message}": average_acc,
        **pass_at_k,
    }
