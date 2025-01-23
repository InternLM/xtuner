from collections import defaultdict
from functools import partial
from typing import Dict, Sequence


def messages_collate_fn(
    instances: Sequence[Dict],
    return_only_messages: bool = True,
):

    return_dict = defaultdict(list)
    messages = []

    for example in instances:
        assert 'conversation' in example.keys()
        messages.append(example['conversation'])
        for k, v in example.items():
            return_dict[k].append(v)

    if return_only_messages:
        return messages
    else:
        return return_dict


def message_data_collator(return_only_messages=True):
    return partial(
        messages_collate_fn, return_only_messages=return_only_messages)
