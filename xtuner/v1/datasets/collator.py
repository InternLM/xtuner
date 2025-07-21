import torch
from torch.nn.utils.rnn import pad_sequence


def sft_llm_collator(instances, pad_token_id=0, ignore_id=-100, batch_pack=True, max_length=None):
    _instances = []
    for ins in instances:
        if isinstance(ins, list):
            _instances.extend(ins)
        else:
            _instances.append(ins)

    instances = _instances

    input_ids = []
    labels = []
    num_tokens = []

    for data in instances:
        _input_ids = data["input_ids"]
        _labels = data["labels"]
        _num_tokens = data["num_tokens"]

        # TODO remove list
        if isinstance(_num_tokens, list):
            assert len(_num_tokens) == 1
            _num_tokens = _num_tokens[0]

        assert isinstance(_num_tokens, int)

        if max_length:
            _input_ids = _input_ids[:max_length]
            _labels = _labels[:max_length]
            _num_tokens = min(_num_tokens, max_length)

        input_ids.append(torch.LongTensor(_input_ids))
        labels.append(torch.LongTensor(_labels))
        num_tokens.append(_num_tokens)

    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    num_tokens = torch.IntTensor(num_tokens)

    if len(instances) > 1 and batch_pack:
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

    elif len(instances) > 1 and not batch_pack:
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=ignore_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)

    data_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "num_tokens": num_tokens,
        "attention_mask": attention_mask.bool(),
    }

    return data_dict
