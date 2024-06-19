# Adopted from: https://github.com/huggingface/transformers/blob/HEAD/src/transformers/generation/utils.py  # noqa: E501
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils.generic import ModelOutput


@dataclass
class PolicyOutput(ModelOutput):
    output_ids: Optional[torch.Tensor] = None
    output_str: Optional[list[str]] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    logits_entropy: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    top_logprobs: Optional[torch.Tensor] = None
    question_mask: Optional[torch.Tensor] = None
    answer_mask: Optional[torch.Tensor] = None

    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)

    def get_tensor_keys(self):
        keys = []
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                keys.append(k)
        return keys


def union_keys_from_policy_outputs(policy_outputs: list[PolicyOutput]) -> list:
    all_keys = set()
    for po in policy_outputs:
        all_keys = all_keys.union(set(po.keys()))
    # e.g., return ["output_str", "output_ids", "loss", ...]
    return list(all_keys)


def union_tensor_keys_from_policy_outputs(
        policy_outputs: list[PolicyOutput]) -> list:
    all_keys = set()
    for po in policy_outputs:
        all_keys = all_keys.union(set(po.get_tensor_keys()))
    # e.g., return ["output_ids", "loss", ...]
    return list(all_keys)


def concat_policy_outputs(policy_outputs: list[PolicyOutput],
                          padding_token_map: dict = None) -> PolicyOutput:
    if isinstance(policy_outputs, PolicyOutput):
        # Wrong input type
        return policy_outputs
    elif policy_outputs is None or len(policy_outputs) == 0:
        return PolicyOutput(None)
    elif len(policy_outputs) == 1:
        return policy_outputs[0]

    # padding
    if padding_token_map is not None:
        policy_outputs = padding_policy_outputs(policy_outputs,
                                                padding_token_map)

    concated = PolicyOutput()
    all_keys = union_keys_from_policy_outputs(policy_outputs)
    for key in all_keys:
        for po in policy_outputs:
            value = po[key]
            if value is not None:
                # get the first non-empty value
                break
        if value is None:
            # skip if all values are None
            continue

        if isinstance(value, torch.Tensor):
            concated[key] = torch.cat(
                [po[key] for po in policy_outputs if po[key] is not None],
                dim=0)
        elif isinstance(value, list):
            # e.g., list[str]
            concated[key] = []
            for po in policy_outputs:
                if po[key] is not None:
                    concated[key].extend(po[key])
        elif isinstance(value, tuple) and isinstance(value[0], torch.Tensor):
            results = []
            for i in range(len(value)):
                beef = [
                    po[key][i] for po in policy_outputs
                    if po[key][i] is not None
                ]
                tensor = torch.cat(
                    beef, dim=0) if len(beef) > 0 else torch.Tensor()
                results.append(tensor)
            concated[key] = tuple(results)
            raise NotImplementedError(
                f'{value}\n{[v.shape for v in value]}\n{results}')
        else:
            raise TypeError(
                f'value: {value} with unsupported type: {type(value)}.')
    return concated


def padding_policy_outputs(policy_outputs: list[PolicyOutput],
                           padding_token_map={},
                           right_padding=True,
                           padding_id=0):
    tensor_keys = union_tensor_keys_from_policy_outputs(policy_outputs)
    for key in tensor_keys:
        padding_id = padding_token_map.get(key, padding_id)
        max_seq_len = find_max_seq_len(policy_outputs, key)
        for policy_output in policy_outputs:
            origin_tensor = policy_output[key]
            padding_size = max_seq_len - origin_tensor.shape[1]
            pad = (0, padding_size) if right_padding else (padding_size, 0)
            padded_tensor = torch.nn.functional.pad(
                origin_tensor, pad, mode='constant', value=padding_id)
            policy_output[key] = padded_tensor
    return policy_outputs


def find_max_seq_len(policy_outputs: list[PolicyOutput], key):
    max_seq_len = 0
    for policy_output in policy_outputs:
        if policy_output[key] is None:
            continue
        batch_size, seq_len = policy_output[key].shape[:2]
        max_seq_len = seq_len if seq_len > max_seq_len else max_seq_len
    return max_seq_len


def logprobs_from_logits(logits: torch.Tensor,
                         labels: torch.Tensor,
                         gather: bool = True) -> torch.Tensor:
    r"""
    Adapted from: https://github.com/huggingface/trl/blob/main/trl/core.py#L131

    Example:

    ```python
    >>> logits, _ = model(**input_kwargs)
    >>> input_ids = input_kwargs["input_ids"]
    >>> logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    ```"""
    logp = torch.nn.functional.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy
