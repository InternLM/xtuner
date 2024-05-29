from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizer


def get_question_answer_mask(
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    tokenizer_pad_token_id: int,
    generate_pad_token_id: int = None,
):
    """
    Example:
        input_ids = torch.tensor([[0, 1, 9]])
        output_ids = torch.tensor([[0, 1, 9, 2, 3, 4, 5]])
        tokenizer_pad_token_id = 0  # set 0 as neither question or answer
        generate_pad_token_id = None
        expected_qst_mask = torch.tensor([[0, 1, 1, 0, 0, 0, 0]])
        expected_ans_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    """
    # seq_mask yields zero where token == pad_token_id
    seq_mask = output_ids.not_equal(tokenizer_pad_token_id).int()
    if generate_pad_token_id is not None:
        seq_mask *= output_ids.not_equal(generate_pad_token_id).int()

    question_len = input_ids.shape[-1]
    question_mask = seq_mask.clone()
    question_mask[:, question_len:] = 0
    answer_mask = seq_mask.clone()
    answer_mask[:, :question_len] = 0
    return question_mask, answer_mask


def partition_by_micro_batch_size(
    input_ids: Union[list[str], torch.Tensor, list[int]],
    micro_batch_size: int,
    attention_mask: torch.Tensor = None,
    labels: Optional[Union[list[torch.Tensor], torch.Tensor,
                           dict[str, torch.Tensor]]] = None,
) -> list[dict[str, torch.Tensor]]:
    micro_batches: list[dict[str, torch.Tensor]] = []
    batch_size = input_ids.shape[0] if isinstance(
        input_ids, torch.Tensor) else len(input_ids)
    if micro_batch_size <= 0 or batch_size == micro_batch_size:
        micro_batch = {}
        micro_batch['input_ids'] = input_ids
        micro_batch['attention_mask'] = attention_mask
        micro_batch['labels'] = labels
        micro_batches.append(micro_batch)
        return micro_batches
    if micro_batch_size > batch_size:
        micro_batch_size = batch_size

    num_splits = int(batch_size // micro_batch_size) + (
        batch_size % micro_batch_size > 0)
    if isinstance(input_ids, torch.Tensor):
        input_ids_split = torch.split(input_ids, micro_batch_size, dim=0)
    else:
        input_ids_split = [
            input_ids[i:i + micro_batch_size]
            for i in range(0, len(input_ids), micro_batch_size)
        ]
    attention_mask_split = (
        torch.split(attention_mask, micro_batch_size, dim=0)
        if attention_mask is not None else [None for _ in range(num_splits)])
    labels_split = (
        partition_label_by_micro_batch_size(labels, micro_batch_size,
                                            num_splits)
        if labels is not None else [None for _ in range(num_splits)])
    for i in range(num_splits):
        micro_batch = {}
        micro_batch['input_ids'] = input_ids_split[i]
        micro_batch['attention_mask'] = attention_mask_split[i]
        micro_batch['labels'] = labels_split[i]
        micro_batches.append(micro_batch)
    return micro_batches


def partition_label_by_micro_batch_size(
    labels: Union[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]],
    micro_batch_size: int,
    num_splits: int = 1,
):
    if isinstance(labels, torch.Tensor):
        return torch.split(labels, micro_batch_size, dim=0)
    if isinstance(labels, list):
        return [
            labels[i:i + micro_batch_size]
            for i in range(0, len(labels), micro_batch_size)
        ]
    if isinstance(labels, dict):
        split = [{} for _ in range(num_splits)]
        for key in labels.keys():
            if key == 'loss_factor':
                for i in range(num_splits):
                    split[i][key] = labels[key]
            else:
                tensors = partition_label_by_micro_batch_size(
                    labels[key], micro_batch_size)
                for i in range(num_splits):
                    split[i][key] = tensors[i]
        return split


def partition_list_by_micro_batch_size(
    input_ids: list[torch.Tensor],
    micro_batch_size: list[int],
    labels: list[torch.Tensor],
    attention_mask: Optional[list[torch.Tensor]] = None,
    loss_weights: Optional[list[float]] = None,
) -> list[dict]:
    length = len(input_ids)
    batch_size = input_ids[0].shape[0]
    num_splits = int(batch_size // micro_batch_size[0]) + (
        batch_size % micro_batch_size[0] > 0)
    micro_batches = [[{} for i in range(length)] for _ in range(num_splits)]
    if loss_weights is None:
        loss_weights = [None for _ in range(length)]
    if attention_mask is None:
        attention_mask = [None for _ in range(length)]
    for i in range(length):
        sub_input_ids = input_ids[i]
        sub_attention_mask = attention_mask[i]
        sub_labels = labels[i]
        sub_loss_weights = loss_weights[i]
        sub_micro_batches = partition_by_micro_batch_size(
            sub_input_ids, micro_batch_size[i], sub_attention_mask, sub_labels)
        for micro_batch_index, sub_micro_batch in enumerate(sub_micro_batches):
            micro_batches[micro_batch_index][i]['input_ids'] = sub_micro_batch[
                'input_ids']
            micro_batches[micro_batch_index][i][
                'attention_mask'] = sub_micro_batch['attention_mask']
            micro_batches[micro_batch_index][i]['labels'] = sub_micro_batch[
                'labels']
            micro_batches[micro_batch_index][i][
                'loss_weights'] = sub_loss_weights
    return micro_batches


def merge_loss_list(loss_list_mb: list[list[torch.Tensor]]):
    micro_batch_num = len(loss_list_mb)
    loss_num = len(loss_list_mb[0])
    loss_list = [i for i in range(loss_num)]
    for loss_index in range(loss_num):
        losses = []
        for batch_index in range(micro_batch_num):
            losses.append(loss_list_mb[batch_index][loss_index])
        loss_list[loss_index] = sum(losses) / micro_batch_num
    return loss_list


def get_answer_str(
    tokenizer: PreTrainedTokenizer,
    output_ids: torch.Tensor,
    answer_mask: torch.Tensor,
):
    answer_ids = output_ids * answer_mask
    zero_mask = answer_ids.eq(0)
    answer_ids = zero_mask * tokenizer.all_special_ids[0] + answer_ids

    answer_str = tokenizer.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return answer_str
