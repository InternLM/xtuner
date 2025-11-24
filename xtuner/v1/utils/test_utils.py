from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import transformers
from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_utils import sort_frames


# from liger-kernel
def assert_verbose_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_print=5):
    """Assert that two tensors are element-wise equal within a tolerance,
    providing detailed information about mismatches.

    Parameters:
    tensor1 (torch.Tensor): First tensor to compare.
    tensor2 (torch.Tensor): Second tensor to compare.
    rtol (float): Relative tolerance.
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    # Check if the shapes of the tensors match
    if tensor1.shape != tensor2.shape:
        raise AssertionError("Input tensors must have the same shape.")

    # Calculate the difference between the tensors
    diff = torch.abs(tensor1 - tensor2)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(tensor2)

    # Find tolerance mismatched elements
    tol_mismatched = diff > tolerance

    # Find nan mismatched elements
    nan_mismatched = torch.logical_xor(torch.isnan(tensor1), torch.isnan(tensor2))

    # Find +inf mismatched elements
    posinf_mismatched = torch.logical_xor(torch.isposinf(tensor1), torch.isposinf(tensor2))
    # Find -inf mismatched elements
    neginf_mismatched = torch.logical_xor(torch.isneginf(tensor1), torch.isneginf(tensor2))

    # Find all mismatched elements
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Check if all elements are close
    all_close = num_mismatched == 0

    # Raise AssertionError with detailed information if there are mismatches
    if not all_close and num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        print_count = min(max_print, num_mismatched)
        for index in mismatched_indices[:print_count]:
            i = tuple(index.tolist())
            mismatch_details.append(f"Mismatch at index {i}: tensor1[{i}] = {tensor1[i]}, tensor2[{i}] = {tensor2[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")

        raise AssertionError("\n".join(mismatch_details))


def init_data_mesh(device, sp_size):
    world_size = dist.get_world_size()
    dp_size = world_size // sp_size
    data_mesh = init_device_mesh(
        device,
        (dp_size, sp_size),
        mesh_dim_names=("dp", "sp"),
    )
    return data_mesh


# only for test

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IGNORE_TOKEN_ID = -100


def preprocess_intern_s1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    ds_name: str = None,  # type: ignore
    num_image: int = 1,
    prompt_only: bool = False,
    system_prompt: str = None,  # type: ignore
    max_length: int | None = None,
) -> Dict[str, list[int]]:
    if max_length is not None:
        tokenizer.model_max_length = max_length
    assert len(sources) == 1, "ERROR: process only the first conversations"
    conversations = sources[0]

    # process pretrain data
    if conversations[0]["from"] == "pretrain":
        assert len(conversations) == 1, "ERROR: pretrain data should only have one message"
        input_text = conversations[0]["value"]
        if not text_only:
            image_cnt = input_text.count("<image>")
            assert image_cnt == num_image, (
                f"ERROR: {ds_name}, image_cnt: {image_cnt} != num_image: {num_image}, {input_text}"
            )
            for i in range(num_image):
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                input_text = input_text.replace("<image>", image_tokens, 1)
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        labels = input_ids.copy()
        # ignore loss for image tokens
        img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        labels[labels == img_start_token_id] = -100
        labels[labels == img_context_token_id] = -100
        labels[labels == img_end_token_id] = -100

        if len(input_ids) > tokenizer.model_max_length:
            print(
                f"WARNING: input_ids length {len(input_ids)} exceeds "
                f"model_max_length {tokenizer.model_max_length}. truncated!"
            )
            input_ids = input_ids[: tokenizer.model_max_length]
            labels = labels[: tokenizer.model_max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        if not text_only:
            assert (input_ids == img_end_token_id).sum() == num_image, (
                f"ERROR: image tokens are truncated, this dataset is {ds_name}"
            )

        assert input_ids.size() == labels.size()

        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return dict(input_ids=input_ids.tolist(), labels=labels.tolist())

    if conversations[0]["from"] == "system":
        system_prompt = conversations[0]["value"]
        conversations = conversations[1:]  # remove system prompt
    else:
        system_prompt = system_prompt

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation["from"] == "human":
                image_cnt = conversation["value"].count("<image>")
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}"
                    conversation["value"] = conversation["value"].replace("<image>", image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        # if current_image_idx < num_image, it means <image> placeholder is less than num_image
        assert current_image_idx == num_image, (
            f"ERROR: current_image_idx: {current_image_idx} != num_image: {num_image}"
        )

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        roles.append("system")
    if not prompt_only:
        for conversation in conversations:
            if conversation["from"] == "human":
                batches.append(f"<|im_start|>user\n{conversation['value']}<|im_end|>\n")
                roles.append("human")
            elif conversation["from"] == "gpt":
                batches.append(f"<|im_start|>assistant\n{conversation['value']}<|im_end|>\n")
                roles.append("gpt")
            elif conversation["from"] == "function":
                batches.append(f"<|im_start|>function\n{conversation['value']}<|im_end|>\n")
                roles.append("function")
            else:
                raise ValueError(f"ERROR role: {conversation['from']}: dataset name: {ds_name}")
    else:
        assert conversations[0]["from"] == "human", (
            "ERROR: prompt_only is True, the first message should be from human"
        )
        batches.append(f"<|im_start|>user\n{conversations[0]['value']}<|im_end|>\n")
        roles.append("human")

    input_ids = tokenizer(
        batches,
        return_tensors="np",
        padding=False,
        truncation=False,
    ).input_ids

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer("<|im_start|>assistant\n", return_tensors="np").input_ids[0]
    ignore_len = ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == "system" or role == "human" or role == "function":
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == "gpt":
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise ValueError(f"ERROR role: {role}: dataset name: {ds_name}")

    input_ids = np.concatenate(final_input_ids)
    targets = np.concatenate(final_targets)

    if len(input_ids) > tokenizer.model_max_length:
        print(
            f"WARNING: input_ids length {len(input_ids)} exceeds "
            f"model_max_length {tokenizer.model_max_length}. truncated!"
        )
        input_ids = input_ids[: tokenizer.model_max_length]
        targets = targets[: tokenizer.model_max_length]

    if not text_only:
        image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (input_ids == image_end_token_id).sum() == num_image, (
            f"ERROR: image tokens are truncated, this dataset is {ds_name}"
        )

    return dict(
        input_ids=cast(List[int], input_ids.tolist()),
        labels=cast(List[int], targets.tolist()),
    )


def add_video_root(messages: list[dict], video_root: Path | str):
    video_root = Path(video_root)
    for msg in messages:
        if "content" not in msg:
            continue
        content_list = msg["content"]
        if not isinstance(content_list, list):
            raise TypeError("content should be a list of dict")
        for content in content_list:
            if "type" not in content or content["type"] != "video":
                continue
            content_path = video_root / content["path"]
            if Path(content_path).is_dir():
                image_list = sort_frames(list(content_path.glob("*.jpg")))
                new_image_list = []
                for image in image_list:
                    new_image_list.append(str(content_path / image))
                content["path"] = new_image_list
            else:
                content["path"] = str(content_path)
