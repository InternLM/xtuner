# Copyright (c) OpenMMLab. All rights reserved.
import io
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers.trainer_pt_utils import LabelSmoother

from .conversation import get_conv_template


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
IGNORE_INDEX = -100


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type="imagenet"):
    if normalize_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )

    return transform


def preprocess_fast(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,  # type: ignore
    num_image: int = 1,
    prompt_only: bool = False,
    system_prompt: str = None,  # type: ignore
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "system": "system", "pretrain": "pretrain"}  # type: ignore

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

    assert len(conv.messages) % 2 == 0, f"{ds_name}, {len(conv.messages)}, {conv.messages}"
    inputs = conv.messages[::2]
    outputs = conv.messages[1::2]

    input_ids, labels = [], []
    # input_texts = ''
    if system_prompt is None:
        system_prompt = conv.system_template.format(system_message=conv.system_message)
    input_text = system_prompt + conv.sep
    # input_texts += input_text
    input_encode = tokenizer.encode(input_text, add_special_tokens=True)
    input_ids += input_encode
    labels += [IGNORE_INDEX] * len(input_encode)

    real_num_images = 0
    for input_, output_ in zip(inputs, outputs):
        # output_[0] = '<|assistant|>\n'
        # 放到 input 而不是 output 是为了和官方对齐
        input_text = "".join(input_) + conv.sep + output_[0]

        if not text_only:
            real_num_images += input_text.count("<image>")
            for i in range(num_image):
                image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}"
                input_text = input_text.replace("<image>", image_tokens, 1)
        assert "<image>" not in input_text, f"error: {ds_name}, {input_text}"
        output_text = output_[1] + conv.sep

        input_encode = tokenizer.encode(input_text, add_special_tokens=False)
        output_encode = tokenizer.encode(output_text, add_special_tokens=False)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)

        if prompt_only is False:
            input_ids += output_encode
            labels += output_encode

        # input_texts += input_text
        # input_texts += output_text

    if not text_only:
        assert real_num_images == num_image, f"{ds_name} data error: {real_num_images} vs. {num_image}"
        # print(input_texts)
        # assert input_ids.count(32013) == num_image_token_list[
        #     0], f'error1: {input_ids}, {num_image_token_list[0]}, {input_texts}'
    if len(input_ids) > tokenizer.model_max_length:
        print(
            f"WARNING: input_ids length {len(input_ids)} exceeds "
            f"model_max_length {tokenizer.model_max_length}. truncated!"
        )
        input_ids = input_ids[: tokenizer.model_max_length]
        labels = labels[: tokenizer.model_max_length]

    input_ids = torch.tensor(input_ids, dtype=torch.long)[None]  # type: ignore
    labels = torch.tensor(labels, dtype=torch.long)[None]  # type: ignore
    assert input_ids.size() == labels.size()  # type: ignore
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(0),  # type: ignore
    )


def preprocess_internvl2_5(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,  # type: ignore
    num_image: int = 1,
    prompt_only: bool = False,
    system_prompt: str = None,  # type: ignore
) -> Dict:
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

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    if conversations[0]["from"] == "system":
        system_prompt = conversations[0]["value"]
        conversations = conversations[1:]  # remove system prompt
    elif system_prompt is not None:
        system_prompt = system_prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

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

    add_bos_token = getattr(tokenizer, "add_bos_token", False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors="np",
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer("<|im_start|>assistant\n", return_tensors="np").input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
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

    input_ids = torch.tensor(np.concatenate(final_input_ids))
    targets = torch.tensor(np.concatenate(final_targets))

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

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(0),  # 0 is default pad token
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if max_num >= i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_num_patch(size, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    if use_thumbnail and blocks > 1:
        blocks += 1
    return blocks


def packing_collate(features, pack_batch=True, pad_id=0):
    input_ids = []
    labels = []
    pixel_values = []
    num_tokens = []
    num_img_tokens = []
    image_flags = []

    for data in features:
        input_ids.append(torch.LongTensor(data["input_ids"]))
        labels.append(torch.LongTensor(data["labels"]))
        num_tokens.extend(data["num_tokens"])
        num_img_tokens.extend(data["num_img_tokens"])
        pixel_values.append(data["pixel_values"])
        image_flags.append(data["image_flags"])

    attention_mask = [ids.ne(pad_id) for ids in input_ids]
    num_tokens = torch.IntTensor(num_tokens)
    num_img_tokens = torch.IntTensor(num_img_tokens)

    if len(features) > 1 and pack_batch:
        # batch packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)
        image_flags = torch.cat(image_flags, dim=0)
        pixel_values = torch.cat(pixel_values, dim=0)
    elif len(features) > 1 and not pack_batch:
        raise NotImplementedError
    else:
        raise NotImplementedError

    data_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask.bool(),
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "num_tokens": num_tokens,
        "num_img_tokens": num_img_tokens,
    }

    return data_dict
