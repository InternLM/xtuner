# Copyright (c) OpenMMLab. All rights reserved.
"""Environment-controlled tokenizer snapshots for data-pipeline debugging."""

import copy
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from xtuner.v1.utils import get_logger


logger = get_logger()
_DEBUG_SAMPLE_ENV = "XTUNER_TOKENIZE_DEBUG_SAMPLES"
_DEBUG_SAMPLE_SEED = 42


def _true_spans(mask: list[bool]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _decode_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _token_pieces(tokenizer, token_ids: list[int]) -> list[str]:
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    if isinstance(pieces, str):
        return [pieces]
    return [str(piece) for piece in pieces]


def _summarize_field(value: Any) -> Any:
    value_module = type(value).__module__
    if value_module.startswith("torch") and hasattr(value, "shape"):
        return {
            "kind": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if value_module.startswith("numpy") and hasattr(value, "shape"):
        if len(value.shape) == 0 and hasattr(value, "item"):
            return value.item()
        return {
            "kind": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def _render_qwen_internal(tokenize_fn, raw_data: dict) -> tuple[str, list[bool]]:
    from xtuner.v1.data_proto.messages.qwen35_chat import qwen35_tokenize_fn_fastspeed
    from xtuner.v1.data_proto.messages.qwen36_chat import qwen36_tokenize_fn_fastspeed

    renderers = {
        "qwen3.5-vl": qwen35_tokenize_fn_fastspeed,
        "qwen3.6-vl": qwen36_tokenize_fn_fastspeed,
    }

    messages = copy.deepcopy(raw_data["messages"])
    chat_template = tokenize_fn.chat_template
    if chat_template.default_system is not None:
        if messages[0]["role"] == "system":
            messages[0]["content"] = chat_template.default_system
        else:
            messages.insert(0, {"role": "system", "content": chat_template.default_system})

    return renderers[tokenize_fn.chat_template_name](
        messages,
        tools=copy.deepcopy(raw_data.get("tools")),
        add_vision_id=tokenize_fn.add_vision_id,
        return_labels=False,
    )


def _build_record(
    *,
    tokenizer,
    raw_data: dict,
    rendered_text: str,
    character_loss_mask: list[bool],
    tokenized_data: dict,
    dataset_name: str,
    dataset_path: str,
    sample_index: int,
) -> dict:
    input_ids = [int(token_id) for token_id in tokenized_data["input_ids"]]
    labels = [int(label) for label in tokenized_data["labels"]]
    if len(input_ids) != len(labels):
        raise ValueError(f"input_ids length {len(input_ids)} != labels length {len(labels)}")

    loss_mask = [label != -100 for label in labels]
    loss_spans = []
    for start, end in _true_spans(loss_mask):
        token_ids = labels[start:end]
        loss_spans.append(
            {
                "start": start,
                "end": end,
                "token_ids": token_ids,
                "token_pieces": _token_pieces(tokenizer, token_ids),
                "decoded_text": _decode_ids(tokenizer, token_ids),
            }
        )

    return {
        "source": {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "sample_index": sample_index,
        },
        "raw_data": raw_data,
        "rendered": {
            "text": rendered_text,
            "character_count": len(rendered_text),
            "loss_character_count": sum(character_loss_mask),
            "loss_character_spans": [
                {
                    "start": start,
                    "end": end,
                    "text": rendered_text[start:end],
                }
                for start, end in _true_spans(character_loss_mask)
            ],
        },
        "tokenized": {
            "input_ids": input_ids,
            "labels": labels,
            "decoded_text": _decode_ids(tokenizer, input_ids),
            "token_count": len(input_ids),
            "other_fields": {
                key: _summarize_field(value)
                for key, value in tokenized_data.items()
                if key not in {"input_ids", "labels"}
            },
        },
        "loss": {
            "token_count": sum(loss_mask),
            "masked_token_count": len(labels) - sum(loss_mask),
            "token_spans": loss_spans,
        },
    }


def _write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


def _read_raw_data(dataset, sample_index: int) -> dict:
    with open(dataset.path) as file:
        file.seek(dataset.offsets[sample_index])
        return json.loads(file.readline())


def _sample_indices(dataset_size: int, sample_count: int) -> list[int]:
    count = min(sample_count, dataset_size)
    return sorted(random.Random(_DEBUG_SAMPLE_SEED).sample(range(dataset_size), count))


def maybe_dump_tokenize_debug_samples(*, dataset, tokenize_fn, dataset_name: str) -> None:
    """Dump the first N records when XTUNER_TOKENIZE_DEBUG_SAMPLES is set."""
    raw_sample_count = os.environ.get(_DEBUG_SAMPLE_ENV, "0")
    try:
        sample_count = int(raw_sample_count)
    except ValueError as error:
        raise ValueError(f"{_DEBUG_SAMPLE_ENV} must be an integer, got {raw_sample_count!r}") from error
    if sample_count <= 0:
        return

    chat_template_name = getattr(tokenize_fn, "chat_template_name", None)
    if chat_template_name not in {"qwen3.5-vl", "qwen3.6-vl"}:
        logger.warning(
            f"[Dataset] Skip tokenizer debug snapshots for [{dataset_name}]: "
            "only chat_template='qwen3.5-vl' or 'qwen3.6-vl' is supported."
        )
        return

    dataset_path = str(dataset.path)
    path_hash = hashlib.sha256(dataset_path.encode()).hexdigest()[:8]
    output_root = Path(os.environ.get("WORK_DIR", os.getcwd())) / "tokenize_debug_samples"
    output_dir = (
        output_root
        / _safe_path_component(chat_template_name)
        / f"{_safe_path_component(dataset_name)}__{_safe_path_component(Path(dataset_path).name)}__{path_hash}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_state = tokenize_fn.state
    tokenize_fn.set_state("runtime")
    generated_files = []
    sample_indices = _sample_indices(len(dataset), sample_count)
    try:
        for sample_index in sample_indices:
            output_path = output_dir / f"sample_{sample_index:06d}.json"
            raw_data = _read_raw_data(dataset, sample_index)
            try:
                rendered_text, character_loss_mask = _render_qwen_internal(tokenize_fn, raw_data)
                tokenized_data = tokenize_fn(
                    copy.deepcopy(raw_data),
                    media_root=getattr(dataset, "media_root", ""),
                )
                record = _build_record(
                    tokenizer=tokenize_fn.tokenizer,
                    raw_data=raw_data,
                    rendered_text=rendered_text,
                    character_loss_mask=character_loss_mask,
                    tokenized_data=tokenized_data,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    sample_index=sample_index,
                )
            except Exception as error:
                record = {
                    "source": {
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "sample_index": sample_index,
                    },
                    "raw_data": raw_data,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
                logger.exception(
                    f"Failed to dump tokenizer debug sample {sample_index} "
                    f"from [{dataset_name}]{dataset_path}"
                )

            _write_json(output_path, record)
            generated_files.append(output_path.name)
    finally:
        tokenize_fn.set_state(previous_state)

    _write_json(
        output_dir / "manifest.json",
        {
            "environment_variable": _DEBUG_SAMPLE_ENV,
            "sampling": "random_without_replacement",
            "random_seed": _DEBUG_SAMPLE_SEED,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "chat_template_name": chat_template_name,
            "requested_sample_count": sample_count,
            "generated_sample_count": len(generated_files),
            "sample_indices": sample_indices,
            "files": generated_files,
        },
    )
    logger.info(
        f"[Dataset] Dumped {len(generated_files)} tokenizer debug samples "
        f"from [{dataset_name}]{dataset_path} to {output_dir}."
    )
