#!/usr/bin/env python3
"""Inspect an official HF chat template, rendered cases, and stop-token sources."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig


DEFAULT_CASES = [
    {
        "name": "final_assistant",
        "messages": [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
        ],
        "kwargs": {"add_generation_prompt": False},
    },
    {
        "name": "assistant_to_user",
        "messages": [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Question two"},
        ],
        "kwargs": {"add_generation_prompt": False},
    },
    {
        "name": "generation_prompt",
        "messages": [{"role": "user", "content": "Question one"}],
        "kwargs": {"add_generation_prompt": True},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an official Hugging Face tokenizer and print a JSON audit of "
            "its selected chat template, EOS sources, special tokens, and rendered cases."
        )
    )
    parser.add_argument("model", help="Official HF repo id or local model directory")
    parser.add_argument("--revision", help="Immutable HF revision or commit")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--template", help="Named template or explicit template passed to apply_chat_template")
    parser.add_argument(
        "--cases",
        type=Path,
        help=(
            "JSON file containing a list of cases. Each case has name, messages, "
            "optional tools, and optional kwargs for apply_chat_template."
        ),
    )
    return parser.parse_args()


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    cases = json.loads(path.read_text())
    if not isinstance(cases, list):
        raise ValueError("--cases must contain a JSON list")
    return cases


def normalize_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def id_details(tokenizer, value: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
        }
        for token_id in normalize_ids(value)
    ]


def load_generation_config(model: str, load_kwargs: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        return GenerationConfig.from_pretrained(model, **load_kwargs), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def render_case(tokenizer, case: dict[str, Any], template: str | None) -> dict[str, Any]:
    kwargs = dict(case.get("kwargs", {}))
    if "tools" in case:
        kwargs["tools"] = case["tools"]
    if template is not None:
        kwargs["chat_template"] = template

    try:
        text = tokenizer.apply_chat_template(case["messages"], tokenize=False, **kwargs)
        applied_ids = tokenizer.apply_chat_template(case["messages"], tokenize=True, **kwargs)
        if isinstance(applied_ids, Mapping):
            applied_ids = applied_ids["input_ids"]
        encoded_ids = tokenizer.encode(text, add_special_tokens=False)

        assistant_mask = None
        assistant_mask_error = None
        try:
            masked = tokenizer.apply_chat_template(
                case["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **kwargs,
            )
            assistant_mask = masked.get("assistant_masks")
            if hasattr(assistant_mask, "tolist"):
                assistant_mask = assistant_mask.tolist()
        except Exception as exc:
            assistant_mask_error = f"{type(exc).__name__}: {exc}"

        return {
            "name": case.get("name", "unnamed"),
            "kwargs": kwargs,
            "rendered": text,
            "apply_chat_template_ids": applied_ids,
            "encode_rendered_ids": encoded_ids,
            "ids_match": applied_ids == encoded_ids,
            "official_assistant_mask": assistant_mask,
            "official_assistant_mask_error": assistant_mask_error,
            "tokens": tokenizer.convert_ids_to_tokens(applied_ids),
            "decoded": tokenizer.decode(applied_ids, skip_special_tokens=False),
        }
    except Exception as exc:
        return {
            "name": case.get("name", "unnamed"),
            "kwargs": kwargs,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    load_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    config = AutoConfig.from_pretrained(args.model, **load_kwargs)
    generation_config, generation_config_error = load_generation_config(args.model, load_kwargs)
    selected_template = tokenizer.get_chat_template(chat_template=args.template)

    raw_template = getattr(tokenizer, "chat_template", None)
    template_sources = [selected_template]
    if isinstance(raw_template, dict):
        template_sources.extend(str(value) for value in raw_template.values())
    elif raw_template is not None:
        template_sources.append(str(raw_template))
    template_text = "\n".join(template_sources)

    eos_sources = {
        "tokenizer": {
            "value": tokenizer.eos_token,
            "ids": id_details(tokenizer, tokenizer.eos_token_id),
        },
        "config": {
            "raw": getattr(config, "eos_token_id", None),
            "ids": id_details(tokenizer, getattr(config, "eos_token_id", None)),
        },
        "generation_config": None,
    }
    if generation_config is not None:
        eos_sources["generation_config"] = {
            "raw": generation_config.eos_token_id,
            "ids": id_details(tokenizer, generation_config.eos_token_id),
            "stop_strings": getattr(generation_config, "stop_strings", None),
        }

    special_tokens = []
    for token in dict.fromkeys(tokenizer.all_special_tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        special_tokens.append(
            {
                "token": token,
                "id": token_id,
                "appears_in_selected_or_raw_template": token in template_text,
                "encoded_ids": tokenizer.encode(token, add_special_tokens=False),
            }
        )

    added_tokens = []
    for token_id, token in sorted(tokenizer.added_tokens_decoder.items()):
        token_text = str(token)
        added_tokens.append(
            {
                "token": token_text,
                "id": token_id,
                "special": bool(getattr(token, "special", False)),
                "appears_in_selected_or_raw_template": token_text in template_text,
                "encoded_ids": tokenizer.encode(token_text, add_special_tokens=False),
            }
        )

    report = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(config, "_commit_hash", None) or tokenizer.init_kwargs.get("_commit_hash"),
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_type": getattr(config, "model_type", None),
        "selected_template": selected_template,
        "raw_chat_template": raw_template,
        "special_tokens_map": tokenizer.special_tokens_map,
        "eos_sources": eos_sources,
        "generation_config_error": generation_config_error,
        "special_tokens": special_tokens,
        "added_tokens": added_tokens,
        "cases": [render_case(tokenizer, case, args.template) for case in load_cases(args.cases)],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
