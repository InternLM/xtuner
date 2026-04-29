import base64
import os
import re
from typing import Any, Dict, List

import ray

from xtuner.v1.utils import get_logger


ENABLE_INTERLEAVED_THINKING = os.getenv("ENABLE_INTERLEAVED_THINKING", "1") == "1"
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "1") == "1"

logger = get_logger()
logger.info(f"[agent_tokenize_fn] ENABLE_INTERLEAVED_THINKING={ENABLE_INTERLEAVED_THINKING}")
logger.info(f"[agent_tokenize_fn] ENABLE_THINKING={ENABLE_THINKING}")


def tokenize(
    tokenizer,
    messages,
    tools=None,
    enable_interleaved_thinking: bool = ENABLE_INTERLEAVED_THINKING,
    enable_thinking: bool = ENABLE_THINKING,
    **kwargs,
) -> dict:
    input_ids = tokenizer.encode("", add_special_tokens=False)
    thinking_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    thinking_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    routed_experts = None
    previous_routed_experts_tasks = set()

    def get_content_index(content_ids) -> int:
        content_ids_str = " ".join([str(content_id) for content_id in content_ids])
        thinking_end_ids_str = " ".join([str(thinking_end_id) for thinking_end_id in thinking_end_ids])
        return len(content_ids) - len(content_ids_str.split(thinking_end_ids_str)[-1].split())

    def split_conversation(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        final_chunks: List[List[Dict[str, Any]]] = []
        context_chunk: List[Dict[str, Any]] = []
        for message in messages:
            if message["role"] == "assistant":
                if context_chunk:
                    final_chunks.append(context_chunk)
                final_chunks.append([message])
                context_chunk = []
            else:
                context_chunk.append(message)
        if context_chunk:
            final_chunks.append(context_chunk)
        return final_chunks

    labels: List[int] = []
    logprobs: List[float] = []
    msg_num = 0
    for idx, msg in enumerate(split_conversation(messages)):
        msg_num += len(msg)
        if msg[0]["role"] == "assistant":
            if msg[0].get("raw_content_ids"):
                token_ids = msg[0]["raw_content_ids"]
            else:
                assistant_with_gen = tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
                )
                assistant_without_gen = tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
                )
                prompt = assistant_without_gen[len(assistant_with_gen) - len(assistant_without_gen) :]
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)

            content_idx = 0
            if not enable_interleaved_thinking and msg_num < len(messages) or msg[0].get("remove_thinking"):
                content_idx = get_content_index(token_ids)
            token_ids = token_ids[content_idx:]

            if msg[0].get("loss", True):
                labels.extend(token_ids)
                logprobs.extend(msg[0]["raw_content_logprobs"][content_idx:])
            else:
                labels.extend([-100] * len(token_ids))
                logprobs.extend([0] * len(token_ids))

            if (
                isinstance(msg[0].get("extra_info"), dict)
                and "routed_experts" in msg[0]["extra_info"]
                and msg[0]["extra_info"]["routed_experts"] is not None
            ):
                routed_experts_ref = msg[0]["extra_info"]["routed_experts"]
                if isinstance(routed_experts_ref, ray.ObjectRef):
                    if routed_experts_ref.hex() in previous_routed_experts_tasks:
                        logger.warning(
                            "[tokenize_fn] Detected repeated routed_experts_ref, setting routed_experts to None to avoid errors."
                        )
                        routed_experts = None
                    else:
                        routed_experts = routed_experts_ref
                        previous_routed_experts_tasks.add(routed_experts_ref.hex())
                else:
                    assert isinstance(routed_experts_ref, str), (
                        f"Expected routed_experts_ref to be a base64 string, but got {type(routed_experts_ref)}"
                    )
                    ref_bytes = base64.b64decode(routed_experts_ref.encode("utf-8"))
                    routed_experts = ref_bytes
            else:
                routed_experts = None

        else:
            prompt = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                tools=tools if idx == 0 else None,
                enable_thinking=enable_thinking,
            )
            if (
                not enable_interleaved_thinking
                and msg_num < len(messages)
                and re.search(r"assistant\n<think>\s*$", prompt)
            ):
                prompt = prompt.rsplit("<think>", 1)[0]
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if idx > 0:
                token_ids = tokenizer.encode("\n", add_special_tokens=False) + token_ids
            if isinstance(msg[-1].get("extra_info"), dict) and msg[-1]["extra_info"].get("previous_completions", []):
                token_ids += tokenizer.encode(
                    "".join(msg[-1]["extra_info"]["previous_completions"]), add_special_tokens=False
                )
                token_ids += thinking_start_ids
            labels.extend([-100] * len(token_ids))
            logprobs.extend([0] * len(token_ids))
        input_ids.extend(token_ids)
    assert len(input_ids) == len(labels) == len(logprobs)
    return {"input_ids": input_ids, "labels": labels, "logprobs": logprobs, "routed_experts": routed_experts}
