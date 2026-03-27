from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
import threading
from typing import Any, Generic, TypeVar

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils import get_logger

@dataclass
class InteractionNode:
    """A single interaction turn inside a session."""

    request_id: str
    session_uid: int
    messages: list[dict[str, Any]]
    tools: list | None = None
    output_message_list: list[dict[str, Any]] | None = None
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    parent: InteractionNode | None = None
    @property
    def parent_messages(self) -> list[dict[str, Any]]:
        """Messages that a child turn should start with (input + output)."""
        return self.messages + (self.output_message_list or [])

    @property
    def is_complete(self) -> bool:
        return self.output_message_list is not None

    def is_parent_of(self, child_messages: list[dict[str, Any]]) -> bool:
        return _is_prefix(self.parent_messages, child_messages)

    def has_similar_last_message(
        self,
        child_messages: list[dict[str, Any]],
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        if not _is_prefix(self.messages, child_messages):
            return False, None, None
        return _is_similar_on_last_message(self.parent_messages, child_messages)


def _is_prefix(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """Return True if *a* is a prefix of *b*."""
    if len(a) > len(b):
        return False
    return b[: len(a)] == a


def _is_similar_on_last_message(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
    if len(a) > len(b):
        return False, None, None
    last_a = a[-1]
    last_b = b[len(a) - 1]
    common_keys = set(last_a.keys()) & set(last_b.keys())
    for key in common_keys:
        if last_a[key] != last_b[key]:
            return False, None, None
    diff_a = {k: v for k, v in last_a.items() if k not in common_keys}
    diff_b = {k: v for k, v in last_b.items() if k not in common_keys}
    return True, diff_a, diff_b


def _find_kth(tokens: list[int], target: int, k: int) -> int:
    """Return the index of the *k*‑th occurrence of *target*, or ``-1``."""
    count = 0
    for idx, tok in enumerate(tokens):
        if tok == target:
            count += 1
            if count == k:
                return idx
    return -1

class InteractionCache:
    """Thread‑safe, LRU‑bounded cache of interaction turns.

    Responsibilities:
      * register / drop / finalize interaction nodes
      * find the best parent node for an incoming interaction
      * build incremental prompt‑ids by reusing the parent's token prefix
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None,
        max_cached: int = 10000,
    ):
        self._tokenizer = tokenizer
        self._max_cached = max_cached
        self._cache: OrderedDict[str, InteractionNode] = OrderedDict()
        self._lock = threading.RLock()

    # ── public API ──────────────────────────────────────────────────

    def register(
        self,
        request_id: str,
        rollout_state: RolloutState,
    ) -> InteractionNode | None:
        """Create, link, and store a new :class:`InteractionNode`.

        Returns ``None`` when the rollout has no ``session_uid`` (i.e. it is
        a single‑turn request that does not need caching).
        """
        if rollout_state.session_uid is None:
            return None

        node = InteractionNode(
            request_id=request_id,
            session_uid=rollout_state.session_uid,
            messages=deepcopy(rollout_state.message),
            tools=deepcopy(rollout_state.tools),
        )
        with self._lock:
            node.parent = self._find_parent(node)
            self._cache[request_id] = node
            self._cache.move_to_end(request_id)
            self._evict()
        return node

    def drop(self, request_id: str) -> None:
        with self._lock:
            self._cache.pop(request_id, None)

    def finalize(
        self,
        node: InteractionNode,
        output_message_list: list[dict[str, Any]],
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> None:
        """Mark *node* as complete so it can serve as a parent later."""
        node.output_message_list = deepcopy(output_message_list)
        node.prompt_ids = list(prompt_ids)
        node.response_ids = list(response_ids)

    def build_prompt_ids(
        self,
        request_id: str,
        node: InteractionNode | None,
        messages: list[dict[str, Any]],
        tools: list | None,
    ) -> list[int]:
        """Return prompt token ids, incremental when a parent exists.

        Falls back to full tokenization when incremental alignment fails.
        """
        full_ids = self._tokenize_messages(messages, tools)

        if node is None or node.parent is None:
            return full_ids

        parent = node.parent
        parent_ids = self._build_parent_token_ids(parent)
        incremental_ids = self._try_incremental_alignment(
            request_id=request_id,
            session_uid=node.session_uid,
            full_prompt_ids=full_ids,
            parent_prompt_ids=parent_ids,
        )
        return incremental_ids

    def _find_parent(self, node: InteractionNode) -> InteractionNode | None:
        candidates = sorted(
            (
                n
                for n in self._cache.values()
                if n.session_uid == node.session_uid and n.is_complete
            ),
            key=lambda n: len(n.messages),
            reverse=True,
        )
        for parent in candidates:
            if parent.is_parent_of(node.messages):
                return parent

            is_similar, diff_parent, diff_child = parent.has_similar_last_message(
                node.messages,
            )
            if is_similar:
                logger.warning(
                    "Found a candidate parent with similar last message but "
                    "not a strict prefix. "
                    f"Parent-only keys: {diff_parent} | "
                    f"child-only keys: {diff_child}"
                )
        return None

    def _build_parent_token_ids(self, parent: InteractionNode) -> list[int]:
        """Concatenate the parent's prompt + response + eos."""
        prompt_ids = list(parent.prompt_ids)
        response_ids = list(parent.response_ids)

        eos = self._eos_token_id()
        if eos is not None and (not response_ids or response_ids[-1] != eos):
            response_ids.append(eos)

        return prompt_ids + response_ids

    def _try_incremental_alignment(
        self,
        request_id: str,
        session_uid: int,
        full_prompt_ids: list[int],
        parent_prompt_ids: list[int],
    ) -> list[int]:
        """Try to splice *parent_prompt_ids* into *full_prompt_ids*.

        Falls back to *full_prompt_ids* on any alignment failure.
        """
        eos = self._eos_token_id()
        if not parent_prompt_ids or eos is None:
            return list(full_prompt_ids)

        parent_eos_count = parent_prompt_ids.count(eos)
        if parent_eos_count <= 0:
            # No eos in parent → cannot determine a safe split point.
            logger.debug(
                f"[{request_id}] Parent has no eos tokens; "
                "falling back to full prompt."
            )
            return list(full_prompt_ids)

        child_split_idx = _find_kth(full_prompt_ids, eos, parent_eos_count)
        if child_split_idx == -1:
            logger.warning(
                f"[{request_id}] Cannot find {parent_eos_count}-th eos in "
                f"child prompt (only {full_prompt_ids.count(eos)} eos found); "
                "falling back to full prompt."
            )
            return list(full_prompt_ids)

        # ── Validate: the overlapping region MUST actually match ─────
        # The parent's tokens should equal full_prompt_ids[:child_split_idx+1]
        # at least up to min(len(parent), child_split_idx+1).
        overlap_len = child_split_idx + 1
        if len(parent_prompt_ids) != overlap_len:
            logger.warning(
                f"[{request_id}] Parent token length ({len(parent_prompt_ids)}) "
                f"differs from expected overlap ({overlap_len}); "
                "falling back to full prompt."
            )
            self._warn_prefix_mismatch(
                request_id, session_uid, full_prompt_ids, parent_prompt_ids,
            )
            return list(full_prompt_ids)

        if full_prompt_ids[:overlap_len] != parent_prompt_ids[:overlap_len]:
            logger.warning(
                f"[{request_id}] Parent prefix tokens do not match the "
                "child prompt tokens in the overlap region; "
                "falling back to full prompt."
            )
            self._warn_prefix_mismatch(
                request_id, session_uid, full_prompt_ids, parent_prompt_ids,
            )
            return list(full_prompt_ids)

        return parent_prompt_ids + full_prompt_ids[overlap_len:]

    def _eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id if self._tokenizer is not None else None

    def _tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list | None,
    ) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer is required for token-in/token-out protocol adapters."
            )
        return list(
            self._tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    def _evict(self) -> None:
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)

    @staticmethod
    def _warn_prefix_mismatch(
        request_id: str,
        session_uid: int,
        full_prompt_ids: list[int],
        parent_prompt_ids: list[int],
    ) -> None:
        prefix_len = min(len(full_prompt_ids), len(parent_prompt_ids))
        if (
            len(full_prompt_ids) < len(parent_prompt_ids)
            or full_prompt_ids[:prefix_len] != parent_prompt_ids[:prefix_len]
        ):
            logger.warning(
                "Parent prefix tokens mismatch after fallback "
                f"for request_id={request_id} session_uid={session_uid}. "
                f"parent_prefix_len={len(parent_prompt_ids)} "
                f"full_prompt_len={len(full_prompt_ids)}"
            )