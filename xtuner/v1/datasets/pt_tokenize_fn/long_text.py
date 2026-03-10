import copy
import hashlib
import inspect
import warnings
from difflib import SequenceMatcher
from typing import Annotated, Any, Union, cast

from cyclopts import Parameter
from pydantic import ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.datasets.data_item import CacheItem, DataItem

from ..utils import CachableTokenizeFunction, tokenizer_xxhash
from .text import PretrainTokenizeFunction, PretrainTokenizeFunctionConfig


class LongTextPretrainTokenizeFunction(PretrainTokenizeFunction):
    """Tokenize function for long texts by splitting into char-level chunks
    with overlapping windows and LCS-based boundary merging."""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        chunk_size: int = 4096,
        tokenizer_chunk_chars: int = 4096,
        overlap_chars: int = 512,
        min_chunk_tokens: int = 0,
        add_eos_token: bool = True,
        add_bos_token: bool = False,
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.tokenizer_chunk_chars = tokenizer_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_tokens = min_chunk_tokens
        # Call grandparent __init__ via CachableTokenizeFunction to avoid
        # re-running PretrainTokenizeFunction.__init__ attribute assignments
        self.add_eos_token = add_eos_token
        self.add_bos_token = add_bos_token
        self._tokenizer_hash = tokenizer_hash
        self._hash = hash
        CachableTokenizeFunction.__init__(self, tokenizer)  # type: ignore[arg-type]

    def _get_text(self, item: dict) -> str:
        if "messages" in item:
            messages = item["messages"]
            assert messages[0]["role"] == "pretrain"
            return messages[0]["content"]
        return item["content"]

    def _make_chunk_text(self, text: str, char_start: int, char_end: int | None) -> str:
        """Build the text for a chunk, prepending BOS only if first chunk,
        appending EOS only if last chunk."""
        is_first = char_start == 0
        is_last = char_end is None or char_end >= len(text)

        chunk = text[char_start:char_end]

        if self.add_bos_token and is_first:
            assert self.tokenizer.bos_token is not None
            chunk = self.tokenizer.bos_token + chunk
        if self.add_eos_token and is_last:
            assert self.tokenizer.eos_token is not None
            chunk = chunk + self.tokenizer.eos_token
        return chunk

    def shard_char_boundaries(self, text: str) -> list[tuple[int, int]]:
        """Return list of (char_start, char_end) covering the full text, where
        each segment corresponds to approximately chunk_size tokens.

        Uses overlapping tokenizer windows + LCS matching to find accurate token boundaries.
        """
        is_fast = isinstance(self.tokenizer, PreTrainedTokenizerFast)

        start = 0
        prev_tokens: list[int] = []
        buffer_tokens: list[int] = []
        buffer_abs_chars: list[int] = []  # absolute char offset of each token in buffer
        chunk_boundaries: list[int] = [0]  # char start positions of each chunk

        while start < len(text):
            piece = text[start : start + self.tokenizer_chunk_chars]

            if is_fast:
                enc = self.tokenizer(
                    piece,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                cur_tokens: list[int] = enc["input_ids"]
                offset_map: list[tuple[int, int]] = enc["offset_mapping"]
                # Compute absolute char offsets
                abs_char_offsets = [start + om[0] for om in offset_map]
            else:
                # Slow tokenizer fallback: no offset mapping
                cur_tokens = self.tokenizer.encode(piece, add_special_tokens=False)
                # Approximate absolute char offsets via decode + prefix search
                abs_char_offsets = self._approx_abs_char_offsets(text, start, cur_tokens)

            if start != 0 and prev_tokens:
                # Overlap fusion via LCS
                overlap_text = text[start : start + self.overlap_chars]
                overlap_tokens = self.tokenizer.encode(overlap_text, add_special_tokens=False)
                overlap_len = len(overlap_tokens)

                prev_overlap = prev_tokens[-overlap_len:] if overlap_len <= len(prev_tokens) else prev_tokens
                cur_overlap = cur_tokens[:overlap_len]

                match = SequenceMatcher(
                    None,
                    list(reversed(prev_overlap)),
                    list(reversed(cur_overlap)),
                    autojunk=False,
                ).find_longest_match(0, len(prev_overlap), 0, len(cur_overlap))

                # Rollback buffer by match.a tokens from tail
                if match.a > 0:
                    buffer_tokens = buffer_tokens[: -match.a]
                    buffer_abs_chars = buffer_abs_chars[: -match.a]

                # Skip match.b tokens from head of cur_tokens
                trim = overlap_len - match.b
                cur_tokens = cur_tokens[trim:]
                abs_char_offsets = abs_char_offsets[trim:]

            buffer_tokens.extend(cur_tokens)
            buffer_abs_chars.extend(abs_char_offsets)

            # Flush complete chunks when we have >= 2 * chunk_size tokens
            while len(buffer_tokens) >= 2 * self.chunk_size:
                # The chunk ends where the (chunk_size+1)-th token starts
                chunk_end_char = buffer_abs_chars[self.chunk_size]
                chunk_boundaries.append(chunk_end_char)
                buffer_tokens = buffer_tokens[self.chunk_size :]
                buffer_abs_chars = buffer_abs_chars[self.chunk_size :]

            prev_tokens = cur_tokens
            # Advance pointer
            next_start = start + self.tokenizer_chunk_chars - self.overlap_chars
            if next_start <= start:
                next_start = start + 1  # safety guard
            start = next_start

        # Emit remaining buffer as final chunks
        while buffer_tokens:
            if len(buffer_tokens) > self.chunk_size:
                chunk_end_char = buffer_abs_chars[self.chunk_size]
                chunk_boundaries.append(chunk_end_char)
                buffer_tokens = buffer_tokens[self.chunk_size :]
                buffer_abs_chars = buffer_abs_chars[self.chunk_size :]
            else:
                # Last (possibly short) chunk
                break

        # Final boundary is end of text
        chunk_boundaries.append(len(text))

        # Convert boundaries to (start, end) pairs
        result = []
        for i in range(len(chunk_boundaries) - 1):
            result.append((chunk_boundaries[i], chunk_boundaries[i + 1]))

        return result

    def _approx_abs_char_offsets(self, text: str, start: int, cur_tokens: list[int]) -> list[int]:
        """Approximate absolute char offsets for slow tokenizers by decoding
        each prefix and searching in original text."""
        warnings.warn(
            "Tokenizer does not support `return_offsets_mapping` (slow tokenizer). "
            "Using approximate char boundary estimation. For better accuracy, "
            "use a fast tokenizer (PreTrainedTokenizerFast).",
            stacklevel=5,
        )
        offsets = []
        running_chars = start
        for i, tok in enumerate(cur_tokens):
            offsets.append(running_chars)
            decoded = self.tokenizer.decode([tok], skip_special_tokens=False)
            running_chars += max(len(decoded), 1)
        return offsets

    def __call__(self, item: Any, **kwargs: Any) -> DataItem | CacheItem:  # type: ignore[override]
        char_start: int | None = kwargs.get("char_start")
        char_end: int | None = kwargs.get("char_end")
        text = self._get_text(item)

        if self.state == "cache":
            boundaries = self.shard_char_boundaries(text)
            chunks = []
            for cs, ce in boundaries:
                chunk_text = self._make_chunk_text(text, cs, ce)
                nt = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
                if nt < self.min_chunk_tokens and (cs, ce) == boundaries[-1]:
                    continue  # drop trailing short chunk
                chunks.append({"char_start": cs, "char_end": ce, "num_tokens": nt})
            total = sum(c["num_tokens"] for c in chunks)
            return cast(CacheItem, {"num_tokens": total, "chunks": chunks})

        # Runtime: tokenize specified char range
        if char_start is not None:
            chunk_text = self._make_chunk_text(text, char_start, char_end)
            input_ids = self.tokenizer.encode(chunk_text, add_special_tokens=False)
            num_tokens = len(input_ids)
            if num_tokens == 0:
                labels = []
            else:
                labels = copy.deepcopy(input_ids)
                labels[0] = -100
            return {"input_ids": input_ids, "labels": labels, "num_tokens": num_tokens}

        # Full-text fallback (no char range specified)
        return super().__call__(item)

    def hash(self) -> str:
        if self._hash is None:
            if self._tokenizer_hash is None:
                _tokenizer_hash = tokenizer_xxhash(self.tokenizer)[:16]
            else:
                _tokenizer_hash = self._tokenizer_hash

            _source_hash = (
                hashlib.sha256(inspect.getsource(self.__class__.__call__).encode()).hexdigest()[:16]
                + hashlib.sha256(inspect.getsource(self.__class__.__init__).encode()).hexdigest()[:16]
            )

            self._hash = (
                f"{_tokenizer_hash}_{_source_hash}"
                f"_{self.add_bos_token}_{self.add_eos_token}"
                f"_{self.chunk_size}_{self.tokenizer_chunk_chars}"
                f"_{self.overlap_chars}_{self.min_chunk_tokens}"
            )
        return self._hash


class LongTextPretrainTokenizeFunctionConfig(PretrainTokenizeFunctionConfig):
    model_config = ConfigDict(extra="forbid")
    chunk_size: Annotated[int, Parameter(group="tokenize_fn")] = 4096
    tokenizer_chunk_chars: Annotated[int, Parameter(group="tokenize_fn")] = 4096
    overlap_chars: Annotated[int, Parameter(group="tokenize_fn")] = 512
    min_chunk_tokens: Annotated[int, Parameter(group="tokenize_fn")] = 0

    def build(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        tokenizer_hash: str | None = None,
        anno_name: str | None = None,
        **kwargs,
    ) -> LongTextPretrainTokenizeFunction:
        return LongTextPretrainTokenizeFunction(
            tokenizer=tokenizer,
            chunk_size=self.chunk_size,
            tokenizer_chunk_chars=self.tokenizer_chunk_chars,
            overlap_chars=self.overlap_chars,
            min_chunk_tokens=self.min_chunk_tokens,
            add_eos_token=self.add_eos_token,
            add_bos_token=self.add_bos_token,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
        )
