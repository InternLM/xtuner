# Copyright (c) OpenMMLab. All rights reserved.
import copy
import hashlib
import inspect
from typing import TYPE_CHECKING, Annotated, Dict, List, cast

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict

from xtuner.utils import IGNORE_INDEX
from xtuner.v1.datasets.data_item import DataItem
from xtuner.v1.utils import get_logger

from .utils import CachableTokenizeFunction, tokenizer_xxhash


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = get_logger()

internlm = dict(
    knowledge=dict(
        begin=dict(
            without_name="<|im_start|>knowledge\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    pretrain_meta=dict(
        begin=dict(with_name="", without_name=""),
        end="",
        loss=dict(icl=False, current=False, prefix=False, end=False),
    ),
    pretrain_content=dict(
        begin=dict(with_name="", without_name=""),
        end="</s>",
        loss=dict(icl=True, current=True, prefix=False, end=True),
    ),
    system=dict(
        begin=dict(
            with_name="<|im_start|>system name={name}\n",
            without_name="<|im_start|>system\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    user=dict(
        begin=dict(
            with_name="<|im_start|>user name={name}\n",
            without_name="<|im_start|>user\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_prefix=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
        ),
        end="",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_middle=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="",
        loss=dict(
            icl=False,
            current=True,
            prefix=False,
        ),
    ),
    answer_postfix=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    assistant=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    environment=dict(
        begin=dict(
            with_name="<|im_start|>environment name={name}\n",
            without_name="<|im_start|>environment\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    tool=dict(
        begin=dict(
            with_name="<|action_start|>{name}\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|action_end|>\n",
        belong="assistant",
    ),
    thought=dict(
        begin=dict(without_name=""),
        end="",
        belong="assistant",
    ),
    max_len=32 * 1024,
)

qwen = dict(
    knowledge=dict(
        begin=dict(
            without_name="<|im_start|>knowledge\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    pretrain_meta=dict(
        begin=dict(with_name="", without_name=""),
        end="",
        loss=dict(icl=False, current=False, prefix=False, end=False),
    ),
    pretrain_content=dict(
        begin=dict(with_name="", without_name=""),
        end="<|endoftext|>",
        loss=dict(icl=True, current=True, prefix=False, end=True),
    ),
    system=dict(
        begin=dict(
            with_name="<|im_start|>system name={name}\n",
            without_name="<|im_start|>system\n",
            name={
                "interpreter": "<|repo_name|>",
                "plugin": "<|file_sep|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    user=dict(
        begin=dict(
            with_name="<|im_start|>user name={name}\n",
            without_name="<|im_start|>user\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_prefix=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
        ),
        end="",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_middle=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="",
        loss=dict(
            icl=False,
            current=True,
            prefix=False,
        ),
    ),
    answer_postfix=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    assistant=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
            name={
                "interpreter": "<|repo_name|>",
                "plugin": "<|file_sep|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    environment=dict(
        begin=dict(
            with_name="<|im_start|>environment name={name}\n",
            without_name="<|im_start|>environment\n",
            name={
                "interpreter": "<|repo_name|>",
                "plugin": "<|file_sep|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    tool=dict(
        begin=dict(
            with_name="<tool_call>{name}\n",
            name={
                "interpreter": "<|repo_name|>",
                "plugin": "<|file_sep|>",
            },
        ),
        end="</tool_call>\n",
        belong="assistant",
    ),
    thought=dict(
        begin=dict(without_name=""),
        end="",
        belong="assistant",
    ),
)

qwen_intern = dict(
    knowledge=dict(
        begin=dict(
            without_name="<|im_start|>knowledge\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    pretrain_meta=dict(
        begin=dict(with_name="", without_name=""),
        end="",
        loss=dict(icl=False, current=False, prefix=False, end=False),
    ),
    pretrain_content=dict(
        begin=dict(with_name="", without_name=""),
        end="<|endoftext|>",
        loss=dict(icl=True, current=True, prefix=False, end=True),
    ),
    system=dict(
        begin=dict(
            with_name="<|im_start|>system name={name}\n",
            without_name="<|im_start|>system\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    user=dict(
        begin=dict(
            with_name="<|im_start|>user name={name}\n",
            without_name="<|im_start|>user\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_prefix=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
        ),
        end="",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_middle=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="",
        loss=dict(
            icl=False,
            current=True,
            prefix=False,
        ),
    ),
    answer_postfix=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    assistant=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    environment=dict(
        begin=dict(
            with_name="<|im_start|>environment name={name}\n",
            without_name="<|im_start|>environment\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    tool=dict(
        begin=dict(
            with_name="<|action_start|>{name}\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|action_end|>\n",
        belong="assistant",
    ),
    thought=dict(
        begin=dict(without_name=""),
        end="",
        belong="assistant",
    ),
)


deepseekv3 = dict(
    knowledge=dict(
        begin=dict(
            without_name="<｜Knowledge｜>",
        ),
        end="<｜end▁of▁sentence｜>",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    pretrain_meta=dict(
        begin=dict(with_name="", without_name=""),
        end="",
        loss=dict(icl=False, current=False, prefix=False, end=False),
    ),
    pretrain_content=dict(
        begin=dict(with_name="", without_name=""),
        end="<｜end▁of▁sentence｜>",
        loss=dict(icl=True, current=True, prefix=False, end=True),
    ),
    system=dict(
        begin=dict(
            with_name="<｜System｜> name={name}",
            without_name="<｜System｜>",
            name={
                "interpreter": "<｜Interpreter｜>",
                "plugin": "<｜Plugin｜>",
            },
        ),
        end="",
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    user=dict(
        begin=dict(
            with_name="<｜User｜> name={name}",
            without_name="<｜User｜>",
        ),
        end="",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_prefix=dict(
        begin=dict(
            with_name="<｜Assistant｜> name={name}",
            without_name="<｜Assistant｜>",
        ),
        end="",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    answer_middle=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="",
        loss=dict(
            icl=False,
            current=True,
            prefix=False,
        ),
    ),
    answer_postfix=dict(
        begin=dict(
            with_name="",
            without_name="",
        ),
        end="<｜end▁of▁sentence｜>",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    assistant=dict(
        begin=dict(
            with_name="<｜Assistant｜> name={name}",
            without_name="<｜Assistant｜>",
            name={
                "interpreter": "<｜Interpreter｜>",
                "plugin": "<｜Plugin｜>",
            },
        ),
        end="<｜end▁of▁sentence｜>",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    environment=dict(
        begin=dict(
            with_name="<｜Environment｜> name={name}",
            without_name="<｜Environment｜>",
            name={
                "interpreter": "<｜Interpreter｜>",
                "plugin": "<｜Plugin｜>",
            },
        ),
        end="",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    tool=dict(
        begin=dict(
            with_name="<｜ActionStart｜>",
            name={
                "interpreter": "<｜Interpreter｜>",
                "plugin": "<｜Plugin｜>",
            },
        ),
        end="<｜ActionEnd｜>",
        belong="assistant",
    ),
    thought=dict(
        begin=dict(without_name=""),
        end="",
        belong="assistant",
    ),
    max_len=32768,
)


internlm_128k = copy.deepcopy(internlm)
internlm_128k["max_len"] = 128 * 1024

ROLE_CONFIG = {
    "internlm2": internlm,
    "qwen": qwen,
    "qwen2": qwen,
    "qwen_intern": qwen_intern,
    "internlm2_128k": internlm_128k,
    "deepseekv3": deepseekv3,
}
MAX_LEN = 32 * 1024


def format_begin(role_cfg, message):
    name = message.get("name", None)
    if name is not None:
        begin = role_cfg["begin"].get("with_name", "")
        if name in role_cfg["begin"].get("name", {}):
            begin = begin.format(name=role_cfg["begin"]["name"][name])
        else:
            begin = begin.format(name=name)
    else:
        begin = role_cfg["begin"].get("without_name", "")
    return begin


def format_sub_role(messages: List[Dict], roles_cfg) -> List[Dict]:
    new_message = list()
    for message in messages:
        if message["role"] in [
            "assistant",
            "user",
            "system",
            "environment",
            "pretrain_content",
            "pretrain_meta",
            "answer_prefix",
            "answer_middle",
            "answer_postfix",
        ]:
            new_message.append(message)
            continue
        role_cfg = roles_cfg[message["role"]]
        begin = format_begin(role_cfg, message)
        new_content = begin + message["content"] + role_cfg["end"]
        if role_cfg.get("fallback_role"):
            new_message.append(dict(role=role_cfg["fallback_role"], content=new_content))
        elif role_cfg.get("belong"):
            if new_message[-1]["role"] != role_cfg.get("belong"):
                new_message.append(dict(role=role_cfg.get("belong"), content=new_content))
            else:
                new_message[-1]["content"] += new_content
        else:
            new_message.append(dict(role=message["role"], content=new_content))

    return new_message


def ftdp_tokenize(tokenizer: "PreTrainedTokenizer", messages: list[dict] | dict, template_config) -> Dict:
    token_ids: list[int] = []

    # HACK for delivery format
    if isinstance(messages, dict) and "dialogs" in messages:
        messages = messages["dialogs"]
    messages = cast(list[dict], messages)
    _processed_data = format_sub_role(messages, template_config)

    for dialog_item in _processed_data:
        role = dialog_item["role"]
        content = dialog_item["content"]
        # TODO: is strip necessary? or use lstrip? 避免开始有\n\n的情况
        # content = content.lstrip()
        begin = format_begin(template_config[role], dialog_item)
        end = template_config[role]["end"]
        begin_token = tokenizer.encode(begin, add_special_tokens=False)
        if not template_config[role]["loss"].get("begin", False):
            begin_token = [-token_id for token_id in begin_token]
        end_token = tokenizer.encode(template_config[role]["end"], add_special_tokens=False)

        content_token = tokenizer.encode(begin + content + end, add_special_tokens=False)
        if end == "":
            content_token = content_token[len(begin_token) :]
        else:
            content_token = content_token[len(begin_token) : -len(end_token)]

        if dialog_item.get("loss", True):
            loss_cfg = template_config[role]["loss"]
        else:
            loss_cfg = dict(icl=False, current=False, meta=False, end=False)

        loss_type = dialog_item.get("type", "current")
        if (loss_type in loss_cfg) and (not loss_cfg[loss_type]):
            content_token = [-token_id for token_id in content_token]
        if not loss_cfg.get("end", False):
            end_token = [-token_id for token_id in end_token]

        if begin == "":
            tokens = content_token
        else:
            tokens = begin_token + content_token
        if end != "":
            tokens = tokens + end_token

        token_ids += tokens

    # Qwen2 models does not use bos token
    if tokenizer.bos_token_id:
        token_ids = [tokenizer.bos_token_id] + token_ids

    max_len = template_config.get("max_len", MAX_LEN)
    token_ids = token_ids[:max_len]
    labels = [x if x >= 0 else IGNORE_INDEX for x in token_ids]

    if labels:
        labels[0] = -100
    token_ids = [abs(x) for x in token_ids]

    training_data = {
        "input_ids": token_ids,
        "labels": labels,
        "num_tokens": len(token_ids),
    }

    return training_data


class FtdpTokenizeFunction(CachableTokenizeFunction):
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        chat_template="internlm2",
        tokenizer_hash: str | None = None,
        hash: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.template_config = ROLE_CONFIG[chat_template]
        self._hash = hash
        self._tokenizer_hash = tokenizer_hash

    def __call__(self, item: dict | list, **kwargs) -> DataItem:
        return ftdp_tokenize(self.tokenizer, item, self.template_config)  # type: ignore[return-value]

    def hash(self) -> str:
        if self._hash is None:
            # truncate to 16 characters prevent too long cache directory
            if self._tokenizer_hash is None:
                _tokenizer_hash = tokenizer_xxhash(self.tokenizer)[:16]
            else:
                _tokenizer_hash = self._tokenizer_hash
            _template_hash = hashlib.sha256(repr(self.template_config).encode()).hexdigest()[:16]
            _source_hash = (
                hashlib.sha256(inspect.getsource(ftdp_tokenize).encode()).hexdigest()[:16]
                + hashlib.sha256(inspect.getsource(self.__class__.__call__).encode()).hexdigest()[:16]
                + hashlib.sha256(inspect.getsource(self.__class__.__init__).encode()).hexdigest()[:16]
            )

            self._hash = f"{_tokenizer_hash}_{_template_hash}_{_source_hash}"
        else:
            assert isinstance(self._hash, str), (
                "hash is not a valid string, it means `FtdpTokenizeFunction._hash` is modified by user."
            )

        return self._hash


class FtdpTokenizedDataMapping(CachableTokenizeFunction):
    def __init__(self, max_length, hash: str | None = None):
        self.max_length = max_length
        self._hash = hash

    def __call__(self, item: dict, **kwargs) -> DataItem:
        item["input_ids"] = item["tokens"]
        del item["tokens"]
        if len(item["input_ids"]) > self.max_length:
            item["input_ids"] = item["input_ids"][: self.max_length]
        labels = [x if x >= 0 else -100 for x in item["input_ids"]]
        item["input_ids"] = [abs(x) for x in item["input_ids"]]
        item["labels"] = labels
        item["num_tokens"] = len(item["input_ids"])

        ret = DataItem(input_ids=item["input_ids"], labels=item["labels"], num_tokens=item["num_tokens"])
        return ret

    def hash(self) -> str:
        if self._hash is None:
            _source_hash = (
                hashlib.sha256((inspect.getsource(self.__class__.__call__)).encode()).hexdigest()
                + hashlib.sha256((inspect.getsource(self.__class__.__init__)).encode()).hexdigest()
            )
            self._hash = f"{self.max_length}_{_source_hash}"
        else:
            assert isinstance(self._hash, str), (
                "hash is not a valid string, it means `FtdpTokenizeFunction._hash` is modified by user."
            )

        return self._hash


# TODO: Maybe rename
class FTDPTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    chat_template: Annotated[str, Parameter(group="tokenize_fn")] = "internlm2"
    hash: Annotated[str | None, Parameter(group="tokenize_fn")] = None

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> "FtdpTokenizeFunction":
        from xtuner.v1.datasets import FtdpTokenizeFunction

        return FtdpTokenizeFunction(
            tokenizer, chat_template=self.chat_template, hash=self.hash, tokenizer_hash=tokenizer_hash
        )
