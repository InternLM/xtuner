import copy

from transformers import PreTrainedTokenizer
from xtuner.v1.datasets.data_item import DataItem
from xtuner.v1.utils import get_logger

from ..utils import CachableTokenizeFunction


logger = get_logger()


class PretrainTokenizeFunction(CachableTokenizeFunction[DataItem]):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, item: dict) -> DataItem:
        try:
            text = item["text"] + self.tokenizer.eos_token
        except:
            if isinstance(item, dict):
                text = item["content"] + self.tokenizer.eos_token
            elif isinstance(item, list):
                text = item[0]["content"] + self.tokenizer.eos_token
            else:
                raise NotImplementedError()
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(input_ids)
        labels = copy.deepcopy(input_ids)
        labels[0] = -100
        return {"input_ids": input_ids, "labels": labels, "num_tokens": num_tokens}
