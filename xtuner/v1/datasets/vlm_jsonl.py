import os

import torch

from xtuner.v1.utils import get_logger

from .jsonl import JsonlDataset


logger = get_logger()


class VLMJsonlDataset(JsonlDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.fake_data = {
            "id": -1,
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好呀！很高兴为你服务～有什么问题或需要帮忙的地方，随时告诉我哦！"},
            ],
        }

    def __getitem__(self, item):
        try:
            data = super().__getitem__(item)
        except Exception as e:
            logger.warning(f"[{os.path.basename(self.path)}]: {e}. Dumping a fake data.")
            data = self.tokenize_fn(self.fake_data)
            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            if "labels" in data:
                data["labels"] = torch.ones_like(data["input_ids"]) * -100
        return data
