from mmengine import Config

from xtuner.model.base import BaseTune
from xtuner.registry import BUILDER


class AutoModel():

    @classmethod
    def from_config(cls, config: str):
        config = Config.fromfile(config)
        model: BaseTune = BUILDER.build(config.model)
        return model

    @classmethod
    def from_pretrained(cls, config: str, checkpoint: str):
        config = Config.fromfile(config)
        model: BaseTune = BUILDER.build(config.model)
        model.load_checkpoint(checkpoint)
        return model
