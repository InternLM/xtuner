from module.eval import Eval
from module.infer import Infer
from module.train import Train


class TestHandler:
    def __init__(self):
        self.type_map = {"eval": Eval, "infer": Infer, "pre_train": Train, "rl": Train, "sft": Train}

    def get_cmd(self, type, config=None):
        if type not in self.type_map:
            raise ValueError(f"Unsupported type: {type}")

        handler_class = self.type_map[type]
        return handler_class.get_cmd(config=None)

    def validate(self, type, config=None):
        if type not in self.type_map:
            raise ValueError(f"Unsupported type: {type}")

        handler_class = self.type_map[type]
        return handler_class.validate(config=None)
