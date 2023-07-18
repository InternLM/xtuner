from .sft import SupervisedFinetune


class DistillFinetune(SupervisedFinetune):

    def __init__(self, llm, tokenizer):
        super().__init__(llm, tokenizer)
