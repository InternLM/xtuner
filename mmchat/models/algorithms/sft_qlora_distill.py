from .sft import SupervisedFinetune

class QloraDistillFinetune(SupervisedFinetune):

    def __init__(self, llm, tokenizer):
        super().__init__(llm, tokenizer)