from .sft_distill import DistillFinetune


class LoraDistillFinetune(DistillFinetune):

    def __init__(self, llm, tokenizer):
        super().__init__(llm, tokenizer)
