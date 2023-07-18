from .sft import SupervisedFinetune


class SupervisedLoraFinetune(SupervisedFinetune):

    def __init__(self, llm, tokenizer, lora):
        super().__init__(llm, tokenizer)
