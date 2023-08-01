from .sft_lora import SupervisedLoraFinetune


# todo: check if class `SupervisedQloraFinetune` is necessary
class SupervisedQloraFinetune(SupervisedLoraFinetune):

    def __init__(self,
                 llm,
                 lora,
                 data_preprocessor=None,
                 tokenizer=None,
                 peft_path=None):
        super().__init__(llm, lora, data_preprocessor, tokenizer, peft_path)
