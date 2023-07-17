import bitsandbytes as bnb
import torch
from peft import get_peft_model, prepare_model_for_kbit_training

from mmchat.registry import MODELS
from .sft import SupervisedFinetune


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SupervisedQloraFinetune(SupervisedFinetune):

    def __init__(self, llm, data_preprocessor, lora):
        super().__init__(llm, data_preprocessor)

        self.llm = prepare_model_for_kbit_training(self.llm)

        modules = find_all_linear_names(self.llm)

        lora = MODELS.build(lora)
        lora.target_modules = modules

        self.llm = get_peft_model(self.llm, lora)

        for name, module in self.llm.named_modules():
            # todo
            # if isinstance(module, LoraLayer):
            #     module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            # if 'lm_head' in name or 'embed_tokens' in name:
            #     if hasattr(module, 'weight'):
            #         if module.weight.dtype == torch.float32:
            #             module = module.to(torch.float16)
        self._is_init = True

    def init_weights(self):
        pass
