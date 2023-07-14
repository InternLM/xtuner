from typing import Optional, Sequence, Dict
import torch
from mmengine.model import BaseDataPreprocessor
from mmchat.registry import TOKENIZER
import copy
from torch.nn.utils.rnn import pad_sequence
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
class DataProcesorForCausalLM(BaseDataPreprocessor):

    def __init__(self, 
                 tokenizer, 
                 source_max_len,
                 target_max_len,
                 train_on_source,
                 predict_with_generate,
                 non_blocking: bool = False):
        super().__init__(non_blocking)
        self.tokenizer = TOKENIZER.build(tokenizer)
        # import pdb;pdb.set_trace()
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate
    
    def forward(self,instances: Sequence[Dict], training=True) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example}" for example in instances['input']]
        targets = [f"{example}{self.tokenizer.eos_token}" for example in instances['output']]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # import pdb;pdb.set_trace()
        
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        
        if labels is not None:
            data_dict['labels'] = labels

        return self.cast_data({'data': data_dict, 'data_samples': None})

