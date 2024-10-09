

import torch

from torch.nn.utils.rnn import pad_sequence

from xtuner._lite import get_logger
from xtuner._lite.chat import ChatMessages
from xtuner._lite.datasets import OPENAI_CONVERT_MAP


logger = get_logger()

class SftTokenizeFunction():

    def __init__(self, tokenizer, chat_template, raw_format='openai'):

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.raw_format = raw_format

    def __call__(self, item):

        formatter = OPENAI_CONVERT_MAP[self.raw_format]
        msg = formatter(item)
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)
        return tokenized


class SftCollator():

    def __init__(self, pad_token_id=0, ignore_id=-100, pack_batch=False):
        self.pack_batch = pack_batch
        self.pad_token_id = pad_token_id
        self.ignore_id = ignore_id

    def __call__(self, instances):
        
        _instances = []
        for ins in instances:
            if isinstance(ins, list):
                _instances.extend(ins)
            else:
                _instances.append(ins)

        instances = _instances

        input_ids = []
        labels = []
        num_tokens = []
        
        for data in instances:
            
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))

            if isinstance(data['num_tokens'], int):
                num_tokens.append(data['num_tokens'])
            else:
                num_tokens.extend(data['num_tokens'])

        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        num_tokens = torch.IntTensor(num_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=self.ignore_id)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if input_ids.shape != labels.shape:
            logger.error(f'[instances] {instances}')
            logger.error(f'[num_tokens] {num_tokens}')
            logger.error(f'[input_ids] {input_ids}')
            logger.error(f'[labels] {labels}')
            raise RuntimeError('The shape of input_ids and labels must be '
                               f'equal, but  found {input_ids.shape} and '
                               f'{labels.shape}.')
        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': num_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict
