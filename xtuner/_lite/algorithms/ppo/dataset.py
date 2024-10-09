import torch

from xtuner._lite.chat.messages.chat import ChatMsg
from xtuner._lite.datasets import OPENAI_CONVERT_MAP
from ..sft import SftCollator, SftTokenizeFunction


class PPODataset(torch.utils.data.Dataset):

    def __init__(self, prompts, responses):
        super().__init__()

        assert len(prompts) == len(responses)
        self.prompts = prompts
        self.responses = responses
        self.policies = None

    def __len__(self):
        return len(self.prompts)

    def update_policy(self, policies):

        assert len(policies) == len(self.prompts)
        self.policies = policies

    def __getitem__(self, item):

        prompt = self.prompts[item]
        response = self.responses[item]
        num_prefill_tokens = len(prompt)

        input_ids = prompt + response
        labels = [-100] * (num_prefill_tokens - 1) + response + [-100]

        if self.policies:
            return {
                'input_ids': input_ids,
                'labels': labels,
                'num_tokens': [len(input_ids)],
                **self.policies[item]
            }
        else:
            return {
                'input_ids': input_ids,
                'labels': labels,
                'num_tokens': [len(input_ids)]
            }


class PPOTokenizeFunction(SftTokenizeFunction):

    def __init__(self,
                 tokenizer,
                 chat_template,
                 raw_format='openai',
                 sys_prompt=None):
        super().__init__(tokenizer, chat_template, raw_format)
        self.sys_prompt = sys_prompt

    def __call__(self, item):

        formatter = OPENAI_CONVERT_MAP[self.raw_format]
        msg = formatter(item)
        if self.sys_prompt is not None:
            sys_msg = ChatMsg(role='system', content=self.sys_prompt)
            msg.messages = [sys_msg] + msg.messages
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)

        return tokenized


class PPOCollator(SftCollator):

    def __call__(self, instances):

        data = super().__call__(instances)

        old_logprobs = [item['old_logprobs'] for item in instances]
        ref_logprobs = [item['ref_logprobs'] for item in instances]
        old_values = [item['old_values'] for item in instances]
        reward_score = [item['reward_score'] for item in instances]

        data['old_logprobs'] = old_logprobs
        data['ref_logprobs'] = ref_logprobs
        data['old_values'] = old_values
        data['reward_score'] = reward_score

        return data
