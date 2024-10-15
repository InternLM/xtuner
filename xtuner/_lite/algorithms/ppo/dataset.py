import torch
import json
import numpy as np
from xtuner._lite.chat.messages.chat import ChatMsg
from xtuner._lite.datasets import OPENAI_CONVERT_MAP
from ..sft import SftCollator, SftTokenizeFunction


class InferDataset(torch.utils.data.Dataset):

    def __init__(self, prompts, responses):
        super().__init__()

        assert len(prompts) == len(responses)
        self.prompts = prompts
        self.responses = responses
        self.policies = None

    def __len__(self):
        return len(self.prompts)


    def __getitem__(self, item):

        prompt = self.prompts[item]
        response = self.responses[item]
        num_prefill_tokens = len(prompt)

        input_ids = prompt + response
        labels = [-100] * (num_prefill_tokens - 1) + response + [-100]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': len(input_ids)
        }



class PolicyDataset(torch.utils.data.Dataset):

    def __init__(self, policies, reward_min=-5,reward_max = 5, reward_normalize=True):
        super().__init__()

        rewards = [data['reward'] for data in policies]
        rewards = np.array(rewards).clip(reward_min, reward_max)


        self.reward_mean =  rewards.mean()
        self.reward_std = rewards.std()

        if reward_normalize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for i in range(len(policies)):
            policies[i]['reward'] = rewards[i]

        # old_logprobs = [data['old_logprobs'] for data in policies]
        # ref_logprobs = [data['ref_logprobs'] for data in policies]
        # entropy = []
        # kl = []
        # for _old, _ref in zip(old_logprobs, ref_logprobs):
        #     _entropy = - _old.mean().item()
        #     _kl = (_ref - _old).mean().item()
        #     entropy.append(_entropy)
        #     kl.append(_kl)
        
        # self.entropy_mean = sum(entropy) / len(entropy)
        # self.kl_mean = sum(kl) / len(kl)

        
        num_action_tokens = 0
        num_total_tokens = 0
        for policy in policies:
            labels = np.array(policy['labels'])
            num_total_tokens += labels.size
            num_action_tokens += (labels >= 0).sum() 
        self.num_action_tokens = num_action_tokens
        self.num_total_tokens = num_total_tokens

        self.polices = policies

    def dump_jsonl(self, path, tokenizer, debug=False):
    
        with open(path, 'w', encoding='utf8') as f:
            for policy in self.polices:
                json_line = {
                    'num_tokens': policy['num_tokens'],
                    # 'entropy': -policy['old_logprobs'].mean().item(),
                    # 'ref_kl': (policy['old_logprobs'] - policy['ref_logprobs']).mean().item(),
                    'reward': policy['reward'],
                    'sequence': tokenizer.decode(policy['input_ids']),
                }

                if debug:
                    # json_line['advantages'] = policy['advantages'].tolist()
                    # json_line['kl_rewards'] = policy['kl_rewards'].tolist()
                    # json_line['returns'] = policy['returns'].tolist()
                    json_line['input_ids'] = policy['input_ids']
                    json_line['labels'] = policy['labels']

                json_str = json.dumps(json_line, ensure_ascii=False)
                f.write(json_str + '\n')

    def __len__(self):
        return len(self.polices)

    
    def __getitem__(self, item):

        return self.polices[item]


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

        # old_logprobs = [item['old_logprobs'] for item in instances]
        # ref_logprobs = [item['ref_logprobs'] for item in instances]
        # old_values = [item['old_values'] for item in instances]
        reward_score = [item['reward'] for item in instances]
        # advantages = [item['advantages'] for item in instances]
        # returns = [item['returns'] for item in instances]
        # kl_rewards = [item['kl_rewards'] for item in instances]

        # data['old_logprobs'] = old_logprobs
        # data['ref_logprobs'] = ref_logprobs
        # data['old_values'] = old_values
        data['rewards'] = reward_score
        # data['advantages'] = advantages
        # data['returns'] = returns
        # data['kl_rewards'] = kl_rewards

        return data
