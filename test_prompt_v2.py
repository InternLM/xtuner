from datasets import load_dataset
from transformers import AutoTokenizer

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.map_fns import (oasst1_map_fn, oasst1_map_fn_v2,
                                    template_map_fn_factory)
from xtuner.utils import PROMPT_TEMPLATE

tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
prompt_template = PROMPT_TEMPLATE.internlm2_chat
dataset = load_dataset('timdettmers/openassistant-guanaco')

dataset_v1 = process_hf_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    max_length=2048,
    dataset_map_fn=oasst1_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    pack_to_max_length=False)

dataset_v2 = process_hf_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    max_length=2048,
    dataset_map_fn=oasst1_map_fn_v2,
    prompt_template=prompt_template,
    pack_to_max_length=False)

index = 4610
sample = dataset['train'][index]
sample_v1 = dataset_v1[index]
sample_v2 = dataset_v2[index]
print('==================================================')
print('Data')
print('==================================================')
print(sample['text'])
print()

print('==================================================')
print('==================================================')
print('V1')
print('==================================================')
print('==================================================')
print()
print('==================================================')
print('Conversation (After applying prompt template)')
print('==================================================')
print(sample_v1['conversation'])
print()

print('==================================================')
print('input_ids')
print('==================================================')
print(tokenizer.decode(sample_v1['input_ids']))
print()

print('==================================================')
print('labels')
print('==================================================')
labels = [i if i > 0 else 0 for i in sample_v1['labels']]
print(tokenizer.decode(labels))
print()

print('==================================================')
print('==================================================')
print('V2')
print('==================================================')
print('==================================================')
print()
print('==================================================')
print('Messages (After applying prompt template)')
print('==================================================')
print(sample_v2['messages'])
print()

print('==================================================')
print('input_ids')
print('==================================================')
print(tokenizer.decode(sample_v2['input_ids']))
print()

print('==================================================')
print('labels')
print('==================================================')
labels = [i if i > 0 else 0 for i in sample_v2['labels']]
print(tokenizer.decode(labels))
print()

breakpoint()
