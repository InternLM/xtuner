# Copyright (c) OpenMMLab. All rights reserved.
def alpaca_zh_map_fn(example):
    PROMPT = {
        'with_input':
        ('Below is an instruction that describes a task, paired with an '
         'input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction_zh}\n\n### Input:\n{input_zh}\n\n'
         '### Response: '),
        'without_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction_zh}\n\n'
         '### Response: ')
    }
    if example.get('input', '') != '':
        prompt_template = PROMPT['with_input']
    else:
        prompt_template = PROMPT['without_input']

    return {
        'input': [prompt_template.format(**example)],
        'output': [example['output_zh']]
    }
