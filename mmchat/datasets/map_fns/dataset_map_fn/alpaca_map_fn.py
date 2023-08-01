def alpaca_map_fn(example):
    PROMPT = {
        'with_input':
        ('Below is an instruction that describes a task, paired with an '
         'input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n'
         '### Response: '),
        'without_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n'
         '### Response: ')
    }
    if example.get('input', '') != '':
        prompt_template = PROMPT['with_input']
    else:
        prompt_template = PROMPT['without_input']
    if example.get('output', '') == '<nooutput>':
        return {'input': '', 'output': ''}
    else:
        return {'input': prompt_template.format(**example)}
