# Copyright (c) OpenMMLab. All rights reserved.
def medical_map_fn(example):
    PROMPT = {
        'with_input': ('如果你是一名医生，请根据患者的描述回答医学问题。\n\n'
                       '### Input: {instruction}. {input}\n\n### Response: '),
        'without_input': ('如果你是一名医生，请根据患者的描述回答医学问题。\n\n'
                          '### Input: {instruction}\n\n### Response: '),
    }
    if example.get('input', '') != '':
        prompt_template = PROMPT['with_input']
    else:
        prompt_template = PROMPT['without_input']

    return {
        'conversation': [{
            'input': prompt_template.format(**example),
            'output': example['output']
        }]
    }
