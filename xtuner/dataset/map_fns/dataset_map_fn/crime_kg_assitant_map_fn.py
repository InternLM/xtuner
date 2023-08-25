# Copyright (c) OpenMMLab. All rights reserved.
def crime_kg_assitant_map_fn(example):
    return {
        'conversation': [{
            'input': ('你现在是一名专业的中国律师，请根据Human的问题给出准确、有理有据的回复。\n\n'
                      '### Human: {input}\n### Bot: ').format(**example),
            'output':
            example['output']
        }]
    }
