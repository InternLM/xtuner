# Copyright (c) OpenMMLab. All rights reserved.
GENTITLE_STSTEM = ('If you are an expert in writing papers, please generate '
                   "a good paper title for this paper based on other authors' "
                   'descriptions of their abstracts.\n')


def arxiv_map_fn(example):
    return {
        'conversation': [{
            'system': GENTITLE_STSTEM,
            'input': example['abstract'],
            'output': example['title']
        }]
    }
