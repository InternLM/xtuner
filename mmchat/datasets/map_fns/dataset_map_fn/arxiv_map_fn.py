# Copyright (c) OpenMMLab. All rights reserved.
def arxiv_map_fn(example):
    PROMPT = ('If you are an expert in writing papers, please generate '
              "a good paper title for this paper based on other authors' "
              'descriptions of their abstracts.\n\n'
              '### Descriptions:\n{abstract}\n\n### Title: ')
    return {'input': [PROMPT.format(**example)], 'output': [example['title']]}
