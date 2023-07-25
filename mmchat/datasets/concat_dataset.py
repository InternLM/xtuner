from torch.utils.data import ConcatDataset as _ConcatDataset

from mmchat.registry import DATASETS


class ConcatDataset(_ConcatDataset):

    def __init__(self, tokenizer, datasets_cfg):
        datasets = []
        names = []
        for name, cfg in datasets_cfg.items():
            if cfg.get('tokenizer', None) is None:
                cfg['tokenizer'] = tokenizer
            datasets.append(DATASETS.build(cfg))
            names.append(name)
        self.names = names
        super().__init__(datasets=datasets)

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += '\n'.join([
            f'{name}: {repr(dataset)},'
            for name, dataset in zip(self.names, self.datasets)
        ])
        return main_str
