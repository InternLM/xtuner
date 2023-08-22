# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import ConcatDataset as _ConcatDataset

from xtuner.registry import BUILDER


class ConcatDataset(_ConcatDataset):

    def __init__(self, datasets_cfg, datasets_kwargs=None):
        datasets = []
        names = []
        for name, cfg in datasets_cfg.items():
            if datasets_kwargs is not None:
                cfg.update(datasets_kwargs)
            datasets.append(BUILDER.build(cfg))
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
