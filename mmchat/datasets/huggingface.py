from mmchat.registry import DATASETS


def process_hf_dataset(dataset,
                       mode='train',
                       map_fn=None,
                       remove_columns=[],
                       rename_maps=[]):

    dataset = DATASETS.build(dataset)
    if isinstance(map_fn, str):
        map_fn = eval(map_fn)
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # Remove unused columns.
    if 'train' in dataset.column_names:
        dataset = dataset.remove_columns([
            col for col in dataset.column_names['train']
            if col not in ['input', 'output']
        ])
    return dataset[mode]
