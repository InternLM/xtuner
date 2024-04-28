import json
import os

from datasets import Dataset, concatenate_datasets


def load_json_file(data_files=None, data_dir=None, suffix=None):
    assert (data_files is not None) != (data_dir is not None)
    if data_dir is not None:
        data_files = os.listdir(data_dir)
        data_files = [os.path.join(data_dir, fn) for fn in data_files]
        if suffix is not None:
            data_files = [fp for fp in data_files if fp.endswith(suffix)]
    elif isinstance(data_files, str):
        data_files = [data_files]

    dataset_list = []
    for fp in data_files:
        with open(fp, encoding='utf-8') as file:
            data = json.load(file)
        ds = Dataset.from_list(data)
        dataset_list.append(ds)
    dataset = concatenate_datasets(dataset_list)
    return dataset
