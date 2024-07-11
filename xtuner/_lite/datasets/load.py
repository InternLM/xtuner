# Copyright (c) OpenMMLab. All rights reserved.
import functools
from torch import distributed as dist
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from xtuner._lite import get_logger
from tqdm import tqdm
import random
import os
from datasets import Dataset
import json
from torch import distributed as dist

logger = get_logger()

def load_json(file):
    with open(file) as f:
        dset = json.load(f)
    return dset

def load_jsonl(file):
    dset = []
    with open(file, 'r') as f:
        for line in f:
            dset.append(json.loads(line))
    return dset

LOAD_FN_MAP = {
    ".json" : load_json,
    ".jsonl": load_jsonl,
}

    

def master_only_load(load_fn):

    @functools.wraps(load_fn)
    def wrapper(*args, **kwargs):

        if not (dist.is_available() and dist.is_initialized()):
            return load_fn(*args, **kwargs)

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=30)))

        logger.info(f'xtuner_dataset_timeout = {timeout}', logger='current')

        gloo_group = dist.new_group(backend='gloo', timeout=timeout)

        if dist.get_rank() == 0:
            dataset = load_fn(*args, **kwargs)
            objects = [dataset]
        else:
            objects = [None]

        dist.monitored_barrier(group=gloo_group, timeout=timeout)
        dist.broadcast_object_list(objects, src=0)

        return objects[0]

    return wrapper



def multi_thread_map(map_fns, dataset, desc, num_proc=8):

    if not isinstance(map_fns, (tuple, list)):
        map_fns = [map_fns]
    
    def sequential_map(item):
        for fn in map_fns:
            item = fn(item)
        return item
    
    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        results = list(
            tqdm(
                executor.map(sequential_map, dataset),
                desc=desc,
                total=len(dataset)))

    return results

def openai_format(item):
    
    item['messages'] = item['instruction']
    return item

@master_only_load
def load_hf_dataset(path, split='train', sample_ratio=1.0, num_proc=8, map_fn=None):
    from datasets import load_dataset
    dataset = load_dataset(path)[split]
    
    if map_fn:
        dataset = dataset.map(map, num_proc=num_proc)
    
    if sample_ratio != 1:
        ori_samples = len(dataset)
        target_samples = int(sample_ratio * ori_samples)
        indices = random.choices([i for i in range(ori_samples)], k=target_samples)
        dataset = dataset.select(indices)
        
    return dataset


def load_local_dataset(path, file_types, sample_ratio=1.0, num_proc=8, map_fn=None, init_fn=None):
    
    data_files = []
    if os.path.isdir(path):
        dir_files = []
        for root, dirs, files in os.walk(path, followlinks=True):
            dirs.sort()
            for relative_path in sorted(files):
                suffix = os.path.splitext(relative_path)[-1]
                
                if suffix in file_types:
                    absolute_path = os.path.join(root, relative_path)
                    dir_files.append(absolute_path)
        
        if len(dir_files) == 0:
            raise RuntimeError(f'There are no files with the suffix {file_types}' 
                            f'in `{path}`.')
        
        logger.info(f'Found {len(dir_files)} files in {path}')
        data_files.extend(dir_files)
        
    elif os.path.isfile(path):
        data_files.append(path)
    else:
        raise RuntimeError(f'`{path}` not found.')


    if dist.is_available():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
        
    if rank == 0:
        logger.debug(f"All files in {path}:\n{data_files}")
        
    file_sizes = [os.path.getsize(file) for file in data_files]
    
    size_order = sorted(enumerate(file_sizes), key=lambda x:x[1], reverse=True)
    sorted_indices = [ind_and_size[0] for ind_and_size in size_order]
    
    per_rank_files = [[] for _ in range(world_size)]
    per_rank_sizes = [0 for _ in range(world_size)]
    
    for ind in sorted_indices:
        
        min_size = min(per_rank_sizes)
        target = per_rank_sizes.index(min_size)
           
        per_rank_files[target].append(ind)
        per_rank_sizes[target] += file_sizes[ind]
    
    logger.debug(f'Assigned Files: {per_rank_files[rank]}')
            
    _local_datasets = []
    desc = f'[RANK {rank}]Load files'
    for i in tqdm(per_rank_files[rank], desc=desc):
    
        file = data_files[i]
        suffix = os.path.splitext(file)[-1]
        dset = LOAD_FN_MAP[suffix](file)
        _local_datasets.append(dset)

    
    if map_fn:
        local_datasets = []
        for i, ind in enumerate(per_rank_files[rank]):
            dset = _local_datasets[i]
            try:
                desc = f'[RANK {rank}]Map local file {ind}'
                mapped = multi_thread_map(map_fn, dset, desc , num_proc)
            except TypeError:
                logger.warning(f'Map {file} failed.')
                continue
            
            local_datasets.append(mapped)
        
        if rank == 0:
            logger.debug(f'Original Sample:\n{_local_datasets[0][0]}')
            logger.debug(f'Mapped Sample:\n{local_datasets[0][0]}')
    else:
        local_datasets = _local_datasets
        if rank == 0:
            logger.debug(f'Original Sample:\n{_local_datasets[0][0]}')
        
    if dist.is_available() and world_size > 1:
        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=30)))
        logger.info(('Waiting for other ranks, it will timeout if it exceeds '
                     f'{timeout}.'))
        group = dist.new_group(backend='gloo', timeout=timeout)
    
        per_rank_datasets = [None] * world_size
        dist.all_gather_object(per_rank_datasets, local_datasets, group=group)
        
        datasets = []
        for dsets in per_rank_datasets:
            datasets.extend(dsets)
    
    else:
        datasets = local_datasets
        
    return [init_fn(dset) for dset in datasets]


def load_datasets(paths, sources, sample_ratios, file_types=LOAD_FN_MAP.keys(), map_fns=None, init_fns=None, num_proc=8):
    
    if isinstance(sample_ratios, (tuple, list)):   
         
        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * len(paths)
            
        if len(sample_ratios) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only '
                               f'{len(sample_ratios)} sample ratios were set.')
    
    if isinstance(sources, str):
        sources = [sources]
    
    if isinstance(sources, (tuple, list)):   
         
        if len(sources) == 1:
            sources = list(sources) * len(paths)
            
        if len(sources) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only '
                               f'{len(sources)} sources were set.')
            
    if map_fns is None:
        map_fns = [None] * len(paths)
    
    
    if isinstance(map_fns, (tuple, list)):   
         
        if len(map_fns) == 1:
            map_fns = list(map_fns) * len(paths)
            
        if len(map_fns) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only'
                               f'{len(map_fns)} map fns were set.')
            
            
    if init_fns is None:
        init_fns = [None] * len(paths)
    
    if isinstance(init_fns, (tuple, list)):   
         
        if len(init_fns) == 1:
            init_fns = list(init_fns) * len(paths)
            
        if len(init_fns) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only'
                               f'{len(init_fns)} init fns were set.')
            

    datasets = []
    
    for i , path in enumerate(paths):
        src = sources[i]
        ratio = sample_ratios[i]
        map_fn = map_fns[i]
        init_fn = init_fns[i]
        
        if src == 'local':
            dsets = load_local_dataset(path, file_types, ratio, num_proc, map_fn, init_fn)
            datasets.extend(dsets)
        elif src == 'huggingface':
            dset = load_hf_dataset(path, sample_ratios=ratio, num_proc=num_proc, map_fn=map_fn )
            datasets.append(dset)
        
    return datasets

@master_only_load
def load_ms_dataset():
    pass