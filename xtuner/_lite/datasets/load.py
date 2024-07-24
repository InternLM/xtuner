# Copyright (c) OpenMMLab. All rights reserved.
import functools
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from datasets import (Dataset, concatenate_datasets, load_dataset,
                      load_from_disk)
from torch import distributed as dist
from tqdm import tqdm

from xtuner._lite import get_logger
from .cache import CacheDataset

logger = get_logger()


def load_json(file):
    with open(file) as f:
        dset = json.load(f)
    return dset


def load_jsonl(file):
    dset = []
    with open(file) as f:
        for line in f:
            dset.append(json.loads(line))
    return dset


def load_bin(file):
    return load_dataset('json', data_files=file, split='train')


LOAD_FN_MAP = {'.json': load_json, '.jsonl': load_jsonl, '.bin': load_bin}


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
def load_hf_dataset(path,
                    split='train',
                    sample_ratio=1.0,
                    num_proc=8,
                    cache_dir=None,
                    map_fn=None,
                    init_fn=None):
    from datasets import load_dataset
    dataset = load_dataset(path)[split]

    if map_fn:
        dataset = dataset.map(map_fn, num_proc=num_proc)

    if sample_ratio != 1:
        ori_samples = len(dataset)
        target_samples = int(sample_ratio * ori_samples)
        indices = random.choices([i for i in range(ori_samples)],
                                 k=target_samples)
        dataset = dataset.select(indices)

    if init_fn:
        dataset = init_fn(dataset)

    if cache_dir:
        dataset.save_to_disk(cache_dir)
        del dataset
        dataset = load_from_disk(cache_dir)

    return dataset


def load_from_cache(cache_dir, init_fn):

    if dist.is_available():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    sub_cache_dirs = []
    for _path in tqdm(os.listdir(cache_dir)):
        path = os.path.join(cache_dir, _path)
        if os.path.isdir(path):
            sub_cache_dirs.append(path)

    num_dsets = len(sub_cache_dirs)
    avg_num = math.ceil(num_dsets / world_size)
    start = rank * avg_num
    end = min((rank + 1) * avg_num, num_dsets)
    desc = f'[Rank {rank}] Loading Cached Dataset'

    rank_datasets = []
    for ind in tqdm(range(start, end), desc=desc):
        dset = init_fn(sub_cache_dirs[ind])
        rank_datasets.append(dset)

    if dist.is_available() and world_size > 1:
        dist.barrier()
        buffers = [None] * world_size
        dist.all_gather_object(buffers, rank_datasets)
        world_datasets = []
        for dsets_per_rank in buffers:
            world_datasets.extend(dsets_per_rank)

        assert len(world_datasets) == num_dsets
    else:
        world_datasets = rank_datasets

    return world_datasets


def load_local_datasets(paths,
                        file_types,
                        file_pattern=None,
                        cache_dir=None,
                        sample_ratios=1.0,
                        num_proc=8,
                        map_fns=None,
                        init_fns=Dataset.from_list):

    if isinstance(paths, str):
        paths = [paths]

    if isinstance(sample_ratios, (tuple, list)):

        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * len(paths)

        if len(sample_ratios) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only '
                               f'{len(sample_ratios)} sample ratios were set.')

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

    files = []
    file_sample_ratios = []
    file_map_fns = []
    file_init_fns = []

    for pid, path in enumerate(paths):
        if os.path.isdir(path):
            dir_files = []
            for root, dirs, _files in os.walk(path, followlinks=True):
                dirs.sort()
                for relative_path in sorted(_files):
                    suffix = os.path.splitext(relative_path)[-1]
                    absolute_path = os.path.join(root, relative_path)
                    if file_pattern is not None:
                        if bool(re.match(file_pattern, absolute_path)):
                            dir_files.append(absolute_path)
                    elif suffix in file_types:
                        dir_files.append(absolute_path)

            _num_dir_files = len(dir_files)
            if _num_dir_files == 0:
                raise RuntimeError(
                    f'There are no files with the suffix {file_types}'
                    f'in `{path}`.')

            logger.info(f'Found {len(dir_files)} files in {path}')
            files.extend(dir_files)
            file_sample_ratios.extend([sample_ratios[pid]] * _num_dir_files)
            file_map_fns.extend([map_fns[pid]] * _num_dir_files)
            file_init_fns.extend([init_fns[pid]] * _num_dir_files)

        elif os.path.isfile(path):
            files.append(path)
            file_sample_ratios.append(sample_ratios[pid])
            file_map_fns.append(map_fns[pid])
            file_init_fns.append(init_fns[pid])
        else:
            raise RuntimeError(f'`{path}` not found.')

    num_files = len(files)

    if dist.is_available():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    else:
        world_size = 1
        rank = 0

    datasets = []
    cached_infos = {}
    for ind in range(num_files):

        file = files[ind]
        suffix = os.path.splitext(file)[-1]
        dset = LOAD_FN_MAP[suffix](file)
        logger.debug(f'[File {ind}] Raw Sample:\n{dset[0]}')

        init_fn = file_init_fns[ind]
        map_fn = file_map_fns[ind]
        if map_fn:
            num_per_shard = math.ceil(len(dset) / world_size)
            shard_start = rank * num_per_shard
            shard_end = min((rank + 1) * num_per_shard, len(dset))

            dset = dset[shard_start:shard_end]
            try:
                desc = f'[RANK {rank}] Map local file {ind}'
                dset = multi_thread_map(map_fn, dset, desc, num_proc)
                logger.debug(f'[File {ind}] Mapped Sample:\n{dset[0]}')
            except:
                raise RuntimeError(f'[RANK {rank}] Map {file} failed.')

            if dist.is_available() and world_size > 1:
                dist.barrier()
                buffers = [None] * world_size
                dist.all_gather_object(buffers, dset)
                dset = concatenate_datasets(
                    [Dataset.from_list(_list) for _list in buffers])

        if init_fn:
            dset = init_fn(dset)

        if cache_dir and isinstance(dset, CacheDataset):

            digits = len(str(abs(num_files)))
            cache_id = (f'cache-local-{ind+1:0{digits}}-of-'
                        f'{num_files:0{digits}}')
            sub_cache_dir = os.path.join(cache_dir, cache_id)

            # if os.path.exists(sub_cache_dir):
            #     shutil.rmtree(sub_cache_dir)
            #     logger.warning(f'Found {sub_cache_dir} exists. '
            #                 'Clear it and re-cache.')
            dset.cache(sub_cache_dir)

            infos = {
                'path': file,
                'num_samples': dset.num_samples,
                'num_tokens': dset.total_tokens
            }
            cached_infos[cache_id] = infos
        datasets.append(dset)

    if cache_dir and rank == 0:
        _path = os.path.join(cache_dir, 'local_infos.json')
        with open(_path, 'w') as f:
            json.dump(cached_infos, f)

    return datasets


def load_datasets(paths,
                  sources,
                  sample_ratios=1.0,
                  file_types=LOAD_FN_MAP.keys(),
                  file_pattern=None,
                  cache_dir=None,
                  map_fns=None,
                  init_fns=None,
                  num_proc=8):

    if isinstance(paths, str):
        paths = [paths]

    num_paths = len(paths)

    if isinstance(sample_ratios, (float, int)):
        sample_ratios = [sample_ratios] * num_paths

    if isinstance(sample_ratios, (tuple, list)):

        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * num_paths

        if len(sample_ratios) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only '
                               f'{len(sample_ratios)} sample ratios were set.')

    if isinstance(sources, str):
        sources = [sources]

    if isinstance(sources, (tuple, list)):

        if len(sources) == 1:
            sources = list(sources) * num_paths

        if len(sources) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only '
                               f'{len(sources)} sources were set.')

    if not isinstance(map_fns, (tuple, list)):
        map_fns = [map_fns] * num_paths

    if isinstance(map_fns, (tuple, list)):

        if len(map_fns) == 1:
            map_fns = list(map_fns) * num_paths

        if len(map_fns) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only'
                               f'{len(map_fns)} map fns were set.')

    if not isinstance(init_fns, (tuple, list)):
        init_fns = [init_fns] * num_paths

    if isinstance(init_fns, (tuple, list)):

        if len(init_fns) == 1:
            init_fns = list(init_fns) * num_paths

        if len(init_fns) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only'
                               f'{len(init_fns)} init fns were set.')

    local_inds = [i for i, src in enumerate(sources) if src == 'local']
    local_paths = [paths[ind] for ind in local_inds]
    local_map_fns = [map_fns[ind] for ind in local_inds]
    local_init_fns = [init_fns[ind] for ind in local_inds]
    local_sample_ratios = [sample_ratios[ind] for ind in local_inds]

    hf_inds = [i for i, src in enumerate(sources) if src == 'huggingface']
    hf_paths = [paths[ind] for ind in hf_inds]
    hf_map_fns = [map_fns[ind] for ind in hf_inds]
    hf_init_fns = [init_fns[ind] for ind in hf_inds]
    hf_sample_ratios = [sample_ratios[ind] for ind in hf_inds]

    datasets = []
    if len(local_inds):
        local_datasets = load_local_datasets(local_paths, file_types,
                                             file_pattern, cache_dir,
                                             local_sample_ratios, num_proc,
                                             local_map_fns, local_init_fns)
        datasets.extend(local_datasets)

    if len(hf_inds):
        cached_infos = {}
        for i in range(len(hf_inds)):
            if cache_dir:
                digits = len(str(abs(len(hf_inds))))
                cache_id = (f'cache-hf-{i+1:0{digits}}-of-'
                            f'{len(hf_inds):0{digits}}')
                sub_cache_dir = os.path.join(cache_dir, cache_id)
            else:
                sub_cache_dir = None
            dset = load_hf_dataset(
                hf_paths[i],
                sample_ratio=hf_sample_ratios[i],
                num_proc=num_proc,
                map_fn=hf_map_fns[i],
                init_fn=hf_init_fns[i],
                cache_dir=sub_cache_dir)
            datasets.append(dset)

            num_tokens = sum(dset['num_tokens'])
            num_samples = len(dset)
            infos = {
                'path': hf_paths[i],
                'num_samples': num_samples,
                'num_tokens': num_tokens
            }
            cached_infos[cache_id] = infos

        if cache_dir:
            _path = os.path.join(cache_dir, 'hf_infos.json')
            with open(_path, 'w') as f:
                json.dump(cached_infos, f)

    return datasets


@master_only_load
def load_ms_dataset():
    pass
