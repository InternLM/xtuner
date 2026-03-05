from os import CLD_STOPPED
from brainpp.rjob import Job
from clusterx import CLUSTER, CLUSTER_MAPPING
from clusterx.launcher.base import JobSchema, JobStatus
from xtuner.v1.datasets import build_datasets, DatasetConfigList
from cyclopts import App
import ray
from pathlib import Path
from xtuner.v1.utils import Config
from more_itertools import batched
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache
from transformers import AutoTokenizer
from math import ceil
import copy
import random
from loguru import logger
import json
import shutil


CACHE_META = ".xpuyu-cache-meta.json"

app = App()


spec = CLUSTER_MAPPING[CLUSTER]
ClusterParams = spec["params"]
Cluster = spec["type"]


assert Cluster is not None, "Please install clusterx correctly"
cluster = Cluster()


def start_ray(nnodes: int, cpus_per_task: int, memory_per_task: int) -> JobSchema:
    assert ClusterParams is not None
    cmd_path = (Path(__file__).parent / "launch_ray.sh").absolute()

    clusterx_params = ClusterParams(
        cmd=str(cmd_path),
        job_name="offline_cache",
        cpus_per_task=cpus_per_task,
        memory_per_task=memory_per_task,
        num_nodes=nnodes,
        no_env=True,
        image="registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt28_latest",
    )
    job = cluster.run(clusterx_params)

    while job.status != JobStatus.RUNNING:
        job = cluster.get_job_info(job.job_id)

    return job


def merge_worker_caches(base_cache_dir: Path, num_workers: int) -> None:
    """Merge all worker cache directories into the base directory.

    Args:
        base_cache_dir (Path): Base directory containing worker-{i} subdirectories
        num_workers (int): Number of worker directories to merge
    """
    merged_meta = {}

    logger.info(f"Starting cache merge into {base_cache_dir}")

    for worker_id in range(num_workers):
        worker_dir = base_cache_dir / f"worker-{worker_id}"
        if not worker_dir.exists():
            logger.warning(f"Worker directory {worker_dir} does not exist, skipping")
            continue

        meta_file = worker_dir / CACHE_META
        if not meta_file.exists():
            logger.warning(f"Meta file {meta_file} does not exist, skipping worker-{worker_id}")
            continue

        # Load worker's meta.json
        with open(meta_file) as f:
            worker_meta = json.load(f)

        logger.info(f"Processing worker-{worker_id}: {len(worker_meta)} hash entries")

        # Copy each hash directory and merge metadata
        for hash_key, hash_meta in worker_meta.items():
            src_hash_dir = worker_dir / hash_key
            dst_hash_dir = base_cache_dir / hash_key

            if not src_hash_dir.exists():
                logger.warning(f"Hash directory {src_hash_dir} does not exist, skipping")
                continue

            if dst_hash_dir.exists():
                logger.warning(f"Hash directory {dst_hash_dir} already exists, skipping copy")
            else:
                shutil.move(src_hash_dir, dst_hash_dir)
                logger.debug(f"Moved {src_hash_dir} -> {dst_hash_dir}")

            # Merge metadata (assuming no conflicts, otherwise we'd need merge logic)
            if hash_key in merged_meta:
                logger.warning(f"Hash key {hash_key} already exists in merged metadata, skipping")
            else:
                merged_meta[hash_key] = hash_meta

        # Clean up worker directory
        try:
            shutil.rmtree(worker_dir)
            logger.info(f"Removed worker directory {worker_dir}")
        except Exception as e:
            logger.error(f"Failed to remove worker directory {worker_dir}: {e}")

    # Write merged metadata
    merged_meta_file = base_cache_dir / CACHE_META
    with open(merged_meta_file, "w") as f:
        json.dump(merged_meta, f, indent=4)

    logger.info(f"Cache merge completed: {len(merged_meta)} hash entries in {merged_meta_file}")


def cache_worker(dataset_config_list: DatasetConfigList, tokenizer_path, worker_id: int):
    import socket
    import os
    from loguru import logger

    hostname = socket.gethostname()
    pid = os.getpid()
    logger.info(
        f"[Worker {worker_id}] Running on {hostname}, PID={pid}, processing {len(dataset_config_list)} datasets"
    )
    monkey_patch_hf_modules_cache()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    build_datasets(dataset_config_list, tokenizer)
    logger.info(f"[Worker {worker_id}] Finished on {hostname}, PID={pid}")


@app.default
def main(
    config_path: Path,
    nnodes: int,
    cpus_per_task: int,
    memory_per_task: int,
    tasks_per_node: int = 1,
):
    job = start_ray(nnodes=nnodes, cpus_per_task=cpus_per_task, memory_per_task=memory_per_task)
    try:
        logger.info(f"ray clsuter job id {job.job_id}")
        assert job.nodes_ip is not None
        ray_ip = f"ray://{job.nodes_ip[0]}:10001"

        logger.info(f"Connect to ray cluster: {ray_ip}")
        ray.init(ray_ip)

        resources = ray.cluster_resources()
        logger.info(f"Ray cluster resources: CPU={resources.get('CPU', 0)}, nodes={len(ray.nodes())}")
        logger.info(f"Ray nodes: {[node['NodeManagerAddress'] for node in ray.nodes()]}")

        config = Config.fromfile(config_path)
        dataset_config = config.dataset_config
        tokenizer_path = config.trainer.tokenizer_path
        base_cache_dir = Path(dataset_config[0]["dataset"].cache_dir)

        global cache_worker
        worker = ray.remote(
            num_cpus=cpus_per_task // tasks_per_node,
            runtime_env={
                "env_vars": {
                    "XTUNER_TOKENIZE_WORKERS": str(cpus_per_task // tasks_per_node),
                    "PYTHONPATH": str((Path(__file__).parent.parent).absolute()),
                }
            },
        )(cache_worker)
        batch_size = ceil(len(dataset_config) / (nnodes * tasks_per_node))
        batch_config_list = list(batched(dataset_config, batch_size))
        random.shuffle(batch_config_list)

        res = []
        logger.info(f"Total tasks: {len(batch_config_list)} (nnodes={nnodes}, tasks_per_node={tasks_per_node})")
        logger.info(f"Batch size: {batch_size}, Total datasets: {len(dataset_config)}")
        for i, batch in enumerate(batch_config_list):
            # each worker should cache tokenized meta data to different paths to avoid CACHE_META file IO conflicts
            batch_copy = copy.deepcopy(list(batch))

            for dataset_cfg in batch_copy:
                if hasattr(dataset_cfg["dataset"], "cache_dir") and dataset_cfg["dataset"].cache_dir is not None:
                    original_cache_dir = Path(dataset_cfg["dataset"].cache_dir)
                    worker_cache_dir = original_cache_dir / f"worker-{i}"
                    dataset_cfg["dataset"].cache_dir = str(worker_cache_dir)

            res.append(worker.remote(batch_copy, tokenizer_path, i))

            logger.info(f"Submitted task {i + 1}/{len(batch_config_list)}")

        logger.info(f"\nWaiting for {len(res)} tasks to complete...")
        for i, obj in enumerate(res):
            ray.get(obj)
            logger.info(f"Task {i + 1}/{len(res)} completed")

        # Merge worker caches after all tasks complete
        logger.info("All tasks completed, starting cache merge...")

        merge_worker_caches(base_cache_dir, num_workers=len(batch_config_list))

    finally:
        cluster.stop(job_id=job.job_id)


if __name__ == "__main__":
    app()
