import copy
import json
import random
import shutil
import threading
import time
from math import ceil
from pathlib import Path
from typing import Annotated

import ray
from clusterx import CLUSTER, CLUSTER_MAPPING
from clusterx.launcher.base import JobSchema, JobStatus
from cyclopts import App, Parameter
from loguru import logger
from more_itertools import batched
from tqdm import tqdm

from transformers import AutoTokenizer
from xtuner.v1.datasets import DatasetConfigList
from xtuner.v1.utils import Config
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


CACHE_META = ".xpuyu-cache-meta.json"


@ray.remote
class ProgressTracker:
    """Ray Actor to track progress across all workers."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0

    def increment(self, count: int = 1) -> None:
        """Increment the completed count."""
        self.completed += count

    def get_progress(self) -> tuple[int, int]:
        """Get current progress (completed, total)."""
        return self.completed, self.total


app = App()


spec = CLUSTER_MAPPING[CLUSTER]
ClusterParams = spec["params"]
Cluster = spec["type"]


assert Cluster is not None, "Please install clusterx correctly"
cluster = Cluster()


def start_ray(
    nnodes: int,
    cpus_per_task: int,
    memory_per_task: int,
    image: str,
    max_wait_minutes: int = 30,
    head_ip: str | None = None,
) -> JobSchema:
    """Start Ray cluster nodes and wait for them to be ready.

    This function supports two modes:
    - Head mode (head_ip=None): Start a single head node. Requires nnodes=1.
    - Worker mode (head_ip=str): Start worker nodes that connect to the specified head IP.

    Args:
        nnodes (int): Number of nodes to request
        cpus_per_task (int): CPUs per task
        memory_per_task (int): Memory per task in GB
        image (str): Docker image to use for the job
        max_wait_minutes (int): Maximum time to wait for job to start (default: 30)
        head_ip (str | None): Ray head node IP address. If None, starts as head node (requires
            nnodes=1). If provided, starts as worker nodes connecting to this head IP.

    Returns:
        JobSchema: The running job information

    Raises:
        ValueError: If head_ip is None but nnodes != 1
        TimeoutError: If job does not start within max_wait_minutes
        RuntimeError: If job fails to start
    """
    assert ClusterParams is not None

    cmd = str((Path(__file__).parent / "launch_ray.sh").absolute())
    if head_ip:
        cmd += f" {head_ip}"
    elif nnodes != 1:
        raise ValueError(f"Head node mode requires nnodes=1, got {nnodes}")

    clusterx_params = ClusterParams(
        cmd=cmd,
        job_name="offline_cache",
        cpus_per_task=cpus_per_task,
        memory_per_task=memory_per_task,
        num_nodes=nnodes,
        no_env=True,
        image=image,
    )
    job = cluster.run(clusterx_params)
    logger.info(f"Submitted job {job.job_id}, waiting for it to start...")

    start_time = time.time()
    timeout_seconds = max_wait_minutes * 60

    while job.status != JobStatus.RUNNING:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Job {job.job_id} did not start within {max_wait_minutes} minutes. Current status: {job.status}"
            )

        time.sleep(10)  # Poll every 10 seconds
        job = cluster.get_job_info(job.job_id)
        logger.info(f"Job status: {job.status}, elapsed: {elapsed:.1f}s")

        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job.job_id} failed")

    logger.info(f"Job {job.job_id} is now RUNNING on nodes: {job.nodes_ip}")

    return job


def merge_worker_caches(base_cache_dir: Path, num_workers: int) -> None:
    """Merge all worker cache directories into the base directory.

    Handles merging when same file_hash exists across workers by:
    - Merging subdirectories (tokenize_hash) as union
    - Deep-merging metadata (offsets lists, num_tokens dicts)

    Args:
        base_cache_dir (Path): Base directory containing worker-{i} subdirectories
        num_workers (int): Number of worker directories to merge
    """
    merged_meta: dict = {}
    merged_tags: dict = {}

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

        with open(meta_file) as f:
            worker_meta = json.load(f)

        logger.info(f"Processing worker-{worker_id}: {len(worker_meta)} hash entries")

        for file_hash, hash_meta in worker_meta.items():
            if file_hash == "tags":
                continue

            src_hash_dir = worker_dir / file_hash
            dst_hash_dir = base_cache_dir / file_hash

            if not src_hash_dir.exists():
                logger.warning(f"Hash directory {src_hash_dir} does not exist, skipping")
                continue

            if dst_hash_dir.exists():
                logger.info(f"Hash directory {dst_hash_dir} already exists, merging subdirectories")

                for tok_hash_dir in src_hash_dir.iterdir():
                    if tok_hash_dir.is_dir():
                        dst_tok_dir = dst_hash_dir / tok_hash_dir.name

                        if not dst_tok_dir.exists():
                            shutil.move(tok_hash_dir, dst_tok_dir)
                            logger.debug(f"Moved tokenize subdir {tok_hash_dir.name} to {dst_tok_dir}")
                        else:
                            logger.warning(f"Tokenize subdir {dst_tok_dir} already exists, skipping")
            else:
                shutil.move(src_hash_dir, dst_hash_dir)
                logger.debug(f"Moved {src_hash_dir} -> {dst_hash_dir}")

            if file_hash in merged_meta:
                existing_offsets = set(merged_meta[file_hash].get("offsets", []))
                new_offsets = set(hash_meta.get("offsets", []))

                merged_meta[file_hash]["offsets"] = list(existing_offsets | new_offsets)

                if "num_tokens" not in merged_meta[file_hash]:
                    merged_meta[file_hash]["num_tokens"] = {}

                merged_meta[file_hash]["num_tokens"].update(hash_meta.get("num_tokens", {}))

                logger.debug(f"Merged metadata for file_hash {file_hash}")
            else:
                merged_meta[file_hash] = hash_meta

        if "tags" in worker_meta:
            for tag_name, tag_data in worker_meta["tags"].items():
                if tag_name not in merged_tags:
                    merged_tags[tag_name] = {}

                for file_path, tok_data in tag_data.items():
                    if file_path not in merged_tags[tag_name]:
                        merged_tags[tag_name][file_path] = {}

                    for tok_hash, cache_info in tok_data.items():
                        updated_cache_info = {}
                        for key, value in cache_info.items():
                            if isinstance(value, str) and f"/worker-{worker_id}/" in value:
                                updated_cache_info[key] = value.replace(f"/worker-{worker_id}/", "/")
                            else:
                                updated_cache_info[key] = value

                        merged_tags[tag_name][file_path][tok_hash] = updated_cache_info

        try:
            shutil.rmtree(worker_dir)
            logger.info(f"Removed worker directory {worker_dir}")
        except Exception as e:
            logger.error(f"Failed to remove worker directory {worker_dir}: {e}")

    if merged_tags:
        merged_meta["tags"] = merged_tags

    merged_meta_file = base_cache_dir / CACHE_META
    with open(merged_meta_file, "w") as f:
        json.dump(merged_meta, f, indent=4, ensure_ascii=False)

    logger.info(f"Cache merge completed: {len(merged_meta)} hash entries in {merged_meta_file}")


def cache_worker(
    dataset_config_list: DatasetConfigList,
    tokenizer_path: str,
    worker_id: int,
    tracker: Any,
):
    """Ray remote worker function to process and cache datasets.

    This function is executed in parallel on Ray workers. Each worker processes a batch
    of dataset configurations, tokenizing the data and caching it to disk. Progress is
    reported to the tracker after each file is processed.

    Args:
        dataset_config_list (DatasetConfigList): List of dataset configurations to process.
            Each item contains a dataset config and tokenize function config.
        tokenizer_path (str): Path to the pretrained tokenizer to load
        worker_id (int): Unique identifier for this worker (used for logging)
        tracker (Any): Ray Actor handle to the ProgressTracker for reporting progress
    """
    import os
    import socket

    from loguru import logger

    from xtuner.v1.datasets.utils import tokenizer_xxhash

    hostname = socket.gethostname()
    pid = os.getpid()

    logger.info(f"[Worker {worker_id}] Starting on {hostname}, PID={pid}")

    monkey_patch_hf_modules_cache()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    tok_hash = tokenizer_xxhash(tokenizer)[:16]

    for config in dataset_config_list:
        _dataset_config = config["dataset"]
        _tokenize_fn_config = config["tokenize_fn"]

        _tokenize_fn = _tokenize_fn_config.build(tokenizer, tokenizer_hash=tok_hash)

        _dataset_config.build(_tokenize_fn)
        tracker.increment.remote(1)

    logger.info(f"[Worker {worker_id}] Finished on {hostname}, PID={pid}")


@app.default
def main(
    config_path: Annotated[Path, Parameter(help="path to XTuner config")],
    nnodes: Annotated[int, Parameter(help="num of node (clusterx params)")],
    cpus_per_task: Annotated[int, Parameter(help="cpus per node (clusterx params)")],
    memory_per_task: Annotated[int, Parameter(help="memory per node (clusterx params)")],
    tasks_per_node: Annotated[int, Parameter(help="num cache workers per node")] = 1,
    image: Annotated[
        str, Parameter(help="image containing XTuner dependencies and ray")
    ] = "registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt28_latest",
    ray_log_to_driver: Annotated[
        bool, Parameter(help="whether or not worker logs are to be logged to driver node")
    ] = False,
):
    head_job = start_ray(nnodes=1, cpus_per_task=cpus_per_task, memory_per_task=memory_per_task, image=image)  # type: ignore[arg-type]
    worker_job = None

    try:
        logger.info(f"ray cluster job id {head_job.job_id}")
        assert head_job.nodes_ip is not None
        ray_ip = f"ray://{head_job.nodes_ip[0]}:10001"

        logger.info(f"Connect to ray cluster: {ray_ip}")

        max_retries = 10
        for attempt in range(max_retries):
            try:
                ray.init(ray_ip, log_to_driver=ray_log_to_driver)
                logger.info(f"Successfully connected to Ray cluster on attempt {attempt + 1}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.warning(
                        f"Ray connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s... (Ray may still be initializing)"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to Ray after {max_retries} attempts")
                    raise

        if nnodes > 1:
            try:
                worker_job = start_ray(
                    nnodes=nnodes - 1,
                    cpus_per_task=cpus_per_task,
                    memory_per_task=memory_per_task,
                    image=image,
                    head_ip=head_job.nodes_ip[0],
                )  # type: ignore[arg-type]
            except Exception as e:
                logger.error(f"Failed to start worker nodes: {e}")
                raise

        resources = ray.cluster_resources()
        logger.info(f"Ray cluster resources: CPU={resources.get('CPU', 0)}, nodes={len(ray.nodes())}")
        logger.info(f"Ray nodes: {[node['NodeManagerAddress'] for node in ray.nodes()]}")

        config = Config.fromfile(config_path)
        dataset_config = config.dataset_config
        tokenizer_path = config.trainer.tokenizer_path
        base_cache_dir = Path(dataset_config[0]["dataset"].cache_dir)

        total_files = len(dataset_config)
        logger.info(f"Total files to process: {total_files}")

        tracker = ProgressTracker.remote(total_files)

        global cache_worker
        worker = ray.remote(
            num_cpus=cpus_per_task // tasks_per_node,
            runtime_env={
                "env_vars": {
                    "XTUNER_TOKENIZE_WORKERS": str(cpus_per_task // tasks_per_node),
                    "PYTHONPATH": str((Path(__file__).parent.parent.parent).absolute()),
                }
            },
        )(cache_worker)
        batch_size = ceil(len(dataset_config) / (nnodes * tasks_per_node))
        batch_config_list = list(batched(dataset_config, batch_size))
        random.seed(0)
        random.shuffle(batch_config_list)

        res = []
        logger.info(f"Total tasks: {len(batch_config_list)} (nnodes={nnodes}, tasks_per_node={tasks_per_node})")
        logger.info(f"Batch size: {batch_size}, Total datasets: {len(dataset_config)}")
        for i, batch in enumerate(batch_config_list):
            batch_copy = copy.deepcopy(list(batch))

            for dataset_cfg in batch_copy:
                if hasattr(dataset_cfg["dataset"], "cache_dir") and dataset_cfg["dataset"].cache_dir is not None:
                    original_cache_dir = Path(dataset_cfg["dataset"].cache_dir)
                    worker_cache_dir = original_cache_dir / f"worker-{i}"
                    dataset_cfg["dataset"].cache_dir = str(worker_cache_dir)

            res.append(worker.remote(batch_copy, tokenizer_path, i, tracker))

            logger.info(f"Submitted task {i + 1}/{len(batch_config_list)}")

        pbar = tqdm(total=total_files, desc="Processing files", unit="file")
        stop_event = threading.Event()

        def update_progress():
            """Background thread to update progress bar."""
            last_completed = 0
            while not stop_event.is_set():
                try:
                    completed, _ = ray.get(tracker.get_progress.remote())
                    delta = completed - last_completed
                    if delta > 0:
                        pbar.update(delta)
                        last_completed = completed
                    time.sleep(0.5)  # Poll every 0.5 seconds
                except Exception as e:
                    logger.warning(f"Error updating progress: {e}")
                    break

        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

        logger.info(f"\nWaiting for {len(res)} tasks to complete...")
        ray.get(res)

        stop_event.set()
        progress_thread.join(timeout=2)
        completed, _ = ray.get(tracker.get_progress.remote())
        pbar.update(completed - pbar.n)  # Final update
        pbar.close()

        logger.info("All tasks completed, starting cache merge...")
        merge_worker_caches(base_cache_dir, num_workers=len(batch_config_list))

    finally:
        if ray.is_initialized():
            ray.shutdown()

        cluster.stop(job_id=head_job.job_id)
        if worker_job:
            cluster.stop(job_id=worker_job.job_id)


if __name__ == "__main__":
    app()
