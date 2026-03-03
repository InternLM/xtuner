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
        nnodes=nnodes,
        no_env=True,
        image="registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt28_latest",
    )
    job = cluster.run(clusterx_params)
    while job.status != JobStatus.RUNNING:
        job = cluster.get_job_info(job.job_id)
    return job


def cache_worker(dataset_config_list: DatasetConfigList, tokenizer_path):
    monkey_patch_hf_modules_cache()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    build_datasets(dataset_config_list, tokenizer)


@app.default
def main(
    config_path: Path,
    nnodes: int,
    cpus_per_task: int,
    memory_per_task: int,
):
    job = start_ray(nnodes=nnodes, cpus_per_task=cpus_per_task, memory_per_task=memory_per_task)
    try:
        print(f"ray clsuter job id {job.job_id}")
        assert job.nodes_ip is not None
        ray_ip = f"ray://{job.nodes_ip[0]}:10001"

        print(f"Connect to ray cluster: {ray_ip}")
        ray.init(ray_ip)

        config = Config.fromfile(config_path)
        dataset_config = config.dataset_config
        tokenizer_path = config.trainer.tokenizer_path

        global cache_worker
        worker = ray.remote(num_cpus=cpus_per_task)(cache_worker)
        batch_size = ceil(len(dataset_config) / nnodes)
        batch_config_list = list(batched(dataset_config, batch_size))

        for batch in batch_config_list:
            worker.remote(batch, tokenizer_path)
    finally:
        cluster.stop(job_id=job.job_id)


if __name__ == "__main__":
    app()
