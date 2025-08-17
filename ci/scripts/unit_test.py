import sys
import time
from pathlib import Path
import os

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clusterx import CLUSTER, CLUSTER_MAPPING
from clusterx.launcher.base import JobStatus
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Unit Test Environment")
    parser.add_argument(
        "image",
        type=str,
    )
    parser.add_argument(
        "env",
        type=str,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cluster_spec = CLUSTER_MAPPING[CLUSTER]
    cluster_cls = cluster_spec["type"]
    params_cls = cluster_spec["params"]

    current_dir = os.getcwd()
    cmd = f"cd {current_dir}; {args.env}; pytest tests; export XTUNER_DETERMINISTIC=true"
    print(f"Running command: {cmd}")

    assert params_cls is not None and cluster_cls is not None, (
        f"Cluster {CLUSTER} is not available in current ci machine!"
    )
    commit_id = os.environ.get('CI_COMMIT_SHORT_SHA')
    commit_branch = os.environ.get('CI_COMMIT_REF_NAME')
    job_id = os.environ.get('CI_JOB_ID')
    params = params_cls(
        job_name=f"xtuner-ci-unittest-{job_id}-{commit_branch}-{commit_id}",
        image=args.image,
        cmd=cmd,
        gpus_per_task=8,
        num_nodes=1,
        no_env=True
    )
    cluster = cluster_cls()
    job_schema = cluster.run(params)
    while True:
        time.sleep(1)
        status = cluster.get_job_info(job_schema.job_id).status
        if status == JobStatus.QUEUING:
            print(f"Job {job_schema.job_id} is queuing...")

        if not isinstance(status, str):
            print(f"Found unrecognized status {status}, waiting...")

        if status == JobStatus.RUNNING:
            print(f"Job {job_schema.job_id} is running...")

        if status == JobStatus.FAILED:
            time.sleep(10)
            # Sleep 10 seconds to wait for aliyun or volc get the full log
            log = cluster.get_log(job_schema.job_id)
            print(log)
            raise RuntimeError(f"Job {job_schema.job_id} failed with status {status}")

        if status == JobStatus.SUCCEEDED:
            print(f"Job {job_schema.job_id} succeeded!")
            break


if __name__ == "__main__":
    main()
