import time
import traceback
from typing import Any, Dict, Optional

from clusterx.config import CLUSTER
from clusterx.launcher import CLUSTER_MAPPING
from clusterx.launcher.base import JobStatus


class ClusterTaskExecutor:
    def __init__(self):
        cluster_spec = CLUSTER_MAPPING[CLUSTER]
        cluster_cls = cluster_spec["type"]
        params_cls = cluster_spec["params"]
        cluster = cluster_cls()

        self.cluster = cluster
        self.params_cls = params_cls

    def execute_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        resource = task_config.get("resource", {})
        command = task_config.get("command", "")
        timeout = task_config.get("timeout", 600)
        envs = resource.get("envs", [])
        job_schema = None

        if not command:
            return False, "Command is empty. Not implemented! Please check!"

        all_command = []
        print(envs, resource)
        for env in envs:
            all_command.append(f"export {env}")

        all_command.append(command)
        run_command = "\n".join(all_command)

        try:
            job_name = "-".join([task_config["type"], task_config["case_name"], task_config["run_id"]])
            params = self.params_cls(
                job_name=job_name,
                cmd=run_command,
                gpus_per_task=resource.get("gpus_per_task", 8),
                cpus_per_task=resource.get("cpus_per_task", 32),
                memory_per_task=resource.get("memory_per_task", 512),
                priority=resource.get("priority", 4),
                priority_preemptible=resource.get("preemptible", False),
                num_nodes=resource.get("num_nodes", 1),
                image=resource.get("image", None),
                partition=resource.get("partition", "llmrazor_gpu"),
                no_env=resource.get("no_env", True),
            )

            job_schema = self.cluster.run(params)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"clusterx job {job_name} start fail, task config is {task_config}, exception is: {e}")

        start_time = time.time()

        while True:
            status = self.get_task_status(job_schema.job_id)
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                self.stop_task(job_schema.job_id)
                return (
                    False,
                    f"Pool timeout: jobname {job_name}, {timeout} seconds, task {job_schema.job_id} status is {status}",
                )
            elif status in [JobStatus.SUCCEEDED]:
                return True, status
            elif status in [JobStatus.FAILED, JobStatus.STOPPED]:
                return False, status
            time.sleep(10)

    def get_task_status(self, job_id: str) -> Optional[JobStatus]:
        try:
            status = self.cluster.get_job_info(job_id).status
        except Exception as e:
            status = JobStatus.UNRECORGNIZED
            print(f"Check job {job_id} status failed, exception is: {e}")

        return status

    def stop_task(self, job_id: str) -> Optional[JobStatus]:
        error_time = 0
        while error_time < 10:
            try:
                self.cluster.stop(job_id=job_id)
                return True
            except Exception as e:
                error_time += 1
                if error_time <= 10:
                    print(
                        f"Stop task {job_id} fail, try time {error_time}, exception is: {e}",
                    )
                    time.sleep(100)
                else:
                    raise Exception(f"Stop task {job_id} failed after retry 10 times, exception is: {e}")
