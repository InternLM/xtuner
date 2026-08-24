import re
import time
import traceback
from typing import Any, Dict, Optional

from clusterx.config import CLUSTER
from clusterx.launcher import CLUSTER_MAPPING
from clusterx.launcher.base import JobSchema, JobStatus


JOB_LOOKUP_RETRY_INTERVAL_S = 5
JOB_LOOKUP_RETRY_TIMES = 6


class ClusterTaskExecutor:
    def __init__(self):
        cluster_spec = CLUSTER_MAPPING[CLUSTER]
        cluster_cls = cluster_spec["type"]
        params_cls = cluster_spec["params"]
        cluster = cluster_cls()

        self.cluster = cluster
        self.params_cls = params_cls

    def execute_task(self, task_config: Dict[str, Any]):
        resource = task_config.get("resource", None)
        command = task_config.get("command", "")
        timeout = task_config.get("timeout", 600)
        envs = resource.get("envs", [])
        job_schema = None

        if not command:
            return False, "Command is empty or resource is None. Not implemented! Please check!"
        if resource is None:
            return False, "Resource is None. Please check!"

        all_command = []
        print(envs, resource)
        for env in envs:
            all_command.append(f"export {env}")

        all_command.append(command)
        run_command = "; ".join(all_command)
        job_name = "-".join([task_config["type"], task_config["case_name"], task_config["run_id"]])

        try:
            params = self.params_cls(
                job_name=job_name,
                cmd=run_command,
                gpus_per_task=resource.get("gpus_per_task", 8),
                cpus_per_task=resource.get("cpus_per_task", 32),
                memory_per_task=resource.get("memory_per_task", 512),
                priority=resource.get("priority", 9),
                priority_preemptible=resource.get("preemptible", False),
                num_nodes=resource.get("num_nodes", 1),
                image=resource.get("image", None),
                no_env=resource.get("no_env", True),
                image_pull_policy=resource.get("image_pull_policy", "Always"),
            )

            job_schema = self.cluster.run(params)
        except Exception as e:
            traceback.print_exc()
            job_schema = self._lookup_job_schema(job_name)
            if job_schema is None:
                raise RuntimeError(
                    f"clusterx job {job_name} start fail and lookup found no matching job, "
                    f"task config is {task_config}, exception is: {e}"
                )
            print(
                f"clusterx job {job_name} submit error recovered via lookup: "
                f"job_id={job_schema.job_id}, status={job_schema.status}, original exception: {e}"
            )

        start_time = time.time()
        run_start_time = None

        while True:
            status = self.get_task_status(job_schema.job_id)
            if status in [JobStatus.RUNNING] and run_start_time is None:
                run_start_time = time.time()
            if status in [JobStatus.SUCCEEDED]:
                run_time = time.time() - run_start_time
                if run_time >= timeout:
                    return False, f"Task succeeded, but run time is {run_time}, exceeding then {timeout}"
                else:
                    return True, "Task succeeded"
            elif status in [JobStatus.FAILED, JobStatus.STOPPED]:
                if status in [JobStatus.FAILED]:
                    time.sleep(10)
                    try:
                        log = self.cluster.get_log(job_schema.job_id)
                        print("=== Task log ===")
                        print(log)
                    except Exception as e:
                        print(f"Get log failed: {e}")
                return False, "Task failed or stopped"
            else:
                start_time = time.time()
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                self.stop_task(job_schema.job_id)
                raise Exception(
                    f"Pool timeout: jobname {job_name}, {timeout} seconds, task {job_schema.job_id} status is {status}"
                )
            time.sleep(10)

    @staticmethod
    def _job_name_matches(candidate: str | None, job_name: str) -> bool:
        if not candidate:
            return False
        return candidate == job_name or candidate.startswith(f"{job_name}-")

    def _pick_latest_job(self, jobs: list[JobSchema]) -> JobSchema:
        return max(jobs, key=lambda job: job.job_id or job.job_name or "")

    def _lookup_job_schema_once(self, job_name: str) -> JobSchema | None:
        try:
            return self.cluster.get_job_info(job_name)
        except Exception:
            pass

        name_regex = rf"^{re.escape(job_name)}(-.*)?$"
        try:
            jobs = self.cluster.list_jobs(regex=name_regex, num=50)
            if jobs:
                return self._pick_latest_job(jobs)
        except Exception as e:
            print(f"list_jobs lookup for {job_name} failed: {e}")

        client = getattr(self.cluster, "client", None)
        get_job_name = getattr(self.cluster, "_get_job_name", None)
        if client is not None and get_job_name is not None:
            try:
                matched_names = [
                    get_job_name(job)
                    for job in (client.list() or [])
                    if self._job_name_matches(get_job_name(job), job_name)
                ]
                if matched_names:
                    return self.cluster.get_job_info(max(matched_names))
            except Exception as e:
                print(f"brainpp client list lookup for {job_name} failed: {e}")

        try:
            jobs = self.cluster.list_jobs(num=100)
            matched = [job for job in jobs if self._job_name_matches(job.job_id, job_name)]
            if matched:
                return self._pick_latest_job(matched)
        except Exception as e:
            print(f"generic list_jobs lookup for {job_name} failed: {e}")

        return None

    def _lookup_job_schema(self, job_name: str) -> JobSchema | None:
        for attempt in range(1, JOB_LOOKUP_RETRY_TIMES + 1):
            job_schema = self._lookup_job_schema_once(job_name)
            if job_schema is not None:
                return job_schema
            if attempt < JOB_LOOKUP_RETRY_TIMES:
                print(
                    f"Job {job_name} not found on attempt {attempt}/{JOB_LOOKUP_RETRY_TIMES}, "
                    f"retry in {JOB_LOOKUP_RETRY_INTERVAL_S}s"
                )
                time.sleep(JOB_LOOKUP_RETRY_INTERVAL_S)
        return None

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
