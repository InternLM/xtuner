import json
import multiprocessing
import time
import uuid
from typing import Callable, Optional

import httpx
import ray
import requests  # type: ignore[import-untyped]
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.accelerator import SingleAcceleratorWorker
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.utils import get_logger


class RolloutWorker(SingleAcceleratorWorker):
    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        self.config = config
        self.rank = rank
        self.master_addr = master_addr  # ray master
        self.master_port = master_port
        self.world_size = world_size
        self.accelerator = accelerator
        self.server_func: Callable
        self.endpoints: dict[str, str] = dict()
        # handle stream response
        self.client = httpx.AsyncClient(timeout=self.config.rollout_timeout)
        self.paused = False
        self.server_task = None
        self.engine_bundle_idxs: list[int] = []
        self.server_process: Optional[multiprocessing.Process] = None
        self.init_dist_port()  # server port, nccl port, dist port
        self.logger = get_logger()

    def init_dist_port(self):
        self.host, self.ports = ray.get(find_master_addr_and_port.remote(3))
        self.dist_port = self.ports[0]
        self.server_port = self.ports[1]
        self.nccl_port = self.ports[2]
        self.dist_init_addr = f"{self.host}:{self.dist_port}"
        self.server_url = f"http://{self.host}:{self.server_port}"
        return self.dist_init_addr

    def init(self, dist_init_addr: str = ""):
        self.dist_init_addr = dist_init_addr if dist_init_addr else self.dist_init_addr
        self.launch_server()
        return (self.rank, self.server_url)

    def set_engine_bundle_idxs(self, engine_bundle_idxs: list[int]):
        self.engine_bundle_idxs = engine_bundle_idxs

    def launch_server(self):
        server_configs = self._transform_rollout_config_to_server_configs()
        timeout = 3600.0  # Increased timeout to 5 minutes for downloading large models
        start_time = time.perf_counter()

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {server_configs.api_key}",
        }

        self.logger.info(f"launch server task on server_url: {self.server_url}")

        # note(@duanyanhui): launch server as multiprocessing for sglang temporarily
        if self.config.launch_server_method == "multiprocessing":
            process = multiprocessing.Process(target=self.server_func, args=(server_configs,))
            process.start()
            self.server_process = process
            time.sleep(60)  # Wait for the server to start
            with requests.Session() as session:
                while time.perf_counter() - start_time < timeout:
                    try:
                        response = session.get(
                            f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                        )
                        if response.status_code == 200:
                            return
                    except requests.RequestException as e:
                        self.logger.error(
                            f"can't connect to server url {self.server_url}/{self.endpoints['health_generate']} because {e}"
                        )
                    time.sleep(5)
            process.terminate()
            raise TimeoutError("Server failed to start within the timeout period.")
        else:
            # launch the server as ray task
            # so that the lmdeploy backend could get externl pg
            current_pg = ray.util.get_current_placement_group()
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=current_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=self.engine_bundle_idxs[0],
            )
            assert ray.is_initialized()
            ray_kwargs = (
                {"runtime_env": server_configs.ray_runtime_env} if hasattr(server_configs, "ray_runtime_env") else {}
            )
            self.server_task = (
                ray.remote(self.server_func)
                .options(scheduling_strategy=scheduling_strategy, num_cpus=1, num_gpus=0.01, **ray_kwargs)
                .remote(server_configs)
            )

            with requests.Session() as session:
                while time.perf_counter() - start_time < timeout:
                    try:
                        response = session.get(
                            f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                        )
                        if response.status_code == 200:
                            return
                    except requests.RequestException:
                        pass

                    try:
                        ray.get(self.server_task, timeout=0.1)
                        raise Exception("Server task terminated unexpectedly.")
                    except ray.exceptions.GetTimeoutError:
                        pass
                    except Exception as e:
                        raise e

                    time.sleep(2)
            ray.cancel(self.server_task)
            raise TimeoutError("Server failed to start within the timeout period.")

    async def rollout_task(self, prompt: str, sample_params):
        uid = str(uuid.uuid4())
        response = None
        try:
            response = await self._create_request(
                f"{self.server_url}/{self.endpoints['generate']}",
                uid,
                prompt,
                sample_params=sample_params or {},
                extra_params={},
            )
            self.logger.debug(f" +++ send request {uid} to worker: {self.rank}")

            if response.status_code != 200:
                error_body = await response.atext()
                self.logger.error(f"Request {uid} failed with status {response.status_code}: {error_body}")
                return "", "failed"  # 返回明确的失败状态

            last_trajectory = ""
            async for chunk in response.aiter_text():
                if chunk == "":
                    continue
                try:
                    if self.paused:
                        await response.aclose()
                        self.logger.debug(f"--- get paused request {uid}")
                        return last_trajectory, "unfinished"
                    chunk_data = chunk[len("data:") :].strip()  # Remove "data: " prefix
                    if chunk_data == "[DONE]":
                        self.logger.debug(f" --- get finished request {uid}")
                        await response.aclose()
                        return last_trajectory, "finished"
                    else:
                        if not (chunk_data.startswith("{") and chunk_data.endswith("}")):
                            continue
                        last_trajectory += json.loads(chunk_data)["choices"][0]["text"]
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error for chunk in request {uid}: {chunk}, error: {e}")
                    continue  # 选择跳过这个损坏的块
                except Exception as e:
                    self.logger.error(f"Error processing chunk for {uid}: {chunk}, error: {e}")
                    return last_trajectory, "failed"  # 出现意外错误时，终止并返回失败
        except httpx.RequestError as e:
            self.logger.error(f"Request {uid} failed with a network error: {e}")
            return "", "failed"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in rollout_task for {uid}: {e}")
            return "", "failed"
        finally:
            # 确保在任何情况下都尝试关闭响应
            if response:
                await response.aclose()

    async def rollout(self, prompt: str, sample_params):
        return await self.rollout_task(prompt, sample_params)

    def pause(self):
        self.paused = True
        self.pause_generation()

    def restart(self):
        self.paused = False
        self.continue_generation()

    def shutdown(self):
        if self.server_task is not None:
            ray.cancel(self.server_task)
            return

        if self.server_process is not None:
            import psutil

            parent = psutil.Process(self.server_process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            gone, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                child.kill()
            parent.terminate()
            parent.wait(timeout=5)
            self.logger.debug(f"Worker {self.rank} server process and its children terminated.")
            return

    # not implemented functions
    async def _create_request(
        self,
        url: str,
        uid: str,
        prompt: str,
        sample_params: dict = dict(),
        extra_params: dict = dict(),
    ):
        raise NotImplementedError("_create_request must be implemented in subclass")

    def _transform_rollout_config_to_server_configs(self):
        raise NotImplementedError("_transform_rollout_config_to_server_configs must be implemented in subclass")

    def get_logprobs(self, input_ids, sampling_params):
        raise NotImplementedError("get_logprobs must be implemented in subclass")

    def update_weights(self):
        raise NotImplementedError("update_weights must be implemented in subclass")

    def reset_prefix_cache(self):
        raise NotImplementedError("reset_prefix_cache must be implemented in subclass")

    def sleep(self):
        raise NotImplementedError("sleep must be implemented in subclass")

    def wake_up(self):
        raise NotImplementedError("wake_up must be implemented in subclass")

    def pause_generation(self):
        raise NotImplementedError("pause_generation must be implemented in subclass")

    def continue_generation(self):
        raise NotImplementedError("continue_generation must be implemented in subclass")
