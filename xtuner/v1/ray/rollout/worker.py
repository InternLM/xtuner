import asyncio
import multiprocessing
import time
from typing import Callable

import httpx
import ray
import requests  # type: ignore[import-untyped]
from ray import ObjectRef
from ray.util.queue import Queue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from starlette.responses import StreamingResponse

from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.accelerator import SingleAcceleratorWorker
from xtuner.v1.ray.config import RolloutConfig


class RolloutWorker(SingleAcceleratorWorker):
    def __init__(
        self,
        infer_config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        bundle_idx: int,
        accelerator: str = "GPU",
    ):
        self.config = infer_config
        self.rank = rank
        self.master_addr = master_addr  # ray master
        self.master_port = master_port
        self.world_size = world_size
        self.bundle_idx = bundle_idx
        self.accelerator = accelerator
        self.server_func: Callable
        self.endpoints: dict[str, str] = dict()
        # handle stream response
        self.client = httpx.AsyncClient()
        self.received_samples = 0
        self.consumed_samples = 0
        self.response_dict: dict[str, tuple[ObjectRef, StreamingResponse]] = {}
        self.buffer_queue = Queue(maxsize=1000)
        self.paused = False
        self.server_task = None
        self.init_dist_port()  # server port, nccl port, dist port

    def init_dist_port(self):
        self.host, self.ports = ray.get(find_master_addr_and_port.remote(3))
        self.dist_port = self.ports[0]
        self.server_port = self.ports[1]
        self.nccl_port = self.ports[2]
        self.dist_init_addr = f"{self.host}:{self.dist_port}"
        self.server_url = f"http://{self.host}:{self.server_port}"
        return self.dist_init_addr

    def init(self, infer_config: RolloutConfig, dist_init_addr: str = ""):
        self.dist_init_addr = dist_init_addr if dist_init_addr else self.dist_init_addr
        self.server_task = self.launch_server(infer_config)
        return self.server_url

    def launch_server(self, infer_config, launch_server_method: str = "multiprocessing"):
        server_configs = self._transform_rollout_config_to_server_configs(infer_config)
        timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
        start_time = time.perf_counter()

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {server_configs.api_key}",
        }

        print(f"launch server task on server_url: {self.server_url}")

        # note(@duanyanhui): launch server as multiprocessing for sglang temporarily
        if launch_server_method == "multiprocessing":
            process = multiprocessing.Process(target=self.server_func, args=(server_configs,))
            process.start()
            time.sleep(60)  # Wait for the server to start
            with requests.Session() as session:
                while time.perf_counter() - start_time < timeout:
                    try:
                        response = session.get(
                            f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                        )
                        if response.status_code == 200:
                            return process
                    except requests.RequestException as e:
                        print(
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
                placement_group_bundle_index=self.bundle_idx,
            )
            assert ray.is_initialized()
            server_task = (
                ray.remote(self.server_func)
                .options(scheduling_strategy=scheduling_strategy, num_cpus=1, num_gpus=0.01)
                .remote(server_configs)
            )

            with requests.Session() as session:
                while time.perf_counter() - start_time < timeout:
                    try:
                        response = session.get(
                            f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                        )
                        if response.status_code == 200:
                            return server_task
                    except requests.RequestException:
                        pass

                    try:
                        ray.get(server_task, timeout=0.1)
                        raise Exception("Server task terminated unexpectedly.")
                    except ray.exceptions.GetTimeoutError:
                        pass
                    except Exception as e:
                        raise e

                    time.sleep(2)
            ray.cancel(server_task, no_restart=True)
            raise TimeoutError("Server failed to start within the timeout period.")

    async def rollout(self, inqueue: Queue, outqueue: Queue):
        self.paused = False
        await self.rollout_task(inqueue, outqueue)

    async def rollout_task(self, inqueue: Queue, outqueue: Queue):
        tasks = dict()
        monitored = set()
        self.response_dict = {}
        while not self.paused:
            # async query response
            for uid, item in self.response_dict.items():
                if uid not in monitored:
                    task = asyncio.create_task(self.fetch_response_task(item, outqueue))
                    tasks[uid] = task
                    monitored.add(uid)

            # check if tasks are done
            done_uids = []
            for uid, task in list(tasks.items()):
                if task.done():
                    done_uids.append(uid)
                    try:
                        await task  # 捕获异常
                    except Exception as e:
                        print(f"Error in stream task {uid}: {e}")

            # delete finished task from response_list
            for uid in done_uids:
                if uid in self.response_dict:
                    del self.response_dict[uid]
                del tasks[uid]
                monitored.remove(uid)

            # check to insert new query
            await self._insert_query(inqueue)
            await asyncio.sleep(0.01)

    async def pause(self):
        self.paused = True
        while self.buffer_queue.qsize() + self.consumed_samples < self.received_samples:
            await asyncio.sleep(0.01)
        print(
            f"Worker {self.rank} paused, buffer queue size: {self.buffer_queue.qsize()}, consumed samples: {self.consumed_samples}, received samples: {self.received_samples}"
        )
        self.pause_generation()

    async def restart(self, inqueue: Queue, outqueue: Queue):
        if self.paused:
            self.paused = False
            self.wake_up()
            await self.rollout_task(inqueue, outqueue)

    async def _insert_query(self, inqueue: Queue):
        if (
            not self.paused
            and len(self.response_dict) < self.config.max_running_requests
            and (inqueue.qsize() + self.buffer_queue.qsize()) > 0
        ):
            rollout_meta, uid, prompt, input_ids, sample_params, extra_params = self._get_sample(inqueue)
            response = await self._create_request(
                f"{self.server_url}/{self.endpoints['generate']}", uid, input_ids, prompt, sample_params, extra_params
            )
            self.response_dict[uid] = (ray.put(rollout_meta), response)
            self.received_samples += 1
            print(f"+++ add query to rank {self.rank}")

    def _get_sample(self, inqueue):
        if not self.buffer_queue.empty():
            rollout_meta_ref = self.buffer_queue.get()
        else:
            rollout_meta_ref = inqueue.get()
        rollout_meta = ray.get(rollout_meta_ref[0])  # tuple(objectref,)
        uid = rollout_meta.uid
        response = rollout_meta.response
        output_ids = rollout_meta.output_ids
        prompt = rollout_meta.prompt + response if response else rollout_meta.prompt
        input_ids = rollout_meta.input_ids + output_ids if output_ids else rollout_meta.input_ids
        sample_params = rollout_meta.sample_params.dict()
        extra_params = rollout_meta.extra_params if rollout_meta.extra_params else {}
        return rollout_meta, uid, prompt, input_ids, sample_params, extra_params

    # not implemented functions
    async def _create_request(
        self,
        url: str,
        uid: str,
        input_ids: list[int],
        prompt: str,
        sample_params: dict = dict(),
        extra_params: dict = dict(),
    ):
        raise NotImplementedError("_create_request must be implemented in subclass")

    async def fetch_response_task(self, item, outqueue: Queue):
        raise NotImplementedError("fetch_response_task must be implemented in subclass")

    def _collect_pending_response(self, meta_ref: ObjectRef, last_trajectory: str):
        raise NotImplementedError("_collect_pending_response must be implemented in subclass")

    # dispatched functions
    def _transform_rollout_config_to_server_configs(self, infer_config: RolloutConfig):
        pass

    def get_lobprobs(self, input_ids, sampling_params):
        pass

    def update_weights(self):
        pass

    def reset_prefix_cache(self):
        pass

    def sleep(self):
        pass

    def wake_up(self):
        pass

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def shutdown(self):
        pass
