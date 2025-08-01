import copy
import json
import os
from typing import List

import ray
from ray import ObjectRef
from ray.util.queue import Queue
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from xtuner.v1.ray.config import RolloutConfig

from .worker import RolloutWorker


@ray.remote
class SGLangWorker(RolloutWorker):
    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        bundle_idx: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, bundle_idx, accelerator)
        self.server_func = launch_server
        self.endpoints["health_generate"] = "health_generate"
        self.endpoints["generate"] = "generate"

    async def _create_request(
        self,
        url: str,
        uid: str,
        input_ids: List[int],
        prompt: str,
        sample_params: dict = dict(),
        extra_params: dict = dict(),
    ):
        # default params
        sample_params["max_new_tokens"] = sample_params.get("max_tokens", 128)
        del sample_params["max_tokens"]
        payload = {
            "stream": True,
            "sampling_params": sample_params,
        }
        if input_ids:
            payload["input_ids"] = input_ids
        else:
            payload["text"] = prompt

        if extra_params:
            payload.update(extra_params)

        req = self.client.build_request(
            "POST",
            url,
            json=payload,
        )
        r = await self.client.send(req, stream=True)
        return StreamingResponse(r.aiter_text(), background=BackgroundTask(r.aclose))

    async def fetch_response_task(self, item, outqueue: Queue):
        last_trajectory = ""
        meta_ref, res = item[0], item[1]
        async for chunk in res.body_iterator:
            chunk = chunk[5:].strip()
            if self.paused:
                self._collect_pending_response(meta_ref, last_trajectory)
                print(f"Worker {self.rank} is paused, skipping when handle step response.")
                break
            if chunk == "[DONE]":
                last_trajectory = json.loads(last_trajectory)
                meta = ray.get(meta_ref)
                meta.output_ids = last_trajectory.get(self.endpoints["output_ids"], [])
                meta.response = last_trajectory.get(self.endpoints["text"], "")
                outqueue.put(ray.put(meta))
                print(f"--- get response from rank {self.rank}")
                self.consumed_samples += 1
                break
            else:
                last_trajectory = copy.deepcopy(chunk)

    def _collect_pending_response(self, meta_ref: ObjectRef, last_trajectory: str):
        meta = ray.get(meta_ref)
        last_trajectory_json = json.loads(last_trajectory)
        meta.output_ids = last_trajectory_json.get(self.endpoints["output_ids"], [])
        meta.response = last_trajectory_json.get(self.endpoints["text"], "")
        self.buffer_queue.put(ray.put(meta))

    def get_logprobs(self, input_ids, sampling_params):
        return self._make_request(
            "generate",
            {"input_ids": input_ids, "sampling_params": sampling_params, "stream": False, "return_logprob": True},
        )

    def sleep(self, level=1):
        return self._make_request("release_memory_occupation")

    def wake_up(self):
        return self._make_request("resume_memory_occupation")

    def pause_generation(self):
        return self._make_request("pause_generation")

    def continue_generation(self):
        return self._make_request("continue_generation")

    def shutdown(self):
        pass

    def update_weights(self, ipc_handles):
        pass

    def reset_prefix_cache(self):
        pass

    def _transform_rollout_config_to_server_configs(self, infer_config):
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        sglang_server_args = ServerArgs(model_path=infer_config.model_path)
        sglang_server_args.host = self.host
        sglang_server_args.port = self.server_port
        sglang_server_args.nccl_port = self.nccl_port
        sglang_server_args.dist_init_addr = self.dist_init_addr
        sglang_server_args.base_gpu_id = self.rank % infer_config.gpus_per_node
        sglang_server_args.gpu_id_step = 1
        sglang_server_args.nnodes = max(1, infer_config.tensor_parallel_size // infer_config.gpus_per_node)

        if sglang_server_args.nnodes > 1:
            sglang_server_args.node_rank = self.rank // infer_config.gpus_per_node
        else:
            sglang_server_args.node_rank = 0
        return sglang_server_args

    def _transform_rollout_config_to_router_configs(self, infer_config):
        pass
