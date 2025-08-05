import json
from typing import List

import ray
import uvloop
from ray import ObjectRef
from ray.util.queue import Queue
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

from .worker import RolloutWorker


def run_vllm_server_wrapper(server_args):
    uvloop.run(run_server(server_args))


@ray.remote
class vLLMWorker(RolloutWorker):
    def __init__(
        self,
        config,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        bundle_idx: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, bundle_idx, accelerator)
        self.server_func = run_vllm_server_wrapper
        self.endpoints["health_generate"] = "health"
        self.endpoints["generate"] = "v1/chat/completions"
        self.endpoints["output_ids"] = "output_ids"
        self.endpoints["response"] = "text"

    async def _create_request(
        self,
        url: str,
        uid: str,
        input_ids: List[int],
        prompt: str,
        sample_params: dict = dict(),
        extra_params: dict = dict(),
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY",  # 如果需要鉴权
        }
        payload = {
            "request_id": uid,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        payload.update(sample_params)
        payload.update(extra_params)

        req = self.client.build_request(
            "POST",
            url,
            headers=headers,
            json=payload,
        )
        r = await self.client.send(req, stream=True)
        return r

    async def fetch_response_task(self, item, outqueue: Queue):
        last_trajectory = ""
        meta_ref, res = item[0], item[1]
        async for chunk in res.aiter_text():
            # chunk contains:
            # 1. " "
            # 2. "data: {\"choices\":[{\"delta\":{\"content\":\"...\"},\"index\":0,\"finish_reason\":null}]}"
            # 3. "[DONE]"
            if chunk == "":
                continue
            try:
                chunk = chunk[5:].strip()  # Remove "data: " prefix
                if self.paused:
                    self._collect_pending_response(meta_ref, last_trajectory)
                    await res.aclose()
                    print(f"Worker {self.rank} is paused, skipping when handle step response.")
                    break
                if chunk == "[DONE]":
                    meta = ray.get(meta_ref)
                    meta.output_ids = []
                    meta.response = last_trajectory
                    # note(@duanyanhui): A tuple(objectref) needs to be placed in the outqueue
                    # here to ensure objectref is passed between multiple actors; otherwise,
                    # outqueue.get() will automatically dereference the objectref
                    outqueue.put((ray.put(meta),))
                    print(f"--- get response from rank {self.rank}")
                    self.consumed_samples += 1
                    await res.aclose()
                else:
                    chunk_json = json.loads(chunk)
                    last_trajectory += chunk_json["choices"][0]["delta"]["content"]
            except Exception as e:
                print(f"Error processing chunk: {chunk}, error: {e}")
                await res.aclose()
                break

    def _collect_pending_response(self, meta_ref: ObjectRef, last_trajectory: str):
        meta = ray.get(meta_ref)
        meta.output_ids = []
        meta.response = last_trajectory
        self.buffer_queue.put((ray.put(meta),))

    def get_logprobs(self, input_ids, sampling_params):
        pass

    def generate(self, input_ids, sampling_params):
        # 直接调用engine.generate方法
        pass

    def sleep(self, level=1):
        pass

    def wake_up(self):
        pass

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def shutdown(self):
        # router rm worker && kill process
        pass

    def update_weights(self, ipc_handles):
        # todo
        pass

    def reset_prefix_cache(self):
        # todo
        pass

    def _transform_rollout_config_to_server_configs(self, infer_config):
        # use vllm FlexibleArgumentParser to parse the config
        # and return the args as the default server config
        parser = FlexibleArgumentParser()
        parser = make_arg_parser(parser)
        args = parser.parse_args([])
        args.__dict__.update(vars(infer_config))

        args.host = self.host
        args.port = self.server_port
        args.model = infer_config.model_path
        args.disable_log_requests = True
        args.disable_log_stats = True
        return args
