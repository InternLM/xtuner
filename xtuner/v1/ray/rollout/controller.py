import threading
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import uvicorn
from fastapi import FastAPI

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RLRolloutRequestItem, RLRolloutResponseItem, RolloutExtraParams, SampleParams
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.utils import get_logger

from .worker import RolloutWorker


@ray.remote
class RolloutController:
    """Controller for managing and coordinating multiple RolloutWorker
    actors."""

    def __init__(
        self,
        infer_config: RolloutConfig,
        workers_bundle_idx_map: Dict[RolloutWorker, Tuple[int, int]],
    ):
        """Initialize the RolloutController.

        Args:
            infer_config (RolloutConfig): The configuration for the rollout.
            workers_bundle_idx_map (Dict[RolloutWorker, Tuple[int, int]]): A map
                from worker actors to their (head_rank, bundle_index) tuple.
        """
        self.config = infer_config
        self.num_workers = 0
        self.worker_server_urls: List[str] = []
        self.active_rollout_workers: List[RolloutWorker] = []
        self.tokenizer = AutoTokenizer.from_pretrained(infer_config.tokenizer_path, trust_remote_code=True)
        self.workers_bundle_idx_map = workers_bundle_idx_map
        self.engine_mesh_list, self.server_url_dict = self.init_workers()
        self.start_api_server()
        # todo(@duanyanhui): add router to replace native round robin
        self.sample_params = SampleParams().dict()
        # note: 目前默认使用return_token_ids和return_logprob，并且不使用流式
        self.extra_params = dict(
            RolloutExtraParams(
                stream=False,
                include_stop_str_in_output=True,
                no_stop_trim=True,
                return_logprob=True,
                return_token_ids=True,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                top_logprobs=1,
            )
        )
        self.print_params_flag = True
        self.logger = get_logger()

    def get_rollout_info(self):
        """Get information about the current rollout setup.

        Returns:
            dict: A dictionary containing the engine mesh list, server URL
                dictionary, and the rollout configuration.
        """
        return dict(
            engine_mesh_list=self.engine_mesh_list,
            server_url_dict=self.server_url_dict,
            rollout_config=self.config,
        )

    def init_workers(self):
        """Initializes and configures the pool of RolloutWorker actors.

        This method configures distributed inference engines by grouping
        workers, where each group forms a tensor-parallel inference engine. It
        determines the `active_workers` to act as the head of each engine,
        constructs the `engine_mesh_list` to define engine topology, acquires
        necessary distributed communication ports, and finally launches servers
        on the `active_workers` to get their addresses.

        Returns:
            Tuple[List, Dict]: A tuple where the first element is
            `engine_mesh_list`, a list of lists containing the ranks of workers
            in each engine, and the second element is `worker_server_urls_map`,
            a dictionary mapping the ID of each active worker to its
            corresponding server URL.
        """
        workers = list(self.workers_bundle_idx_map.keys())
        active_servers_count, nodes_per_engine = self._get_active_servers_count(self.config, len(workers))
        interval = len(workers) // active_servers_count
        self.active_rollout_workers = workers[::interval]
        self.num_workers = len(self.active_rollout_workers)

        set_bundle_idxs_objectref = []
        engine_mesh_list = []
        for active_worker in self.active_rollout_workers:
            head_rank, _ = self.workers_bundle_idx_map[active_worker]
            engine_workers = workers[head_rank : head_rank + interval]
            engine_bundle_idxs = [self.workers_bundle_idx_map[worker][1] for worker in engine_workers]
            set_bundle_idxs_objectref.append(active_worker.set_engine_bundle_idxs.remote(engine_bundle_idxs))  # type: ignore[attr-defined]
            engine_mesh_list.append([self.workers_bundle_idx_map[worker][0] for worker in engine_workers])
        ray.get(set_bundle_idxs_objectref)
        # init dist_init_addr for each worker according to parallel settings
        init_dist_init_addrs = ray.get([worker.init_dist_port.remote() for worker in self.active_rollout_workers])  # type: ignore[attr-defined]
        dist_init_addrs = self._update_dist_init_addr(
            nodes_per_engine, init_dist_init_addrs, self.config.tensor_parallel_size
        )
        # launch rollout servers
        worker_server_urls_map = dict(
            ray.get(
                [
                    worker.init.remote(dist_init_addrs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(self.active_rollout_workers)
                ]
            )
        )
        self.worker_server_urls = list(worker_server_urls_map.values())
        self.worker_cycler = cycle(self.active_rollout_workers)
        return engine_mesh_list, worker_server_urls_map

    async def rollout(
        self,
        prompt: Union[str, List[Dict[str, Any]]] | None = None,
        input_ids: Optional[List[int]] | None = None,
        tools: List = [],
        tool_choice: str = "auto",
        sample_params: Optional[SampleParams] = None,
        extra_params: dict = dict(),
        format: str = "openai",
    ) -> RLRolloutResponseItem:
        # 这个函数接受标准的openapi chat create接口，所以不需要再额外定义输入的形式
        """Perform a rollout using one of the workers in a round-robin fashion.

        Args:
            prompt (List[str]): The prompt to send to the model.
            tools (List, optional): A list of tools the model can call.
                Defaults to [].
            tool_choice (str, optional): The tool choice strategy.
                Defaults to "auto".
            sample_params (Optional[SampleParams], optional): The sampling
                parameters for generation. If None, the default `sample_params`
                of the controller will be used. Defaults to None.
            extra_params (dict, optional): Extra parameters for the worker.
                Defaults to dict().
            format (str, optional): The format of the response.
                Defaults to "openai".

        Returns:
            The response from the rollout worker.
        """
        worker = next(self.worker_cycler)
        # update sample params and extra params
        self.sample_params.update(sample_params.dict() if sample_params else {})
        self.extra_params.update(extra_params if extra_params else {})
        if self.print_params_flag:
            # 通过print_params_flag控制只打印一次参数
            self.logger.info(f"Rollout with sample params: {self.sample_params}, extra params: {self.extra_params}")
            self.print_params_flag = False
        assert prompt is not None or input_ids is not None, "Either prompt or input_ids must be provided."
        response_ref = worker.rollout.remote(  # type: ignore[attr-defined]
            prompt=prompt,
            input_ids=input_ids,
            tools=tools,
            tool_choice=tool_choice,
            sample_params=self.sample_params,
            extra_params=self.extra_params,
            format=format,
        )
        return await response_ref

    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Starts the API server to expose the rollout functionality."""
        app = FastAPI()
        port = self.config.api_port if self.config.api_port else port

        @app.post("/v1/chat/completions")
        async def chat_completions(request: RLRolloutRequestItem) -> RLRolloutResponseItem:
            response = await self.rollout(
                prompt=request.messages,
                tools=request.tools,
                tool_choice=request.tool_choice,
                sample_params=request.sample_params,
                extra_params=request.extra_params,
            )
            return response

        config = uvicorn.Config(app, host=host, port=port)
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

    # internal functions
    def _update_dist_init_addr(self, nodes_per_engine, dist_init_addrs, tp_size):
        """Update the distributed initialization addresses for workers.

        This is used to group workers that belong to the same inference engine.

        Args:
            nodes_per_engine (int): The number of nodes per inference engine.
            dist_init_addrs (list): The list of initial addresses.
            tp_size (int): The tensor parallel size.

        Returns:
            list: The updated list of distributed initialization addresses.
        """
        if nodes_per_engine > 1:
            index = list(range(0, self.num_workers + 1, tp_size)) + [self.num_workers]
            for i in range(1, len(index)):
                dist_init_addrs[index[i - 1] : index[i]] = [dist_init_addrs[index[i - 1]]] * (index[i] - index[i - 1])
        return dist_init_addrs

    def _get_active_servers_count(self, infer_config: RolloutConfig, gpu_nums: int):
        """Calculate the number of active servers and nodes per engine.

        This calculation depends on the inference backend and parallelism settings.

        Args:
            infer_config (RolloutConfig): The rollout configuration.
            gpu_nums (int): The total number of GPUs available.

        Returns:
            Tuple[int, int]: A tuple containing the number of active servers
                and the number of nodes per engine.
        """
        # NOTE：Since different inference engines have different launch methods,
        # the number of nodes contained in each engine is not consistent.
        # For example: sglang requires starting an inference engine for each node,
        # while lmdeploy and vllm does not. Therefore, we calculate the number
        # of active servers based on the configuration.
        support_cross_node_comm = infer_config.rollout_cross_node_comm
        gpus_per_node = infer_config.gpus_per_node
        tp_size = infer_config.tensor_parallel_size
        nodes_per_engine = 1 if support_cross_node_comm or tp_size < gpus_per_node else tp_size // gpus_per_node
        active_servers_count = int((gpu_nums // tp_size) * nodes_per_engine)
        return active_servers_count, nodes_per_engine

    def _broadcast_to_active_workers(self, method_name: str, block: bool):
        """Helper function to call a method on all active workers.

        Args:
            method_name (str): The name of the method to call.
            block (bool): Whether to block until the call completes.

        Returns:
            A list of futures if `block` is False, otherwise a list of results.
        """
        futures = [getattr(worker, method_name).remote() for worker in self.active_rollout_workers]
        if not block:
            return futures

        results = ray.get(futures)
        return results

    def pause(self, block=True):
        """Pauses all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("pause", block)

    def restart(self, block=True):
        """Restarts all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("restart", block)

    def reset_prefix_cache(self, block=True):
        """Resets the prefix cache on all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("reset_prefix_cache", block)

    def offload(self, block=True):
        """Offloads model weights and KV cache on all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("offload", block)

    def onload_weights(self, block=True):
        """Onloads model weights on all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("onload_weights", block)

    def onload_kvcache(self, block=True):
        """Onloads KV cache on all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("onload_kvcache", block)

    def shutdown(self, block=True):
        """Shuts down all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._broadcast_to_active_workers("shutdown", block)
