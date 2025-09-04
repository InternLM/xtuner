from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from transformers import AutoTokenizer
from xtuner.v1.ray.config.worker import RolloutConfig

from .worker import RolloutWorker


class SampleParams(BaseModel):
    """Sampling parameters configuration for text generation in XTuner.

    Args:
        n (int): Number of samples to generate for each input. Defaults to 1.
        top_k (int): Number of highest probability vocabulary tokens to keep for
            top-k filtering. Set to 0 to disable. Defaults to 0.
        top_p (float): Cumulative probability threshold for nucleus (top-p) sampling.
            Defaults to 1.0.
        temperature (float): Sampling temperature to control randomness. Lower values
            make output more deterministic. Defaults to 1.0.
        repetition_penalty (float): Penalty applied to tokens that have already
            appeared in the sequence. Defaults to 1.0 (no penalty).
        presence_penalty (float): Penalty applied based on token presence in the
            generated text. Defaults to 0.0.
        frequency_penalty (float): Penalty applied based on token frequency in the
            generated text. Defaults to 0.0.
        min_tokens (int): Minimum number of tokens to generate before considering
            stop conditions. Defaults to 0.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 2048.
        stops (List[str]): List of string sequences that will stop generation when
            encountered. Defaults to empty list.
        stop_token_ids (List[int]): List of token IDs that will stop generation when
            encountered. Defaults to empty list.
        logprobs (int): Number of log probabilities to return for each token.
            Set to 0 to disable. Defaults to 0.
        skip_special_tokens (bool): Whether to skip special tokens during decoding.
            Defaults to True.
        do_sample (bool): Whether to use sampling (True) or greedy decoding (False).
            Defaults to True.
    """

    n: Annotated[int, Parameter(help="Number of samples to generate.")] = 1
    top_k: Annotated[
        int, Parameter(help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    ] = 0
    top_p: Annotated[float, Parameter(help="The cumulative probability for nucleus sampling.")] = 1.0
    temperature: Annotated[float, Parameter(help="The value used to module the next token probabilities.")] = 1.0
    repetition_penalty: Annotated[float, Parameter(help="The parameter for repetition penalty.")] = 1.0
    presence_penalty: Annotated[float, Parameter(help="The parameter for presence penalty.")] = 0.0
    frequency_penalty: Annotated[float, Parameter(help="The parameter for frequency penalty.")] = 0.0
    min_tokens: Annotated[int, Parameter(help="Minimum number of tokens to generate.")] = 0
    max_tokens: Annotated[int, Parameter(help="Maximum number of tokens to generate.")] = 2048
    stops: Annotated[List[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[List[int], Parameter(help="List of stop token IDs.")] = []
    logprobs: Annotated[int, Parameter(help="Number of log probabilities to return.")] = 0
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True
    do_sample: Annotated[bool, Parameter(help="Whether to sample or not.")] = True


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
        self.tokenizer = (
            AutoTokenizer.from_pretrained(infer_config.model_path, trust_remote_code=True)
            if infer_config.tokenizer_path
            else None
        )
        self.workers_bundle_idx_map = workers_bundle_idx_map
        self.engine_mesh_list, self.server_url_dict = self.init_workers()
        # todo(@duanyanhui): add router to replace native round robin
        self.worker_index = 0  # round robin index
        self.sample_params = SampleParams()

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
            head_rank, start_bundle_idx = self.workers_bundle_idx_map[active_worker]
            engine_workers = workers[start_bundle_idx : start_bundle_idx + interval]
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
        return engine_mesh_list, worker_server_urls_map

    async def rollout(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List = [],
        tool_choice: str = "auto",
        sample_params: Optional[SampleParams] = None,
        extra_params: dict = dict(),
        format: str = "openai",
    ):
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
        index = self.worker_index % len(self.active_rollout_workers)
        final_sample_params = sample_params if sample_params else self.sample_params
        response_ref = self.active_rollout_workers[index].rollout.remote(  # type: ignore[attr-defined]
            prompt,
            tools=tools,
            tool_choice=tool_choice,
            sample_params=final_sample_params,
            extra_params=extra_params,
            format=format,
        )
        self.worker_index += 1
        return await response_ref

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
