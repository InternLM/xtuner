import asyncio
from functools import partial
from typing import Any, Dict, List, Optional

import ray

from transformers import AutoTokenizer
from xtuner.v1.ray.environment.lagent.tokenize import tokenize as lagent_tokenize
from xtuner.v1.utils.executor import SharedPoolExecutor
from xtuner.v1.utils import get_logger


_PROCESS_TOKENIZER = None


def _tokenize_in_process(
    task: tuple[List[Dict[str, Any]], Optional[List[Any]]],
    tokenizer_path: str,
    enable_interleaved_thinking: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    messages, tools = task
    global _PROCESS_TOKENIZER
    if _PROCESS_TOKENIZER is None:
        _PROCESS_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    assert _PROCESS_TOKENIZER is not None, "Process tokenizer is not initialized."
    return lagent_tokenize(
        _PROCESS_TOKENIZER,
        messages,
        tools=tools,
        enable_interleaved_thinking=enable_interleaved_thinking,
        enable_thinking=enable_thinking,
    )


class TokenizeWorker:
    def __init__(
        self,
        tokenizer_path: str,
        num_processes: int = 1,
        enable_interleaved_thinking: bool = True,
        enable_thinking: bool = True,
    ):
        self.logger = get_logger(tag="TokenizeWorker")
        self.tokenizer_path = tokenizer_path
        self.enable_interleaved_thinking = enable_interleaved_thinking
        self.enable_thinking = enable_thinking
        self.num_processes = max(1, num_processes)
        self.pool: Optional[SharedPoolExecutor] = None
        self.tokenizer = None

        if self.num_processes > 1:
            self.pool = SharedPoolExecutor(
                fn=_tokenize_in_process,
                partial_kwargs={
                    "tokenizer_path": tokenizer_path,
                    "enable_interleaved_thinking": self.enable_interleaved_thinking,
                    "enable_thinking": self.enable_thinking,
                },
                max_workers=self.num_processes,
                mp_context="fork",
            )
            self.logger.info(f"Tokenize worker starts process pool, num_processes={self.num_processes}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    async def tokenize(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None) -> Dict[str, Any]:
        if self.pool is not None:
            return await asyncio.wrap_future(self.pool.submit((messages, tools)))

        assert self.tokenizer is not None, "Tokenizer is not initialized."
        tokenize_call = partial(
            lagent_tokenize,
            self.tokenizer,
            messages,
            tools=tools,
            enable_interleaved_thinking=self.enable_interleaved_thinking,
            enable_thinking=self.enable_thinking,
        )
        return tokenize_call()

    def shutdown(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


class TokenizeController:
    def __init__(
        self,
        tokenizer_path: str,
        num_ray_actors: int = 0,
        num_cpus_per_actor: int = 1,
        num_processes_per_actor: int = 1,
        request_timeout: float = 300.0,
        enable_spread_scheduling: bool = True,
    ):
        self.logger = get_logger(tag="TokenizeController")
        self.request_timeout = request_timeout
        self._lock = asyncio.Lock()
        self._next_actor_idx = 0
        self._actors: List[ray.actor.ActorHandle] = []

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if num_ray_actors <= 0:
            return

        worker_cls = ray.remote(TokenizeWorker)
        actor_options: Dict[str, Any] = dict(num_cpus=num_cpus_per_actor)
        if enable_spread_scheduling and num_ray_actors > 1:
            actor_options["scheduling_strategy"] = "SPREAD"
            self.logger.info("TokenizeController enables SPREAD scheduling for tokenize workers.")
        for _ in range(num_ray_actors):
            actor = worker_cls.options(**actor_options).remote(
                tokenizer_path=tokenizer_path,
                num_processes=num_processes_per_actor,
            )
            self._actors.append(actor)
        self.logger.info(
            "TokenizeController starts %d ray actors, %d processes per actor.",
            len(self._actors),
            max(1, num_processes_per_actor),
        )

    async def tokenize(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None) -> Dict[str, Any]:
        if not self._actors:
            return lagent_tokenize(self.tokenizer, messages, tools=tools)

        async with self._lock:
            actor = self._actors[self._next_actor_idx]
            self._next_actor_idx = (self._next_actor_idx + 1) % len(self._actors)

        response_ref = actor.tokenize.remote(messages, tools)
        return await asyncio.wait_for(asyncio.shield(response_ref), timeout=self.request_timeout)

    def shutdown(self):
        tasks = [actor.shutdown.remote() for actor in self._actors]
        if tasks:
            ray.get(tasks)
