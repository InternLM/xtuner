"""CPU-only Ray simulation for rollout pause latency.

Run a small sanity case:

    XTUNER_RUN_ROLLOUT_PAUSE_SIM=1 \
    XTUNER_PAUSE_SIM_WORKERS=4 \
    XTUNER_PAUSE_SIM_BATCH_PER_WORKER=2 \
    python tests/rl/test_rollout_pause_simulation.py -v

Run the default 128-worker, 1x oversend case:

    XTUNER_RUN_ROLLOUT_PAUSE_SIM=1 \
    python tests/rl/test_rollout_pause_simulation.py -v
"""

import asyncio
import math
import os
import random
import threading
import time
import unittest
from collections import Counter
from dataclasses import dataclass

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray


GENERATE_GROUP = "generate"
CONTROL_GROUP = "control"


@dataclass(frozen=True)
class PauseSimulationConfig:
    worker_count: int
    batch_per_worker: int
    oversend_ratio: float
    pending_per_worker: int
    total_pending: int
    abort_timeout_s: float
    collect_timeout_s: float
    abort_ack_delay_s: float
    fast_return_ratio: float
    fast_return_max_s: float
    slow_return_s: float
    seed: int
    start_timeout_s: float

    @classmethod
    def from_env(cls) -> "PauseSimulationConfig":
        worker_count = int(os.environ.get("XTUNER_PAUSE_SIM_WORKERS", "128"))
        batch_per_worker = int(os.environ.get("XTUNER_PAUSE_SIM_BATCH_PER_WORKER", "16"))
        oversend_ratio = float(os.environ.get("XTUNER_PAUSE_SIM_OVERSEND_RATIO", "1.0"))
        pending_per_worker = int(math.ceil(batch_per_worker * (1.0 + oversend_ratio)))
        return cls(
            worker_count=worker_count,
            batch_per_worker=batch_per_worker,
            oversend_ratio=oversend_ratio,
            pending_per_worker=pending_per_worker,
            total_pending=worker_count * pending_per_worker,
            abort_timeout_s=float(os.environ.get("XTUNER_PAUSE_SIM_ABORT_TIMEOUT_S", "10.0")),
            collect_timeout_s=float(os.environ.get("XTUNER_PAUSE_SIM_COLLECT_TIMEOUT_S", "10.0")),
            abort_ack_delay_s=float(os.environ.get("XTUNER_PAUSE_SIM_ABORT_ACK_DELAY_S", "0.0")),
            fast_return_ratio=float(os.environ.get("XTUNER_PAUSE_SIM_FAST_RETURN_RATIO", "0.95")),
            fast_return_max_s=float(os.environ.get("XTUNER_PAUSE_SIM_FAST_RETURN_MAX_S", "2.0")),
            slow_return_s=float(os.environ.get("XTUNER_PAUSE_SIM_SLOW_RETURN_S", "30.0")),
            seed=int(os.environ.get("XTUNER_PAUSE_SIM_SEED", "0")),
            start_timeout_s=float(os.environ.get("XTUNER_PAUSE_SIM_START_TIMEOUT_S", "60.0")),
        )


class SimulatedRolloutWorker:
    def __init__(self, abort_timeout_s: float, abort_ack_delay_s: float) -> None:
        self.abort_timeout_s = abort_timeout_s
        self.abort_ack_delay_s = abort_ack_delay_s
        self.abort_event = threading.Event()
        self.started_count = 0

    @ray.method(concurrency_group=GENERATE_GROUP)
    async def generate(self, request_id: int, abort_return_delay_s: float) -> dict:
        self.started_count += 1
        while not self.abort_event.is_set():
            await asyncio.sleep(0.01)
        if abort_return_delay_s <= self.abort_timeout_s:
            await asyncio.sleep(abort_return_delay_s)
            return {
                "request_id": request_id,
                "status": "backend_returned_after_abort",
                "delay_s": abort_return_delay_s,
            }

        await asyncio.sleep(self.abort_timeout_s)
        return {
            "request_id": request_id,
            "status": "client_cancelled_after_abort_timeout",
            "delay_s": self.abort_timeout_s,
        }

    @ray.method(concurrency_group=CONTROL_GROUP)
    async def pause_generation(self) -> dict:
        self.abort_event.set()
        if self.abort_ack_delay_s > 0:
            await asyncio.sleep(self.abort_ack_delay_s)
        return {"started_count": self.started_count}

    @ray.method(concurrency_group=CONTROL_GROUP)
    def get_started_count(self) -> int:
        return self.started_count


class SimulatedRolloutController:
    def __init__(self, workers: list) -> None:
        self.workers = workers

    @ray.method(concurrency_group=CONTROL_GROUP)
    def pause_generation(self) -> dict:
        start = time.perf_counter()
        results = ray.get([worker.pause_generation.remote() for worker in self.workers])
        return {
            "worker_count": len(results),
            "started_count": sum(item["started_count"] for item in results),
            "fanout_s": time.perf_counter() - start,
        }


def _build_abort_return_delays(config: PauseSimulationConfig) -> list[float]:
    rng = random.Random(config.seed)
    delays = []
    for _ in range(config.total_pending):
        if rng.random() < config.fast_return_ratio:
            delays.append(rng.random() * config.fast_return_max_s)
        else:
            delays.append(config.slow_return_s)
    return delays


def _ensure_ray_initialized() -> bool:
    if ray.is_initialized():
        return False
    address = os.environ.get("XTUNER_PAUSE_SIM_RAY_ADDRESS", "local")
    init_kwargs = dict(
        address=address,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    if address == "local":
        init_kwargs["num_cpus"] = max(1, min(os.cpu_count() or 1, 32))
    ray.init(**init_kwargs)
    return True


def _wait_until_all_requests_start(workers: list, expected_total: int, timeout_s: float) -> int:
    deadline = time.perf_counter() + timeout_s
    started_total = 0
    while time.perf_counter() < deadline:
        started_counts = ray.get([worker.get_started_count.remote() for worker in workers])
        started_total = sum(started_counts)
        if started_total >= expected_total:
            return started_total
        time.sleep(0.1)
    return started_total


def _collect_until_timeout(refs: list, timeout_s: float) -> tuple[list[dict], list]:
    remaining = refs
    collected = []
    deadline = time.perf_counter() + timeout_s
    while remaining:
        wait_s = min(1.0, max(0.0, deadline - time.perf_counter()))
        if wait_s <= 0:
            break
        ready, remaining = ray.wait(remaining, num_returns=len(remaining), timeout=wait_s)
        if not ready:
            continue
        collected.extend(ray.get(ready))
    return collected, remaining


@unittest.skipUnless(
    os.environ.get("XTUNER_RUN_ROLLOUT_PAUSE_SIM") == "1",
    "Set XTUNER_RUN_ROLLOUT_PAUSE_SIM=1 to run the CPU-only Ray rollout pause simulation.",
)
class TestRolloutPauseSimulation(unittest.TestCase):
    def test_pause_with_many_workers_and_oversend(self):
        config = PauseSimulationConfig.from_env()
        started_ray = _ensure_ray_initialized()
        try:
            worker_cls = ray.remote(
                num_cpus=0,
                max_concurrency=config.pending_per_worker + 4,
                concurrency_groups={
                    GENERATE_GROUP: config.pending_per_worker,
                    CONTROL_GROUP: 4,
                },
            )(SimulatedRolloutWorker)
            workers = [
                worker_cls.remote(config.abort_timeout_s, config.abort_ack_delay_s)
                for _ in range(config.worker_count)
            ]
            controller_cls = ray.remote(
                num_cpus=0,
                max_concurrency=max(4, config.worker_count),
                concurrency_groups={
                    GENERATE_GROUP: max(1, config.total_pending),
                    CONTROL_GROUP: max(1, config.worker_count),
                },
            )(SimulatedRolloutController)
            controller = controller_cls.remote(workers)

            delays = _build_abort_return_delays(config)
            request_refs = []
            request_id = 0
            for worker in workers:
                for _ in range(config.pending_per_worker):
                    request_refs.append(worker.generate.remote(request_id, delays[request_id]))
                    request_id += 1

            started_total = _wait_until_all_requests_start(
                workers,
                expected_total=config.total_pending,
                timeout_s=config.start_timeout_s,
            )
            self.assertEqual(started_total, config.total_pending)

            pause_start = time.perf_counter()
            pause_result = ray.get(controller.pause_generation.remote())
            abort_fanout_s = time.perf_counter() - pause_start

            collect_start = time.perf_counter()
            collected, remaining = _collect_until_timeout(request_refs, config.collect_timeout_s)
            collect_s = time.perf_counter() - collect_start
            total_pause_s = time.perf_counter() - pause_start

            status_counts = Counter(item["status"] for item in collected)
            expected_fast = sum(delay <= config.abort_timeout_s for delay in delays)
            expected_slow = config.total_pending - expected_fast

            print("\nRollout pause simulation")
            print(f"  workers                    : {config.worker_count}")
            print(f"  batch_per_worker           : {config.batch_per_worker}")
            print(f"  oversend_ratio             : {config.oversend_ratio:.2f}")
            print(f"  pending_per_worker         : {config.pending_per_worker}")
            print(f"  total_pending              : {config.total_pending}")
            print(f"  abort_timeout_s            : {config.abort_timeout_s:.2f}")
            print(f"  collect_timeout_s          : {config.collect_timeout_s:.2f}")
            print(f"  fast_return_ratio          : {config.fast_return_ratio:.2f}")
            print(f"  expected_backend_returned  : {expected_fast}")
            print(f"  expected_client_cancelled  : {expected_slow}")
            print(f"  abort_fanout_s             : {abort_fanout_s:.3f}")
            print(f"  controller_fanout_s        : {pause_result['fanout_s']:.3f}")
            print(f"  collect_s                  : {collect_s:.3f}")
            print(f"  total_pause_s              : {total_pause_s:.3f}")
            print(f"  collected                  : {len(collected)}")
            print(f"  remaining_after_collect    : {len(remaining)}")
            print(f"  status_counts              : {dict(status_counts)}")

            self.assertEqual(pause_result["worker_count"], config.worker_count)
            self.assertEqual(pause_result["started_count"], config.total_pending)
            self.assertEqual(len(collected) + len(remaining), config.total_pending)

            expected_max_pause_s = os.environ.get("XTUNER_PAUSE_SIM_EXPECT_MAX_PAUSE_S")
            if expected_max_pause_s is not None:
                self.assertLessEqual(total_pause_s, float(expected_max_pause_s))

            for ref in remaining:
                ray.cancel(ref)
        finally:
            if started_ray:
                ray.shutdown()


if __name__ == "__main__":
    unittest.main()
