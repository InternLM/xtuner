#!/usr/bin/env python3
import argparse
import asyncio
import time
from dataclasses import dataclass
from statistics import mean
from typing import Awaitable, Callable, Dict, List, Tuple

import ray
from transformers import AutoTokenizer

from xtuner.v1.ray.config.tokenize import TokenizeControllerConfig
from xtuner.v1.ray.environment.lagent.tokenize import tokenize as lagent_tokenize


TARGET_LENGTHS = [16384, 32768, 65536, 131072, 262144]
DEFAULT_CONCURRENCY_LEVELS = [512, 1024, 4096]


@dataclass
class SweepResult:
    target_length: int
    actual_length: int
    repeat_count: int
    message_chars: int


@dataclass
class BaselineTimingResult:
    target_length: int
    actual_length: int
    elapsed_sec: float


@dataclass
class ThroughputResult:
    mode: str
    target_length: int
    concurrency: int
    total_requests: int
    total_elapsed_sec: float
    rps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


def build_messages(repeat_count: int) -> List[Dict[str, str]]:
    chunk = "Token length sweep stress test segment. "
    prefix = "Please summarize the following synthetic corpus:\n"
    body = chunk * repeat_count
    return [{"role": "user", "content": prefix + body}]


async def get_input_length(controller, repeat_count: int) -> int:
    messages = build_messages(repeat_count)
    tokenized = await controller.tokenize(messages, tools=None)
    return len(tokenized["input_ids"])


async def find_repeat_for_target(
    get_input_length_func: Callable[[int], Awaitable[int]],
    target_len: int,
    init_guess: int,
) -> Tuple[int, int]:
    low = 1
    high = max(2, init_guess)
    high_len = await get_input_length_func(high)
    while high_len < target_len:
        low = high + 1
        high *= 2
        high_len = await get_input_length_func(high)

    best_repeat = high
    best_len = high_len
    while low <= high:
        mid = (low + high) // 2
        mid_len = await get_input_length_func(mid)
        if mid_len >= target_len:
            best_repeat = mid
            best_len = mid_len
            high = mid - 1
        else:
            low = mid + 1
    return best_repeat, best_len


async def run_sweep(args):
    controller = None
    baseline_tokenizer = None
    if args.baseline_only:
        baseline_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

        async def get_length_func(repeat_count: int) -> int:
            messages = build_messages(repeat_count)
            tokenized = lagent_tokenize(baseline_tokenizer, messages, tools=None)
            return len(tokenized["input_ids"])
    else:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        cfg = TokenizeControllerConfig(
            num_ray_actors=args.num_ray_actors,
            num_cpus_per_actor=args.num_cpus_per_actor,
            num_processes_per_actor=args.num_processes_per_actor,
            request_timeout=args.request_timeout,
            enable_spread_scheduling=args.enable_spread_scheduling,
        )
        controller = cfg.build(tokenizer_path=args.tokenizer_path)

        async def get_length_func(repeat_count: int) -> int:
            return await get_input_length(controller, repeat_count)

    results: List[SweepResult] = []
    guess = 64
    for target in TARGET_LENGTHS:
        repeat_count, actual_len = await find_repeat_for_target(get_length_func, target, guess)
        message_chars = len(build_messages(repeat_count)[0]["content"])
        results.append(
            SweepResult(
                target_length=target,
                actual_length=actual_len,
                repeat_count=repeat_count,
                message_chars=message_chars,
            )
        )
        guess = repeat_count

    print(
        "target_len\tactual_len\trepeat_count\tmessage_chars\tstatus\n"
        + "-" * 80
    )
    for row in results:
        status = "OK" if row.actual_length >= row.target_length else "FAILED"
        print(
            f"{row.target_length}\t{row.actual_length}\t{row.repeat_count}\t"
            f"{row.message_chars}\t{status}"
        )

    if args.baseline_only:
        # Baseline: direct synchronous tokenize path.
        print("\nDirect synchronous tokenize baseline")
        print("target_len\tactual_len\telapsed_sec")
        print("-" * 60)
        baseline_results: List[BaselineTimingResult] = []
        total_elapsed = 0.0
        for row in results:
            messages = build_messages(row.repeat_count)
            start = time.perf_counter()
            baseline_tokenized = lagent_tokenize(baseline_tokenizer, messages, tools=None)
            elapsed = time.perf_counter() - start
            actual_len = len(baseline_tokenized["input_ids"])
            total_elapsed += elapsed
            baseline_results.append(
                BaselineTimingResult(
                    target_length=row.target_length,
                    actual_length=actual_len,
                    elapsed_sec=elapsed,
                )
            )
            print(f"{row.target_length}\t{actual_len}\t{elapsed:.6f}")

        avg_elapsed = total_elapsed / len(baseline_results)
        print("-" * 60)
        print(f"baseline_total_sec={total_elapsed:.6f}")
        print(f"baseline_avg_sec={avg_elapsed:.6f}")

    async def baseline_call(messages: List[Dict[str, str]]) -> int:
        tokenized = await asyncio.to_thread(lagent_tokenize, baseline_tokenizer, messages, None)
        return len(tokenized["input_ids"])

    async def controller_call(messages: List[Dict[str, str]]) -> int:
        assert controller is not None
        tokenized = await controller.tokenize(messages, tools=None)
        return len(tokenized["input_ids"])

    def percentile(values: List[float], ratio: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int((len(ordered) - 1) * ratio))
        return ordered[idx]

    async def benchmark_one_mode(
        mode: str,
        tokenize_func: Callable[[List[Dict[str, str]]], Awaitable[int]],
        messages: List[Dict[str, str]],
        expected_length: int,
        concurrency: int,
        total_requests: int,
    ) -> ThroughputResult:
        semaphore = asyncio.Semaphore(concurrency)
        latencies: List[float] = []

        async def one_request():
            async with semaphore:
                start = time.perf_counter()
                actual_len = await tokenize_func(messages)
                elapsed = time.perf_counter() - start
                if actual_len != expected_length:
                    raise RuntimeError(
                        f"{mode} output length mismatch: expected={expected_length}, actual={actual_len}"
                    )
                latencies.append(elapsed)

        begin = time.perf_counter()
        await asyncio.gather(*(one_request() for _ in range(total_requests)))
        total_elapsed_sec = time.perf_counter() - begin

        avg_latency_ms = mean(latencies) * 1000
        return ThroughputResult(
            mode=mode,
            target_length=expected_length,
            concurrency=concurrency,
            total_requests=total_requests,
            total_elapsed_sec=total_elapsed_sec,
            rps=(total_requests / total_elapsed_sec) if total_elapsed_sec > 0 else 0.0,
            avg_latency_ms=avg_latency_ms,
            p50_latency_ms=percentile(latencies, 0.50) * 1000,
            p95_latency_ms=percentile(latencies, 0.95) * 1000,
            p99_latency_ms=percentile(latencies, 0.99) * 1000,
        )

    print("\nHigh concurrency E2E benchmark")
    print("mode\ttarget_len\tconcurrency\trequests\ttotal_sec\trps\tavg_ms\tp50_ms\tp95_ms\tp99_ms")
    print("-" * 140)
    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]
    total_requests_factor = max(1, args.requests_factor)

    bench_rows: List[ThroughputResult] = []
    for row in results:
        messages = build_messages(row.repeat_count)
        expected_length = row.actual_length
        for concurrency in concurrency_levels:
            total_requests = concurrency * total_requests_factor
            if args.baseline_only:
                baseline_metrics = await benchmark_one_mode(
                    mode="baseline_sync",
                    tokenize_func=baseline_call,
                    messages=messages,
                    expected_length=expected_length,
                    concurrency=concurrency,
                    total_requests=total_requests,
                )
                bench_rows.append(baseline_metrics)
                print(
                    f"{baseline_metrics.mode}\t{row.target_length}\t{concurrency}\t{total_requests}\t"
                    f"{baseline_metrics.total_elapsed_sec:.4f}\t{baseline_metrics.rps:.2f}\t"
                    f"{baseline_metrics.avg_latency_ms:.2f}\t{baseline_metrics.p50_latency_ms:.2f}\t"
                    f"{baseline_metrics.p95_latency_ms:.2f}\t{baseline_metrics.p99_latency_ms:.2f}"
                )
            elif controller is not None:
                controller_metrics = await benchmark_one_mode(
                    mode="controller",
                    tokenize_func=controller_call,
                    messages=messages,
                    expected_length=expected_length,
                    concurrency=concurrency,
                    total_requests=total_requests,
                )
                bench_rows.append(controller_metrics)
                print(
                    f"{controller_metrics.mode}\t{row.target_length}\t{concurrency}\t{total_requests}\t"
                    f"{controller_metrics.total_elapsed_sec:.4f}\t{controller_metrics.rps:.2f}\t"
                    f"{controller_metrics.avg_latency_ms:.2f}\t{controller_metrics.p50_latency_ms:.2f}\t"
                    f"{controller_metrics.p95_latency_ms:.2f}\t{controller_metrics.p99_latency_ms:.2f}"
                )

    if controller is not None:
        controller.shutdown()
    if ray.is_initialized():
        ray.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep long input_ids and benchmark high-concurrency tokenize e2e latency."
    )
    parser.add_argument("--tokenizer-path", type=str, default="/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/interns2-35ba3-base05-20260424a-rl-data260428rc0-56k-badword-mtp4-resume800/20260430074140/hf-40", help="HuggingFace tokenizer path or local path.")
    parser.add_argument("--num-ray-actors", type=int, default=0, help="Tokenize ray actor count (0 means local mode).")
    parser.add_argument("--num-cpus-per-actor", type=int, default=1, help="CPU cores per tokenize actor.")
    parser.add_argument(
        "--num-processes-per-actor", type=int, default=1, help="Subprocess count inside each tokenize actor."
    )
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Tokenize request timeout in seconds.")
    parser.add_argument(
        "--enable-spread-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Ray SPREAD scheduling for tokenize actors.",
    )
    parser.add_argument(
        "--baseline-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only direct synchronous tokenize path, without building tokenize controller.",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CONCURRENCY_LEVELS),
        help="Comma-separated concurrency levels, e.g. 512,1024,4096.",
    )
    parser.add_argument(
        "--requests-factor",
        type=int,
        default=1,
        help="Total requests per level = concurrency * requests_factor.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_sweep(parse_args()))
