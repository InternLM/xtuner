import argparse
import json
import os
import time
from collections import defaultdict

import psutil
import ray
from ray.util.state import list_actors

from xtuner.v1._writer import TensorboardWriter


try:
    import pynvml
except ImportError:
    pynvml = None


def _get_current_node_id() -> str:
    node_id = ray.get_runtime_context().get_node_id()
    if hasattr(node_id, "hex"):
        return node_id.hex()
    return str(node_id)


def _get_actor_value(actor_info, *keys):
    for key in keys:
        value = actor_info.get(key)
        if value is not None:
            return value
    return None


def _get_nvml_process_getter(api_candidates):
    """Pick the first available NVML process API from candidates.

    Why this is needed:
    Different driver/NVML bindings expose different function variants
    (v3/v2/legacy), so we need graceful runtime fallback instead of
    hardcoding a single symbol.
    """
    for api_name in api_candidates:
        getter = getattr(pynvml, api_name, None)
        if getter is not None:
            return getter
    return None


def _collect_local_gpu_memory_by_pid_gb() -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """Collect local GPU memory by pid, including per-device breakdown.

    Why this is needed:
    - We need per-process attribution (not only per-GPU totals) to map memory
      back to rollout/training actors.
    - We also need per-device numbers to report 8 GPUs separately.
    """
    pid_to_gpu_memory_gb: dict[int, float] = defaultdict(float)
    pid_to_gpu_memory_by_device_gb: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            process_getters = [
                _get_nvml_process_getter(
                    (
                        "nvmlDeviceGetComputeRunningProcesses_v3",
                        "nvmlDeviceGetComputeRunningProcesses_v2",
                        "nvmlDeviceGetComputeRunningProcesses",
                    )
                ),
                _get_nvml_process_getter(
                    (
                        "nvmlDeviceGetGraphicsRunningProcesses_v3",
                        "nvmlDeviceGetGraphicsRunningProcesses_v2",
                        "nvmlDeviceGetGraphicsRunningProcesses",
                    )
                ),
            ]

            for getter in process_getters:
                if getter is None:
                    continue
                try:
                    running_procs = getter(handle)
                except pynvml.NVMLError:
                    continue
                for proc in running_procs:
                    pid = getattr(proc, "pid", None)
                    used_gpu_memory = getattr(proc, "usedGpuMemory", None)
                    if pid is None or used_gpu_memory is None or used_gpu_memory < 0:
                        continue
                    pid_int = int(pid)
                    used_gpu_memory_gb = used_gpu_memory / 1024 / 1024 / 1024
                    pid_to_gpu_memory_gb[pid_int] += used_gpu_memory_gb
                    pid_to_gpu_memory_by_device_gb[pid_int][i] += used_gpu_memory_gb
        return dict(pid_to_gpu_memory_gb), {pid: dict(mem) for pid, mem in pid_to_gpu_memory_by_device_gb.items()}
    finally:
        pynvml.nvmlShutdown()


def _get_proc_ngid(pid: int) -> int | None:
    """Get namespace-global pid (Ngid) for cross-namespace pid matching.

    Why this is needed:
    In containerized setups, Ray actor pid and NVML pid may differ because
    they are reported from different pid namespaces. Ngid lets us bridge them.
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("Ngid:"):
                    continue
                ngid_str = line.split(":", 1)[1].strip()
                if not ngid_str:
                    return None
                ngid = int(ngid_str)
                return ngid if ngid > 0 else None
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def _format_memory_info_for_console(memory_info: dict) -> str:
    def _round_recursive(value):
        if isinstance(value, float):
            return round(value, 1)
        if isinstance(value, list):
            return [_round_recursive(item) for item in value]
        if isinstance(value, dict):
            return {k: _round_recursive(v) for k, v in value.items()}
        return value

    formatted = _round_recursive(memory_info)
    lines = [
        f"[RL_MEM] time={formatted.get('time', '')}",
        f"[RL_MEM] node_id={formatted.get('node_id', '')}",
    ]
    for actor_name, actor_info in formatted.items():
        if actor_name in {"time", "node_id"}:
            continue
        if not isinstance(actor_info, dict):
            continue
        lines.append(
            f"[RL_MEM] {actor_name}: "
            f"cpu_gb={actor_info.get('mem_gb', [])}, "
            f"gpu_gb={actor_info.get('gpu_mem_gb', [])}, "
            f"pid={actor_info.get('pid', [])}"
        )
    return "\n".join(lines)


def _round_gb(value: float) -> float:
    return round(value, 1)


def monitor_actor_memory(work_dir: str, interval: int = 60):
    if pynvml is None:
        raise ImportError("pynvml 未安装，无法监控 GPU 内存")

    current_node_id = _get_current_node_id()
    print(f"开始监控当前节点 Actor 内存使用情况，间隔 {interval} 秒...")
    print(f"当前节点 ID: {current_node_id}")
    print("说明: 该监控只统计运行在当前节点上的 Ray actor，当前节点通常是 head 节点，但不保证。")
    print("=" * 80)
    os.makedirs(f"{work_dir}/tb", exist_ok=True)
    f = open(f"{work_dir}/actor_memory.json", "w")

    pynvml.nvmlInit()
    try:
        local_gpus = int(pynvml.nvmlDeviceGetCount())
    finally:
        pynvml.nvmlShutdown()

    print(f"当前节点 GPU 数量: {local_gpus}")
    tb_writer_list = [TensorboardWriter(log_dir=f"{work_dir}/tb/{rank}") for rank in range(max(local_gpus, 1))]

    count = 0
    try:
        while True:
            count += 1
            memory_info = {}
            pid_to_gpu_memory_gb, pid_to_gpu_memory_by_device_gb = _collect_local_gpu_memory_by_pid_gb()
            unmatched_actor_pid_samples: list[tuple[int, int | None]] = []

            # Only collect actors on this node. Ray actor metadata is cluster-wide,
            # while psutil and NVML can only query local processes and GPUs.
            actors = list_actors(detail=True, filters=[("state", "=", "ALIVE")], limit=10000)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            memory_info["time"] = current_time
            memory_info["node_id"] = current_node_id

            for actor_info in actors:
                actor_node_id = _get_actor_value(actor_info, "node_id", "NodeID", "NodeId")
                if str(actor_node_id) != current_node_id:
                    continue

                actor_name = _get_actor_value(actor_info, "class_name", "ActorClassName") or "Unnamed"
                pid = _get_actor_value(actor_info, "pid", "Pid")
                memory_gb = 0
                gpu_memory_gb = 0
                gpu_memory_by_device = [0.0 for _ in range(local_gpus)]

                if pid:
                    try:
                        pid = int(pid)
                        process = psutil.Process(pid)
                        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
                        actor_pid_set = {pid}
                        actor_ngid = _get_proc_ngid(pid)
                        if actor_ngid is not None:
                            actor_pid_set.add(actor_ngid)
                        for child in process.children(recursive=True):
                            actor_pid_set.add(child.pid)
                            child_ngid = _get_proc_ngid(child.pid)
                            if child_ngid is not None:
                                actor_pid_set.add(child_ngid)
                        for actor_pid in actor_pid_set:
                            device_memory = pid_to_gpu_memory_by_device_gb.get(actor_pid, {})
                            for device_idx, device_mem_gb in device_memory.items():
                                if 0 <= device_idx < local_gpus:
                                    gpu_memory_by_device[device_idx] += device_mem_gb
                        gpu_memory_gb = sum(gpu_memory_by_device)  # type: ignore
                        if gpu_memory_gb <= 0 and len(unmatched_actor_pid_samples) < 10:
                            unmatched_actor_pid_samples.append((pid, actor_ngid))

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                memory_gb = _round_gb(memory_gb)  # type: ignore
                gpu_memory_gb = _round_gb(gpu_memory_gb)  # type: ignore

                if actor_name in memory_info:
                    memory_info[actor_name]["mem_gb"].append(memory_gb)  # type: ignore
                    memory_info[actor_name]["pid"].append(str(pid))  # type: ignore
                    memory_info[actor_name]["gpu_mem_gb"].append(gpu_memory_gb)  # type: ignore
                else:
                    memory_info[actor_name] = {  # type: ignore
                        "mem_gb": [memory_gb],
                        "pid": [str(pid)],
                        "gpu_mem_gb": [gpu_memory_gb],
                    }

            # 写入文件
            json.dump(memory_info, f, ensure_ascii=False)
            f.write("\n")
            f.flush()

            if pid_to_gpu_memory_gb and all(
                actor_name in ["time", "node_id"] or max(memory_mb_info.get("gpu_mem_gb", [0.0])) <= 0.0  # type: ignore
                for actor_name, memory_mb_info in memory_info.items()
            ):
                sample_gpu_pids = list(pid_to_gpu_memory_gb.items())[:10]
                print(
                    "[RL_MEM_MONITOR][WARN] NVML sees GPU memory but no actor pid matched. "
                    f"sample_gpu_pid_mem={sample_gpu_pids}, sample_actor_pid_ngid={unmatched_actor_pid_samples}"
                )

            for actor_name, memory_mb_info in memory_info.items():
                if actor_name in ["time", "node_id"]:
                    continue
                memory_mb: list[float] = memory_mb_info["mem_gb"]  # type: ignore
                gpu_memory_mb: list[float] = memory_mb_info["gpu_mem_gb"]  # type: ignore

                if len(memory_mb) == 1:
                    tb_writer_list[0].add_scalar(
                        tag=f"{actor_name}/cpu_gb",
                        scalar_value=_round_gb(memory_mb[-1]),
                        global_step=count,
                    )
                    tb_writer_list[0].add_scalar(
                        tag=f"{actor_name}/gpu_gb",
                        scalar_value=_round_gb(gpu_memory_mb[-1]),
                        global_step=count,
                    )
                else:
                    for i in range(len(memory_mb)):
                        tb_writer_list[i % len(tb_writer_list)].add_scalar(
                            tag=f"{actor_name}/cpu_gb",
                            scalar_value=_round_gb(memory_mb[i]),
                            global_step=count,
                        )

                if len(gpu_memory_mb) == 1:
                    tb_writer_list[0].add_scalar(
                        tag=f"{actor_name}/gpu_gb",
                        scalar_value=_round_gb(gpu_memory_mb[-1]),
                        global_step=count,
                    )
                else:
                    for i in range(len(gpu_memory_mb)):
                        tb_writer_list[i % len(tb_writer_list)].add_scalar(
                            tag=f"{actor_name}/gpu_gb",
                            scalar_value=_round_gb(gpu_memory_mb[i]),
                            global_step=count,
                        )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n监控已停止")
    finally:
        f.close()
        for tb_writer in tb_writer_list:
            tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL MEMORY MONITOR")
    parser.add_argument("--work_dir", type=str, default="dense_8b")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    work_dir = args.work_dir
    interval = args.interval

    while True:
        try:
            ray.init(address="auto")
            time.sleep(interval)
            break
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception:
            print("连接 Ray 集群失败, 等等")

    monitor_actor_memory(work_dir=work_dir, interval=interval)
