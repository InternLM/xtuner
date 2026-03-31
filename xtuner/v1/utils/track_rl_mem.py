import argparse
import dataclasses
import json
import os
import time
from collections import defaultdict
from typing import Any

import psutil
import ray

from xtuner.v1._writer import TensorboardWriter


try:
    import pynvml
except ImportError:
    pynvml = None


def _maybe_init_nvml():
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _maybe_shutdown_nvml(initialized: bool):
    if initialized:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _state_obj_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _sanitize_tag_component(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")


def _get_object_store_stats(object_limit: int = 5000, top_k: int = 10):
    stats: dict[str, Any] = {
        "available": False,
        "total_objects": 0,
        "total_size_mb": 0.0,
        "callsite_enabled": 0,
        "summary_by": "",
        "ref_type_counts": {},
        "task_state_counts": {},
        "top_callsites": [],
        "detail_truncated": 0,
        "detail_object_count": 0,
        "detail_total_size_mb": 0.0,
        "detail_ref_type_counts": {},
        "detail_ref_type_size_mb": {},
        "top_call_sites_from_objects": [],
        "top_pids": [],
        "top_ips": [],
    }

    try:
        from ray.util import state as ray_state

        summary_raw = ray_state.summarize_objects(timeout=30, raise_on_missing_output=False)
        summary_data = _state_obj_to_dict(summary_raw)
        stats["available"] = True
        stats["total_objects"] = summary_data.get("total_objects", 0)
        stats["total_size_mb"] = float(summary_data.get("total_size_mb", 0.0) or 0.0)
        stats["callsite_enabled"] = int(bool(summary_data.get("callsite_enabled", False)))
        stats["summary_by"] = summary_data.get("summary_by", "")

        ref_type_counts = defaultdict(int)
        task_state_counts = defaultdict(int)
        callsite_items = []
        for callsite, item in (summary_data.get("summary", {}) or {}).items():
            item_dict = _state_obj_to_dict(item)
            callsite_items.append(
                {
                    "callsite": callsite,
                    "total_size_mb": float(item_dict.get("total_size_mb", 0.0) or 0.0),
                    "total_objects": int(item_dict.get("total_objects", 0) or 0),
                    "total_num_workers": int(item_dict.get("total_num_workers", 0) or 0),
                    "total_num_nodes": int(item_dict.get("total_num_nodes", 0) or 0),
                    "ref_type_counts": item_dict.get("ref_type_counts", {}) or {},
                    "task_state_counts": item_dict.get("task_state_counts", {}) or {},
                }
            )
            for ref_type, count in (item_dict.get("ref_type_counts", {}) or {}).items():
                ref_type_counts[str(ref_type)] += int(count)
            for task_state, count in (item_dict.get("task_state_counts", {}) or {}).items():
                task_state_counts[str(task_state)] += int(count)

        callsite_items.sort(key=lambda x: (x["total_size_mb"], x["total_objects"]), reverse=True)
        stats["top_callsites"] = callsite_items[:top_k]
        stats["ref_type_counts"] = dict(ref_type_counts)
        stats["task_state_counts"] = dict(task_state_counts)

        try:
            object_states = ray_state.list_objects(
                limit=object_limit, timeout=30, detail=False, raise_on_missing_output=False
            )
            pid_size_mb = defaultdict(float)
            pid_count = defaultdict(int)
            ip_size_mb = defaultdict(float)
            ip_count = defaultdict(int)
            ref_type_size_mb = defaultdict(float)
            ref_type_count = defaultdict(int)
            callsite_size_mb = defaultdict(float)
            callsite_count = defaultdict(int)

            object_state_dicts = [_state_obj_to_dict(obj) for obj in object_states]
            stats["detail_object_count"] = len(object_state_dicts)
            stats["detail_truncated"] = int(len(object_state_dicts) >= object_limit)

            for obj in object_state_dicts:
                size_mb = float(obj.get("object_size", 0) or 0) / (1024**2)
                pid = str(obj.get("pid", "unknown"))
                ip = str(obj.get("ip", "unknown"))
                ref_type = str(obj.get("reference_type", "UNKNOWN"))
                call_site = str(obj.get("call_site", "unknown"))

                stats["detail_total_size_mb"] += size_mb
                pid_size_mb[pid] += size_mb
                pid_count[pid] += 1
                ip_size_mb[ip] += size_mb
                ip_count[ip] += 1
                ref_type_size_mb[ref_type] += size_mb
                ref_type_count[ref_type] += 1
                callsite_size_mb[call_site] += size_mb
                callsite_count[call_site] += 1

            stats["detail_ref_type_counts"] = dict(ref_type_count)
            stats["detail_ref_type_size_mb"] = dict(ref_type_size_mb)
            stats["top_call_sites_from_objects"] = [
                {"callsite": k, "size_mb": v, "count": callsite_count[k]}
                for k, v in sorted(callsite_size_mb.items(), key=lambda item: item[1], reverse=True)[:top_k]
            ]
            stats["top_pids"] = [
                {"pid": k, "size_mb": v, "count": pid_count[k]}
                for k, v in sorted(pid_size_mb.items(), key=lambda item: item[1], reverse=True)[:top_k]
            ]
            stats["top_ips"] = [
                {"ip": k, "size_mb": v, "count": ip_count[k]}
                for k, v in sorted(ip_size_mb.items(), key=lambda item: item[1], reverse=True)[:top_k]
            ]
        except Exception as e:
            stats["detail_error"] = str(e)

    except Exception as e:
        stats["error"] = str(e)

    return stats


def monitor_actor_memory(work_dir: str, interval: int = 60, object_limit: int = 5000, top_k: int = 10):

    print(f"开始监控 Actor 内存使用情况，间隔 {interval} 秒...")
    print("=" * 80)
    os.makedirs(f"{work_dir}/tb", exist_ok=True)
    actor_f = open(f"{work_dir}/actor_memory.jsonl", "w", encoding="utf-8")
    object_f = open(f"{work_dir}/object_store.jsonl", "w", encoding="utf-8")

    cluster_resources = ray.cluster_resources()
    total_gpus = int(cluster_resources.get("GPU", 0))

    print(f"集群总GPU数量: {total_gpus}")
    tb_writer_list = [TensorboardWriter(log_dir=f"{work_dir}/tb/{rank}") for rank in range(total_gpus)]

    count = 0
    try:
        while True:
            count += 1
            memory_info = {}
            object_store_info = {}

            # 获取所有 Actor
            actors = ray.state.actors()

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            memory_info["time"] = current_time
            print(f"\n时间: {current_time}")
            print("-" * 80)

            for actor_id, actor_info in actors.items():
                actor_name = actor_info.get("ActorClassName", "Unnamed")
                pid = actor_info.get("Pid")
                memory_gb = 0
                gpu_memory_gb = 0

                if pid:
                    try:
                        process = psutil.Process(pid)
                        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
                        nvml_initialized = _maybe_init_nvml()
                        if nvml_initialized:
                            device_count = pynvml.nvmlDeviceGetCount()
                            for i in range(device_count):
                                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                                # 检查该GPU是否被当前进程使用
                                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                                if any(proc.pid == pid for proc in compute_procs):
                                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                    gpu_memory_gb = mem_info.used / 1024 / 1024 / 1024
                                    break
                        _maybe_shutdown_nvml(nvml_initialized)

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if actor_name in memory_info:
                    memory_info[actor_name]["mem_gb"].append(memory_gb)  # type: ignore
                    memory_info[actor_name]["pid"].append(str(pid)[:6])  # type: ignore
                    memory_info[actor_name]["gpu_mem_gb"].append(gpu_memory_gb)  # type: ignore
                else:
                    memory_info[actor_name] = {  # type: ignore
                        "mem_gb": [memory_gb],
                        "pid": [str(pid)[:6]],
                        "gpu_mem_gb": [gpu_memory_gb],
                    }

            object_store_info = _get_object_store_stats(object_limit=object_limit, top_k=top_k)
            object_store_info["time"] = current_time
            object_store_info["object_limit"] = object_limit
            object_store_info["top_k"] = top_k

            # 写入文件
            json.dump(memory_info, actor_f, ensure_ascii=False)
            actor_f.write("\n")
            actor_f.flush()
            json.dump(object_store_info, object_f, ensure_ascii=False)
            object_f.write("\n")
            object_f.flush()

            for actor_name, memory_mb_info in memory_info.items():
                if actor_name == "time":
                    continue
                memory_mb: list[float] = memory_mb_info["mem_gb"]  # type: ignore
                gpu_memory_mb: list[float] = memory_mb_info["gpu_mem_gb"]  # type: ignore

                if len(memory_mb) == 1:
                    tb_writer_list[0].add_scalar(
                        tag=f"{actor_name}/cpu_gb",
                        scalar_value=memory_mb[-1],
                        global_step=count,
                    )
                    tb_writer_list[0].add_scalar(
                        tag=f"{actor_name}/gpu_gb",
                        scalar_value=gpu_memory_mb[-1],
                        global_step=count,
                    )
                else:
                    assert total_gpus % len(memory_mb) == 0, f"{total_gpus}, {len(memory_mb)}"
                    multi_factor = total_gpus // len(memory_mb)
                    for i in range(len(memory_mb)):
                        tb_writer_list[i * multi_factor].add_scalar(
                            tag=f"{actor_name}/cpu_gb",
                            scalar_value=memory_mb[i],
                            global_step=count,
                        )

                if len(gpu_memory_mb) == 1:
                    tb_writer_list[0].add_scalar(
                        tag=f"{actor_name}/gpu_gb",
                        scalar_value=gpu_memory_mb[-1],
                        global_step=count,
                    )
                else:
                    assert total_gpus % len(gpu_memory_mb) == 0, f"{total_gpus}, {len(gpu_memory_mb)}"
                    multi_factor = total_gpus // len(gpu_memory_mb)
                    for i in range(len(gpu_memory_mb)):
                        tb_writer_list[i * multi_factor].add_scalar(
                            tag=f"{actor_name}/gpu_gb",
                            scalar_value=gpu_memory_mb[i],
                            global_step=count,
                        )

            tb_writer_list[0].add_scalar(
                tag="ray_object_store/total_size_mb",
                scalar_value=float(object_store_info.get("total_size_mb", 0.0) or 0.0),
                global_step=count,
            )
            tb_writer_list[0].add_scalar(
                tag="ray_object_store/total_objects",
                scalar_value=float(object_store_info.get("total_objects", 0) or 0),
                global_step=count,
            )
            tb_writer_list[0].add_scalar(
                tag="ray_object_store/detail_total_size_mb",
                scalar_value=float(object_store_info.get("detail_total_size_mb", 0.0) or 0.0),
                global_step=count,
            )
            tb_writer_list[0].add_scalar(
                tag="ray_object_store/detail_object_count",
                scalar_value=float(object_store_info.get("detail_object_count", 0) or 0),
                global_step=count,
            )
            tb_writer_list[0].add_scalar(
                tag="ray_object_store/detail_truncated",
                scalar_value=float(object_store_info.get("detail_truncated", 0) or 0),
                global_step=count,
            )

            for ref_type, value in (object_store_info.get("ref_type_counts", {}) or {}).items():
                tb_writer_list[0].add_scalar(
                    tag=f"ray_object_store/ref_type_count/{_sanitize_tag_component(str(ref_type))}",
                    scalar_value=float(value),
                    global_step=count,
                )
            for ref_type, value in (object_store_info.get("detail_ref_type_size_mb", {}) or {}).items():
                tb_writer_list[0].add_scalar(
                    tag=f"ray_object_store/ref_type_size_mb/{_sanitize_tag_component(str(ref_type))}",
                    scalar_value=float(value),
                    global_step=count,
                )
            for task_state, value in (object_store_info.get("task_state_counts", {}) or {}).items():
                tb_writer_list[0].add_scalar(
                    tag=f"ray_object_store/task_state_count/{_sanitize_tag_component(str(task_state))}",
                    scalar_value=float(value),
                    global_step=count,
                )

            time.sleep(interval)
            print(memory_info)
            print(object_store_info)

    except KeyboardInterrupt:
        print("\n监控已停止")
    finally:
        actor_f.close()
        object_f.close()
        for tb_writer in tb_writer_list:
            tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL MEMORY MONITOR")
    parser.add_argument("--work_dir", type=str, default="dense_8b")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--object_limit", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    work_dir = args.work_dir
    interval = args.interval
    object_limit = args.object_limit
    top_k = args.top_k

    while True:
        try:
            if not ray.is_initialized():
                ray.init(address="auto")
                time.sleep(interval)
            break
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception:
            print("连接 Ray 集群失败, 等等")

    monitor_actor_memory(work_dir=work_dir, interval=interval, object_limit=object_limit, top_k=top_k)
