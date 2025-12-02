import os
import ray
import time
import psutil
import json
from xtuner.v1._writer import TensorboardWriter
import pynvml
import argparse


def monitor_actor_memory(work_dir, interval: int = 60):
    print(f"开始监控 Actor 内存使用情况，间隔 {interval} 秒...")
    print("=" * 80)
    os.makedirs(f'{work_dir}/tb', exist_ok=True)
    f = open(f'{work_dir}/actor_memory.json', 'w')

    cluster_resources = ray.cluster_resources()
    total_gpus = int(cluster_resources.get("GPU", 0))

    print(f"集群总GPU数量: {total_gpus}")
    tb_writer_list = [TensorboardWriter(log_dir=f'{work_dir}/tb/{rank}') for rank in range(total_gpus)]

    count = 0
    try:
        while True:
            count += 1
            memory_info = {}

            # 获取所有 Actor
            actors = ray.state.actors()

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            memory_info['time'] = current_time
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
                        pynvml.nvmlInit()
                        device_count = pynvml.nvmlDeviceGetCount()
                        for i in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            # 检查该GPU是否被当前进程使用
                            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            if any(proc.pid == pid for proc in compute_procs):
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_memory_gb = mem_info.used / 1024 / 1024 / 1024
                                break
                        pynvml.nvmlShutdown()

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if actor_name in memory_info:
                    memory_info[actor_name]['mem_gb'].append(memory_gb)
                    memory_info[actor_name]['pid'].append(str(pid)[:6])
                    memory_info[actor_name]['gpu_mem_gb'].append(gpu_memory_gb)
                else:
                    memory_info[actor_name] = {'mem_gb': [memory_gb], 'pid': [str(pid)[:6]],
                                               'gpu_mem_gb': [gpu_memory_gb]}

            # 写入文件
            json.dump(memory_info, f, ensure_ascii=False)
            f.write("\n")
            f.flush()

            for actor_name, memory_mb_info in memory_info.items():
                if actor_name == 'time':
                    continue
                memory_mb = memory_mb_info['mem_gb']
                gpu_memory_mb = memory_mb_info['gpu_mem_gb']

                if len(memory_mb) == 1:
                    tb_writer_list[0].add_scalar(
                        tag=f'{actor_name}/cpu_gb',
                        scalar_value=memory_mb[-1],
                        global_step=count,
                    )
                    tb_writer_list[0].add_scalar(
                        tag=f'{actor_name}/gpu_gb',
                        scalar_value=gpu_memory_mb[-1],
                        global_step=count,
                    )
                else:
                    assert total_gpus % len(memory_mb) == 0, f'{total_gpus}, {len(memory_mb)}'
                    multi_factor = total_gpus // len(memory_mb)
                    for i in range(len(memory_mb)):
                        tb_writer_list[i * multi_factor].add_scalar(
                            tag=f'{actor_name}/cpu_gb',
                            scalar_value=memory_mb[i],
                            global_step=count,
                        )

                if len(gpu_memory_mb) == 1:
                    tb_writer_list[0].add_scalar(
                        tag=f'{actor_name}/gpu_gb',
                        scalar_value=gpu_memory_mb[-1],
                        global_step=count,
                    )
                else:
                    assert total_gpus % len(gpu_memory_mb) == 0, f'{total_gpus}, {len(gpu_memory_mb)}'
                    multi_factor = total_gpus // len(gpu_memory_mb)
                    for i in range(len(gpu_memory_mb)):
                        tb_writer_list[i * multi_factor].add_scalar(
                            tag=f'{actor_name}/gpu_gb',
                            scalar_value=gpu_memory_mb[i],
                            global_step=count,
                        )

            time.sleep(interval)
            print(memory_info)

    except KeyboardInterrupt:
        print("\n监控已停止")
    finally:
        f.close()
        for tb_writer in tb_writer_list:
            tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL MEMORY MONITOR')
    parser.add_argument('--work_dir', type=str, default='dense_8b')
    parser.add_argument('--interval', type=int, default=60)
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
        except Exception as e:
            print(f"连接 Ray 集群失败, 等等")

    monitor_actor_memory(work_dir=work_dir, interval=interval)
