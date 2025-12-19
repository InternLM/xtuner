import json
import re
import sys
import os
import csv
import glob
import multiprocessing as mp
import time
import argparse


def find_min_ts_end(json_file, pattern_list, cat_str, max_count):

    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {e}")
        return None
    
    # 提取所有匹配的事件
    events = []
    for event in data.get("traceEvents", []):
        if event.get("cat") != cat_str:
            continue
        name = event.get("name", "")
        match = False
        for pattern_str in pattern_list:
            match = match or name == pattern_str
        # match = name == pattern_str
        if match:
            events.append(event)
    
    if len(events) == 0:
        raise RuntimeError()
    
    events.sort(key=lambda x: x['ts'])
    return events


def worker_process(args):
    file_batch, pattern_list, cat_str, worker_id, total_workers, total_files, max_count = args
    results = []
    processed_count = 0
    
    for json_file in file_batch:
        result = find_min_ts_end(json_file, pattern_list, cat_str, max_count)
        if result:
            results.append(result)
        
        processed_count += 1
        # 每处理完一定数量的文件，打印进度
        if processed_count % max(1, len(file_batch) // 10) == 0:
            print(f"进程 {worker_id}: 已处理 {processed_count}/{len(file_batch)} 个文件")
    
    return results, processed_count


def extract_rank_number(file_path):
    import re
    # 使用正则表达式提取 "rankXXX" 中的数字部分
    match = re.search(r"rank(\d+)", file_path)
    if match:
        return int(match.group(1))  # 提取数字并转为整数
    return float('inf')  # 如果没有匹配到 rank，返回一个较大的值以确保排在最后


def batch_find_min_ts_end_mp(directory, output_json, pattern_list, cat_str, num_processes=16, max_count=62):
    # 获取目录中的所有JSON文件
    json_files = glob.glob(os.path.join(directory, "*.json"))
    json_files = sorted(json_files, key=lambda x: extract_rank_number(os.path.basename(x)))
    
    if not json_files:
        print(f"在目录 {directory} 中未找到JSON文件")
        return
    
    total_files = len(json_files)
    print(f"找到 {total_files} 个JSON文件")
    
    # 将文件分成多个批次
    file_batches = []
    batch_size = (total_files + num_processes - 1) // num_processes
    for i in range(0, total_files, batch_size):
        file_batches.append(json_files[i:i+batch_size])
    
    # 准备工作进程的参数
    worker_args = []
    for i, batch in enumerate(file_batches):
        worker_args.append((batch, pattern_list, cat_str, i, len(file_batches), total_files, max_count))
    
    # 创建进程池
    pool = mp.Pool(processes=min(num_processes, len(file_batches)))
    
    # 并行处理文件
    print(f"使用 {min(num_processes, len(file_batches))} 个进程处理文件...")
    start_time = time.time()
    
    # 使用进程池处理文件
    results = pool.map(worker_process, worker_args)
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    end_time = time.time()
    
    # 计算总处理文件数
    total_processed = sum(count for _, count in results)
    print(f"处理完成，共处理 {total_processed}/{total_files} 个文件，耗时: {end_time - start_time:.2f} 秒")
    
    
    profile_results = {'traceEvents': []}
    rank = 0
    for worker_id in range(len(results)):
        for job_id in range(len(results[worker_id][0])):
            events = results[worker_id][0][job_id]
            for event in events:
                event['pid'] = 0
                event['tid'] = rank
            profile_results['traceEvents'].extend(events)
            rank += 1
    # breakpoint()
    
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(profile_results, file, indent=2, ensure_ascii=False)


# pattern_list = [
#     "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",  # nccl ag
#     "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",  # nccl rs
#     "void deep_ep::intranode::notify_dispatch<8>(int const*, int*, int const*, int*, int, int, int, bool const*, int*, int*, int, int, void**, int**, int)",  # notify_dispatch
#     "void deep_ep::intranode::dispatch<8, 768, 8192>(int4*, float*, int*, long*, float*, int*, int*, int4 const*, float const*, long const*, float const*, bool const*, int const*, int, int, int, int, int, int, int, int, void**, int, int, int)",  # dispatch
# ]


def parse_args():
    parser = argparse.ArgumentParser(description='从 JSON profiling 文件中提取特定的 kernel 事件')
    
    parser.add_argument(
        '--json_directory',
        type=str,
        required=True,
        help='包含 JSON profiling 文件的目录路径'
    )
    
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='输出 JSON 文件路径（默认为 json_directory/nccl.json）'
    )
    
    parser.add_argument(
        '--pattern_list',
        type=str,
        nargs='+',
        default=["ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"],
        help='要匹配的 kernel 名称列表'
    )
    
    parser.add_argument(
        '--num_processes',
        type=int,
        default=8,
        help='并行处理的进程数（默认: 8）'
    )
    
    parser.add_argument(
        '--max_count',
        type=int,
        default=500,
        help='最大算子数量（默认: 500）'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    # 如果没有指定输出文件，使用默认路径
    output_json = args.output_json
    if output_json is None:
        output_json = os.path.join(args.json_directory, "nccl.json")

    json_directory = args.json_directory
    
    pattern_list = args.pattern_list
    
    cat_str = "kernel"
    num_processes = args.num_processes  # 默认进程数
    max_count = args.max_count # allgather 最大算子数量
    batch_find_min_ts_end_mp(json_directory, output_json, pattern_list, cat_str, num_processes, max_count)
