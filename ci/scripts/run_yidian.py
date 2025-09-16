#!/usr/bin/env python3
"""
Yidian集群任务提交脚本
基于test_clusterx_sft.py的成功经验，专门用于yidian集群
保持与run.py相同的接口，但包含yidian特定的配置
"""
import sys
import time
import os
import re
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clusterx import CLUSTER, CLUSTER_MAPPING
from clusterx.launcher.base import JobStatus


def parse_args():
    """解析命令行参数，与run.py保持一致"""
    parser = argparse.ArgumentParser(description="Yidian Cluster Submission")
    parser.add_argument("image", type=str)
    parser.add_argument("env", type=str) 
    parser.add_argument("cmd", type=str)
    parser.add_argument("--nodes", type=int, default=1)
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    cluster_spec = CLUSTER_MAPPING[CLUSTER]
    cluster_cls = cluster_spec["type"]
    params_cls = cluster_spec["params"]
    
    current_dir = os.getcwd()
    cmd = f"cd {current_dir}; {args.env}; {args.cmd}"
    print(f"Running command: {cmd}")
    
    assert params_cls is not None and cluster_cls is not None, (
        f"Cluster {CLUSTER} is not available in current ci machine!"
    )
    
    # 获取GitLab CI环境变量
    commit_id = os.environ.get('CI_COMMIT_SHORT_SHA', 'test')
    commit_branch = os.environ.get('CI_COMMIT_REF_NAME', 'test')
    job_id = os.environ.get('CI_JOB_ID', '0')
    
    # 清理分支名中的非法字符，避免Kubernetes标签错误
    commit_branch = commit_branch.replace('/', '-').replace('\\', '-').replace('_', '-')
    
    # Yidian特定的资源配置（基于test_clusterx_sft.py的成功经验）
    params = params_cls(
        job_name=f"xtuner-ci-{job_id}-{commit_branch}-{commit_id}",
        image=args.image,
        cmd=cmd,
        gpus_per_task=16,       # Yidian使用16个GPU per node
        cpus_per_task=512,      # 更多CPU资源（基于test_clusterx_sft.py）
        memory_per_task="1800", # 更大内存（基于test_clusterx_sft.py的1800G配置）
        num_nodes=args.nodes,
        no_env=True             # 不继承环境变量，使用自定义环境
    )
    
    print(f"创建集群任务: {params.job_name}")
    print(f"节点数: {params.num_nodes}")
    print(f"每节点GPU数: {params.gpus_per_task}")
    print(f"每节点CPU数: {params.cpus_per_task}")
    print(f"每节点内存: {params.memory_per_task}G")
    
    cluster = cluster_cls()
    job_schema = cluster.run(params)
    print(f"任务已提交: {job_schema.job_id}")
    
    # 监控任务状态（与run.py保持一致的逻辑）
    while True:
        time.sleep(10)  # 增加监控间隔，减少API调用频率

        job_info = cluster.get_job_info(job_schema.job_id)
        status = job_info.status

        if status == JobStatus.QUEUING:
            print(f"Job {job_schema.job_id} is queuing...")
        elif status == JobStatus.RUNNING:
            print(f"Job {job_schema.job_id} is running...")
        elif status == JobStatus.FAILED:
            print(f"Job {job_schema.job_id} failed!")
            # 等待10秒确保日志完全收集
            time.sleep(10)
            try:
                log = cluster.get_log(job_schema.job_id)
                print("=== 任务失败日志 ===")
                print(log)
            except Exception as e:
                print(f"获取日志失败: {e}")
            raise RuntimeError(f"Job {job_schema.job_id} failed with status {status}")
        elif status == JobStatus.SUCCEEDED:
            print(f"Job {job_schema.job_id} succeeded!")
            break
        else:
            print(f"Found unrecognized status {status}, waiting...")


if __name__ == "__main__":
    main()