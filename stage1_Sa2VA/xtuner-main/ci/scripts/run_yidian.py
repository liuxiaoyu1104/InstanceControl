#!/usr/bin/env python3
"""
Yidian cluster job submission script.
Based on the successful test_clusterx_sft.py setup, specialized for the Yidian cluster.
Keeps the same interface as run.py while including Yidian-specific settings.
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
    """Parse command-line arguments and keep them consistent with run.py."""
    parser = argparse.ArgumentParser(description="Yidian Cluster Submission")
    parser.add_argument("image", type=str)
    parser.add_argument("env", type=str) 
    parser.add_argument("cmd", type=str)
    parser.add_argument("--nodes", type=int, default=1)
    return parser.parse_args()

def main():
    """Main function."""
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
    
    # Get GitLab CI environment variables.
    commit_id = os.environ.get('CI_COMMIT_SHORT_SHA', 'test')
    commit_branch = os.environ.get('CI_COMMIT_REF_NAME', 'test')
    job_id = os.environ.get('CI_JOB_ID', '0')
    
    # Clean invalid characters in the branch name to avoid Kubernetes label errors.
    commit_branch = commit_branch.replace('/', '-').replace('\\', '-').replace('_', '-')
    
    # Yidian-specific resource settings based on the successful test_clusterx_sft.py setup.
    params = params_cls(
        job_name=f"xtuner-ci-{job_id}-{commit_branch}-{commit_id}",
        image=args.image,
        cmd=cmd,
        gpus_per_task=16,       # Yidian uses 16 GPUs per node.
        cpus_per_task=512,      # More CPU resources based on test_clusterx_sft.py.
        memory_per_task="1800", # Larger memory based on the 1800 GB setting in test_clusterx_sft.py.
        num_nodes=args.nodes,
        no_env=True             # Do not inherit environment variables; use a custom environment.
    )
    
    print(f"创建集群任务: {params.job_name}")
    print(f"节点数: {params.num_nodes}")
    print(f"每节点GPU数: {params.gpus_per_task}")
    print(f"每节点CPU数: {params.cpus_per_task}")
    print(f"每节点内存: {params.memory_per_task}G")
    
    cluster = cluster_cls()
    job_schema = cluster.run(params)
    print(f"任务已提交: {job_schema.job_id}")
    
    # Monitor job status with logic consistent with run.py.
    while True:
        time.sleep(10)  # Increase the monitoring interval to reduce API call frequency.

        job_info = cluster.get_job_info(job_schema.job_id)
        status = job_info.status

        if status == JobStatus.QUEUING:
            print(f"Job {job_schema.job_id} is queuing...")
        elif status == JobStatus.RUNNING:
            print(f"Job {job_schema.job_id} is running...")
        elif status == JobStatus.FAILED:
            print(f"Job {job_schema.job_id} failed!")
            # Wait 10 seconds to ensure logs are fully collected.
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