import re
import time
import psutil
import subprocess

import torch


def get_system_info() -> dict:
    """
    Get CPU, GPU infomation
    """
    cpu_percent = psutil.cpu_percent()
    total_memory = round(psutil.virtual_memory().total / (1024**3), 1)
    mem_usage = round(psutil.virtual_memory().used / (1024**3), 1)

    try:
        cmd = 'nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits'
        output = subprocess.check_output(cmd, shell=True)
        gpu_utilization, gpu_total_memory, gpu_memory_used = re.findall(r'\d+', output.decode('utf-8'))
        gpu_total_memory = int(gpu_total_memory)
        gpu_memory_used = int(gpu_memory_used)
        gpu_name = torch.cuda.get_device_name(0)
    except:
        gpu_utilization, gpu_total_memory, gpu_memory_used, gpu_name = '-', '-', '-', 'N/A'

    return {
        '\033[32mCPU': f'{cpu_percent}% | {mem_usage}/{total_memory}G\033[0m',
        f'\033[35m{gpu_name}': f'{gpu_utilization}% | {gpu_memory_used}/{gpu_total_memory}M\033[0m',
    }


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time
    return wrapper


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.size(1)
    K = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
    return K

def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, **kernel_args):
    XX = rbf_kernel(X, X, **kernel_args)
    YY = rbf_kernel(Y, Y, **kernel_args)
    XY = rbf_kernel(X, Y, **kernel_args)
    return XX.mean() + YY.mean() - 2 * XY.mean()