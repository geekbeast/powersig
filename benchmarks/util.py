import csv
import os
import time
from enum import Enum

from contextlib import contextmanager
from cupy.cuda.memory import MemoryPool
import psutil
import torch
import cupy as cp
import jax

from benchmarks.configuration import CPU_MEMORY, CUPY_MEMORY, DURATION, GPU_MEMORY

class Backend(Enum):
    CPU = "cpu"
    TORCH = "torch"
    JAX = "jax"
    CUPY = "cupy_backend"
    TORCH_CUDA = "torch_cuda"
    JAX_CUDA = "jax_cuda"


def save_stats(stats, filename):
    # Determine if file exists to decide if we need to write headers
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)


class TrackingMemoryPool(MemoryPool):
    def __init__(self, allocator=None):
        super().__init__(allocator)
        self.peak_usage = 0

    def reset_peak_usage(self):
        # Reset the peak usage to the current allocated memory (assumes memory is freed).
        self.peak_usage = self.used_bytes()

    def malloc(self, size):
        # Perform the allocation via parent class
        memptr = super().malloc(size)

        # Now check how much memory this pool is actually using in total
        self.peak_usage = max(self.peak_usage, self.used_bytes())

        return memptr

# Global tracking pool - only initialized once when module is first imported
tracking_pool = TrackingMemoryPool()
cp.cuda.set_allocator(tracking_pool.malloc)

@contextmanager
def track_peak_memory(backend: Backend, stats, device=None):
    cupy_initial_mem = 0

    process = psutil.Process(os.getpid())
    # Get CPU initial memory in byte
    cpu_initial_mem = process.memory_info().rss

    if cp.is_available():
        tracking_pool.reset_peak_usage()
        # Get cupy_backend initial memory in bytes
        cupy_initial_mem = tracking_pool.peak_usage

    # Context manager to track peak GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    jax_cuda_available = any(device.platform == 'gpu' for device in jax.devices())

    if jax_cuda_available:
        start_jax_memory = get_jax_memory(device)
    else:
        start_jax_memory = 0
    start = time.time()

    try:
        yield
    finally:
        stats[DURATION] = time.time() - start

        print(f"{backend} computation took: {stats[DURATION]}s")

        peak_cpu_mem = (process.memory_info().rss - cpu_initial_mem) / (1024 * 1024)  # MB
        stats[CPU_MEMORY] = peak_cpu_mem
        print(f"Peak CPU memory usage: {peak_cpu_mem:.1f} MB")
        
        if  cp.is_available() and backend == Backend.CUPY:
            peak_cupy_mem = tracking_pool.peak_usage / (1024 * 1024)  # MB
            stats[GPU_MEMORY] = peak_cupy_mem
            # Shim for existing code that expects CUPY_MEMORY
            stats[CUPY_MEMORY] = peak_cupy_mem
            print(f"Peak cupy_backend memory usage: {peak_cupy_mem:.1f} MB")

        if torch.cuda.is_available() and backend == Backend.TORCH_CUDA:
            torch.cuda.synchronize()
            peak_pytorch_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            stats[GPU_MEMORY] = peak_pytorch_memory
            print(f"Peak pytorch memory usage: {peak_pytorch_memory:.2f} MB")
        
        # TODO: This is a hack to get the peak memory usage for Jax, the problem is we have to reset device memory before each run of the kernel
        # and that means we have to rewarm the compilation cache to be fair.
        if jax_cuda_available and backend == Backend.JAX_CUDA:
            # Set new peak, we have to subtract the initial memory to get the delta. 
            # If we didn't set new peak, then we don't know what the delta is, so we may be overcounting (less likely to be an issue with these benchmarks)
            peak_jax_memory = get_peak_jax_memory(device) - start_jax_memory
            stats[GPU_MEMORY] = peak_jax_memory
            print(f"Peak jax memory usage: {peak_jax_memory:.2f} MB")


def get_peak_jax_memory(device = None):
    peak_jax_memory = 0
    for device in jax.devices():
        if device.platform == 'gpu' and (device!=None and device.id == device.id):
            peak_jax_memory += device.memory_stats()['peak_bytes_in_use'] / (1024 ** 2)  # Convert to MB

    return peak_jax_memory

def get_jax_memory(device = None):
    jax_memory = 0
    for device in jax.devices():
        if device.platform == 'gpu' and (device!=None and device.id == device.id):
            jax_memory += device.memory_stats()['bytes_in_use'] / (1024 ** 2)  # Convert to MB
    return jax_memory