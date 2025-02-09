import csv
import os
import time

import cupy as cp
from contextlib import contextmanager

import psutil
import torch.cuda
from cupy.cuda.memory import OutOfMemoryError

from benchmarks.configuration import signature_kernel, CPU_MEMORY, SIG_KERNEL_MAX_LENGTH, dyadic_order, \
    ksig_kernel, ORDER, SIGNATURE_KERNEL, DURATION, CSV_FIELDS, POWERSIG_MAX_LENGTH, KSIG_MAX_LENGTH, MAX_LENGTH, \
    PYTORCH_MEMORY, CUPY_MEMORY
from benchmarks.util import generate_brownian_motion, TrackingMemoryPool
from powersig.matrixsig import build_scaling_for_integration, tensor_compute_gram_entry
from powersig.util.series import torch_compute_derivative_batch
from tests.utils import setup_torch

tracking_pool = TrackingMemoryPool()

@contextmanager
def track_peak_memory(backend, stats):
    cupy_initial_mem = 0

    process = psutil.Process(os.getpid())
    # Get CPU initial memory in byte
    cpu_initial_mem = process.memory_info().rss

    if cp.is_available():
        tracking_pool.reset_peak_usage()
        # Get cupy initial memory in bytes
        cupy_initial_mem = tracking_pool.peak_usage

    # Context manager to track peak GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()

    try:
        yield
    finally:
        stats[DURATION] = time.time() - start

        print(f"{backend} computation took: {stats[DURATION]}s")

        peak_cpu_mem = (process.memory_info().rss - cpu_initial_mem) / (1024 * 1024)  # MB
        stats[CPU_MEMORY] = peak_cpu_mem
        print(f"Peak CPU memory usage: {peak_cpu_mem:.1f} MB")
        
        if cp.is_available():
            peak_cupy_mem = tracking_pool.peak_usage / (1024 * 1024)  # MB
            stats[CUPY_MEMORY] = peak_cupy_mem
            print(f"Peak cupy memory usage: {peak_cupy_mem:.1f} MB")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_pytorch_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            stats[PYTORCH_MEMORY] = peak_pytorch_memory
            print(f"Peak pytorch memory usage: {peak_pytorch_memory:.2f} MB")


def benchmark_sigkernel_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Time series length: {X.shape[1]}")
    print(f"Dyadic Order: {dyadic_order}")

    with track_peak_memory("SigKernel", stats):
        sk = signature_kernel.compute_Gram(X, X)

        if sk.shape[0] == 1 and sk.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = sk.item()

        print(f"SigKernel: \n {sk.tolist()}")

    return stats

def benchmark_ksig_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Dyadic Order: {dyadic_order}")

    with track_peak_memory("KSigPDESignatureKernel", stats):
        result = ksig_kernel(X,X)
        if result.shape[0] == 1 and result.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = result.item()
        print(f"KSigPDESignatureKernelKSigPDE computation of gram Matrix: \n {result}")

    return stats

def benchmark_powersig_on_length(X: torch.Tensor, scales: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": ORDER}

    print(f"Order: {ORDER}")
    tcge = torch.compile(tensor_compute_gram_entry)
    """Context manager to track peak CPU memory usage"""
    with track_peak_memory("PowerSig", stats):
        dX_i = torch_compute_derivative_batch(X).squeeze()
        result = tcge(dX_i, torch.clone(dX_i), scales, ORDER)
        stats[SIGNATURE_KERNEL] = result

        print(f"PowerSig computation of gram Matrix: \n {result}")

    return stats


if __name__== '__main__':
    setup_torch()

    # Setup new tracking pool allocator for cp
    cp.cuda.set_allocator(tracking_pool.malloc)

    sk_filename = "sigkernel.csv"
    ps_filename = "powersig.csv"
    ks_filename = "ksig.csv"

    sk_file_exists = os.path.isfile(sk_filename)
    ps_file_exists = os.path.isfile(ps_filename)
    ks_file_exists = os.path.isfile(ks_filename)

    with open(sk_filename, 'a', newline='') as skf, open(ps_filename, 'a', newline='') as psf, open(ks_filename, 'a', newline='') as ksf:
        writer_skf = csv.DictWriter(skf, fieldnames=CSV_FIELDS)
        writer_psf = csv.DictWriter(psf, fieldnames=CSV_FIELDS)
        writer_ksf = csv.DictWriter(ksf, fieldnames=CSV_FIELDS)

        if not sk_file_exists:
            writer_skf.writeheader()

        if not ps_file_exists:
            writer_psf.writeheader()

        if not ks_file_exists:
            writer_ksf.writeheader()

        length = 2
        scales = build_scaling_for_integration(ORDER, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), dtype=torch.float64)
        while length <= MAX_LENGTH:
            X, _ = generate_brownian_motion(length)
            if X.shape[1] < 4:
                print(f"X = {X.tolist()}")
                print(f"dX = {torch_compute_derivative_batch(X.unsqueeze(2))}")
            X = X.unsqueeze(2)

            print(f"Time series shape: {X.shape}")

            if length <= SIG_KERNEL_MAX_LENGTH:
                try:
                    stats = benchmark_sigkernel_on_length(X)
                    writer_skf.writerow(stats)
                    skf.flush()
                except OutOfMemoryError as ex:
                    print(f"SigKernel ran out of memory for time series of length {X.shape[1]}: {ex}")

            if length <= KSIG_MAX_LENGTH:
                try:
                    stats = benchmark_ksig_on_length(X)
                    writer_ksf.writerow(stats)
                    ksf.flush()
                except OutOfMemoryError as ex:
                    print(f"KSig ran out of memory for time series of length {X.shape[1]}: {ex}")
            #
            if length <= POWERSIG_MAX_LENGTH:
                try:
                    stats = benchmark_powersig_on_length(X, scales)
                    writer_psf.writerow(stats)
                    psf.flush()
                except Exception as ex:
                    print(f"PowerSig ran out of memory for time series of length {X.shape[1]}: {ex}")

            length<<=1


