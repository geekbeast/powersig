import csv
import os
import time
import unittest
from contextlib import contextmanager
from operator import lshift

import psutil
import torch.cuda
from jinja2.compiler import generate
from mpmath.libmp.libintmath import powers

from benchmarks.configuration import signature_kernel, GPU_MEMORY, CPU_MEMORY, SIG_KERNEL_MAX_LENGTH, dyadic_order, \
    ksig_kernel, ORDER, SIGNATURE_KERNEL, DURATION, CSV_FIELDS, POWERSIG_MAX_LENGTH, KSIG_MAX_LENGTH, MAX_LENGTH
from benchmarks.util import generate_brownian_motion
from powersig.matrixsig import build_scaling_for_integration, tensor_compute_gram_entry
from powersig.util.series import torch_compute_derivative_batch
from tests.utils import setup_torch


@contextmanager
def track_peak_memory(stats):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    try:
        yield
    finally:
        peak_mem = (process.memory_info().rss - start_mem) / (1024 * 1024)  # MB
        stats[CPU_MEMORY] = peak_mem
        print(f"Peak memory usage: {peak_mem:.1f} MB")


def benchmark_sigkernel_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Time series length: {X.shape[1]}")
    print(f"Dyadic Order: {dyadic_order}")
    """Context manager to track peak GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    max_batch = 10
    start = time.time()

    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(stats):
        sk = signature_kernel.compute_Gram(X, X)
        stats[DURATION] = time.time() - start
        if sk.shape[0] == 1 and sk.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = sk.item()
        print(f"SigKernel computation took: {stats[DURATION]}s")
        print(f"SigKernel: \n {sk.tolist()}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        stats[GPU_MEMORY] = peak_memory
        print(f"Peak GPU memory usage: {peak_memory:.2f} MB")

    return stats

def benchmark_ksig_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Time series length: {X.shape[1]}")
    print(f"Dyadic Order: {dyadic_order}")
    """Context manager to track peak GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    max_batch = 10
    start = time.time()

    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(stats):
        result = ksig_kernel(X,X)
        stats[DURATION] = time.time() - start
        if result.shape[0] == 1 and result.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = result.item()
        print(f"KSigPDESignatureKernel computation took: {stats[DURATION]}s")
        print(f"KSigPDESignatureKernelKSigPDE computation of gram Matrix: \n {result}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        stats[GPU_MEMORY] = peak_memory
        print(f"Peak GPU memory usage: {peak_memory:.2f} MB")

    return stats

def benchmark_powersig_on_length(X: torch.Tensor, dt: float, scales: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": ORDER}

    print(f"Time series length: {X.shape[1]}")
    print(f"Order: {ORDER}")
    """Context manager to track peak GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    max_batch = 10
    start = time.time()

    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(stats):
        dX_i = torch_compute_derivative_batch(X).squeeze()
        result = tensor_compute_gram_entry(dX_i, torch.clone(dX_i), scales, ORDER)

        stats[DURATION] = time.time() - start
        stats[SIGNATURE_KERNEL] = result
        print(f"PowerSig computation took: {stats[DURATION]}s")
        print(f"PowerSig computation of gram Matrix: \n {result}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        stats[GPU_MEMORY] = peak_memory
        print(f"Peak GPU memory usage: {peak_memory:.2f} MB")

    return stats

def benchmark_sigkernel(filename="sigkernel.csv"):
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        length = 2
        while length <= SIG_KERNEL_MAX_LENGTH:
            X, _ = generate_brownian_motion(length)
            length<<=1



if __name__== '__main__':
    setup_torch()

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
            X, dt = generate_brownian_motion(length)
            X = X.unsqueeze(2)
            if length <= SIG_KERNEL_MAX_LENGTH:
                stats = benchmark_sigkernel_on_length(X)
                writer_skf.writerow(stats)

            if length <= KSIG_MAX_LENGTH:
                stats = benchmark_ksig_on_length(X)
                writer_ksf.writerow(stats)

            if length <= POWERSIG_MAX_LENGTH:
                stats = benchmark_powersig_on_length(X, dt, scales)
                writer_psf.writerow(stats)

            length<<=1


