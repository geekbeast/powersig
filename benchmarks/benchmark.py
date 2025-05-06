import csv
import os
import time
import argparse

import cupy as cp
# Import our JAX configuration first
import powersig.jax_config
from powersig.util.cupy_series import cupy_compute_derivative_batch
from powersig.util.jax_series import jax_compute_derivative_vmap

# Configure JAX with optimal settings for benchmarking
# Using maximum speed optimization
jax_config = powersig.jax_config.configure_jax()

import jax
import numpy as np

import powersig
import jax.numpy as jnp
from contextlib import contextmanager
from operator import lshift

import psutil
import torch.cuda
from cupy.cuda.memory import OutOfMemoryError
from jinja2.compiler import generate
from mpmath.libmp.libintmath import powers

from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR,
    KSIG_BACKEND,
    KSIG_PDE_BACKEND,
    POLYNOMIAL_ORDER,
    POLYSIG_BACKEND,
    POLYSIG_MAX_LENGTH,
    POWERSIG_BACKEND,
    SIGKERNEL_BACKEND,
    SIGKERNEL_RESULTS,
    POWERSIG_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    POLYSIG_RESULTS,
    polysig_sk,
    signature_kernel, CPU_MEMORY, SIG_KERNEL_MAX_LENGTH, dyadic_order, \
    ksig_pde_kernel, ORDER, SIGNATURE_KERNEL, DURATION, CSV_FIELDS, POWERSIG_MAX_LENGTH, KSIG_MAX_LENGTH, MAX_LENGTH, \
    GPU_MEMORY, CUPY_MEMORY, ksig_kernel, NUM_PATHS, RUN_ID)
from benchmarks.util import generate_brownian_motion, TrackingMemoryPool
from powersig.matrixsig import build_scaling_for_integration, tensor_compute_gram_entry, centered_compute_gram_entry
from powersig.torch import batch_compute_gram_entry as torch_batch_compute_gram_entry, build_stencil, compute_vandermonde_vectors
from powersig.jax import batch_compute_gram_entry
from powersig.cuda import cuda_compute_gram_entry
from powersig.util.series import torch_compute_derivative_batch
from tests.utils import setup_torch

tracking_pool = TrackingMemoryPool()
# tcge = torch.compile(tensor_compute_gram_entry)
tcge = torch.compile(batch_compute_gram_entry)

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
    
    jax_cuda_avaiable = any(device.platform == 'gpu' for device in jax.devices())

    start = time.time()

    try:
        yield
    finally:
        stats[DURATION] = time.time() - start

        print(f"{backend} computation took: {stats[DURATION]}s")

        peak_cpu_mem = (process.memory_info().rss - cpu_initial_mem) / (1024 * 1024)  # MB
        stats[CPU_MEMORY] = peak_cpu_mem
        print(f"Peak CPU memory usage: {peak_cpu_mem:.1f} MB")
        
        if  cp.is_available():
            peak_cupy_mem = tracking_pool.peak_usage / (1024 * 1024)  # MB
            stats[CUPY_MEMORY] = peak_cupy_mem
            print(f"Peak cupy memory usage: {peak_cupy_mem:.1f} MB")

        if torch.cuda.is_available() and backend != POLYSIG_BACKEND:
            torch.cuda.synchronize()
            peak_pytorch_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            stats[GPU_MEMORY] = peak_pytorch_memory
            print(f"Peak pytorch memory usage: {peak_pytorch_memory:.2f} MB")
        
        # TODO: This is a hack to get the peak memory usage for PolySig, the problem is we have to reset device memory before each run of the kernel
        # and that means we have to rewarm the compilation cache to be fair.
        if jax_cuda_avaiable and backend == POLYSIG_BACKEND:
            peak_jax_memory = 0
            for device in jax.devices():
                if device.platform == 'gpu':
                    peak_jax_memory += device.memory_stats()['peak_bytes_in_use'] / (1024 ** 2)  # Convert to MB
            stats[GPU_MEMORY] = peak_jax_memory
            print(f"Peak jax memory usage: {peak_jax_memory:.2f} MB")


def benchmark_sigkernel_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Time series length: {X.shape[1]}")
    print(f"Dyadic Order: {dyadic_order}")

    with track_peak_memory(SIGKERNEL_BACKEND, stats):
        sk = signature_kernel.compute_Gram(X, X)

        if sk.shape[0] == 1 and sk.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = sk.item()

        print(f"SigKernel: \n {sk.tolist()}")

    return stats

def benchmark_ksig_pde_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Dyadic Order: {dyadic_order}")

    with track_peak_memory(KSIG_PDE_BACKEND, stats):
        result = ksig_pde_kernel(X, X)
        if result.shape[0] == 1 and result.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = result.item()
        print(f"KSigPDESignatureKernelKSigPDE computation of gram Matrix: \n {result}")

    return stats

def benchmark_ksig_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "dyadic_order": dyadic_order}

    print(f"Dyadic Order: {dyadic_order}")

    with track_peak_memory(KSIG_BACKEND, stats):
        result = ksig_kernel(X, X)
        if result.shape[0] == 1 and result.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = result.item()
        print(f"KSigSignatureKernel computation of gram Matrix: \n {result}")

    return stats

def benchmark_powersig_on_length_cp(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": POLYNOMIAL_ORDER}

    print(f"Order: {POLYNOMIAL_ORDER}")
    # Convert PyTorch tensor to NumPy array first
    X_np = X.cpu().numpy()   
    # Convert NumPy array to CuPy array
    X_cp = cp.array(X_np)
    dX_i = cupy_compute_derivative_batch(X_cp).squeeze()
    dX_i_clone = cp.copy(dX_i)
    # ds = 1 / dX_i.shape[0]
    # dt = 1 / dX_i.shape[0]
    # v_s, v_t = compute_vandermonde_vectors(ds, dt, POLYNOMIAL_ORDER, X.dtype, X.device)
    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(POWERSIG_BACKEND, stats):
        result = powersig.powersig_cupy.batch_compute_gram_entry(dX_i, dX_i_clone, None, POLYNOMIAL_ORDER).item()
        stats[SIGNATURE_KERNEL] = result

        print(f"PowerSig computation of gram Matrix: \n {result}")

    return stats

def benchmark_powersig_on_length_cuda(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": POLYNOMIAL_ORDER}

    print(f"Order: {POLYNOMIAL_ORDER}")
    # Convert PyTorch tensor to NumPy array first
    X_np = X.cpu().numpy()   
    # Convert NumPy array to CuPy array
    X_cp = cp.array(X_np)
    dX_i = cupy_compute_derivative_batch(X_cp).squeeze()
    dX_i_clone = cp.copy(dX_i)
    # ds = 1 / dX_i.shape[0]
    # dt = 1 / dX_i.shape[0]
    # v_s, v_t = compute_vandermonde_vectors(ds, dt, POLYNOMIAL_ORDER, X.dtype, X.device)
    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(POWERSIG_BACKEND, stats):
        result = cuda_compute_gram_entry(dX_i, dX_i_clone, POLYNOMIAL_ORDER).item()
        stats[SIGNATURE_KERNEL] = result

        print(f"PowerSig computation of gram Matrix: \n {result}")

    return stats

def benchmark_powersig_on_length_jax(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": POLYNOMIAL_ORDER}

    print(f"Order: {POLYNOMIAL_ORDER}")
    X_np = X.cpu().numpy()
    dX_i = jax_compute_derivative_vmap(X_np).squeeze()
    dX_i_clone = jnp.copy(dX_i)
    # ds = 1 / dX_i.shape[0]
    # dt = 1 / dX_i.shape[0]
    # v_s, v_t = compute_vandermonde_vectors(ds, dt, POLYNOMIAL_ORDER, X.dtype, X.device)
    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(POWERSIG_BACKEND, stats):
        result = powersig.jax.compute_gram_entry(dX_i, dX_i_clone, POLYNOMIAL_ORDER).item()
        stats[SIGNATURE_KERNEL] = result

        print(f"PowerSig computation of gram Matrix: \n {result}")

    return stats

def benchmark_powersig_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": POLYNOMIAL_ORDER}

    print(f"Order: {POLYNOMIAL_ORDER}")
    dX_i = torch_compute_derivative_batch(X).squeeze()
    dX_i_clone = torch.clone(dX_i)
    # ds = 1 / dX_i.shape[0]
    # dt = 1 / dX_i.shape[0]
    # v_s, v_t = compute_vandermonde_vectors(ds, dt, POLYNOMIAL_ORDER, X.dtype, X.device)
    """Context manager to track peak CPU memory usage"""
    with track_peak_memory(POWERSIG_BACKEND, stats):
        result = powersig.torch.compute_gram_entry(dX_i, dX_i_clone, POLYNOMIAL_ORDER).item()
        # result = tcge(dX_i, dX_i_clone, None, POLYNOMIAL_ORDER).item()
        stats[SIGNATURE_KERNEL] = result

        print(f"PowerSig computation of gram Matrix: \n {result}")

    return stats

def benchmark_polysig_on_length(X: torch.Tensor) -> dict[str, float]:
    stats = {"length": X.shape[1], "order": POLYNOMIAL_ORDER}

    print(f"Order: {POLYNOMIAL_ORDER}")

    # Convert PyTorch tensor to JAX array outside of tracking loop
    X_jax = jnp.array(X.cpu().numpy())
    
    with track_peak_memory(POLYSIG_BACKEND, stats):
        result = polysig_sk.kernel_matrix(X_jax, X_jax)
        assert result.dtype == jnp.float64, "Result dtype is not float64"    
        print(f"PolySig result dtype: {result.dtype}")
        if result.shape[0] == 1 and result.shape[1] == 1:
            stats[SIGNATURE_KERNEL] = float(result[0, 0])
        print(f"PolySig computation of gram Matrix: \n {result}")

    return stats


if __name__== '__main__':
    setup_torch()

    # Setup new tracking pool allocator for cp
    cp.cuda.set_allocator(tracking_pool.malloc)
    
    sk_filename = os.path.join(BENCHMARKS_RESULTS_DIR, SIGKERNEL_RESULTS)
    ps_filename = os.path.join(BENCHMARKS_RESULTS_DIR, POWERSIG_RESULTS)
    polysig_filename = os.path.join(BENCHMARKS_RESULTS_DIR, POLYSIG_RESULTS)
    ks_filename = os.path.join(BENCHMARKS_RESULTS_DIR, KSIG_RESULTS)
    kspde_filename = os.path.join(BENCHMARKS_RESULTS_DIR, KSIG_PDE_RESULTS)

    sk_file_exists = os.path.isfile(sk_filename)
    ps_file_exists = os.path.isfile(ps_filename)
    polysig_file_exists = os.path.isfile(polysig_filename)
    ks_file_exists = os.path.isfile(ks_filename)
    kspde_file_exists = os.path.isfile(kspde_filename)

    with open(sk_filename, 'a', newline='') as skf, open(ps_filename, 'a', newline='') as psf, open(kspde_filename, 'a', newline='') as ksfpde, open(ks_filename, 'a', newline='') as ksf, open(polysig_filename, 'a', newline='') as polysig:
        writer_skf = csv.DictWriter(skf, fieldnames=CSV_FIELDS)
        writer_psf = csv.DictWriter(psf, fieldnames=CSV_FIELDS)
        writer_ksf_pde = csv.DictWriter(ksfpde, fieldnames=CSV_FIELDS)
        writer_ksf = csv.DictWriter(ksf, fieldnames=CSV_FIELDS)
        writer_polysig = csv.DictWriter(polysig, fieldnames=CSV_FIELDS)

        if not sk_file_exists:
            writer_skf.writeheader()

        if not ps_file_exists:
            writer_psf.writeheader()

        if not kspde_file_exists:
            writer_ksf_pde.writeheader()

        if not ks_file_exists:
            writer_ksf.writeheader()

        if not polysig_file_exists:
            writer_polysig.writeheader()

        length = 2
        # scales = build_scaling_for_integration(POLYNOMIAL_ORDER, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), dtype=torch.float64)
        # scales = build_stencil(POLYNOMIAL_ORDER, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), dtype=torch.float64)
        while length <= MAX_LENGTH:
            X, _ = generate_brownian_motion(length,n_paths=NUM_PATHS, dim=2)
            if X.shape[1] < 4:
                print(f"X = {X.tolist()}")
                print(f"dX = {torch_compute_derivative_batch(X.unsqueeze(2))}")

            print(f"Time series shape: {X.shape}")
            for run_id in range(X.shape[0]):
                # if length <= SIG_KERNEL_MAX_LENGTH:
                #     try:
                #         stats = benchmark_sigkernel_on_length(X[run_id:run_id+1])
                #         stats[RUN_ID] = run_id
                #         writer_skf.writerow(stats)
                #         skf.flush()
                #     except OutOfMemoryError as ex:
                #         print(f"SigKernel ran out of memory for time series of length {X.shape[1]}: {ex}")

                # if length <= KSIG_MAX_LENGTH:
                #     try:
                #         stats = benchmark_ksig_pde_on_length(X[run_id:run_id+1])
                #         stats[RUN_ID] = run_id
                #         writer_ksf_pde.writerow(stats)
                #         ksfpde.flush()
                #     except OutOfMemoryError as ex:
                #         print(f"KSigPDE ran out of memory for time series of length {X.shape[1]}: {ex}")

                #     try:
                #         stats = benchmark_ksig_on_length(X[run_id:run_id+1])
                #         stats[RUN_ID] = run_id
                #         writer_ksf.writerow(stats)
                #         ksfpde.flush()
                #     except OutOfMemoryError as ex:
                #         print(f"KSig ran out of memory for time series of length {X.shape[1]}: {ex}")

                # if length <= POLYSIG_MAX_LENGTH:
                #     try:
                #         stats = benchmark_polysig_on_length(X[run_id:run_id+1])
                #         stats[RUN_ID] = run_id
                #         writer_polysig.writerow(stats)
                #         polysig.flush()
                #     except OutOfMemoryError as ex:
                #         print(f"PolySig ran out of memory for time series of length {X.shape[1]}: {ex}")

                if length <= POWERSIG_MAX_LENGTH:
                    try:
                        stats = benchmark_powersig_on_length_jax(X.to('cuda:1')[run_id:run_id+1])
                        stats[RUN_ID] = run_id
                        writer_psf.writerow(stats)
                        psf.flush()
                    except OutOfMemoryError as ex:
                        print(f"PowerSig ran out of memory for time series of length {X.shape[1]}: {ex}")
                    except Exception as ex:
                        print(f"PowerSig ran into an error for time series of length {X.shape[1]}: {ex}")

            length<<=1


