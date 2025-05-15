from math import log2
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from benchmarks import generators
from benchmarks.benchmark import Benchmark
# Import our JAX configuration first

from benchmarks.generators import fractional_brownian_motion
from benchmarks.kernel_benchmarks import (
    KSigBenchmark,
    KSigPDEBenchmark,
    PolySigBenchmark,
    PowerSigBenchmark,
    PowerSigCupyBenchmark,
    PowerSigTorchBenchmark,
    SigKernelBenchmark
)

import torch.multiprocessing as mp
# Configure JAX with optimal settings for benchmarking
# Using maximum speed optimization

import torch.cuda


from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR,
    HURST,
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
    CPU_MEMORY, SIG_KERNEL_MAX_LENGTH, dyadic_order, \
    ksig_pde_kernel, ORDER, SIGNATURE_KERNEL, DURATION, CSV_FIELDS, POWERSIG_MAX_LENGTH, KSIG_MAX_LENGTH, MAX_LENGTH, \
    GPU_MEMORY, CUPY_MEMORY, ksig_kernel, NUM_PATHS, RUN_ID)

from tests.utils import setup_torch

def mp_benchmark(type: str, benchmark: Benchmark, data: torch.Tensor, hurst: float):
    print(f"Benchmarking {benchmark.name} on {type} for {data.shape[0]} rounds with length {data.shape[1]} and hurst value {hurst}")
    for run_id in range(data.shape[0]):
        benchmark.benchmark(data[run_id:run_id+1], run_id, {HURST: hurst})


if __name__== '__main__':
    print("========== Starting benchmarks! ==========")
    
    setup_torch()
    generators.set_seed(42)
    benchmark_length = True
    benchmark_accuracy = True
    benchmark_rough_accuracy = True
    ctx = mp.get_context('spawn')
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    
    length_filter = set()


    if benchmark_accuracy:
        for length in [ 2**i for i in range(1, 10)]:
            active_benchmarks : list[Benchmark] = [
                KSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy"),
                KSigPDEBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy"),
                SigKernelBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy"),
                PowerSigCupyBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy"),
                PowerSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy"),
                PolySigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy"),
            ]
            X, _ = fractional_brownian_motion(length,n_paths=NUM_PATHS, dim=2)
            for benchmark in active_benchmarks:
                print(f"Spawning {benchmark.name} for length {length}")
                p = ctx.Process(target=mp_benchmark, args=("accuracy", benchmark, X, .5))
                p.start()
                p.join()


    if benchmark_rough_accuracy:  
        for length in [ 2**i for i in range(1, 10)]:
            for hurst in [ 1/i-.0000000001 for i in range(1, 100)]:
                active_benchmarks : list[Benchmark] = [
                    KSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    KSigPDEBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    SigKernelBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    PowerSigCupyBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    PowerSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    PolySigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                ]
                num_paths = 2 # This will take forever otherwise. (2^13 - 1) * 99 = 810 K signature kernels to evaluate 
                X, _ = fractional_brownian_motion(length,n_paths=num_paths, dim=2, hurst=hurst)
                for benchmark in active_benchmarks:
                    print(f"Spawning {benchmark.name} for length {length} and hurst {hurst}")
                    p = ctx.Process(target=mp_benchmark, args=("roughness", benchmark, X, hurst))
                    p.start()
                    p.join()

    if (benchmark_length):
        for length in [ 2**i for i in range(1, 20)]:
            active_benchmarks : list[Benchmark] = [
                KSigBenchmark(debug=False),
                KSigPDEBenchmark(debug=False),
                SigKernelBenchmark(debug=False),
                PowerSigCupyBenchmark(debug=False),
                PowerSigBenchmark(debug=False),
                PolySigBenchmark(debug=False),
            ]
            num_paths = max(1,min(10, 21 - log2(length))) # Longer paths have less variance so we need less samples.
            X, _ = fractional_brownian_motion(length,n_paths=num_paths, dim=2)
            if length in length_filter:
                continue
            for benchmark in active_benchmarks:
                print(f"Spawning {benchmark.name} for length {length}")
                p = ctx.Process(target=mp_benchmark, args=("length", benchmark, X, .5))
                p.start()
                p.join()
        


