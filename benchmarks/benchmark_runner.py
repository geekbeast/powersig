import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from benchmarks import generators
from benchmarks.benchmark import Benchmark

from powersig.torch.utils import fractional_brownian_motion
from benchmarks.kernel_benchmarks import (
    KSigBenchmark,
    KSigPDEBenchmark,
    PowerSigBenchmark,
    PowerSigCupyBenchmark,
    SigKernelBenchmark
)

import torch.multiprocessing as mp
# Configure JAX with optimal settings for benchmarking
# Using maximum speed optimization

import torch.cuda
import numpy as np


from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR,
    HURST,
    NUM_PATHS)

from tests.utils import setup_torch

def mp_benchmark(type: str, benchmark: Benchmark, data: torch.Tensor, hurst: float):
    print(f"Benchmarking {benchmark.name} on {type} for {data.shape[0]} rounds with length {data.shape[1]} and hurst value {hurst}")
    for run_id in range(data.shape[0]):
        benchmark.benchmark(data[run_id:run_id+1], run_id, {HURST: hurst})


if __name__== '__main__':
    print("========== Starting benchmarks! ==========")
    
    setup_torch()
    generators.set_seed(42)
    benchmark_accuracy = True
    benchmark_rough_accuracy = True
    benchmark_length = True
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
                PowerSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/accuracy",order=8,device_index=1),
            ]
            X, _ = fractional_brownian_motion(length,n_paths=NUM_PATHS, dim=2)
            for benchmark in active_benchmarks:
                print(f"Spawning {benchmark.name} for length {length}")
                p = ctx.Process(target=mp_benchmark, args=("accuracy", benchmark, X, .5))
                p.start()
                p.join()


    if benchmark_rough_accuracy:  
        for length in range(50,100,10):
            for hurst in np.logspace(-2, 0, 100) - 5e-3:
                active_benchmarks : list[Benchmark] = [
                    KSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough", levels=180),
                    KSigPDEBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    SigKernelBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    PowerSigCupyBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough"),
                    PowerSigBenchmark(debug=False,results_dir=f"{BENCHMARKS_RESULTS_DIR}/rough",order=8,device_index=1),
                ]
                num_paths = 2 # This will take forever otherwise. (2^13 - 1) * 99 = 810 K signature kernels to evaluate 
                X, _ = fractional_brownian_motion(50,n_paths=num_paths, dim=2, hurst=hurst)
                print(f"variation norm: {np.linalg.norm(X[0,1:,:].cpu().numpy()-X[0,:-1,:].cpu().numpy(),ord = 1)}")      
                for benchmark in active_benchmarks:
                    # We don't care about the multiprocessing here, we just want to run the benchmark
                    mp_benchmark("roughness", benchmark, X, hurst)
        

    if (benchmark_length):
        for length in [ 2**i for i in range(1, 21)]:
            active_benchmarks : list[Benchmark] = [
                KSigBenchmark(debug=False),
                KSigPDEBenchmark(debug=False),
                SigKernelBenchmark(debug=False),
                # PowerSigCupyBenchmark(debug=False),
                PowerSigBenchmark(debug=False,order=8,device_index=1),
            ]
            X, _ = fractional_brownian_motion(length,n_paths=NUM_PATHS, dim=2)
            if length in length_filter:
                continue
            for benchmark in active_benchmarks:
                print(f"Spawning {benchmark.name} for length {length}")
                p = ctx.Process(target=mp_benchmark, args=("length", benchmark, X, .5))
                p.start()
                p.join()
        


