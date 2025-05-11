import os
import pickle

import ksig
import torch
from sigkernel import sigkernel
from polysigkernel import SigKernel

_batch, _len_x, _len_y, _dim = 1, 100000, 100, 2
_fresh = True
torch.random.manual_seed(1)
dyadic_order = 0

ksig_static_kernel = ksig.static.kernels.LinearKernel()
POLYNOMIAL_ORDER = 8
LEVELS = 21
# Instantiate the signature kernel, which takes as input the static kernel.
ksig_pde_kernel = ksig.kernels.SignaturePDEKernel(normalize = False, static_kernel=ksig_static_kernel)
ksig_kernel = ksig.kernels.SignatureKernel(n_levels = LEVELS, order = 0, normalize = False, static_kernel=ksig_static_kernel)
polysig_sk = SigKernel(order=POLYNOMIAL_ORDER, static_kernel="linear")

RUN_ID = "run_id"
GPU_MEMORY = "gpu_memory"
CUPY_MEMORY = "cupy_memory"
CPU_MEMORY = "cpu_memory"
SIGNATURE_KERNEL = "signature_kernel"
DURATION = "duration"
LENGTH = "length"
ORDER = "order"
DYADIC_ORDER = "dyadic_order"
CSV_FIELDS = [LENGTH, RUN_ID, DURATION, GPU_MEMORY, CUPY_MEMORY, CPU_MEMORY, ORDER, DYADIC_ORDER, SIGNATURE_KERNEL ]

POLYSIG_BACKEND = "PolySig"
POWERSIG_BACKEND = "PowerSig"
KSIG_BACKEND = "KSigSignatureKernel"
KSIG_PDE_BACKEND = "KSigPDESignatureKernel"
SIGKERNEL_BACKEND = "SigKernel"

MAX_LENGTH = 1<<20
POWERSIG_MAX_LENGTH = 1<<16
SIG_KERNEL_MAX_LENGTH = 1022
KSIG_MAX_LENGTH = 1<<15

POLYSIG_MAX_LENGTH = 1000000
POWERSIG_MAX_LENGTH = 1000000
KSIG_PDE_MAX_LENGTH = 50000


NUM_PATHS = 11

# Directory paths
BENCHMARKS_RESULTS_DIR = "benchmarks/results"

# File extensions
CSV_EXTENSION = ".csv"

# Benchmark result files
POWERSIG_RESULTS = "powersig_results.csv"
POWERSIG_TORCH_RESULTS = "powersig_torch_results.csv"
POWERSIG_CUPY_RESULTS = "powersig_cupy_results.csv"
SIGKERNEL_RESULTS = "sigkernel_results.csv"
KSIG_RESULTS = "ksig_results.csv"
KSIG_CPU_RESULTS = "ksig_cpu_results.csv"
KSIG_PDE_RESULTS = "ksig_pde_results.csv"
KSIG_PDE_CPU_RESULTS = "ksig_pde_cpu_results.csv"
POLYSIG_RESULTS = "polysig_results.csv"

def get_benchmark_config(fresh=_fresh):
    if os.path.exists("tests.run"):
        return pickle.load(open("tests.run", "rb"))
    else:
        config = BenchmarkRun()
        pickle.dump(config, open("tests.run", "wb"))
        return config

class BenchmarkRun:
    def __init__(self, batch: int = _batch,
                 len_x: int = _len_x, len_y: int = _len_y, dim: int = _dim,
                 cuda: bool = False):
        self.batch = batch
        self.len_x = len_x
        self.len_y = len_y
        self.dim = dim
        self.cuda = cuda
        self.X = torch.rand((batch, len_x, dim), dtype=torch.float64) / 50000 # shape (batch,len_x,dim)
        self.Y = torch.rand((batch, len_y, dim), dtype=torch.float64)  # shape (batch,len_y,dim)
        self.Z = torch.rand((batch, len_x, dim), dtype=torch.float64)  # shape

        if self.cuda:
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
            self.Z = self.Z.cuda()
