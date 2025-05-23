import jax
import numpy as np
import torch
import jax.numpy as jnp
import cupy as cp
import os
import ksig
from benchmarks.benchmark import Benchmark
from benchmarks.util import Backend
from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR,
    KSIG_CPU_RESULTS,
    KSIG_PDE_CPU_RESULTS,
    LEVELS,
    ORDER,
    POWERSIG_CUPY_RESULTS,
    POWERSIG_TORCH_RESULTS,
    SIGKERNEL_BACKEND,
    SIGKERNEL_RESULTS,
    POWERSIG_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    CSV_FIELDS,
    POLYNOMIAL_ORDER,
    ksig_pde_kernel
)
import configuration
import powersig
from powersig.cupy_backend.cupy_series import cupy_compute_derivative_batch

import sigkernel

from powersig.jax.algorithm import PowerSigJax




class SigKernelBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR, device_index: int = -1):
        super().__init__(
            filename=os.path.join(results_dir, SIGKERNEL_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.TORCH_CUDA,
            name=SIGKERNEL_BACKEND,
            debug=debug
        )
        self.static_kernel = sigkernel.LinearKernel()
        self.dyadic_order = configuration.dyadic_order
        self.signature_kernel = sigkernel.SigKernel(self.static_kernel, self.dyadic_order)
        self.device_index = device_index
        self.device = None

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        if self.device is None:
            if torch.cuda.is_available():
                device_idx = self.device_index if self.device_index != -1 else torch.cuda.device_count() - 1
                self.device = torch.device(f'cuda:{device_idx}')
            else:
                self.device = torch.device('cpu')

        return data

    def compute_signature_kernel(self, data: torch.Tensor) -> float:
        sk = self.signature_kernel.compute_Gram(data, data)
        if sk.shape[0] == 1 and sk.shape[1] == 1:
            return sk.item()
        return sk.tolist()


class PowerSigBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR, file: str = POWERSIG_RESULTS, order: int = POLYNOMIAL_ORDER, device_index: int = -1):
        super().__init__(
            filename=os.path.join(results_dir, file),
            csv_fields=CSV_FIELDS,
            backend=Backend.JAX_CUDA,
            name="PowerSigJax",
            debug=debug
        )
        self.powersig = None
        self.order = order
        self.device_index = device_index
        self.device = None

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        if self.powersig is None:
            self.powersig = PowerSigJax(self.order)
        
        if self.device is None:
            devices = jax.devices("cuda")
            self.device = devices[self.device_index if self.device_index != -1 else jax.device_count() - 1]

        stats["order"] = self.order
        # Convert torch tensor to numpy array
        X_np = data.cpu().numpy()
        
        # Convert numpy array to JAX array
        return jnp.array(X_np, device=self.device)
        

    def compute_signature_kernel(self, data) -> float:
        if data.shape[1] > 1024:
            return self.powersig.compute_signature_kernel_chunked(data, data).item()
        else:
            return self.powersig.compute_signature_kernel(data, data).item()


class PowerSigCupyBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, POWERSIG_CUPY_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CUPY,
            name="PowerSigCuPy",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> cp.ndarray:
        stats["order"] = POLYNOMIAL_ORDER
        X_np = data.cpu().numpy()
        X_cp = cp.array(X_np)
        dX_i = cupy_compute_derivative_batch(X_cp).squeeze()
        return cp.copy(dX_i)

    def compute_signature_kernel(self, data: cp.ndarray) -> float:
        return powersig_cupy.batch_compute_gram_entry(data, data, POLYNOMIAL_ORDER).item()
        

class PowerSigTorchBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR, device_index: int = -1):
        super().__init__(
            filename=os.path.join(results_dir, POWERSIG_TORCH_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.TORCH_CUDA,
            name="PowerSigTorch",
            debug=debug
        )
        self.powersig = None
        self.device_index = device_index
        self.device = None

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> jnp.ndarray:
        if self.powersig is None:
            self.powersig = powersig.torch.PowerSigTorch(POLYNOMIAL_ORDER)
        
        if self.device is None:
            assert torch.cuda.is_available(), "CUDA is not available"
            device_idx = self.device_index if self.device_index != -1 else torch.cuda.device_count() - 1
            self.device = torch.device(f'cuda:{device_idx}')

        stats["order"] = POLYNOMIAL_ORDER
        return data

    def compute_signature_kernel(self, data: jnp.ndarray) -> float:
        return self.powersig.compute_signature_kernel(data,data).item()


ksig_static_kernel = ksig.static.kernels.LinearKernel()
class KSigBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR, levels = LEVELS):
        super().__init__(
            filename=os.path.join(results_dir, KSIG_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CUPY,
            name="KSig",
            debug=debug
        )
        self.levels = levels
        self.ksig_kernel = None

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        stats[ORDER] = self.levels
        return data

    def compute_signature_kernel(self, data: torch.Tensor) -> float:
        if self.ksig_kernel == None:
            self.ksig_kernel = ksig.kernels.SignatureKernel(n_levels = self.levels, order = 0, normalize = False, static_kernel=ksig_static_kernel)

        result = self.ksig_kernel(data, data)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return result.item()
        else:
            raise ValueError("Result is not a scalar")

# KSigCPU is not support at the moment.
class KSigCPUBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, KSIG_CPU_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CPU,
            name="KSigCPU",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> np.ndarray:
        return data.cpu().numpy()

    def compute_signature_kernel(self, data: np.ndarray) -> float:
        result = ksig_kernel(data, data)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return result.item()
        else:
            raise ValueError("Result is not a scalar")

class KSigPDEBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, KSIG_PDE_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CUPY,
            name="KSigPDE",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        return cp.array(data.cpu().numpy())

    def compute_signature_kernel(self, data: cp.ndarray) -> float:
        result = ksig_pde_kernel(data, data)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return result.item()
        else:
            raise ValueError("Result is not a scalar")


class KSigPDECPUBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, KSIG_PDE_CPU_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CPU,
            name="KSigPDE_CPU",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> cp.ndarray:
        return cp.array(data.cpu().numpy())

    def compute_signature_kernel(self, data: cp.ndarray) -> float:
        result = ksig_pde_kernel(data, data)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return result.item()
        else:
            raise ValueError("Result is not a scalar")
