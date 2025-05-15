import jax
import numpy as np
import torch
import jax.numpy as jnp
import cupy as cp
import os

from benchmarks.benchmark import Benchmark
from benchmarks.util import Backend
from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR,
    KSIG_CPU_RESULTS,
    KSIG_PDE_CPU_RESULTS,
    POWERSIG_CUPY_RESULTS,
    POWERSIG_TORCH_RESULTS,
    SIGKERNEL_BACKEND,
    SIGKERNEL_RESULTS,
    POWERSIG_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    POLYSIG_RESULTS,
    CSV_FIELDS,
    POLYNOMIAL_ORDER,
    polysig_sk,
    ksig_kernel,
    ksig_pde_kernel
)
import configuration
from powersig import powersig_cupy
import powersig
from powersig.util.cupy_series import cupy_compute_derivative_batch

from powersig.torch import compute_gram_entry as torch_compute_gram_entry
from powersig.jax import compute_gram_entry as jax_compute_gram_entry
import sigkernel




class SigKernelBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
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

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        return data

    def compute_signature_kernel(self, data: torch.Tensor) -> float:
        sk = self.signature_kernel.compute_Gram(data, data)
        if sk.shape[0] == 1 and sk.shape[1] == 1:
            return sk.item()
        return sk.tolist()


class PowerSigBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, POWERSIG_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.JAX_CUDA,
            name="PowerSigJax",
            debug=debug
        )
        self.powersig = None

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        if self.powersig is None:
            self.powersig = powersig.jax.PowerSigJax(POLYNOMIAL_ORDER)
        stats["order"] = POLYNOMIAL_ORDER
        # Convert torch tensor to numpy array
        X_np = data.cpu().numpy()
        
        # Convert numpy array to JAX array
        return jnp.array(X_np, device=jax.devices("cuda")[jax.device_count() - 1])
        

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
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, POWERSIG_TORCH_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.TORCH_CUDA,
            name="PowerSigTorch",
            debug=debug
        )
        self.powersig = None

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> jnp.ndarray:
        if self.powersig is None:
            self.powersig = powersig.torch.PowerSigTorch(POLYNOMIAL_ORDER)
        
        stats["order"] = POLYNOMIAL_ORDER
        return data

    def compute_signature_kernel(self, data: jnp.ndarray) -> float:
        return self.powersig.compute_signature_kernel(data,data).item()


class PolySigBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, POLYSIG_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.JAX_CUDA,
            name="PolySig",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> jnp.ndarray:
        stats["order"] = POLYNOMIAL_ORDER
        return jnp.array(data.cpu().numpy())

    def compute_signature_kernel(self, data: jnp.ndarray) -> float:
        result = polysig_sk.kernel_matrix(data, data)
        assert result.dtype == jnp.float64, "Result dtype is not float64"
        if result.shape[0] == 1 and result.shape[1] == 1:
            return float(result[0, 0])
        else:
            raise ValueError("Result is not a scalar")


class KSigBenchmark(Benchmark):
    def __init__(self, debug: bool = False, results_dir: str = BENCHMARKS_RESULTS_DIR):
        super().__init__(
            filename=os.path.join(results_dir, KSIG_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CUPY,
            name="KSig",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        return data

    def compute_signature_kernel(self, data: torch.Tensor) -> float:
        result = ksig_kernel(data, data)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return result.item()
        else:
            raise ValueError("Result is not a scalar")

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
