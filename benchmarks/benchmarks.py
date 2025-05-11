import jax
import torch
import jax.numpy as jnp
import cupy as cp
import os

from benchmarks.base import Benchmark
from benchmarks.util import Backend
from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR,
    KSIG_CPU_RESULTS,
    POWERSIG_TORCH_RESULTS,
    SIGKERNEL_RESULTS,
    POWERSIG_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    POLYSIG_RESULTS,
    CSV_FIELDS,
    POLYNOMIAL_ORDER,
    SIGNATURE_KERNEL,
    polysig_sk,
    ksig_kernel,
    ksig_pde_kernel
)
import configuration
from powersig import powersig_cupy
import powersig
from powersig.util.series import torch_compute_derivative_batch
from powersig.util.cupy_series import cupy_compute_derivative_batch
from powersig.util.jax_series import jax_compute_derivative_vmap
from powersig.torch import compute_gram_entry as torch_compute_gram_entry
from powersig.cuda import cuda_compute_gram_entry_cooperative
from powersig.jax import compute_gram_entry as jax_compute_gram_entry
import sigkernel




class SigKernelBenchmark(Benchmark):
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, SIGKERNEL_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.TORCH,
            name="SigKernel",
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
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, POWERSIG_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.JAX_CUDA,
            name="PowerSig",
            debug=debug
        )
        self.powersig = powersig.jax.PowerSig(POLYNOMIAL_ORDER)

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        stats["order"] = POLYNOMIAL_ORDER
        # Convert torch tensor to numpy array
        X_np = data.cpu().numpy()
        # Convert numpy array to JAX array
        return jnp.array(X_np, device=jax.devices("cuda")[1])
        

    def compute_signature_kernel(self, data) -> float:
        return torch_compute_gram_entry(data, data, POLYNOMIAL_ORDER).item()


class PowerSigCupyBenchmark(Benchmark):
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, POWERSIG_RESULTS),
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
        return powersig_cupy.compute_gram_entry(data, data, POLYNOMIAL_ORDER).item()
        

class PowerSigTorchBenchmark(Benchmark):
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, POWERSIG_TORCH_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.TORCH_CUDA,
            name="PowerSigJAX",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> jnp.ndarray:
        stats["order"] = POLYNOMIAL_ORDER
        X_np = data.cpu().numpy()
        dX_i = jax_compute_derivative_vmap(X_np).squeeze()
        return jnp.copy(dX_i)

    def compute_signature_kernel(self, data: jnp.ndarray) -> float:
        return jax_compute_gram_entry(data, data, POLYNOMIAL_ORDER).item()


class PolySigBenchmark(Benchmark):
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, POLYSIG_RESULTS),
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
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, KSIG_RESULTS),
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
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, KSIG_CPU_RESULTS),
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

class KSigPDEBenchmark(Benchmark):
    def __init__(self, debug: bool = False):
        super().__init__(
            filename=os.path.join(BENCHMARKS_RESULTS_DIR, KSIG_PDE_RESULTS),
            csv_fields=CSV_FIELDS,
            backend=Backend.CUPY,
            name="KSigPDE",
            debug=debug
        )

    def setup(self) -> None:
        pass

    def before_run(self, data: torch.Tensor, stats: dict) -> torch.Tensor:
        return data

    def compute_signature_kernel(self, data: torch.Tensor) -> float:

        result = ksig_pde_kernel(data, data)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return result.item()
        else:
            raise ValueError("Result is not a scalar")
