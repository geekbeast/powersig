# Power-series based computation of Signature Kernels

[![CI](https://github.com/geekbeast/powersig/actions/workflows/ci.yml/badge.svg)](https://github.com/geekbeast/powersig/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/geekbeast/powersig)

Using ADM-derived Neumann series to compute signature kernels.

PowerSig ships two interchangeable backends — **JAX** and **PyTorch** — behind the
same API. Pick whichever matches the framework you already use; both produce the
same kernel values to machine precision (see [Choosing a backend](#choosing-a-backend)).

## Installation

Requires Python 3.12+. Install the extra for the backend you want — the backend
frameworks are optional dependencies, so nothing heavyweight is pulled in by default.

```bash
# JAX, CPU only
pip install "powersig[jax-cpu]"

# JAX, CUDA 13 GPU
pip install "powersig[jax-gpu]"

# PyTorch (CPU or CUDA, depending on the torch wheel you install)
pip install "powersig[torch]"
```

To install from source, use the same extras with a direct reference:

```bash
pip install "powersig[torch] @ git+https://github.com/geekbeast/powersig.git"
```

| Extra | Backend | Pulls in |
| --- | --- | --- |
| `jax-cpu` | JAX | `jax[cpu]>=0.4.34` |
| `jax-gpu` | JAX | `jax[cuda13]>=0.10.0` |
| `torch` | PyTorch | `torch>=2.5.0` |
| `cupy` / `cupy-cuda13` | CuPy | `cupy-cuda12x` / `cupy-cuda13x` |
| `all` | JAX + PyTorch + CuPy | all of the above |

For a specific PyTorch build (a CPU-only wheel, or a CUDA version other than the
PyPI default), install `torch` first from the PyTorch index and then install
PowerSig:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "powersig[torch]"
```

Importing `powersig` does not import any backend, so having only one installed is
fine — backends are resolved lazily on first use.

## Getting Started

### JAX

```python
import jax.numpy as jnp
from powersig.jax.utils import fractional_brownian_motion
from powersig.jax.algorithm import PowerSigJax

def main():
    # Generate fBM paths
    n_steps = 1000
    n_paths = 2
    hurst = 0.7

    # Generate fBM using the jax wrapper
    fbm_paths, dt = fractional_brownian_motion(
        n_steps=n_steps,
        n_paths=n_paths,
        hurst=hurst,
        dim=1
    )

    # Initialize PowerSigJax with polynomial order 8
    powersig = PowerSigJax(order=8)

    # Compute the signature kernel
    kernel_matrix = powersig(fbm_paths)

    print("Shape of fBM paths:", fbm_paths.shape)
    print("Shape of kernel matrix:", kernel_matrix.shape)
    print("\nKernel matrix:")
    print(kernel_matrix)

if __name__ == "__main__":
    main()
```

This example is also available in the repo under [examples/simple.py](examples/simple.py).

### PyTorch

```python
import torch
from powersig.torch.utils import fractional_brownian_motion
from powersig.torch.algorithm import PowerSigTorch

def main():
    # Generate fBM paths
    n_steps = 1000
    n_paths = 2
    hurst = 0.7

    # Generate fBM using the torch wrapper
    fbm_paths, dt = fractional_brownian_motion(
        n_steps=n_steps,
        n_paths=n_paths,
        hurst=hurst,
        dim=1
    )

    # Initialize PowerSigTorch with polynomial order 8
    powersig = PowerSigTorch(order=8)

    # Compute the signature kernel
    kernel_matrix = powersig(fbm_paths)

    print("Shape of fBM paths:", tuple(fbm_paths.shape))
    print("Shape of kernel matrix:", tuple(kernel_matrix.shape))
    print("\nKernel matrix:")
    print(kernel_matrix)

if __name__ == "__main__":
    main()
```

This example is also available in the repo under [examples/simple_torch.py](examples/simple_torch.py).

## Choosing a backend

The two backends expose the same surface, so switching is a matter of swapping the
import and the array type:

| | JAX | PyTorch |
| --- | --- | --- |
| Estimator | `powersig.jax.algorithm.PowerSigJax` | `powersig.torch.algorithm.PowerSigTorch` |
| fBM helper | `powersig.jax.utils.fractional_brownian_motion` | `powersig.torch.utils.fractional_brownian_motion` |
| Static kernels | `powersig.jax.static_kernels` | `powersig.torch.static_kernels` |
| Gram matrix | `ps(X)` / `ps(X, Y)` | `ps(X)` / `ps(X, Y)` |
| Single pair | `ps.compute_signature_kernel(x, y)` | `ps.compute_signature_kernel(x, y)` |
| Gradients | `jax.grad` / `jax.value_and_grad` | `torch.autograd` (`.backward()`) |

Both constructors take the same arguments:

```python
PowerSigJax(order=32, static_kernel=linear_kernel, device=None, dtype=jnp.float64)
PowerSigTorch(order=32, static_kernel=linear_kernel, device=None, dtype=torch.float64)
```

- `order` — truncation order of the power series. Higher is more accurate and more
  expensive; 8–32 is the usual range.
- `static_kernel` — the static kernel lifted to a signature kernel. `linear_kernel`
  (default) and `rbf_kernel` are provided by each backend's `static_kernels` module.
- `device` — defaults to the first available GPU, else CPU.
- `dtype` — defaults to float64. Use float32 to trade accuracy for speed.

Paths are `(batch, length, dim)` for Gram matrices and `(length, dim)` for a single
pair. The two paths in a pair need not have the same length.

Which one to pick:

- **PyTorch** if your model, data loading, or training loop is already in PyTorch —
  the kernel is differentiable through `torch.autograd`, so it drops into an
  existing training loop without a framework boundary.
- **JAX** if you want `jit`/`vmap`/`grad` composition, or are already in a JAX
  codebase.

A **CuPy** backend also exists under `powersig.cupy_backend`. It covers the forward
Gram computation only — no autodiff and no pluggable static kernel — so the JAX and
PyTorch backends are the supported choices for general use.

## Testing

```bash
pip install ".[jax-cpu,dev]"          # add "torch" for the PyTorch suite
pytest tests/test_core_jax.py         # JAX backend
pytest tests/test_core_torch.py \
       tests/test_autodiff_torch.py \
       tests/test_prefix_family_torch.py   # PyTorch backend
```

The PyTorch suite cross-checks its results against the JAX implementation, so it
needs both backends installed. CI runs both on every push and pull request.
