"""
PowerSig - Efficient Computation of Signature Kernels

This package provides efficient implementations of signature kernels
using both JAX and CuPy for GPU acceleration.
"""

# Main implementations
from . import jax
from . import powersig_cupy

# Utility functions
from .util.cupy_series import cupy_compute_derivative_batch

__all__ = [
    'jax',
    'powersig_cupy',
    'jax_compute_derivative_vmap',
    'cupy_compute_derivative_batch',
]
