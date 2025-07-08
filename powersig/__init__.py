"""
PowerSig - Efficient Computation of Signature Kernels

This package provides efficient implementations of signature kernels
using both JAX and CuPy for GPU acceleration.
"""

# Import submodules first
from . import jax
from . import util

# Main implementations
from .jax.algorithm import PowerSigJax
from .jax.utils import fractional_brownian_motion

# Utility functions
from .util.fbm_utils import fractional_brownian_motion as fbm

__all__ = [
    'PowerSigJax',
    'fractional_brownian_motion',
    'fbm',
    'jax',
    'util'
]
