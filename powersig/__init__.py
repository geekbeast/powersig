"""
PowerSig - Efficient Computation of Signature Kernels

This package provides efficient implementations of signature kernels
using JAX, PyTorch, or CuPy backends. Install only the backend you need:

    pip install powersig[jax-cpu]    # JAX on CPU
    pip install powersig[jax-gpu]    # JAX on GPU
    pip install powersig[torch]      # PyTorch
    pip install powersig[cupy]       # CuPy
"""

import importlib as _importlib


def __getattr__(name):
    """Lazy-import backend submodules so missing optional deps don't break import."""
    _submodules = {"jax", "torch", "cupy_backend", "util"}
    if name in _submodules:
        return _importlib.import_module(f".{name}", __name__)

    # Convenience re-exports — only attempt if the backend is installed
    _lazy_imports = {
        "PowerSigJax": (".jax.algorithm", "PowerSigJax"),
        "fractional_brownian_motion": (".jax.utils", "fractional_brownian_motion"),
        "fbm": (".util.fbm_utils", "fractional_brownian_motion"),
    }
    if name in _lazy_imports:
        module_path, attr = _lazy_imports[name]
        mod = _importlib.import_module(module_path, __name__)
        return getattr(mod, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PowerSigJax",
    "fractional_brownian_motion",
    "fbm",
    "jax",
    "torch",
    "util",
    "cupy_backend",
]
