"""Numerical precision regimes for every PowerSig backend.

PowerSig historically computed in float64: importing the JAX backend turns on
``jax_enable_x64`` and sets the default matmul precision to ``highest``, and both
``PowerSigJax`` and ``PowerSigTorch`` default to float64 arrays.  Running the
kernels in float32 *reliably* takes four settings at once, and until now every
caller had to re-implement them outside PowerSig:

1. ``NVIDIA_TF32_OVERRIDE=0`` in the process environment before the CUDA
   libraries initialise, so cuBLAS and cuDNN cannot round float32 products to
   TensorFloat-32 (a 10-bit mantissa) on Ampere and newer GPUs.
2. ``jax_enable_x64`` off, so Python scalars and untyped constants created by
   callers do not promote float32 paths to float64.  (The kernel recurrence
   itself keeps the dtype of its inputs either way.)
3. ``jax_default_matmul_precision = 'highest'``, so XLA dot products run in
   true float32 rather than the TF32 ``default``.
4. The PyTorch switches ``torch.backends.cuda.matmul.allow_tf32``,
   ``torch.backends.cudnn.allow_tf32`` and ``torch.set_float32_matmul_precision``.

``configure('float32')`` applies all four, makes float32 the default dtype of
the backend classes, and returns the record a run should store for provenance;
``configure('float64')`` restores the historical regime.  Setting
``POWERSIG_PRECISION=float32`` in the environment has the same effect without
code (it is read when a backend is imported).  Call ``configure`` before the
first kernel evaluation: the TF32 override is read when CUDA initialises, and
the x64 flag must not change underneath arrays that already exist.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

ENV_VAR = "POWERSIG_PRECISION"
TF32_ENV_VAR = "NVIDIA_TF32_OVERRIDE"
REGIMES = ("float32", "float64")
_ALIASES = {"float32": "float32", "f32": "float32", "fp32": "float32", "32": "float32", "single": "float32",
            "float64": "float64", "f64": "float64", "fp64": "float64", "64": "float64", "double": "float64"}
# JAX matmul precision -> torch float32 matmul precision when TF32 is allowed.
_TORCH_MATMUL = {"highest": "highest", "float32": "highest", "high": "high", "tensorfloat32": "high"}

_state = {"dtype": None, "tf32": None, "matmul_precision": None}


def active():
    """The explicitly configured regime ('float32' or 'float64'), or None before ``configure`` is called."""
    return _state["dtype"]


def normalize(dtype):
    """'float32' or 'float64' from a regime name, a bit width, or a numpy / JAX / torch dtype."""
    if isinstance(dtype, bool) or dtype is None:
        raise ValueError(f"unsupported precision regime {dtype!r}; expected float32 or float64")
    if isinstance(dtype, int):
        text = f"float{dtype}"
    elif isinstance(dtype, str):
        text = dtype
    else:
        try:
            text = np.dtype(dtype).name
        except TypeError:
            text = str(dtype)
    key = text.strip().lower().rsplit(".", 1)[-1]
    if key not in _ALIASES:
        raise ValueError(f"unsupported precision regime {dtype!r}; expected float32 or float64")
    return _ALIASES[key]


def _cuda_initialised():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return True
    if "jax" in sys.modules:
        try:
            from jax._src import xla_bridge
            return bool(getattr(xla_bridge, "_backends", None))
        except Exception:  # private API; absence just means "unknown"
            return False
    return False


def _apply_jax(regime, matmul_precision):
    try:
        import jax
    except ImportError:
        return False
    jax.config.update("jax_enable_x64", regime == "float64")
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    return True


def _apply_torch(tf32, matmul_precision):
    """Apply the torch switches if torch is loaded.  A torch that is installed but not imported is left
    alone (importing it only to flip flags is a heavy side effect); PowerSig's torch backend applies the
    active regime when it is imported, see ``apply_backend_import``."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision(_TORCH_MATMUL.get(matmul_precision, "high") if tf32 else "highest")
    return True


def configure(dtype="float64", *, tf32=False, matmul_precision="highest"):
    """Make every PowerSig backend compute in ``dtype``; TF32 is off unless ``tf32=True``.

    Returns ``record()``.  Safe to call repeatedly; call it before the first kernel evaluation.
    """
    regime = normalize(dtype)
    tf32 = bool(tf32)
    desired = "1" if tf32 else "0"
    if os.environ.get(TF32_ENV_VAR) != desired:
        if _cuda_initialised():
            warnings.warn(f"{TF32_ENV_VAR} is read when the CUDA libraries initialise and a CUDA backend is already "
                          "initialised in this process; the new value may not take effect until restart",
                          RuntimeWarning, stacklevel=2)
        os.environ[TF32_ENV_VAR] = desired
    _apply_jax(regime, matmul_precision)
    _apply_torch(tf32, matmul_precision)
    _state.update(dtype=regime, tf32=tf32, matmul_precision=matmul_precision)
    return record()


def apply_backend_import(backend):
    """Called by a backend module at import.  An explicit regime wins; otherwise ``POWERSIG_PRECISION`` in
    the environment configures the full regime; otherwise the historical default (JAX: x64 on and matmul
    ``highest``; torch: untouched).  Returns which of the three applied."""
    if active() is not None:
        if backend == "jax":
            _apply_jax(_state["dtype"], _state["matmul_precision"])
        elif backend == "torch":
            _apply_torch(_state["tf32"], _state["matmul_precision"])
        return "explicit"
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        configure(env)
        return "environment"
    if backend == "jax":
        _apply_jax("float64", "highest")
    return "default"


def default_dtype(backend):
    """The active regime's dtype for ``backend`` ('jax', 'torch' or 'numpy'); float64 when nothing is configured."""
    regime = active() or "float64"
    if backend == "jax":
        import jax.numpy as jnp
        return jnp.float32 if regime == "float32" else jnp.float64
    if backend == "torch":
        import torch
        return torch.float32 if regime == "float32" else torch.float64
    if backend == "numpy":
        return np.float32 if regime == "float32" else np.float64
    raise ValueError(f"unknown backend {backend!r}")


def record():
    """The current precision switches of this process, for run provenance (None = backend not loaded)."""
    out = {"regime": active(), "tf32": _state["tf32"], "matmul_precision": _state["matmul_precision"],
           TF32_ENV_VAR: os.environ.get(TF32_ENV_VAR), ENV_VAR: os.environ.get(ENV_VAR)}
    jax = sys.modules.get("jax")
    out["jax_enable_x64"] = bool(jax.config.jax_enable_x64) if jax is not None else None
    precision = jax.config.jax_default_matmul_precision if jax is not None else None
    out["jax_default_matmul_precision"] = None if precision is None else str(precision)
    torch = sys.modules.get("torch")
    out["float32_matmul_precision"] = torch.get_float32_matmul_precision() if torch is not None else None
    out["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32) if torch is not None else None
    out["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32) if torch is not None else None
    return out


def reset():
    """Forget the explicit regime (test helper; does not touch the backends' flags or the environment)."""
    _state.update(dtype=None, tf32=None, matmul_precision=None)
