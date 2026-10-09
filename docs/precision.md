# Precision regimes

PowerSig computes in float64 unless told otherwise: importing the JAX backend
(`powersig.jax.algorithm` or `powersig.jax.autodiff`) enables `jax_enable_x64`
and sets the default matmul precision to `highest`, and both `PowerSigJax` and
`PowerSigTorch` default to float64 arrays. Nothing in the library used to turn
TensorFloat-32 off, so a float32 run was only as exact as the caller's own
setup. `powersig.precision` makes the regime an explicit, recorded choice.

## The four switches

A float32 signature kernel is exact float32 only when all four hold:

| switch | what it prevents | set by |
|---|---|---|
| `NVIDIA_TF32_OVERRIDE=0` in the environment | cuBLAS / cuDNN rounding float32 products to TensorFloat-32 (10-bit mantissa) on Ampere and newer GPUs; read when the CUDA libraries initialise, so it must precede the first CUDA computation | `configure()` |
| `jax_enable_x64 = False` | callers' Python scalars and untyped constants promoting float32 paths to float64 (the recurrence itself keeps its inputs' dtype either way) | `configure('float32')` |
| `jax_default_matmul_precision = 'highest'` | XLA's `default` dot precision, which is TF32 on GPUs | `configure()` |
| `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `torch.set_float32_matmul_precision('highest')` | the same rounding inside the torch backend (torch's default allows TF32 for cuDNN) | `configure()` when torch is loaded, and the torch backend at import |

## Usage

```python
from powersig import precision

record = precision.configure("float32")   # before the first kernel evaluation
ps = PowerSigJax(order=8)                 # dtype now defaults to float32
```

`record` (also available at any time as `precision.record()`) holds every
switch — the environment value, the JAX flags and the torch switches — and is
what a run should store next to its results. `configure("float64")` restores
the historical regime (TF32 stays off); `configure("float32", tf32=True,
matmul_precision="high")` is the explicit opt-in to TF32.

Without code, `POWERSIG_PRECISION=float32` in the environment configures the
same regime when a backend is imported. An explicit `configure` call always
wins over the environment, and both win over the import-time default, so the
order of imports no longer matters.

An explicit `dtype=` argument to a backend class still overrides the regime's
default for that instance.

## Order of operations

Call `configure` before any CUDA work in the process. The TF32 override is read
once, when the CUDA libraries initialise; `configure` warns when it detects an
already initialised CUDA backend. Changing `jax_enable_x64` after arrays exist
is allowed by JAX but leaves those arrays in their old dtype, so switch regimes
before building paths.

## What was measured

On the CPU, a float32 `PowerSigJax(order=8)` kernel of two short random paths
returns a float32 value within 2e-8 relative of the float64 value whether
`jax_enable_x64` is on or off, and the float32 torch backend within 3e-7; the
regime tests assert float32 output dtypes and agreement with float64 to 1e-5.
The x64 flag therefore matters for what callers do around the kernel, and the
TF32 switches for what GPUs do inside it.
