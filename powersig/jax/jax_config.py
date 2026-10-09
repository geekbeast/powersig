"""
JAX Configuration settings for PowerSig.
Import this module before any other JAX imports.
Optimized for high-performance systems with multiple cores, high memory, and GPUs.
"""
import os
import jax

from powersig import precision as _precision

# Assume high-end hardware
CPU_COUNT = 32  # High core count
TOTAL_MEMORY_GB = 64  # High memory (64GB)

def _update_first_supported(candidates, what):
    """Apply the first config key the installed JAX actually recognises.

    jax.config keys are not stable across minor releases and jax.config.update
    raises AttributeError on an unknown one, which aborts the rest of this
    function. JAX 0.11 removed the continuous `*_effort` floats in favour of
    O0-O3 enums; PowerSig declares jax>=0.10.0, so both spellings have to work.
    """
    for key, value in candidates:
        try:
            jax.config.update(key, value)
            return key
        except (AttributeError, ValueError):
            continue
    print(f"JAX {jax.__version__}: no supported config key for {what}, using the default")
    return None


def configure_jax():
    # Precision regime: an explicit powersig.precision.configure() call wins, then POWERSIG_PRECISION in the
    # environment, then the historical default (64-bit enabled, matmul precision 'highest').
    _precision.apply_backend_import('jax')
    # jax.config.update('jax_default_dtype_bits', '64')

    # Create XLA flags for GPU optimization
    xla_flags = [
        '--xla_gpu_autotune_level=4',
        '--xla_gpu_collective_permute_decomposer_threshold=128'
    ]

    # Set the XLA flags
    os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

    # Enable optimizations for speed
    jax.config.update('jax_disable_most_optimizations', False)
    # Maximum execution-time optimization. JAX 0.11 replaced the float
    # jax_exec_time_optimization_effort (0.0-1.0) with the jax_optimization_level
    # enum (O0-O3); 1.0 was the maximum, so O3.
    _update_first_supported(
        [('jax_optimization_level', 'O3'),
         ('jax_exec_time_optimization_effort', 1.0)],
        'execution-time optimization',
    )

    # Enable and configure compilation cache
    jax.config.update('jax_enable_compilation_cache', True)
    jax.config.update('jax_compilation_cache_max_size', 2048 * 1024 * 1024)  # 2GB cache

    # Set memory fitting effort for high-memory systems. Same rename:
    # jax_memory_fitting_effort -> jax_memory_fitting_level. 0.3 was deliberately
    # low ("plenty of RAM, don't burn compile time squeezing"), and the new
    # default is O2, so O1 keeps it below default.
    _update_first_supported(
        [('jax_memory_fitting_level', 'O1'),
         ('jax_memory_fitting_effort', 0.3)],
        'memory fitting',
    )

    # Set persistent cache directory
    if not os.path.exists('/tmp/jax_cache'):
        os.makedirs('/tmp/jax_cache', exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')

    # Print configuration summary
    print("JAX configured with high-performance settings:")
    print(f"- 64-bit enabled: {jax.config.jax_enable_x64}")
    print(f"- XLA Flags: {os.environ.get('XLA_FLAGS', '')}")
    print(f"- Default matmul precision: {jax.config.jax_default_matmul_precision}")

    try:
        devices = jax.devices()
        gpu_available = any(d.platform == 'gpu' for d in devices)
        print(f"- Available devices: {devices}")
        print(f"- GPU available: {gpu_available}")
    except:
        print("- Could not detect JAX devices") 